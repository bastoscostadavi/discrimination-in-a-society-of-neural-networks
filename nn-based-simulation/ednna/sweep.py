"""Phase-diagram sweeps over the ``(d, f_d)`` plane.

A sweep evaluates the five order parameters on a grid of prejudice field
``d`` and fraction of prejudiced agents ``f_d``.  Grid points are flattened,
split into batches, and each batch is run as a single vectorized
:class:`~ednna.society.SocietyBatch` inside a worker process.  Results are
cached in ``data/`` keyed by a hash of the configuration, so re-plotting never
re-simulates.
"""

from __future__ import annotations

# Set before numpy is imported in *worker* processes (macOS spawns, so each
# worker imports this module fresh).  The inner loop is memory-bandwidth bound
# and single-threaded; letting each of ten workers spin up its own BLAS pool
# only causes contention.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .config import ModelConfig, SweepConfig  # noqa: E402
from .order_params import (  # noqa: E402
    ORDER_PARAM_NAMES, balance, correlations, measure, overlaps, trust,
)
from .society import SocietyBatch  # noqa: E402

__all__ = [
    "sweep", "sweep_shard", "merge_sweep_shards", "DATA_DIR", "cache_path",
    "shard_cache_path", "snapshot_cache_dir",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _ensure_writable(directory):
    """Create ``directory``, but refuse to resurrect a vanished tree.

    Package paths are resolved at import time.  If the project directory is
    renamed or moved while a multi-hour sweep is running, a plain
    ``mkdir(parents=True)`` silently recreates the old tree and writes the
    results into a directory nobody is looking at.  Requiring the parent to
    still exist turns that into a loud failure at the end of the run --- which
    is bad, but far better than a quiet one.
    """
    if not directory.parent.is_dir():
        raise FileNotFoundError(
            f"{directory.parent} no longer exists: the project directory was "
            f"probably moved or renamed after this run started. Results are "
            f"still in memory but cannot be cached to the original path; "
            f"re-run from the new location."
        )
    directory.mkdir(exist_ok=True)


def cache_path(model, sweep_cfg, tag=""):
    """Deterministic cache filename for a (model, sweep) pair."""
    payload = json.dumps(
        {"model": asdict(model), "sweep": asdict(sweep_cfg), "tag": tag},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    label = tag or f"P{model.n_issues}"
    return DATA_DIR / f"sweep_{label}_{sweep_cfg.n_d}x{sweep_cfg.n_fd}_{digest}.npz"


def shard_cache_path(model, sweep_cfg, tag, shard_index, n_shards):
    """Path for one resumable piece of a sweep's standard cache."""
    path = cache_path(model, sweep_cfg, tag)
    return path.with_name(
        f"{path.stem}.part-{shard_index:03d}-of-{n_shards:03d}.npz"
    )


def snapshot_cache_dir(model, sweep_cfg, tag=""):
    """Directory containing batch-wise lightweight final-state snapshots."""
    path = cache_path(model, sweep_cfg, tag)
    return path.with_suffix("").with_name(f"{path.stem}.snapshots")


def _atomic_savez(path, **arrays):
    """Write an npz without leaving a valid-looking partial file on failure."""
    _ensure_writable(path.parent)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
        ) as tmp:
            tmp_name = tmp.name
            np.savez_compressed(tmp, **arrays)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_text(path, text):
    """Atomically write a small UTF-8 metadata file."""
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(text)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _prepare_snapshot_dir(model, sweep_cfg, tag, n_total):
    """Create and validate the manifest for a snapshot chunk directory."""
    _ensure_writable(DATA_DIR)
    directory = snapshot_cache_dir(model, sweep_cfg, tag)
    _ensure_writable(directory)
    manifest = {
        "schema_version": 1,
        "description": "Lightweight final-state analysis snapshots",
        "model": asdict(model),
        "sweep": asdict(sweep_cfg),
        "tag": tag,
        "n_total": int(n_total),
        "storage_dtype": "float32",
        "arrays": {
            "w": "(society, agent, dimension): final opinion weights",
            "eta": "(society, receiver, emitter): final directed trust",
            "X": "(society, issue, dimension): agenda vectors",
            "discriminates": "(society, agent): biased-agent mask",
            "d": "(society,): discrimination strength",
            "fd": "(society,): biased fraction",
        },
        "class_of": [0] * (model.n_agents // 2)
        + [1] * (model.n_agents - model.n_agents // 2),
    }
    encoded = json.dumps(manifest, sort_keys=True, indent=2, default=str) + "\n"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != json.loads(encoded):
            raise ValueError(f"snapshot manifest mismatch: {manifest_path}")
    else:
        _atomic_write_text(manifest_path, encoded)
    return directory


def _snapshot_chunk_path(directory, start, stop):
    return directory / f"batch-{start:07d}-{stop:07d}.npz"


def _make_jobs(model, sweep_cfg):
    """Return axes and globally indexed batches for a complete sweep."""
    d_axis, fd_axis = sweep_cfg.grids()
    D, F = np.meshgrid(d_axis, fd_axis)  # (n_fd, n_d)
    d_flat = np.repeat(D.ravel(), sweep_cfg.n_repeats)
    fd_flat = np.repeat(F.ravel(), sweep_cfg.n_repeats)
    n_total = d_flat.size

    jobs = []
    model_dict = asdict(model)
    for start in range(0, n_total, sweep_cfg.batch_size):
        stop = min(start + sweep_cfg.batch_size, n_total)
        job = (
            model_dict,
            d_flat[start:stop],
            fd_flat[start:stop],
            sweep_cfg.seed + start,
        )
        jobs.append((start, stop, job))
    return d_axis, fd_axis, n_total, jobs


def _write_snapshot_chunk(path, start, stop, seed, result, clips, snapshot):
    arrays = {
        "start": np.asarray(start, dtype=np.int64),
        "stop": np.asarray(stop, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "n_psd_clips": np.asarray(clips, dtype=np.int64),
    }
    arrays.update(result)
    arrays.update(snapshot)
    _atomic_savez(path, **arrays)


def _load_snapshot_chunk(path, start, stop, model):
    """Load cached measurements while validating the retained final state."""
    n = stop - start
    with np.load(path) as chunk:
        if int(chunk["start"]) != start or int(chunk["stop"]) != stop:
            raise ValueError(f"snapshot batch range mismatch: {path}")
        expected = {
            "w": (n, model.n_agents, model.n_dim),
            "eta": (n, model.n_agents, model.n_agents),
            "X": (n, model.n_issues, model.n_dim),
            "discriminates": (n, model.n_agents),
            "d": (n,),
            "fd": (n,),
        }
        for name, shape in expected.items():
            if chunk[name].shape != shape:
                raise ValueError(
                    f"snapshot {name} has shape {chunk[name].shape}, expected {shape}: {path}"
                )
        result = {
            name: np.asarray(chunk[name], dtype=np.float64)
            for name in ORDER_PARAM_NAMES
        }
        clips = int(chunk["n_psd_clips"])
    return result, clips


def _run_snapshot_jobs(
    indexed_jobs, model, sweep_cfg, tag, n_total, *, verbose, label
):
    """Run missing snapshot batches and reuse completed atomic chunks."""
    directory = _prepare_snapshot_dir(model, sweep_cfg, tag, n_total)
    pieces = [None] * len(indexed_jobs)
    clips = 0
    missing = []
    cached_societies = 0
    for position, (start, stop, job) in enumerate(indexed_jobs):
        path = _snapshot_chunk_path(directory, start, stop)
        if path.exists():
            pieces[position], cached_clips = _load_snapshot_chunk(
                path, start, stop, model
            )
            clips += cached_clips
            cached_societies += stop - start
        else:
            missing.append((position, start, stop, job, path))

    if verbose:
        print(
            f"[{label}] snapshots -> {directory.name}; "
            f"resuming {cached_societies}/{sum(b-a for a, b, _ in indexed_jobs)} societies"
        )
    if not missing:
        return pieces, clips

    t0 = time.time()
    missing_total = sum(stop - start for _, start, stop, _, _ in missing)
    completed = 0

    def keep(item, output):
        nonlocal clips, completed
        position, start, stop, job, path = item
        result, n_clip, snapshot = output
        _write_snapshot_chunk(path, start, stop, job[3], result, n_clip, snapshot)
        pieces[position] = result
        clips += n_clip
        completed += stop - start
        if verbose:
            rate = completed / max(time.time() - t0, 1e-9)
            eta = (missing_total - completed) / max(rate, 1e-9)
            print(
                f"\r[{label}] {completed}/{missing_total} new societies "
                f"({rate:.1f}/s, eta {eta/60:.1f} min)",
                end="", flush=True,
            )

    if sweep_cfg.n_workers > 1 and len(missing) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(sweep_cfg.n_workers, len(missing))) as pool:
            outputs = pool.imap(_run_batch_with_snapshot, [item[3] for item in missing])
            for item, output in zip(missing, outputs):
                keep(item, output)
    else:
        for item in missing:
            keep(item, _run_batch_with_snapshot(item[3]))
    if verbose:
        print()
    return pieces, clips


def _simulate_batch(job, save_snapshot=False):
    """Simulate and measure one batch, optionally retaining its final state."""
    model_dict, d_values, fd_values, seed = job
    model = ModelConfig(**model_dict)
    batch = SocietyBatch(
        n_agents=model.n_agents,
        n_dim=model.n_dim,
        n_issues=model.n_issues,
        d=np.asarray(d_values),
        f_d=np.asarray(fd_values),
        case=model.case,
        seed=seed,
        literal_draft_sign=model.literal_draft_sign,
        dtype=model.numpy_dtype(),
        shared_schedule=model.shared_schedule,
    )
    batch.run(model.n_steps())
    if save_snapshot:
        rho = overlaps(batch)
        eta = trust(batch)
        out = correlations(
            batch, rho=rho, eta=eta, class_indicator=model.class_indicator,
            literal_norm=model.literal_norm,
        )
        out.update(balance(batch, rho=rho, eta=eta))
        snapshot = {
            "w": np.moveaxis(batch.w, 1, 0).astype(np.float32),
            "eta": eta.astype(np.float32),
            "X": np.moveaxis(batch.X, 1, 0).astype(np.float32),
            "discriminates": np.ascontiguousarray(batch.discriminates.T),
            "d": np.asarray(batch.d, dtype=np.float32),
            "fd": np.asarray(batch.f_d, dtype=np.float32),
        }
    else:
        out = measure(
            batch, class_indicator=model.class_indicator,
            literal_norm=model.literal_norm,
        )
        snapshot = None
    result = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    return result, batch.n_psd_clips, snapshot


def _run_batch(job):
    """Simulate one batch of societies and measure them. Runs in a worker."""
    result, clips, _ = _simulate_batch(job, save_snapshot=False)
    return result, clips


def _run_batch_with_snapshot(job):
    """Worker entry point that also returns a float32 final-state snapshot."""
    return _simulate_batch(job, save_snapshot=True)


#: The balances were called ``B_I`` and ``B_A`` when the long sweeps were run, after
#: the older names for the two sectors.  They are now named for the pairwise quantity
#: each is built from, and caches written under the old names are still readable: the
#: 200x200 grids cost about 45 minutes each and the arrays themselves did not change.
LEGACY_KEYS = {"B_I": "B_rho", "B_A": "B_eta"}


def sweep(
    model, sweep_cfg=None, tag="", use_cache=True, verbose=True,
    save_snapshots=False,
):
    """Run (or load) a ``(d, f_d)`` sweep.

    Returns a dict with one ``(n_fd, n_d)`` array per order parameter, plus the
    ``d`` and ``fd`` axes.  Rows are indexed by ``f_d`` and columns by ``d``, so
    the arrays are ready for ``imshow`` with ``origin="lower"``, which draws row 0
    at the bottom so that ``f_d`` increases upwards.
    """
    sweep_cfg = sweep_cfg or SweepConfig()
    path = cache_path(model, sweep_cfg, tag)
    if use_cache and path.exists() and not save_snapshots:
        with np.load(path) as z:
            result = {LEGACY_KEYS.get(k, k): z[k] for k in z.files}
        if verbose:
            print(f"[sweep] loaded cache {path.name}")
        return result

    d_axis, fd_axis, n_total, indexed_jobs = _make_jobs(model, sweep_cfg)
    jobs = [job for _, _, job in indexed_jobs]

    if verbose:
        steps = model.n_steps()
        print(
            f"[sweep] {n_total} societies of N={model.n_agents} "
            f"(P={model.n_issues}, alpha={model.alpha:.3g}), "
            f"{steps:,} interactions each, {len(jobs)} batches x {sweep_cfg.n_workers} workers"
        )

    t0 = time.time()
    if save_snapshots:
        pieces, clips = _run_snapshot_jobs(
            indexed_jobs, model, sweep_cfg, tag, n_total,
            verbose=verbose, label="sweep",
        )
    else:
        pieces = [None] * len(jobs)
        clips = 0
        if sweep_cfg.n_workers > 1 and len(jobs) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(sweep_cfg.n_workers, len(jobs))) as pool:
                for i, (res, n_clip) in enumerate(pool.imap(_run_batch, jobs)):
                    pieces[i] = res
                    clips += n_clip
                    if verbose:
                        done = sum(len(j[1]) for j in jobs[: i + 1])
                        rate = done / max(time.time() - t0, 1e-9)
                        eta = (n_total - done) / max(rate, 1e-9)
                        print(
                            f"\r[sweep] {done}/{n_total} societies "
                            f"({rate:.1f}/s, eta {eta/60:.1f} min)",
                            end="",
                            flush=True,
                        )
        else:
            for i, job in enumerate(jobs):
                res, n_clip = _run_batch(job)
                pieces[i] = res
                clips += n_clip
    if verbose:
        print(f"\r[sweep] done in {(time.time()-t0)/60:.2f} min; PSD clips: {clips:,}")

    shape = (sweep_cfg.n_fd, sweep_cfg.n_d, sweep_cfg.n_repeats)
    result = {"d": d_axis, "fd": fd_axis}
    for name in ORDER_PARAM_NAMES:
        flat = np.concatenate([p[name] for p in pieces])
        result[name] = flat.reshape(shape).mean(axis=2)

    _atomic_savez(path, **result)
    if verbose:
        print(f"[sweep] cached -> {path.name}")
    return result


def sweep_shard(
    model, sweep_cfg, tag="", *, shard_index, n_shards, use_cache=True, verbose=True,
    save_snapshots=False,
):
    """Run one batch-aligned shard of a sweep and cache its flat results.

    Shards preserve the same batches and seeds as :func:`sweep`, so merging them
    is numerically identical to an unsharded run.  A completed standard cache
    takes precedence, followed by an already completed shard cache.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    if not 0 <= shard_index < n_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < n_shards")

    final_path = cache_path(model, sweep_cfg, tag)
    part_path = shard_cache_path(model, sweep_cfg, tag, shard_index, n_shards)
    if use_cache and final_path.exists() and not save_snapshots:
        if verbose:
            print(f"[sweep shard] complete cache already exists: {final_path.name}")
        return final_path
    if use_cache and part_path.exists() and not save_snapshots:
        if verbose:
            print(f"[sweep shard] loaded completed shard {part_path.name}")
        return part_path

    _, _, n_total, all_jobs = _make_jobs(model, sweep_cfg)
    if n_shards > len(all_jobs):
        raise ValueError(
            f"n_shards={n_shards} exceeds the {len(all_jobs)} available batches"
        )
    boundaries = np.linspace(0, len(all_jobs), n_shards + 1, dtype=int)
    selected = all_jobs[boundaries[shard_index]:boundaries[shard_index + 1]]
    jobs = [job for _, _, job in selected]
    n_selected = sum(stop - start for start, stop, _ in selected)

    if verbose:
        print(
            f"[sweep shard] {shard_index + 1}/{n_shards}: {n_selected}/{n_total} "
            f"societies in {len(jobs)} batches x {sweep_cfg.n_workers} workers"
        )

    t0 = time.time()
    if save_snapshots:
        pieces, clips = _run_snapshot_jobs(
            selected, model, sweep_cfg, tag, n_total,
            verbose=verbose, label="sweep shard",
        )
    else:
        pieces = [None] * len(jobs)
        clips = 0
        if sweep_cfg.n_workers > 1 and len(jobs) > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(sweep_cfg.n_workers, len(jobs))) as pool:
                for i, (res, n_clip) in enumerate(pool.imap(_run_batch, jobs)):
                    pieces[i] = res
                    clips += n_clip
                    if verbose:
                        done = sum(stop - start for start, stop, _ in selected[: i + 1])
                        rate = done / max(time.time() - t0, 1e-9)
                        eta = (n_selected - done) / max(rate, 1e-9)
                        print(
                            f"\r[sweep shard] {done}/{n_selected} societies "
                            f"({rate:.1f}/s, eta {eta/60:.1f} min)",
                            end="", flush=True,
                        )
        else:
            for i, job in enumerate(jobs):
                pieces[i], n_clip = _run_batch(job)
                clips += n_clip

    indices = np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop, _ in selected]
    )
    arrays = {
        "indices": indices,
        "n_total": np.asarray(n_total, dtype=np.int64),
        "n_shards": np.asarray(n_shards, dtype=np.int64),
        "shard_index": np.asarray(shard_index, dtype=np.int64),
    }
    for name in ORDER_PARAM_NAMES:
        arrays[name] = np.concatenate([piece[name] for piece in pieces])
    _atomic_savez(part_path, **arrays)
    if verbose:
        print(
            f"\r[sweep shard] done in {(time.time()-t0)/60:.2f} min; "
            f"PSD clips: {clips:,}"
        )
        print(f"[sweep shard] cached -> {part_path.name}")
    return part_path


def merge_sweep_shards(model, sweep_cfg, tag="", *, n_shards, verbose=True):
    """Validate and merge all shard caches into the standard sweep cache."""
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    final_path = cache_path(model, sweep_cfg, tag)
    _, _, n_total, _ = _make_jobs(model, sweep_cfg)
    values = {
        name: np.empty(n_total, dtype=np.float64) for name in ORDER_PARAM_NAMES
    }
    seen = np.zeros(n_total, dtype=bool)

    for shard_index in range(n_shards):
        part_path = shard_cache_path(
            model, sweep_cfg, tag, shard_index, n_shards
        )
        if not part_path.exists():
            raise FileNotFoundError(f"missing sweep shard: {part_path}")
        with np.load(part_path) as part:
            if int(part["n_total"]) != n_total:
                raise ValueError(f"wrong total size in {part_path.name}")
            if int(part["n_shards"]) != n_shards:
                raise ValueError(f"wrong shard count in {part_path.name}")
            if int(part["shard_index"]) != shard_index:
                raise ValueError(f"wrong shard index in {part_path.name}")
            indices = np.asarray(part["indices"], dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= n_total):
                raise ValueError(f"out-of-range indices in {part_path.name}")
            if seen[indices].any() or np.unique(indices).size != indices.size:
                raise ValueError(f"duplicate indices in {part_path.name}")
            for name in ORDER_PARAM_NAMES:
                if part[name].shape != indices.shape:
                    raise ValueError(f"wrong {name} shape in {part_path.name}")
                values[name][indices] = part[name]
            seen[indices] = True

    if not seen.all():
        raise ValueError(f"sweep shards leave {(~seen).sum()} societies uncovered")

    d_axis, fd_axis = sweep_cfg.grids()
    shape = (sweep_cfg.n_fd, sweep_cfg.n_d, sweep_cfg.n_repeats)
    result = {"d": d_axis, "fd": fd_axis}
    for name in ORDER_PARAM_NAMES:
        result[name] = values[name].reshape(shape).mean(axis=2)
    _atomic_savez(final_path, **result)
    if verbose:
        print(f"[sweep merge] {n_shards} shards -> {final_path.name}")
    return result
