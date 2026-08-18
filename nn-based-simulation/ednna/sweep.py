"""Phase-diagram sweeps over the ``(d, f_d)`` plane.

A sweep evaluates the five order parameters on a grid of discrimination field
``d`` and fraction of discriminating agents ``f_d``.  Grid points are flattened,
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
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .config import ModelConfig, SweepConfig  # noqa: E402
from .order_params import ORDER_PARAM_NAMES, measure  # noqa: E402
from .society import SocietyBatch  # noqa: E402

__all__ = ["sweep", "DATA_DIR", "cache_path"]

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


def _run_batch(job):
    """Simulate one batch of societies and measure them. Runs in a worker."""
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
    out = measure(
        batch, class_indicator=model.class_indicator, literal_norm=model.literal_norm
    )
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, batch.n_psd_clips


#: The balances were called ``B_I`` and ``B_A`` when the long sweeps were run, after
#: the older names for the two sectors.  They are now named for the pairwise quantity
#: each is built from, and caches written under the old names are still readable: the
#: 200x200 grids cost about 45 minutes each and the arrays themselves did not change.
LEGACY_KEYS = {"B_I": "B_rho", "B_A": "B_eta"}


def sweep(model, sweep_cfg=None, tag="", use_cache=True, verbose=True):
    """Run (or load) a ``(d, f_d)`` sweep.

    Returns a dict with one ``(n_fd, n_d)`` array per order parameter, plus the
    ``d`` and ``fd`` axes.  Rows are indexed by ``f_d`` and columns by ``d``, so
    the arrays are ready for ``imshow`` with ``origin="lower"``, which draws row 0
    at the bottom so that ``f_d`` increases upwards.
    """
    sweep_cfg = sweep_cfg or SweepConfig()
    path = cache_path(model, sweep_cfg, tag)
    if use_cache and path.exists():
        with np.load(path) as z:
            result = {LEGACY_KEYS.get(k, k): z[k] for k in z.files}
        if verbose:
            print(f"[sweep] loaded cache {path.name}")
        return result

    d_axis, fd_axis = sweep_cfg.grids()
    D, F = np.meshgrid(d_axis, fd_axis)  # (n_fd, n_d)
    d_flat = np.repeat(D.ravel(), sweep_cfg.n_repeats)
    fd_flat = np.repeat(F.ravel(), sweep_cfg.n_repeats)
    n_total = d_flat.size

    jobs = []
    model_dict = asdict(model)
    for start in range(0, n_total, sweep_cfg.batch_size):
        stop = min(start + sweep_cfg.batch_size, n_total)
        jobs.append(
            (model_dict, d_flat[start:stop], fd_flat[start:stop], sweep_cfg.seed + start)
        )

    if verbose:
        steps = model.n_steps()
        print(
            f"[sweep] {n_total} societies of N={model.n_agents} "
            f"(P={model.n_issues}, alpha={model.alpha:.3g}), "
            f"{steps:,} interactions each, {len(jobs)} batches x {sweep_cfg.n_workers} workers"
        )

    t0 = time.time()
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

    _ensure_writable(DATA_DIR)
    np.savez_compressed(path, **result)
    if verbose:
        print(f"[sweep] cached -> {path.name}")
    return result
