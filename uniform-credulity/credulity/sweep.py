"""Phase-diagram sweeps over the ``(a, f_a)`` plane.

A sweep evaluates every order parameter on a grid of field strength and fraction
of biased agents.  Grid points are flattened, split into batches, and each batch
is run as a single vectorized :class:`~credulity.society.SocietyBatch` inside a
worker process.  Results are cached in ``data/`` keyed by a hash of the
configuration, so re-plotting never re-simulates.
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

__all__ = ["sweep", "sweep_in_strips", "strip_configs", "DATA_DIR", "cache_path"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _ensure_writable(directory):
    """Create ``directory``, but refuse to resurrect a vanished tree.

    Package paths are resolved at import time.  If the project directory is
    renamed or moved while a long sweep is running, a plain
    ``mkdir(parents=True)`` silently recreates the old tree and writes the
    results into a directory nobody is looking at.  Requiring the parent to
    still exist turns that into a loud failure at the end of the run.
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
    # The measured set is part of the key.  Without it, adding an order
    # parameter leaves every existing cache file looking valid and then failing
    # on the missing array at read time -- or worse, being silently re-plotted
    # from a stale set of quantities.
    payload = json.dumps(
        {"model": asdict(model), "sweep": asdict(sweep_cfg), "tag": tag,
         "params": list(ORDER_PARAM_NAMES)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    label = tag or f"a_P{model.n_issues}_N{model.n_agents}"
    return DATA_DIR / f"sweep_{label}_{sweep_cfg.n_a}x{sweep_cfg.n_f}_{digest}.npz"


def _run_batch(job):
    """Simulate one batch of societies and measure them. Runs in a worker."""
    model_dict, a_values, f_values, seed = job
    model = ModelConfig(**model_dict)
    batch = SocietyBatch(
        n_agents=model.n_agents,
        n_dim=model.n_dim,
        n_issues=model.n_issues,
        a=np.asarray(a_values),
        f=np.asarray(f_values),
        seed=seed,
        dtype=model.numpy_dtype(),
        shared_schedule=model.shared_schedule,
    )
    batch.run(model.n_steps())
    return ({k: np.asarray(v, dtype=np.float64) for k, v in measure(batch).items()},
            batch.n_psd_clips)


def sweep(model, sweep_cfg=None, tag="", use_cache=True, verbose=True):
    """Run (or load) an ``(a, f_a)`` sweep.

    Returns a dict with one ``(n_f, n_a)`` array per order parameter, plus the
    ``a`` and ``f`` axes.  Rows are indexed by the fraction and columns by the
    strength, so the arrays are ready for ``imshow`` with ``origin="lower"``,
    which draws row 0 at the bottom so the fraction increases upwards.
    """
    sweep_cfg = sweep_cfg or SweepConfig()
    path = cache_path(model, sweep_cfg, tag)
    if use_cache and path.exists():
        with np.load(path) as z:
            result = {k: z[k] for k in z.files}
        if verbose:
            print(f"[sweep] loaded cache {path.name}")
        return result

    a_axis, f_axis = sweep_cfg.grids()
    A, F = np.meshgrid(a_axis, f_axis)  # (n_f, n_a)
    a_flat = np.repeat(A.ravel(), sweep_cfg.n_repeats)
    f_flat = np.repeat(F.ravel(), sweep_cfg.n_repeats)
    n_total = a_flat.size

    jobs = []
    model_dict = asdict(model)
    for start in range(0, n_total, sweep_cfg.batch_size):
        stop = min(start + sweep_cfg.batch_size, n_total)
        jobs.append(
            (model_dict, a_flat[start:stop], f_flat[start:stop], sweep_cfg.seed + start)
        )

    if verbose:
        print(
            f"[sweep] {n_total} societies of N={model.n_agents} "
            f"(P={model.n_issues}, alpha={model.alpha:.3g}), uniform field 'a' "
            f"over {sweep_cfg.a_range}, {model.n_steps():,} interactions each, "
            f"{len(jobs)} batches x {sweep_cfg.n_workers} workers"
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

    shape = (sweep_cfg.n_f, sweep_cfg.n_a, sweep_cfg.n_repeats)
    result = {"a": a_axis, "f": f_axis}
    for name in ORDER_PARAM_NAMES:
        flat = np.concatenate([p[name] for p in pieces])
        # Group quantities are nan where their group is empty, which is a whole
        # row of the grid at f = 0 and at f = 1.  Averaging repeats with
        # `mean` would propagate that correctly but also warn on every all-nan
        # slice; `nanmean` would hide a partial group instead.  Keep `mean`: nan
        # is the answer there, and the warning is silenced rather than the value
        # changed.
        with np.errstate(invalid="ignore"):
            result[name] = flat.reshape(shape).mean(axis=2)

    _ensure_writable(DATA_DIR)
    np.savez_compressed(path, **result)
    if verbose:
        print(f"[sweep] cached -> {path.name}")
    return result


# --- running a plane in horizontal strips ----------------------------------
#
# `sweep` writes its cache once, at the very end.  It builds every job, runs
# them all through the pool, and only then calls `savez_compressed`, so an
# interrupt at any point before that loses the whole run: a kill at three hours
# fifty-five costs exactly what a kill at thirty seconds costs.  At the `full`
# preset that is four hours of a machine with nothing to show for it.
#
# Splitting the prevalence axis into bands and caching each separately bounds
# that loss to one band, and makes a re-run reload the finished ones instead of
# repeating them.  Two things about it are easy to get subtly wrong, and both
# produce a plane that looks right:
#
#   * The bands must be read off the *full* axis, not carved out of the unit
#     interval.  Slicing (0, 0.2), (0.2, 0.4), ... and calling linspace inside
#     each duplicates every boundary row and gives a different row spacing
#     within a band than between bands, so the concatenation is not the grid it
#     appears to be.
#   * The bands must not share a seed.  `sweep` seeds each batch as
#     `seed + <flat offset within this sweep>`, and that offset restarts at zero
#     in every band -- so bands at a common base seed draw the *same* societies
#     over and over, and the plane comes out looking converged when it is one
#     band repeated.  Offsetting each band by its own position in the full grid
#     fixes it and has a second useful property: the seed of a society is a
#     function of where it sits in the plane, so re-running one band reproduces
#     it exactly and does not renumber any other.


def strip_configs(sweep_cfg, n_strips):
    """Split a sweep into ``n_strips`` bands of the prevalence axis.

    Returns a list of :class:`~credulity.config.SweepConfig`, each covering a
    contiguous block of rows of the full grid.  The blocks partition the axis
    exactly -- concatenating their ``f`` axes reproduces the original to
    floating-point equality -- and no two share a seed.
    """
    if n_strips <= 1:
        return [sweep_cfg]
    f_all = np.linspace(*sweep_cfg.f_range, sweep_cfg.n_f)
    # grid points per row of the full plane, which is the stride the flat
    # offsets inside `sweep` advance by
    per_row = sweep_cfg.n_a * sweep_cfg.n_repeats
    out = []
    for rows in np.array_split(np.arange(sweep_cfg.n_f), n_strips):
        if rows.size == 0:
            continue
        lo, hi = int(rows[0]), int(rows[-1])
        out.append(sweep_cfg.with_(
            n_f=int(rows.size),
            f_range=(float(f_all[lo]), float(f_all[hi])),
            seed=int(sweep_cfg.seed + lo * per_row),
        ))
    return out


def sweep_in_strips(model, sweep_cfg=None, n_strips=1, tag="", use_cache=True,
                    verbose=True):
    """Run (or load) a sweep as a sequence of independently cached strips.

    ``n_strips <= 1`` is exactly :func:`sweep`, cache file and all, so the
    striping costs nothing when it is not wanted.  Otherwise each strip is a
    separate cached sweep and the results are concatenated along the prevalence
    axis; the strips carry different ``f_range``, ``n_f`` and ``seed``, all three
    of which are in the cache key, so they cannot collide with each other or with
    an unstriped run of the same plane.
    """
    sweep_cfg = sweep_cfg or SweepConfig()
    cfgs = strip_configs(sweep_cfg, n_strips)
    if len(cfgs) == 1:
        return sweep(model, cfgs[0], tag=tag, use_cache=use_cache, verbose=verbose)

    parts = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs, 1):
        if verbose:
            print(f"[sweep] strip {i}/{len(cfgs)}: {cfg.n_f} rows, "
                  f"f in [{cfg.f_range[0]:.4f}, {cfg.f_range[1]:.4f}]"
                  + (f", {(time.time()-t0)/60:.0f} min so far" if i > 1 else ""))
        parts.append(sweep(model, cfg, tag=tag, use_cache=use_cache,
                           verbose=verbose))

    out = {"a": parts[0]["a"],
           "f": np.concatenate([p["f"] for p in parts])}
    for name in ORDER_PARAM_NAMES:
        out[name] = np.concatenate([p[name] for p in parts], axis=0)
    return out
