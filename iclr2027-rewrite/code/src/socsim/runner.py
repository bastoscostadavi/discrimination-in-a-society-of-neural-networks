"""Executing a campaign: batching, parallelism, resumption.

Two constraints shape this.  First, the inner loop gets its speed from advancing
many societies in lockstep under a shared interaction schedule, so **a batch must
hold exactly one replicate index and many grid points** --- mixing replicates
would share schedules between them and understate the seed-to-seed spread.
:class:`~socsim.society.SocietyBatch` refuses such a batch outright.

Second, a campaign runs for hours in the background on a machine that may sleep,
be killed, or be interrupted.  So chunks are sized by *time* rather than by
count, each is written the moment it finishes, and a resumed run recomputes only
what is missing.
"""

from __future__ import annotations

# Set before numpy is imported in worker processes: the inner loop is
# memory-bandwidth bound and single-threaded, so letting each worker start its
# own BLAS pool only causes contention.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .config import RunSpec  # noqa: E402
from .discrimination import FieldSpec  # noqa: E402
from .observables import measure  # noqa: E402
from .seeds import RunKey, point_id  # noqa: E402
from .society import SocietyBatch  # noqa: E402
from .store import merge, pending, write_shard  # noqa: E402

__all__ = ["enumerate_keys", "run_campaign", "auto_chunk", "DATA_DIR"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def enumerate_keys(spec):
    """Every ``(point, disorder, init)`` this run needs, with its parameters."""
    d_axis, fd_axis = spec.grid.axes()
    keys, params = [], []
    for fd in fd_axis:
        for d in d_axis:
            pid = point_id({"d": d, "f_d": fd, "case": spec.case, "kind": spec.field_kind})
            for dis in range(spec.grid.n_disorder):
                for ini in range(spec.grid.n_init):
                    keys.append(RunKey(spec.name, spec.crn_group, pid, dis, ini))
                    params.append({"d": float(d), "f_d": float(fd)})
    return keys, params


def auto_chunk(spec, n_workers, target_seconds=600.0, rate=None):
    """Choose a chunk size from a time target, not a fixed count.

    A killed background run then loses at most ``target_seconds`` of work
    regardless of how expensive one society happens to be at this ``N``.
    """
    if rate is None:
        # Societies per second per worker, scaled from a measured anchor of
        # ~0.15/s/worker at N=40, K=30, 500 interactions per channel.
        anchor = 0.15 * (40 * 39) / (spec.model.n_agents * (spec.model.n_agents - 1))
        anchor *= 500.0 / spec.model.interactions_per_channel
        if spec.model.dtype == "float32":
            anchor *= 1.8
        rate = anchor
    return int(np.clip(round(target_seconds * rate), 16, 512))


def _memory_cap(spec, chunk):
    """Keep one worker's covariance tensor under a few hundred megabytes."""
    N, K = spec.model.n_agents, spec.model.n_dim
    item = 4 if spec.model.dtype == "float32" else 8
    per_society = N * K * K * item
    return max(1, min(chunk, int(256e6 // max(per_society, 1))))


def _occupancy_cap(chunk, n_todo, n_buckets, n_workers, per_worker=3):
    """Shrink the chunk so every worker actually gets work.

    Chunks never span replicates, so the achievable count is bounded by the
    number of replicate buckets.  Sizing purely by a time target can therefore
    produce fewer chunks than workers --- a small sweep would then run on two
    cores while eight sat idle, which is easy to miss because the run still
    finishes, just slowly.
    """
    if n_buckets >= per_worker * n_workers:
        return chunk
    want = max(1, (per_worker * n_workers) // max(n_buckets, 1))
    per_bucket = max(1, n_todo // max(n_buckets, 1))
    return max(1, min(chunk, -(-per_bucket // want)))


def _simulate(job):
    """One chunk, in a worker. Returns rows, observables and diagnostics."""
    spec, keys, params = job
    specs = [
        FieldSpec(
            kind=spec.field_kind,
            case=spec.case,
            d=p["d"],
            f_d=p["f_d"],
            convention=spec.convention,
        )
        for p in params
    ]
    batch = SocietyBatch.from_keys(spec.model, keys, specs, master=spec.master)
    batch.run(spec.model.n_steps())
    obs = measure(
        batch,
        n_perm=spec.n_permutations,
        rng=np.random.default_rng(spec.master + 977),
    )
    rows = [{**k.as_row(), **p} for k, p in zip(keys, params)]
    diag = {
        "n_psd_clips": batch.n_psd_clips,
        "n_zfloor_hits": batch.n_zfloor_hits,
    }
    return rows, {k: np.asarray(v) for k, v in obs.items()}, diag


def _shard_name(rows):
    """Name a shard by the societies it holds, never by a loop index.

    Naming by position was a real defect: on resume the index restarts at zero,
    so a resumed run's first shards silently overwrite the shards written by the
    original run, and the societies they held are lost.  That is exactly what
    happened on the first production run, which finished 990 societies short.

    Hashing the chunk's keys makes the name a function of the contents, so
    re-running the same chunk overwrites itself harmlessly and two different
    chunks can never collide, however many times a campaign is interrupted.
    """
    ident = "|".join(
        f"{r['point_id']}:{r['replicate_dis']}:{r['replicate_init']}" for r in rows
    )
    return f"chunk_{hashlib.blake2b(ident.encode(), digest_size=8).hexdigest()}.npz"


def _chunks(keys, params, size):
    """Group into chunks that hold one ``(crn_group, init)`` pair each."""
    buckets = {}
    for k, p in zip(keys, params):
        buckets.setdefault((k.crn_group, k.disorder, k.init), []).append((k, p))
    for _, items in sorted(buckets.items()):
        for i in range(0, len(items), size):
            part = items[i : i + size]
            yield [a for a, _ in part], [b for _, b in part]


def run_campaign(
    spec,
    n_workers=10,
    chunk_size=None,
    data_dir=None,
    verbose=True,
    resume=True,
):
    """Simulate a run to completion, writing shards as they land."""
    data_dir = Path(data_dir or DATA_DIR)
    shard_dir = data_dir / "shards" / spec.name
    shard_dir.mkdir(parents=True, exist_ok=True)

    keys, params = enumerate_keys(spec)
    lookup = {(k.point_id, k.disorder, k.init): p for k, p in zip(keys, params)}
    todo = pending(keys, shard_dir) if resume else keys
    n_done = len(keys) - len(todo)
    if not todo:
        if verbose:
            print(f"[{spec.name}] already complete ({len(keys)} societies)")
        return merge(shard_dir, data_dir / f"results_{spec.name}.npz", spec)

    todo_params = [lookup[(k.point_id, k.disorder, k.init)] for k in todo]
    size = chunk_size or auto_chunk(spec, n_workers)
    size = _memory_cap(spec, size)
    n_buckets = len({(k.crn_group, k.disorder, k.init) for k in todo})
    size = _occupancy_cap(size, len(todo), n_buckets, n_workers)
    jobs = [(spec, ks, ps) for ks, ps in _chunks(todo, todo_params, size)]

    if verbose:
        print(
            f"[{spec.name}] {len(todo):,} societies to run "
            f"({n_done:,} already done), N={spec.model.n_agents}, "
            f"P={spec.model.n_issues}, alpha={spec.model.alpha:.3g}, "
            f"{spec.model.n_steps():,} interactions each, "
            f"{len(jobs)} chunks x {n_workers} workers"
        )

    t0 = time.time()
    done = 0
    param_names = ("d", "f_d")

    def _write(result):
        rows, obs, diag = result
        write_shard(shard_dir / _shard_name(rows), spec, rows, obs, param_names, diag)

    if n_workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(n_workers, len(jobs))) as pool:
            for result in pool.imap_unordered(_simulate, jobs):
                _write(result)
                done += len(result[0])
                if verbose:
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(
                        f"\r[{spec.name}] {done:,}/{len(todo):,} "
                        f"({rate:.1f}/s, eta {eta/60:.1f} min)",
                        end="",
                        flush=True,
                    )
    else:
        for job in jobs:
            _write(_simulate(job))
            done += len(job[1])

    if verbose:
        print(f"\r[{spec.name}] done in {(time.time()-t0)/60:.2f} min" + " " * 30)
    return merge(shard_dir, data_dir / f"results_{spec.name}.npz", spec)
