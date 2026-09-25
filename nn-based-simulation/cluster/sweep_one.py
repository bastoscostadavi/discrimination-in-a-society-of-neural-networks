#!/usr/bin/env python3
"""Run one ``(d, f_d)`` sweep and cache it, then exit.

``make_all.py`` runs both agenda sizes in one process, which is the right shape
on a laptop and the wrong one on a batch queue: at large ``N`` the two sweeps
together do not fit inside a single job's walltime.  This script does exactly
one of them, so the two can be an array job, and it writes the cache under the
same tag the figure scripts read.  Once both caches exist, ``make_all.py``
--- run anywhere, including back on the laptop --- draws every figure without
re-simulating.

    python cluster/sweep_one.py --issues 5 --preset full --n-agents 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ednna.config import PRESETS, get_preset  # noqa: E402
from ednna.sweep import (  # noqa: E402
    cache_path, merge_sweep_shards, sweep, sweep_shard,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--issues", type=int, required=True,
                   help="agenda size P for this sweep; the figure scripts want "
                        "the preset's p_small and p_large")
    p.add_argument("--preset", default="full", choices=sorted(PRESETS))
    p.add_argument("--n-agents", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--n-d", type=int, default=None,
                   help="grid points along the d axis")
    p.add_argument("--n-fd", type=int, default=None,
                   help="grid points along the f_d axis")
    p.add_argument("--repeats", type=int, default=None,
                   help="independent societies averaged at each grid point")
    p.add_argument("--n-shards", type=int, default=1,
                   help="number of batch-aligned pieces in the complete sweep")
    p.add_argument("--shard-index", type=int, default=None,
                   help="zero-based piece to run; omit for an unsharded sweep")
    p.add_argument("--merge", action="store_true",
                   help="validate and merge all shard caches, without simulating")
    p.add_argument("--save-snapshots", action="store_true",
                   help="retain batch-wise float32 final w, eta, agenda, and masks")
    p.add_argument("--dtype", default=None, choices=("float32", "float64"),
                   help="float32 roughly halves the run and changes no order "
                        "parameter by more than 0.1 (see config.py)")
    args = p.parse_args()

    preset = get_preset(args.preset)
    model = preset.model.with_(n_issues=args.issues)
    if args.n_agents is not None:
        model = model.with_(n_agents=args.n_agents)
    if args.dtype is not None:
        model = model.with_(dtype=args.dtype)

    cfg = preset.sweep
    if args.batch_size is not None:
        cfg = cfg.with_(batch_size=args.batch_size)
    if args.workers is not None:
        cfg = cfg.with_(n_workers=args.workers)
    if args.n_d is not None:
        cfg = cfg.with_(n_d=args.n_d)
    if args.n_fd is not None:
        cfg = cfg.with_(n_fd=args.n_fd)
    if args.repeats is not None:
        cfg = cfg.with_(n_repeats=args.repeats)

    path = cache_path(model, cfg, tag=f"P{args.issues}")
    print(f"[sweep_one] N={model.n_agents} P={model.n_issues} K={model.n_dim} "
          f"dtype={model.dtype} grid={cfg.n_d}x{cfg.n_fd} "
          f"batch={cfg.batch_size} workers={cfg.n_workers}")
    print(f"[sweep_one] cache -> {path.name}")

    if args.merge and args.shard_index is not None:
        p.error("--merge and --shard-index cannot be used together")
    if not args.merge and args.n_shards > 1 and args.shard_index is None:
        p.error("--shard-index is required when --n-shards is greater than 1")
    t0 = time.time()
    if args.merge:
        merge_sweep_shards(model, cfg, tag=f"P{args.issues}",
                           n_shards=args.n_shards)
    elif args.shard_index is not None and args.n_shards > 1:
        sweep_shard(model, cfg, tag=f"P{args.issues}",
                    shard_index=args.shard_index, n_shards=args.n_shards,
                    save_snapshots=args.save_snapshots)
    else:
        sweep(model, cfg, tag=f"P{args.issues}",
              save_snapshots=args.save_snapshots)
    print(f"[sweep_one] total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
