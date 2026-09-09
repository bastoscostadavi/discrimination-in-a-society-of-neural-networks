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
from ednna.sweep import cache_path, sweep  # noqa: E402


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

    path = cache_path(model, cfg, tag=f"P{args.issues}")
    print(f"[sweep_one] N={model.n_agents} P={model.n_issues} K={model.n_dim} "
          f"dtype={model.dtype} grid={cfg.n_d}x{cfg.n_fd} "
          f"batch={cfg.batch_size} workers={cfg.n_workers}")
    print(f"[sweep_one] cache -> {path.name}")

    t0 = time.time()
    sweep(model, cfg, tag=f"P{args.issues}")
    print(f"[sweep_one] total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
