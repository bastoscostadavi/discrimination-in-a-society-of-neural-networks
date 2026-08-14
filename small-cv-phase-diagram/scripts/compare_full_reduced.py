#!/usr/bin/env python3
"""Compare full and reduced heatmaps as c,v decrease."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from smallcv.config import ModelConfig, SweepConfig
from smallcv.sweep import sweep
from smallcv.plotting import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-values", default="0.2,0.1,0.05")
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--n", type=int, default=21)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()

    sweep_cfg = SweepConfig(n_d=args.n, n_fd=args.n, n_repeats=args.repeats)
    c_values = [float(x) for x in args.c_values.split(",")]
    rows = []
    for c in c_values:
        base = dict(initial_c=c, initial_v=c * args.ratio)
        reduced = sweep(ModelConfig(**base, dynamics="small_cv"), sweep_cfg, tag=f"cmp_reduced_c{c:g}")
        full = sweep(ModelConfig(**base, dynamics="full"), sweep_cfg, tag=f"cmp_full_c{c:g}")
        rows.append([c] + [float(np.mean(np.abs(reduced[k] - full[k]))) for k in ("R_wmu", "R_muc", "R_cw")])

    rows = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    for i, key in enumerate(("R_wmu", "R_muc", "R_cw"), start=1):
        ax.plot(rows[:, 0], rows[:, i], marker="o", label=key)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("initial c, with v/c fixed")
    ax.set_ylabel("mean absolute full-reduced difference")
    ax.legend()
    print(save(fig, "full_vs_small_cv_convergence"))


if __name__ == "__main__":
    main()
