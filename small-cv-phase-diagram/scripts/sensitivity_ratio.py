#!/usr/bin/env python3
"""Measure sensitivity of the small-C,V phase diagram to r=v/c."""

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
    p.add_argument("--c", type=float, default=0.05)
    p.add_argument("--ratios", default="0.25,1,4")
    p.add_argument("--n", type=int, default=21)
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    sweep_cfg = SweepConfig(n_d=args.n, n_fd=args.n, n_repeats=args.repeats)
    ref = None
    rows = []
    for ratio in ratios:
        data = sweep(ModelConfig(initial_c=args.c, initial_v=args.c * ratio), sweep_cfg, tag=f"ratio_{ratio:g}")
        if ref is None:
            ref = data
        rows.append([ratio] + [float(np.mean(np.abs(data[k] - ref[k]))) for k in ("R_wmu", "R_muc", "R_cw")])

    rows = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    for i, key in enumerate(("R_wmu", "R_muc", "R_cw"), start=1):
        ax.plot(rows[:, 0], rows[:, i], marker="o", label=key)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r=v/c$")
    ax.set_ylabel("mean absolute difference from first ratio")
    ax.legend()
    print(save(fig, "ratio_sensitivity"))


if __name__ == "__main__":
    main()
