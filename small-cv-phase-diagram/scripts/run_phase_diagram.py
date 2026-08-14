#!/usr/bin/env python3
"""Generate small-C,V heatmaps over the external (d, f_d) plane."""

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
from matplotlib import pyplot as plt

from smallcv.config import ModelConfig, SweepConfig
from smallcv.plotting import phase_map, rgb_composite, save
from smallcv.sweep import sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-d", type=int, default=41)
    p.add_argument("--n-fd", type=int, default=41)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--c", type=float, default=0.05)
    p.add_argument("--ratio", type=float, default=1.0, help="v/c")
    p.add_argument("--steps-at-c1", type=float, default=250.0)
    p.add_argument("--case", type=int, default=6)
    p.add_argument("--dynamics", choices=("small_cv", "full"), default="small_cv")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="societies per checkpointed batch; smaller values resume with less lost work",
    )
    args = p.parse_args()

    model = ModelConfig(
        case=args.case,
        initial_c=args.c,
        initial_v=args.c * args.ratio,
        interactions_per_channel_at_c1=args.steps_at_c1,
        dynamics=args.dynamics,
    )
    sweep_cfg = SweepConfig(
        n_d=args.n_d,
        n_fd=args.n_fd,
        n_repeats=args.repeats,
        batch_size=args.batch_size,
    )
    data = sweep(model, sweep_cfg, tag=f"{args.dynamics}_c{args.c:g}_r{args.ratio:g}", use_cache=not args.no_cache)

    keys = ("R_wmu", "R_muc", "R_cw")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, key in zip(axes, keys):
        im = phase_map(ax, data[key], data["d"], data["fd"], key)
        fig.colorbar(im, ax=ax, shrink=0.82)
    print(save(fig, f"small_cv_heatmaps_c{args.c:g}_r{args.ratio:g}"))

    fig, ax = plt.subplots(figsize=(3.8, 3.3), constrained_layout=True)
    ax.imshow(rgb_composite(data["R_muc"], data["R_cw"], data["R_wmu"]), origin="upper", extent=[data["d"][0], data["d"][-1], data["fd"][-1], data["fd"][0]], aspect="auto")
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$f_d$")
    ax.set_title("phase composite")
    print(save(fig, f"small_cv_phase_composite_c{args.c:g}_r{args.ratio:g}"))


if __name__ == "__main__":
    main()
