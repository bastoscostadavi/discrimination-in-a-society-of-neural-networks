"""Shared command-line handling for the figure scripts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ednna.config import PRESETS, get_preset  # noqa: E402
from ednna.plotting import use_style  # noqa: E402


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--style",
        default="paper",
        choices=("paper", "iclr"),
        help="figure proportions: 'paper' matches the source draft, "
        "'iclr' matches the ICLR single-column text width",
    )
    p.add_argument(
        "--preset",
        default="medium",
        choices=sorted(PRESETS),
        help="resolution/compute preset (simulation figures only)",
    )
    p.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="society size N for the swept societies, overriding the preset. "
        "Cost scales as N^2, because the run length is "
        "interactions_per_channel * N * (N - 1); memory as batch * N * K^2 for "
        "the covariances plus batch * N^2 for the trust matrices. Does not "
        "touch `polarization_agents`, which the polarization figure sets "
        "separately",
    )
    p.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "float64"),
        help="overriding the preset: float32 roughly halves the run and changes "
        "no order parameter by more than 0.1 (see config.py). Part of the cache "
        "key, as every model field is",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="societies per vectorized batch, overriding the preset. Memory "
        "scales as batch * N * K^2, so this is the knob to turn when a node has "
        "less headroom than the preset assumes. NOTE: it is part of the cache "
        "key, because the key hashes the whole sweep configuration; and since "
        "each batch is seeded from its offset, two batch sizes genuinely draw "
        "different realizations of the same grid rather than sharing one",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes, overriding the preset. Does not affect results, "
        "but it IS part of the cache key, since the key hashes the whole sweep "
        "configuration -- so a run at a different worker count writes a new file "
        "rather than reusing an existing one",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="re-simulate even if a cached sweep exists",
    )
    return p


def setup(description):
    """Parse arguments, activate the style, and return ``(args, preset)``."""
    args = parser(description).parse_args()
    use_style(args.style)
    preset = get_preset(args.preset)

    model = preset.model
    if args.n_agents is not None:
        model = model.with_(n_agents=args.n_agents)
    if args.dtype is not None:
        model = model.with_(dtype=args.dtype)

    sweep_cfg = preset.sweep
    if args.batch_size is not None:
        sweep_cfg = sweep_cfg.with_(batch_size=args.batch_size)
    if args.workers is not None:
        sweep_cfg = sweep_cfg.with_(n_workers=args.workers)

    return args, replace(preset, model=model, sweep=sweep_cfg)
