"""Shared command-line handling for the figure scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dirfield.config import PRESETS, get_preset  # noqa: E402
from dirfield.fields import COMPONENTS  # noqa: E402
from dirfield.plotting import set_component, use_style  # noqa: E402


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
        "--component",
        default="c",
        choices=COMPONENTS,
        help="which field component the strength axis drives (default: c, status)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="societies per vectorized batch, overriding the preset. Memory "
        "scales as batch * N * K^2, so this is the knob to turn on a machine with "
        "less headroom than the preset assumes. NOTE: it changes the cache key, "
        "because each batch is seeded from its offset, so two batch sizes draw "
        "different realizations of the same grid",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes, overriding the preset. Does not affect results "
        "or the cache key",
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
    set_component(args.component)
    preset = get_preset(args.preset)
    sweep_cfg = preset.sweep
    if args.batch_size is not None:
        sweep_cfg = sweep_cfg.with_(batch_size=args.batch_size)
    if args.workers is not None:
        sweep_cfg = sweep_cfg.with_(n_workers=args.workers)
    return args, preset.__class__(**{
        **preset.__dict__,
        "model": preset.model.with_(component=args.component),
        "sweep": sweep_cfg,
    })
