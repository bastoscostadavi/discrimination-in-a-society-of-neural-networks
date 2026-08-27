"""Shared command-line handling for the figure scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credfield.config import PRESETS, get_preset  # noqa: E402
from credfield.fields import COMPONENTS  # noqa: E402
from credfield.plotting import set_component, use_style  # noqa: E402


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--style",
        default="iclr",
        choices=("paper", "iclr"),
        help="figure proportions: 'paper' matches the source draft, "
        "'iclr' matches the ICLR single-column text width",
    )
    p.add_argument(
        "--preset",
        default="full",
        choices=sorted(PRESETS),
        help="resolution/compute preset (simulation figures only).  The default "
        "is 'full': 200x200 at N=40, the resolution of the paper's own phase "
        "diagram, which is what this directory exists to produce.  Use 'quick' "
        "or 'medium' while iterating on a figure",
    )
    p.add_argument(
        "--component",
        default="b",
        choices=COMPONENTS,
        help="which field component the strength axis drives "
        "(default: b, the credulity asymmetry this directory is about)",
    )
    p.add_argument(
        "--strips",
        type=int,
        default=5,
        help="sweep the prevalence axis in this many contiguous bands, each "
        "cached on its own.  Total work is identical; a kill loses at most one "
        "band, and re-running reloads the finished ones.  The committed figures "
        "are built from 5 strips.  Pass 1 for a single sweep with a single cache "
        "write at the very end",
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
    return args, preset.__class__(**{**preset.__dict__,
                                     "model": preset.model.with_(component=args.component)})
