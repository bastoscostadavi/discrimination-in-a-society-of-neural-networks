"""Shared command-line handling for the figure scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credulity.config import PRESETS, get_preset  # noqa: E402
from credulity.plotting import use_style  # noqa: E402


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
        help="resolution/compute preset. 'full' is the paper's own resolution, "
        "200x200 at N=40, and takes about four hours on ten cores",
    )
    p.add_argument(
        "--strips",
        type=int,
        default=None,
        help="cut the prevalence axis into this many independently cached "
        "bands, so an interrupted run loses at most one of them (default: the "
        "preset's own value, 5 for 'full' and 1 otherwise)",
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
    return args, get_preset(args.preset)
