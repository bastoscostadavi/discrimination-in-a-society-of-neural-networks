#!/usr/bin/env python3
"""Side-by-side maps of the two readings of the prejudice field.

The source draft's Eq. 25 (``h_w^D = h_w + D``) together with its Table I
(in-group entry ``-d``) puts the discriminatory phase at ``d < 0``, while its
text and every one of its figures put it at ``d > 0``.  Exactly one global sign
separates the two readings.  This script runs the same sweep under both and
plots the trust-class correlation, so the difference can be seen rather than
argued about.

See ``docs/prejudice-field-sign.md`` for the argument.  Writes
``sign_convention_comparison``.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.config import SweepConfig  # noqa: E402
from ednna.plotting import panel, phase_map, save  # noqa: E402
from ednna.sweep import sweep  # noqa: E402


def figure(preset, style, n_grid=32, use_cache=True):
    # A coarse grid is enough here: the point is that the two rows are mirror
    # images in d, not the fine structure of either.
    cfg = SweepConfig(
        n_d=n_grid, n_fd=n_grid, batch_size=256,
        n_workers=preset.sweep.n_workers, seed=preset.sweep.seed,
    )
    base = preset.model.with_(n_issues=preset.p_small)
    variants = (
        ("consistent convention", False),
        ("draft's Eq. 25 + Table I, literally", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=panel(0.92, 0.78/0.86))
    for row, (label, literal) in enumerate(variants):
        model = base.with_(literal_draft_sign=literal)
        data = sweep(
            model, cfg, tag=f"sign_{'literal' if literal else 'consistent'}",
            use_cache=use_cache,
        )
        for col, key in enumerate(("R_muc", "B_eta")):
            ax = axes[row][col]
            phase_map(ax, data[key], data["d"], data["fd"], key)
            if col == 0:
                ax.text(
                    -0.34, 0.5, label, transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=6.5,
                )
    fig.tight_layout(pad=0.5)
    return save(fig, "sign_convention_comparison", style)


def main():
    args, preset = setup(__doc__)
    figure(preset, args.style, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
