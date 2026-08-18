#!/usr/bin/env python3
"""Order parameters over the ``(d, f_d)`` plane.

The same data as ``correlation_maps`` and ``frustration_maps`` (which reproduce
the source draft's two separate figures), on one grid.  Two cuts of it are
written:

Two cuts are written:

``correlation_maps_both``
    the three pair correlations, both agenda sizes.  These three are also the three
    colour channels of the phase diagram, so this is the figure that composite is
    assembled from;
``order_parameter_maps``
    all five, for reference.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from correlation_maps import agenda_sweeps  # noqa: E402

from ednna.plotting import phase_map, save, text_width  # noqa: E402

KEYS = ("R_wmu", "R_muc", "R_cw", "B_I", "B_A")
CORRELATIONS = ("R_wmu", "R_muc", "R_cw")


def figure(rows, style, name="order_parameter_maps", keys=KEYS):
    """Five columns is too narrow for per-panel colourbars, so each column gets
    one thin horizontal bar underneath: the range and colour map depend on the
    order parameter, not on the agenda size, so one bar per column is exact."""
    n_rows, n_cols = len(rows), len(keys)
    # Square panels at any grid shape.  ``wspace``/``hspace`` are fractions of the
    # *average* axis size, so with the colourbar row carrying a ratio of its own the
    # vertical arithmetic is not symmetric with the horizontal: solve both and set
    # the figure height that makes a ratio-1 row exactly as tall as a column is wide.
    left, right, top, bottom = 0.085, 0.985, 0.93, 0.10
    # hspace has to clear the shared d label between the last row and the colourbars,
    # and it is a fraction of the average axis height, so it grows as the panels shrink
    hspace, wspace, cbar = 0.62, 0.30, 0.07
    W = text_width()
    ax_w = W * (right - left) / (n_cols + wspace * (n_cols - 1))
    ratios = [1.0] * n_rows + [cbar]
    units = sum(ratios) + hspace * (sum(ratios) / len(ratios)) * n_rows
    H = ax_w * units / (top - bottom)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=ratios,
        hspace=hspace, wspace=wspace,
        left=left, right=right, top=top, bottom=bottom,
    )
    images = {}
    for i, (label, P, alpha, data) in enumerate(rows):
        for j, key in enumerate(keys):
            ax = fig.add_subplot(gs[i, j])
            images[key] = phase_map(
                ax, data[key], data["d"], data["fd"], key,
                ylabel=(j == 0), colorbar=False, sparse_ticks=True,
            )
            # the two axes are shared across the grid, so label them once
            if i < n_rows - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)
            if j == 0 and n_rows > 1:
                ax.text(
                    -0.52, 0.5, rf"$\alpha={alpha:.3g}$",
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=7.5,
                )
    for j, key in enumerate(keys):
        cax = fig.add_subplot(gs[n_rows, j])
        cb = fig.colorbar(images[key], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
        cb.outline.set_linewidth(0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    rows = agenda_sweeps(preset, use_cache=not args.no_cache)
    figure(rows, args.style, name="correlation_maps_both", keys=CORRELATIONS)
    figure(rows, args.style)


if __name__ == "__main__":
    main()
