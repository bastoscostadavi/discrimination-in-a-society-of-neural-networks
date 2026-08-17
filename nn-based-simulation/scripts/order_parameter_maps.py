#!/usr/bin/env python3
"""Order parameters over the ``(d, f_d)`` plane.

The same data as ``correlation_maps`` and ``frustration_maps`` (which reproduce
the source draft's two separate figures), on one grid.  Two cuts of it are
written:

``correlation_maps_complex``
    the three pair correlations for the complex agenda alone, which is what the
    body of the paper leads with;
``order_parameter_maps``
    all five for both agenda sizes, which is where the balances and the
    simple-agenda panels live.  Seeing them together is what makes the phases
    readable in one pass -- in particular that the frustrated region at ``d < 0``
    shows up in ``B_A`` and nowhere else.
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
    # square panels at any grid shape: solve for the figure height that makes the
    # axes as tall as the gridspec makes them wide, given the same margins
    left, right, top, bottom = 0.085, 0.985, 0.93, 0.10
    hspace, wspace, cbar = 0.42, 0.30, 0.07
    W = text_width()
    ax_w = W * (right - left) / (n_cols + wspace * (n_cols - 1))
    H = ax_w * (n_rows * (1 + hspace) + cbar) / (top - bottom)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=[1] * n_rows + [cbar],
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
    complex_row = [r for r in rows if r[2] > 1]
    figure(complex_row, args.style, name="correlation_maps_complex", keys=CORRELATIONS)
    figure(rows, args.style)


if __name__ == "__main__":
    main()
