#!/usr/bin/env python3
"""All five order parameters over the ``(d, f_d)`` plane, both agenda sizes.

The same data as ``correlation_maps`` and ``frustration_maps`` (which reproduce
the source draft's two separate figures), combined into a single 2x5 grid: the
three pair correlations and the two balance aggregates, for a simple agenda on
top and a complex one below.  Putting them together is what makes the phases
readable in one pass -- in particular that the frustrated region at ``d < 0``
shows up in ``B_A`` and nowhere else.

Writes ``order_parameter_maps``.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from correlation_maps import agenda_sweeps  # noqa: E402

from ednna.plotting import phase_map, save, text_width  # noqa: E402

KEYS = ("R_wmu", "R_muc", "R_cw", "B_I", "B_A")


def figure(rows, style, name="order_parameter_maps"):
    """Five columns is too narrow for per-panel colourbars, so each column gets
    one thin horizontal bar underneath: the range and colour map depend on the
    order parameter, not on the agenda size, so one bar per column is exact."""
    width = text_width()
    n_rows, n_cols = len(rows), len(KEYS)
    fig = plt.figure(figsize=(width, width * 0.27 * n_rows))
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=[1] * n_rows + [0.07],
        hspace=0.42, wspace=0.30,
        left=0.085, right=0.985, top=0.93, bottom=0.10,
    )
    images = {}
    for i, (label, P, alpha, data) in enumerate(rows):
        for j, key in enumerate(KEYS):
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
            if j == 0:
                ax.text(
                    -0.52, 0.5, rf"$\alpha={alpha:.3g}$",
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=7.5,
                )
    for j, key in enumerate(KEYS):
        cax = fig.add_subplot(gs[n_rows, j])
        cb = fig.colorbar(images[key], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
        cb.outline.set_linewidth(0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    rows = agenda_sweeps(preset, use_cache=not args.no_cache)
    figure(rows, args.style)


if __name__ == "__main__":
    main()
