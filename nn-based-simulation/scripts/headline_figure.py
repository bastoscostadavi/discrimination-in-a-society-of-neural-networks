#!/usr/bin/env python3
"""Headline figure: where the four states occur and what they look like.

The phase map and the microscopic portraits are drawn together so the Roman
numerals connect a region of parameter space directly to a representative
population.  This is intentionally a lighter version of ``state_portraits``:
the projection diagnostics and parameter values remain in that source figure,
while the headline keeps only what is needed to distinguish the states.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

import phase_diagram  # noqa: E402
import state_portraits  # noqa: E402
from ednna.plotting import (  # noqa: E402
    HIST_BLUE, HIST_RED, add_phase_axes, rgb_composite, save, text_width,
)
from ednna.sweep import sweep  # noqa: E402


def _portrait(ax, coords, ref, kappa, prejudiced):
    circle = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(circle), np.sin(circle), color="0.8", lw=0.45,
            ls=(0, (3, 3)))
    state_portraits._draw_reference(ax, ref)
    state_portraits._draw_agents(ax, coords, kappa, prejudiced,
                                 lw=0.60, ms=2.2)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(0.45)
        ax.spines[side].set_color("0.72")


def _portrait_legend(fig):
    handles = [
        plt.Line2D([], [], color=HIST_RED[1], lw=0.9, label="class $A$"),
        plt.Line2D([], [], color=HIST_BLUE[1], lw=0.9, label="class $B$"),
        plt.Line2D([], [], color="0.35", lw=0, marker="o", ms=2.5,
                   label="prejudiced"),
        plt.Line2D([], [], color="0.35", lw=0, marker="o", ms=2.5,
                   mfc="white", mew=0.45, label="class-blind"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.685, 0.035), frameon=False, fontsize=5.8,
               handlelength=1.2, columnspacing=0.9, labelspacing=0.25,
               borderpad=0.05)


def figure(phase, portraits, style, name="headline_states"):
    width = text_width()
    fig = plt.figure(figsize=(width, 0.46 * width))
    outer = fig.add_gridspec(
        1, 2, width_ratios=(1.03, 2.0),
        left=0.075, right=0.99, bottom=0.215, top=0.88, wspace=0.13,
    )

    # (a) The collective map.
    ax = fig.add_subplot(outer[0])
    rgb = rgb_composite(phase["R_muc"], phase["R_cw"], phase["R_wmu"])
    d, fd = phase["d"], phase["fd"]
    ax.imshow(rgb, origin="lower", extent=[d[0], d[-1], fd[0], fd[-1]],
              aspect="auto")
    ax.set_box_aspect(1)
    add_phase_axes(ax, sparse_ticks=True)
    phase_diagram._draw_regions(ax, rgb, d, fd, phase_diagram.REGIONS)

    # (b) The same four labels, now as microscopic populations.
    right = outer[1].subgridspec(2, 4, wspace=0.08, hspace=0.18)
    kappa = np.asarray(portraits["kappa"], float)
    short = {
        "(I)": "Frustration",
        "(II)": "Polarization",
        "(III)": "Coupled",
        "(IV)": "Decoupled",
    }
    display_label = {"(III)": "(IIIa)", "(IV)": "(IIIb)"}
    for col, label in enumerate(portraits["labels"]):
        prejudiced = np.asarray(portraits["prejudiced"][col], bool)
        for row, (coords, _, ref) in enumerate(
                state_portraits._sectors(portraits, col, 2)):
            pax = fig.add_subplot(right[row, col])
            _portrait(pax, coords, ref, kappa, prejudiced)
            if row == 0:
                shown = display_label.get(label, label)
                pax.set_title(f"{shown}\n{short[label]}", fontsize=6.2, pad=2,
                              linespacing=0.9)
            if col == 0:
                pax.set_ylabel(("opinion", "trust")[row], fontsize=7,
                               labelpad=2)
    _portrait_legend(fig)
    return save(fig, name, style, bbox=None)


def main():
    args, preset = setup(__doc__)
    model = preset.model.with_(n_issues=preset.p_small)
    phase = sweep(model, preset.sweep, tag=f"P{preset.p_small}",
                  use_cache=not args.no_cache)
    portraits = state_portraits.run(
        preset, preset.p_small, use_cache=not args.no_cache)
    figure(phase, portraits, args.style)


if __name__ == "__main__":
    main()
