#!/usr/bin/env python3
"""The phase diagram: three order parameters composited into one picture.

The three pair correlations are mapped to the three colour channels -- red for
trust-class, green for opinion-class, blue for opinion-trust -- so that the
colour of a point names the collective state of the society:

(I)   dark          reverse discrimination.  A frustrated, spin-glass-like state:
                    agents favour the out-group, no coherent faction can form,
                    and no correlation survives.
(II)  blue          neutral.  The society polarizes, but along a split that has
                    nothing to do with class.
(III) pale          discriminatory, ideological.  Distrust follows class *and*
                    opinions follow class: two coherent, opposed camps.
(IV)  magenta       discriminatory, class-only.  Distrust follows class while
                    opinions do not: hostility needs no disagreement to sustain
                    it.

Writes ``phase_diagram``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from _cli import setup  # noqa: E402

from ednna.plotting import (  # noqa: E402
    LABELS, add_phase_axes, panel, phase_map, rgb_composite, save, text_width,
)
from ednna.sweep import sweep  # noqa: E402

#: label -> (d, f_d) placement.  Both agendas have all four regions; what differs is
#: how much of the d > 0 half each of (III) and (IV) takes.  The discriminant is the
#: ratio R_cw/R_muc -- whether opinion follows class as strongly as trust does -- and
#: it rises with f_d in both, from 0.03 to 0.48 along d = 0.75 for the simple agenda
#: and from 0.57 to 1.00 for the complex one.  So (IV) sits at moderate f_d and high
#: d, and (III) at the top of the discriminatory band, which the simple agenda barely
#: reaches and the complex one owns.
REGIONS = {
    "(I)": (-0.55, 0.68),
    "(II)": (-0.06, 0.42),
    "(III)": (0.50, 0.94),
    "(IV)": (0.76, 0.55),
}

REGIONS_COMPLEX = {
    "(I)": (-0.55, 0.70),
    "(II)": (-0.04, 0.30),
    "(III)": (0.33, 0.88),
    "(IV)": (0.75, 0.37),
}


def figure(data, style, name="phase_diagram", regions=REGIONS):
    rgb = rgb_composite(data["R_muc"], data["R_cw"], data["R_wmu"])
    d, fd = data["d"], data["fd"]
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    ax.imshow(rgb, origin="lower", extent=[d[0], d[-1], fd[0], fd[-1]], aspect="auto")
    ax.set_box_aspect(1)
    add_phase_axes(ax)
    _draw_regions(ax, rgb, d, fd, regions)
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)


def _draw_regions(ax, rgb, d, fd, regions):
    """Roman numerals at their (d, f_d) placements.

    Region (III) is nearly white for a complex agenda and the frustrated corner is
    nearly black, so each label takes its colour from the pixel underneath rather
    than being fixed.
    """
    for label, (x, y) in (regions or {}).items():
        i = int(np.clip(np.searchsorted(fd, y), 0, len(fd) - 1))
        j = int(np.clip(np.searchsorted(d, x), 0, len(d) - 1))
        luma = float(rgb[i, j] @ (0.299, 0.587, 0.114))
        ax.text(
            x, y, label, color="black" if luma > 0.55 else "white",
            fontsize=8, ha="center", va="center", path_effects=None,
        )



#: f_d values the line cut draws, as fractions.  The lowest is one per cent, where
#: a single society cannot be told from one with no biased agent in it; the rest
#: span the range to a fully prejudiced population.  Four, not more: each is two
#: curves, and the point is read off their spacing rather than off any one of them.
CUT_FRACTIONS = (0.01, 0.10, 0.50, 1.00)

#: half-width of the band of rows pooled around each of them.  One pixel row is one
#: realization, so a cut through a single row is unreadable; the band is narrow
#: enough that f_d barely varies across it.
CUT_HALFWIDTH = 0.02

#: The map and the cut print side by side at the same width, so they are generated
#: at one size and one axes rectangle rather than each cropped to its own content:
#: the cut's y label is the longer of the two, and letting it set the crop leaves
#: that panel visibly shorter than the map on the page.
PAIR_ASPECT = 1.0
PAIR_RECT = dict(left=0.235, right=0.975, bottom=0.165, top=0.975)


def _line_cut(ax, data, fractions=CUT_FRACTIONS, half=CUT_HALFWIDTH):
    """Both class correlations along d, one colour per f_d.

    The phase diagram shows region (II) running along both d = 0 at every f_d and
    small f_d at every d.  That is the statement the audit argument rests on -- the
    collective state turns on f_d, which is not an attribute any single agent has --
    and it is easier to read off a cut than off a colour.

    Both class correlations are drawn, solid for R_muc and dotted for R_cw.
    Neutral is a claim about both: a curve showing only trust would leave open that
    opinion had followed the label instead.  Their sum would carry the same headline
    -- over both sweeps no pixel has the two at opposite signs -- but two curves say
    which of the sectors the label has been written into, and on the narrow agenda
    that is the whole difference between regions (III) and (IV).
    """
    d, fd = data["d"], data["fd"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(fractions)))
    k = np.ones(7) / 7
    # mode="same" pads with zeros, which drags both ends of every curve towards the
    # axis and invents a droop at |d| = 1 that is not in the data; dividing by the
    # same convolution of a ones-vector is the edge-correct running mean
    norm = np.convolve(np.ones_like(d), k, mode="same")

    for frac, c in zip(fractions, colors):
        m = (fd >= frac - half) & (fd <= frac + half)
        if not m.any():
            continue
        for key, ls in (("R_muc", "-"), ("R_cw", ":")):
            rows = data[key][m]
            mu = np.convolve(rows.mean(0), k, mode="same") / norm
            se = np.convolve(rows.std(0) / np.sqrt(rows.shape[0]), k, mode="same") / norm
            ax.plot(d, mu, ls, color=c, lw=1.1,
                    # the fraction, not a percentage: it is the same variable as
                    # the vertical axis of the map beside this panel, and a reader
                    # should be able to find a curve there without converting
                    label=f"{frac:.2f}" if key == "R_muc" else None)
            ax.fill_between(d, mu - se, mu + se, color=c, alpha=0.16, lw=0)

    ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
    ax.axvline(0.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(d[0], d[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_xlabel("$p$")
    ax.set_ylabel(f"{LABELS['R_muc']},  {LABELS['R_cw']}")
    ax.set_box_aspect(1)

    style = [Line2D([], [], color="0.35", ls=ls, lw=1.1) for ls in ("-", ":")]
    first = ax.legend(title="$f_p$", fontsize=6, title_fontsize=6.5,
                      loc="upper left", frameon=False, handlelength=1.1,
                      labelspacing=0.22, borderpad=0.2)
    first._legend_box.align = "left"
    ax.add_artist(first)
    ax.legend(style, [LABELS["R_muc"], LABELS["R_cw"]], fontsize=6,
              loc="lower right", frameon=False, handlelength=1.4,
              labelspacing=0.22, borderpad=0.2)


def figure_cut(data, style, name="phase_diagram_cut"):
    """The two class correlations along d at several f_d, as its own figure."""
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    _line_cut(ax, data)
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)



def figure_with_maps(data, style, name="correlations_and_phase", regions=REGIONS):
    """The three correlations and the composite they make, on one row.

    Four square panels: ``R_wmu``, ``R_muc`` and ``R_cw`` with a colour bar under
    each, then the composite of the three at the right with none, since its scale is
    the three bars to its left.  Putting them on one row is the point: the reader can
    see which panel contributes which channel instead of holding one figure in mind
    while reading the next.
    """
    keys = ("R_wmu", "R_muc", "R_cw")
    n_cols = len(keys) + 1
    left, right, bottom, top = 0.055, 0.995, 0.085, 0.90
    # hspace has to clear the d label between a panel and its colour bar, and it is
    # a fraction of the average row height, which the thin bar row drags down
    wspace, hspace, cbar = 0.28, 0.62, 0.075
    W = text_width()
    panel_w = W * (right - left) / (n_cols + (n_cols - 1) * wspace)
    # one square row plus a thin colour-bar row, with the gap between them
    H = panel_w * (1 + cbar) * (1 + hspace / 2) / (top - bottom)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1, cbar],
                          left=left, right=right, bottom=bottom, top=top,
                          wspace=wspace, hspace=hspace)

    d, fd = data["d"], data["fd"]
    for j, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        im = phase_map(ax, data[key], d, fd, key, ylabel=(j == 0),
                       colorbar=False, sparse_ticks=True)
        cax = fig.add_subplot(gs[1, j])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.8, pad=1)
        cb.outline.set_linewidth(0.4)

    ax = fig.add_subplot(gs[0, -1])
    rgb = rgb_composite(data["R_muc"], data["R_cw"], data["R_wmu"])
    ax.imshow(rgb, origin="lower", extent=[d[0], d[-1], fd[0], fd[-1]], aspect="auto")
    add_phase_axes(ax, ylabel=False, sparse_ticks=True)
    ax.tick_params(labelleft=False)
    # the three to the left carry math labels, which sit lower than upright text of
    # the same size, so the word is set a little smaller to read as the same weight
    ax.set_title("composite", pad=3, fontsize=8.5)
    _draw_regions(ax, rgb, d, fd, regions)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    # one panel per agenda: the paper carries the simple agenda and sends the complex
    # one to an appendix, since the four regions are the same and only their extents
    # differ
    for P, regions, name in ((preset.p_small, REGIONS, "phase_diagram"),
                             (preset.p_large, REGIONS_COMPLEX, "phase_diagram_large_agenda")):
        model = preset.model.with_(n_issues=P)
        data = sweep(model, preset.sweep, tag=f"P{P}", use_cache=not args.no_cache)
        figure(data, args.style, name=name, regions=regions)
        figure_cut(data, args.style, name=f"{name}_cut")
        suffix = "" if P == preset.p_small else "_large_agenda"
        figure_with_maps(data, args.style, name=f"correlations_and_phase{suffix}",
                         regions=regions)


if __name__ == "__main__":
    main()
