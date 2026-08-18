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

from _cli import setup  # noqa: E402

from ednna.plotting import (  # noqa: E402
    add_phase_axes, panel, rgb_composite, save, text_width,
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
    fig, ax = plt.subplots(figsize=panel(0.55, 0.52/0.62))
    ax.imshow(rgb, origin="lower", extent=[d[0], d[-1], fd[0], fd[-1]], aspect="auto")
    add_phase_axes(ax)
    _draw_regions(ax, rgb, d, fd, regions)
    _draw_key(ax)
    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


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


def _draw_key(ax):
    """Which correlation is which colour, so the composite reads without the caption.

    Top left, which is the frustrated corner and therefore dark under both agendas;
    the bottom left is bright magenta for a complex agenda and would swallow it.
    """
    for i, (txt, colour) in enumerate(
        ((r"$R_{\mu,c}$", "#ff5555"), (r"$R_{c,w}$", "#55dd55"), (r"$R_{w,\mu}$", "#7777ff"))
    ):
        ax.text(
            0.015, 0.975 - 0.075 * i, txt, transform=ax.transAxes, color=colour,
            fontsize=6.5, ha="left", va="top",
        )


def pair_figure(rows, style, name="phase_diagram"):
    """The two agendas side by side, which is the comparison that carries (IV).

    Only the simple agenda has a class-only region: at ``alpha > 1`` opinion is
    free to follow class everywhere, so (III) never gives way to (IV) and the
    right-hand panel has three states where the left has four.
    """
    # square panels, as in the order-parameter maps: set the box aspect and let the
    # figure height follow from the width one panel gets
    W = 0.92 * text_width()
    fig, axes = plt.subplots(1, len(rows), figsize=(W, W / len(rows) * 1.24))
    for ax, (alpha, data, regions) in zip(np.atleast_1d(axes), rows):
        rgb = rgb_composite(data["R_muc"], data["R_cw"], data["R_wmu"])
        d, fd = data["d"], data["fd"]
        ax.imshow(rgb, origin="lower", extent=[d[0], d[-1], fd[0], fd[-1]], aspect="auto")
        add_phase_axes(ax, ylabel=(ax is axes[0]))
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
        ax.set_title(rf"$\alpha={alpha:.3g}$", fontsize=8, pad=3)
        ax.set_box_aspect(1)
        _draw_regions(ax, rgb, d, fd, regions)
    _draw_key(axes[0])
    fig.tight_layout(pad=0.4, w_pad=1.2)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    rows = []
    for P, regions in ((preset.p_small, REGIONS), (preset.p_large, REGIONS_COMPLEX)):
        model = preset.model.with_(n_issues=P)
        data = sweep(model, preset.sweep, tag=f"P{P}", use_cache=not args.no_cache)
        rows.append((model.alpha, data, regions))
    pair_figure(rows, args.style)


if __name__ == "__main__":
    main()
