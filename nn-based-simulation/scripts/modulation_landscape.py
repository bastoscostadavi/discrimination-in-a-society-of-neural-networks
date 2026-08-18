#!/usr/bin/env python3
"""The modulation functions over the (h_w, h_mu) plane.

Produces three figures, none of which needs a simulation:

``modulation_surfaces``
    The two first-order modulation functions as surfaces with the evidence Z
    drawn on the floor, showing that the large changes happen exactly where the
    evidence is low -- the dissonant quadrants.

``modulation_contours``
    All four modulation functions as filled contour maps.  The mirror symmetry
    between the sectors, F_w(x, y) = F_mu(y, x), is visible as a reflection
    across the diagonal, which is drawn.

``modulation_shift``
    ``F_mu`` under a discrimination field of either sign, in the same layout as
    ``modulation_contours``, so that the displacement of the separatrix -- the whole
    mechanism of the paper -- can be read off by comparison with it.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402  (adds the package root to sys.path)

from ednna.modulation import F_C, F_V, F_mu, F_w, evidence  # noqa: E402
from ednna.plotting import framed_axes, panel, pastel, save  # noqa: E402

LIM = 4.0
GRID = 400


def _plane(lim=LIM, n=GRID):
    ax = np.linspace(-lim, lim, n)
    return np.meshgrid(ax, ax), ax


def _decorate(ax, lim=LIM, diagonal=True):
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    if diagonal:
        ax.plot([-lim, lim], [-lim, lim], color="tab:green", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"disagree $\;\leftarrow\;h_w\;\rightarrow\;$ agree")
    ax.set_ylabel(r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust")
    ax.set_aspect("equal")


def figure_surfaces(style):
    (HW, HMU), _ = _plane(n=120)
    Z = evidence(HW, HMU)
    fig = plt.figure(figsize=panel(0.90, 0.44))
    for i, (F, name) in enumerate(((F_mu, r"$F_\mu$"), (F_w, r"$F_w$"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        values = np.clip(F(HW, HMU), -3.0, 3.0)
        ax.plot_surface(
            HW, HMU, values, cmap="coolwarm", linewidth=0, antialiased=True,
            rcount=120, ccount=120, alpha=0.95,
        )
        # the evidence on the floor, light where the message is surprising: the
        # large excursions of F sit directly above the two light quadrants
        ax.contourf(HW, HMU, Z, levels=24, zdir="z", offset=-4.2, cmap="Greys", alpha=0.7)
        ax.set_zlim(-4.2, 3.2)
        ax.set_xlabel(r"$h_w$", labelpad=-6)
        ax.set_ylabel(r"$h_\mu$", labelpad=-6)
        ax.set_title(name, pad=-2)
        ax.tick_params(labelsize=5, pad=-2)
        ax.view_init(elev=26, azim=-52)
        ax.set_box_aspect((1, 1, 0.62))
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.02)
    return save(fig, "modulation_surfaces", style)


def _contour_panel(ax, F, name, cap, ylabel=False, shift=0.0):
    """One filled-contour panel over the ``(h_w, h_mu)`` plane.

    ``shift`` is a discrimination field: the receiver sits at ``h_w`` but evaluates
    the modulation at ``h_w + D``, so the panel is the same function seen through a
    displaced coordinate, and the separatrix moves with it.
    """
    (HW, HMU), _ = _plane()
    levels = np.linspace(-cap, cap, 25)
    values = np.clip(F(HW + shift, HMU), -cap, cap)
    im = ax.contourf(HW, HMU, values, levels=levels, cmap=pastel("coolwarm", 0.30))
    ax.contour(HW, HMU, values, levels=levels[::4], colors="k", linewidths=0.25,
               alpha=0.5)
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.plot([-LIM, LIM], [-LIM + shift, LIM + shift], color="#5aa469", lw=0.9)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", labelpad=1)
    if ylabel:
        ax.set_ylabel(r"trust $\leftarrow h_\mu \rightarrow$ distrust", labelpad=1)
    ax.text(0.05, 0.93, name, transform=ax.transAxes, color="#2f4f7f",
            ha="left", va="top")
    framed_axes(ax, minor=False)
    return im


def figure_contours(style):
    """The two first-order modulation functions, side by side.

    ``F_w`` left and ``F_mu`` right, which is the order the sectors are introduced
    in and puts the opinion sector first.  They share a range exactly --- the
    sectors are mirror images, ``F_w(x, y) = F_mu(y, x)`` --- so one colour bar
    serves both, and the symmetry reads as one panel being the other reflected
    across the diagonal.  The second-order pair is in
    ``modulation_contours_all``.
    """
    fig, axes = plt.subplots(1, 2, figsize=panel(0.86, 0.35/0.66), sharey=True)
    im = None
    for k, (F, name) in enumerate(((F_w, r"$F_w$"), (F_mu, r"$F_\mu$"))):
        im = _contour_panel(axes[k], F, name, 3.2, ylabel=(k == 0))
        axes[k].set_xlabel("")   # one shared label below, or the two collide
    fig.supxlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", fontsize=8, y=0.04)
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02,
                      ticks=np.linspace(-3.2, 3.2, 5))
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return save(fig, "modulation_contours", style)


def figure_contours_all(style):
    """All four modulation functions, for the appendix.

    The two that move the means on top, the two that anneal the uncertainties
    below, one colour bar per row since each row shares a range.
    """
    fig, axes = plt.subplots(2, 2, figsize=panel(0.92, 0.60/0.68),
                             sharex=True, sharey=True)
    rows = (((F_w, r"$F_w$"), (F_mu, r"$F_\mu$"), 3.2),
            ((F_C, r"$F_C$"), (F_V, r"$F_V$"), 2.5))
    for i, (left, right, cap) in enumerate(rows):
        for j, (F, name) in enumerate((left, right)):
            im = _contour_panel(axes[i][j], F, name, cap, ylabel=(j == 0))
            if i == 0:
                axes[i][j].set_xlabel("")
        cb = fig.colorbar(im, ax=axes[i], fraction=0.042, pad=0.02,
                          ticks=np.linspace(-cap, cap, 5))
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
    return save(fig, "modulation_contours_all", style)


def figure_shift(style, d=1.0):
    """What the discrimination field does to the trust sector, in one picture.

    The contour layout of :func:`figure_contours` applied to ``F_mu`` alone, at the
    three values of the field the flow figure uses: out-group, none, in-group.
    ``F_mu`` is the right function to show, because the field enters through ``h_w``
    and the prefactor ``1 - 2*Phi(h_w)`` in ``F_mu`` is what carries it into the
    trust sector, so this is the panel on which the mechanism of the paper is a
    displacement of one line.  The green separatrix ``h_mu = h_w + D`` is where blame
    for a surprise passes from one sector to the other; the middle panel is the
    unbiased case and the outer two are the same function seen through a shifted
    coordinate.
    """
    shifts = (-d, 0.0, +d)
    fig, axes = plt.subplots(1, len(shifts), figsize=panel(1.0, 0.42), sharey=True)
    im = None
    for k, shift in enumerate(shifts):
        name = rf"$D = {shift:+.0f}$" if shift else r"$D = 0$"
        im = _contour_panel(axes[k], F_mu, name, 3.2, ylabel=(k == 0), shift=shift)
        axes[k].set_xlabel("")
        axes[k].set_title(("out-group", "no bias", "in-group")[k], fontsize=7.5, pad=3)
    fig.supxlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", fontsize=8, y=0.02)
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.015,
                      ticks=np.linspace(-3.2, 3.2, 5))
    cb.set_label(r"$F_\mu$", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return save(fig, "modulation_shift", style)


def main():
    args, _ = setup(__doc__)
    figure_surfaces(args.style)
    figure_contours(args.style)
    figure_contours_all(args.style)
    figure_shift(args.style)


if __name__ == "__main__":
    main()
