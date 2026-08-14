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

``modulation_slices``
    Cuts at fixed opinion field, showing which sector absorbs a surprise.  This
    is the figure the source draft discusses in its text but never includes (its
    "figure ??"): for |h_mu| < |h_w| the affective sector moves, and beyond the
    crossover at h_mu = h_w the ideological sector moves instead.
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


def _contour_panel(ax, F, name, cap, ylabel=False):
    (HW, HMU), _ = _plane()
    levels = np.linspace(-cap, cap, 25)
    values = np.clip(F(HW, HMU), -cap, cap)
    im = ax.contourf(HW, HMU, values, levels=levels, cmap=pastel("coolwarm", 0.30))
    ax.contour(HW, HMU, values, levels=levels[::4], colors="k", linewidths=0.25,
               alpha=0.5)
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.plot([-LIM, LIM], [-LIM, LIM], color="#5aa469", lw=0.9)
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


def figure_slices(style, h_w0=6.0, d=2.0, lim=9.0):
    """Blame attribution: which sector yields when the message is surprising.

    At fixed opinion field the two sectors trade places at ``h_mu = h_w``: below
    the crossover the affective sector absorbs the surprise (solid), above it
    the ideological sector does (dashed) and the receiver unlearns.  The
    discrimination field moves the crossover to ``h_w + D``, which is the whole
    mechanism of the model in one picture.
    """
    h_mu = np.linspace(-lim, lim, 1200)
    fig, axes = plt.subplots(1, 2, figsize=panel(0.95, 0.38), sharey=True)
    for ax, sign, title in (
        (axes[0], +1.0, rf"emitter agrees ($h_w = {h_w0:.0f}$)"),
        (axes[1], -1.0, rf"emitter disagrees ($h_w = {-h_w0:.0f}$)"),
    ):
        base = sign * h_w0
        for shift, colour, label in (
            (+d, "#5b8ec4", rf"$D = {d:+.0f}$ (tolerant)"),
            (0.0, "#5aa469", r"$D = 0$"),
            (-d, "#c96a63", rf"$D = {-d:+.0f}$ (intolerant)"),
        ):
            hw = base + shift
            ax.plot(h_mu, F_mu(hw, h_mu), color=colour, lw=1.1, label=label)
            ax.plot(h_mu, F_w(hw, h_mu), color=colour, lw=1.0, ls="--")
            ax.axvline(hw, color=colour, lw=0.5, ls=":", alpha=0.8)
        ax.axhline(0.0, color="0.55", lw=0.5)
        ax.axvline(0.0, color="0.55", lw=0.5)
        framed_axes(ax, minor=False)
        ax.set_xlabel(r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust")
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-lim, lim)
    axes[0].set_ylabel(r"$F_\mu$ (solid), $F_w$ (dashed)")
    axes[0].legend(loc="lower left", fontsize=6)
    axes[1].annotate(
        "crossover at $h_\\mu = h_w + D$",
        xy=(0.5, 0.94), xycoords="axes fraction", ha="center", fontsize=6, color="0.3",
    )
    fig.tight_layout(pad=0.4)
    return save(fig, "modulation_slices", style)


def main():
    args, _ = setup(__doc__)
    figure_surfaces(args.style)
    figure_contours(args.style)
    figure_contours_all(args.style)
    figure_slices(args.style)


if __name__ == "__main__":
    main()
