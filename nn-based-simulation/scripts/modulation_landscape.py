#!/usr/bin/env python3
"""The modulation functions over the (h_w, h_mu) plane.

Produces three figures, none of which needs a simulation:

``modulation_surfaces``
    The two first-order modulation functions as surfaces, each standing over its
    own contour map on the floor: the same two panels as ``modulation_contours``,
    seen from the side, so that what the colour bar there encodes as a hue is here
    a height.

``modulation_contours``
    All four modulation functions as filled contour maps.  The mirror symmetry
    between the sectors, F_w(x, y) = F_mu(y, x), is visible as a reflection
    across the diagonal, which is drawn.

``modulation_shift``
    ``F_mu`` under a prejudice field of either sign, in the same layout as
    ``modulation_contours``, so that the displacement of the separatrix -- the whole
    mechanism of the paper -- can be read off by comparison with it.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402  (adds the package root to sys.path)

from ednna.modulation import F_C, F_V, F_mu, F_w, evidence  # noqa: E402
from ednna.plotting import (  # noqa: E402
    framed_axes, matched_colorbar, panel, pastel, save, text_width,
)

LIM = 4.0
GRID = 400

#: Both axis labels are set here rather than left to ``axes.labelsize``.  The x label
#: is shared across the row and so has to be a ``supxlabel``, which does not inherit
#: that setting; matching the y label to it keeps the two the same size.
AXIS_LABEL_PT = 8


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


def _surface_panel(fig, position, HW, HMU, values, name, cmap, norm, levels,
                   line_levels, lim, ticks, zticks, rulings=25, air=0.35):
    """One surface standing over its own contour map, in the style of Figure 2.

    ``air`` is the gap below the surface, as a fraction of the surface's own
    range, and the floor goes at the bottom of it.  Shared by every figure drawn
    this way, so that a reader moving between them is only ever asked to learn
    the arrangement once.
    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    floor = lo - air * (hi - lo)
    # ``computed_zorder=False`` because the depth mplot3d computes cannot separate
    # artists that share a plane: the floor's fills and its contour lines all sit
    # at one z and the fills swallow the lines.  The stacking is fixed anyway --
    # the surface is everywhere above the floor -- so saying so loses nothing.
    ax = fig.add_subplot(*position, projection="3d", computed_zorder=False)
    ax.contourf(HW, HMU, values, levels=levels, zdir="z", offset=floor,
                cmap=cmap, norm=norm, zorder=1)
    ax.contour(HW, HMU, values, levels=line_levels, zdir="z", offset=floor,
               colors="k", linewidths=0.25, alpha=0.5, zorder=2)
    ax.plot_surface(HW, HMU, values, cmap=cmap, norm=norm, linewidth=0,
                    antialiased=True, rcount=HW.shape[0], ccount=HW.shape[1],
                    shade=False, zorder=5)
    step = max(1, (HW.shape[0] - 1) // (rulings - 1))
    ax.plot_wireframe(HW, HMU, values, rstride=step, cstride=0, colors="k",
                      linewidths=0.25, alpha=0.4, zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(floor, hi)
    # Numbered, but only over the inner part of the range: the frame is
    # deliberately wider than the last tick, as in the flat figures.
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=5.0, pad=-5.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _pane_style(ax)
    _height_axis(ax, lim, floor, hi, name, zticks)
    # This azimuth puts the dissonant corners at the back and the front, with the
    # consonant sheet between them, and brings both axes of the plane to the near
    # edges.
    ax.view_init(elev=22, azim=34)
    ax.set_box_aspect((1, 1, 1.02), zoom=0.94)
    _axis_words(ax, lim, floor)    # last: it projects, so the view must be set
    return ax


def figure_surfaces(style, lim=5.0, ticks=(-2, 0, 2), zticks=(-2, 0, 2)):
    """The two first-order modulation functions as surfaces over their own maps.

    The height is the whole point of this figure, so nothing here is clipped and
    the plane is taken out to ``lim``: far enough that the evidence ``Z`` is small
    in two of its four corners, and the peak and the pit those corners raise set
    the vertical scale themselves.  Everything between them is nearly flat by
    comparison, which is the statement -- a message that agrees with a trusted
    emitter moves almost nothing, and a dissonant one moves a great deal.

    The flat map is still there, dropped to the floor and left bare: with the
    surface standing over it, lines ruled across it read as part of the solid
    rather than as part of the map.  Floor and surface share one normalization,
    so a colour means the same thing in both and no bar is needed to say so.
    ``ticks`` and ``zticks`` are numbered rather than bare, since with no colour
    bar the height axis is the figure's only scale.
    """
    from matplotlib.colors import FuncNorm

    (HW, HMU), _ = _plane(lim=lim, n=241)
    cmap = pastel("coolwarm", 0.30)
    peak = float(np.abs(F_w(HW, HMU)).max())
    # A linear ramp over the full range would leave the whole middle of the plane
    # one pale colour, the two corners being several times anything else on it.
    # Colour and contour levels are therefore stretched about zero by the same
    # power law -- gentle enough to invent no structure, steep enough that the
    # floor reads as the flat figure's panels do.
    gamma = 2.2
    t = np.linspace(-1.0, 1.0, 41)
    levels = np.sign(t) * np.abs(t) ** gamma * peak
    # Every fifth level gets a line, except the zero level.  Each function
    # vanishes on one of the two axes -- F_w on h_mu = 0 and F_mu on h_w = 0 --
    # so drawing it puts a line across one floor and a line at right angles to it
    # across the other, which reads as an inconsistency rather than as a contour.
    line_levels = levels[::5]
    line_levels = line_levels[np.abs(line_levels) > 1e-12]
    norm = FuncNorm(
        (lambda v: 0.5 * (1.0 + np.sign(v) * np.abs(v / peak) ** (1.0 / gamma)),
         lambda u: np.sign(2 * u - 1) * np.abs(2 * u - 1) ** gamma * peak),
        vmin=-peak, vmax=peak)
    W = 0.94 * text_width()
    fig = plt.figure(figsize=(W, 0.45 * W))
    for k, (F, name) in enumerate(((F_w, r"$F_w$"), (F_mu, r"$F_\mu$"))):
        _surface_panel(fig, (1, 2, k + 1), HW, HMU, F(HW, HMU), name, cmap, norm,
                       levels, line_levels, lim, ticks, zticks)
    # Margins by hand: text placed from a projection is not counted by
    # ``bbox_inches="tight"``, so the axis names need room left for them or they
    # are cropped at the figure edge.
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.03, top=1.0, wspace=0.0)
    return save(fig, "modulation_surfaces", style)


def figure_evidence(style, lim=3.0, ticks=(-2, 0, 2)):
    """The evidence and its logarithm, in the arrangement of the surface figure.

    ``Z`` is what the receiver thinks the message was worth believing before it
    hears it, and the four modulation functions are all derivatives of ``log Z``,
    so ``-log Z`` is the surface the learning step descends.  Put side by side the
    two say the same thing twice: the ridge of ``Z`` along the consonant diagonal
    is the valley of ``-log Z``, and the dissonant corners where ``Z`` vanishes are
    where the descent is steepest, which is why the modulation functions are
    largest there.

    A sequential palette, not the diverging one of :func:`figure_surfaces`.  Both
    quantities here are positive and have no distinguished middle value, and the
    change of palette also says at a glance that this figure is not another pair
    of modulation functions.
    """
    from matplotlib.colors import PowerNorm

    (HW, HMU), _ = _plane(lim=lim, n=241)
    cmap = pastel("Purples", 0.12)
    Z = evidence(HW, HMU)
    #: ``gamma`` is the colour stretch.  ``Z`` fills its range evenly and wants
    #: none; ``-log Z`` is near zero over most of the plane and climbs only in the
    #: two corners, so a linear ramp would leave its floor blank.
    fields = ((Z, r"$Z$", (0.0, 0.5, 1.0), 1.0),
              (-np.log(np.maximum(Z, 1e-12)), r"$-\ln Z$", (0, 2, 4), 0.5))
    W = 0.94 * text_width()
    fig = plt.figure(figsize=(W, 0.45 * W))
    for k, (values, name, zticks, gamma) in enumerate(fields):
        lo, hi = float(values.min()), float(values.max())
        t = np.linspace(0.0, 1.0, 41)
        levels = lo + (hi - lo) * t ** (1.0 / gamma)
        norm = PowerNorm(gamma, vmin=lo, vmax=hi)
        _surface_panel(fig, (1, 2, k + 1), HW, HMU, values, name, cmap, norm,
                       levels, levels[::5], lim, ticks, zticks)
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.03, top=1.0, wspace=0.0)
    return save(fig, "evidence_surfaces", style)


def figure_landscape(style, lim=4.0, ticks=(-2, 0, 2)):
    """The energy the learning step descends, and the two steps it produces.

    One plane, three surfaces, read left to right as the derivation runs.  The
    first is ``-log Z``, the energy: the update is a gradient step in ``log Z``,
    so this is the surface the step runs down, and it diverges in the two
    dissonant corners -- agreeing with a distrusted emitter, disagreeing with a
    trusted one -- where the message was least expected.  The second and third are
    its two first derivatives, the modulation functions that move the opinion and
    the trust, and their peaks sit exactly on the walls of the first: the step is
    large where the energy is steep, which is the whole content of the pair.

    The palettes differ on purpose.  ``-log Z`` is positive with no distinguished
    middle value and takes a sequential map; the modulation functions are signed
    and take the diverging one, centred on the zero that separates learning from
    unlearning.  ``lim`` is a compromise: the modulation functions want a wide
    plane to show their peaks as peaks, while ``-log Z`` grows without bound and
    past about four takes the whole vertical scale for itself.
    """
    from matplotlib.colors import FuncNorm, PowerNorm

    (HW, HMU), _ = _plane(lim=lim, n=241)
    energy = -np.log(np.maximum(evidence(HW, HMU), 1e-12))
    hi = float(energy.max())
    t = np.linspace(0.0, 1.0, 41)
    #: A square-root ramp: the energy is near zero over most of the plane and
    #: climbs only in the corners, so a linear one leaves its floor blank.
    e_levels = hi * t ** 2
    e_norm = PowerNorm(0.5, vmin=0.0, vmax=hi)

    peak = float(np.abs(F_w(HW, HMU)).max())
    gamma = 2.2
    u = np.linspace(-1.0, 1.0, 41)
    f_levels = np.sign(u) * np.abs(u) ** gamma * peak
    f_lines = f_levels[::5][np.abs(f_levels[::5]) > 1e-12]
    f_norm = FuncNorm(
        (lambda v: 0.5 * (1.0 + np.sign(v) * np.abs(v / peak) ** (1.0 / gamma)),
         lambda w: np.sign(2 * w - 1) * np.abs(2 * w - 1) ** gamma * peak),
        vmin=-peak, vmax=peak)

    zt = (-2, 0, 2)
    panels = (
        (energy, r"$-\ln Z$", pastel("Purples", 0.12), e_norm, e_levels,
         e_levels[::5], (0, 4, 8)),
        (F_w(HW, HMU), r"$F_w$", pastel("coolwarm", 0.30), f_norm, f_levels,
         f_lines, zt),
        (F_mu(HW, HMU), r"$F_\mu$", pastel("coolwarm", 0.30), f_norm, f_levels,
         f_lines, zt),
    )
    W = text_width()
    fig = plt.figure(figsize=(W, 0.32 * W))
    for k, (values, name, cmap, norm, levels, lines, zticks) in enumerate(panels):
        _surface_panel(fig, (1, 3, k + 1), HW, HMU, values, name, cmap, norm,
                       levels, lines, lim, ticks, zticks)
    # More room on the left than the two-panel figures need: the name on the first
    # height axis is "-ln Z", not a single letter.  Saved at exactly this size
    # rather than cropped to content, because a tight box measures the artists it
    # knows about and the height axes are drawn from a projection; cropping to it
    # takes back the very margin left for them.
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.04, top=1.0, wspace=0.0)
    return save(fig, "learning_landscape", style, bbox=None)


def _height_axis(ax, lim, floor, top, name, zticks):
    """A height axis on the left-hand corner, carrying the name and a few marks.

    Matplotlib puts the ``z`` axis on whichever corner the azimuth chooses, which
    at this view is a back one, where it reads as belonging to the box rather
    than to the surface.  It is hidden and redrawn here on the leftmost corner,
    ``(h_w, h_mu) = (+lim, -lim)``, with its own ticks: there is no colour bar, so
    this axis is the only scale in the figure, and without a number or two on it
    the reader cannot tell a peak of three from a peak of thirty.  Only the inner
    marks are drawn, not the divergent ends, and the name sits at the top rather
    than over the panel, because here the height is what it labels.
    """
    # One tick kept and blanked rather than none at all: an axis with no ticks at
    # all gives ``bbox_inches="tight"`` an empty bounding box to union and it
    # raises.
    ax.set_zticks([0.0])
    ax.set_zticklabels([""])
    ax.zaxis.set_tick_params(length=0, width=0)
    ax.zaxis.line.set_visible(False)
    x0, y0 = lim, -lim

    # outward from that corner is +h_w and -h_mu together
    def out(d, z, text, **kw):
        ax.text(x0 + d * lim, y0 - d * lim, z, text, zorder=7,
                **{"color": "0.15", "ha": "right", "va": "center", **kw})

    ax.plot([x0, x0], [y0, y0], [floor, top], color="0.25", lw=0.6, zorder=7)
    for z in zticks:
        ax.plot([x0, x0 + 0.045 * lim], [y0, y0 - 0.045 * lim], [z, z],
                color="0.25", lw=0.6, zorder=7)
        out(0.115, z, f"${z:g}$", fontsize=AXIS_LABEL_PT - 2.0)
    out(0.10, top + 0.06 * (top - floor), name, fontsize=AXIS_LABEL_PT + 1.5, color="#2f4f7f")


#: Each axis name, as ``(text, which axis, where along it in units of lim, how
#: far out in points)``.  Only the names: the flat figures spell out which end is
#: which in words, and here those words have to sit where the tick numbers
#: already are.
_WORDS = (
    (r"$h_w$",    "w",   0.00, 16.0),
    (r"$h_\mu$",  "mu",  0.00, 16.0),
)


def _axis_words(ax, lim, floor):
    """The wording of the flat figures' axis labels, spelled along the 3-D axes.

    ``set_xlabel`` on a 3-D axes gives one string, centred and horizontal, which
    is no place for ``disagree <- h_w -> agree``.  The parts are therefore placed
    by hand, on the page rather than in the plane: each is projected to its point
    on the floor's near edge and then stepped off that edge along its *outward
    normal, measured on the page*.  Offsetting in the plane instead does not work
    --- the projection is oblique, so a step in ``+h_w`` slides the word along the
    edge into the tick numbers rather than away from it --- and neither does a
    fixed screen direction, which would have to be retuned for every view angle.
    """
    from mpl_toolkits.mplot3d import proj3d

    ax.figure.canvas.draw()          # the projection is not final until a draw
    to_axes = ax.transAxes.inverted()

    def at(hw, hmu):
        px, py, _ = proj3d.proj_transform(hw, hmu, floor, ax.get_proj())
        return np.array(ax.transData.transform((px, py)), dtype=float)

    # The floor projects to a rhombus, so "away from the centre" is not the same
    # as "perpendicular to this edge"; the normal is taken from the edge itself
    # and only its sign is settled by the centre.
    middle = sum(at(u * lim, v * lim)
                 for u, v in ((-1, -1), (-1, 1), (1, -1), (1, 1))) / 4.0
    edges = {}
    for axis, (p0, p1) in (("w", ((-lim, lim), (lim, lim))),
                           ("mu", ((lim, -lim), (lim, lim)))):
        a, b = at(*p0), at(*p1)
        d = (b - a) / np.hypot(*(b - a))
        n = np.array([d[1], -d[0]])
        if n @ (0.5 * (a + b) - middle) < 0:
            n = -n
        edges[axis] = n

    px_per_pt = ax.figure.dpi / 72.0
    for text, axis, u, pad in _WORDS:
        anchor = at(u * lim, lim) if axis == "w" else at(lim, u * lim)
        # Stepped off in pixels, then handed over as a fraction of the axes: the
        # pixels of a draw belong to that renderer, and this figure is drawn once
        # on screen at one dpi and saved to PDF at another.  A fraction of the
        # axes is the same place in both, and since the axes has a fixed physical
        # size the pad stays the 5 or 13 points it was asked for.
        fx, fy = to_axes.transform(anchor + edges[axis] * pad * px_per_pt)
        # ``text2D``, not ``text``: the latter on a 3-D axes wants a z as well and
        # would place this back in the plane.
        ax.text2D(fx, fy, text, transform=ax.transAxes, zorder=7, color="0.15",
                  ha="center", va="center",
                  fontsize=AXIS_LABEL_PT - (0.0 if text.startswith("$") else 1.0))


def _pane_style(ax):
    """The frame of :func:`framed_axes`, as near as a 3-D axes can come to it.

    Three white panes with a thin grey edge and a faint grid, instead of
    Matplotlib's shaded panes and heavy grid, so the surfaces sit in the same
    light box the flat panels are drawn in.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("0.25")
        axis.pane.set_linewidth(0.6)
        axis.line.set_color("0.25")
        axis.line.set_linewidth(0.6)
        axis._axinfo["grid"].update(color="0.85", linewidth=0.4)
    ax.tick_params(color="0.25", width=0.6, length=2.2)


def _contour_panel(ax, F, name, cap, ylabel=False, shift=0.0, lim=LIM,
                   ticks=(-4, -2, 0, 2, 4)):
    """One filled-contour panel over the ``(h_w, h_mu)`` plane.

    ``shift`` is a group-bias field: the receiver sits at ``h_w`` but evaluates
    the modulation at ``h_w + b``, so the panel is the same function seen through a
    displaced coordinate, and the separatrix moves with it.
    """
    (HW, HMU), _ = _plane(lim=lim)
    levels = np.linspace(-cap, cap, 25)
    values = np.clip(F(HW + shift, HMU), -cap, cap)
    im = ax.contourf(HW, HMU, values, levels=levels, cmap=pastel("coolwarm", 0.30))
    ax.contour(HW, HMU, values, levels=levels[::4], colors="k", linewidths=0.25,
               alpha=0.5)
    # Nothing is drawn over the field.  The separatrix of the trust update -- the
    # line where F_mu changes sign, and so where trust stops growing and starts
    # decaying -- is already in the picture: F_mu carries its sign in
    # 1 - 2 Phi(h_w + b) and in nothing else, so that line is h_w = -b, which is
    # exactly the zero contour the contour set draws.  Drawing it again in a
    # colour would only be an annotation that could drift out of step with the
    # field behind it.
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", labelpad=1)
    if ylabel:
        ax.set_ylabel(r"trust $\leftarrow h_\mu \rightarrow$ distrust", labelpad=1,
                      fontsize=AXIS_LABEL_PT)
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
    # Height chosen so the square panels fill their vertical allocation.  Left with a
    # taller figure they sit centred in it, and the shared x label ends up floating a
    # long way below them.
    left, right, bottom, top, wspace = 0.115, 0.87, 0.20, 0.98, 0.12
    W = 0.86 * text_width()
    panel_w = W * (right - left) / (2 + wspace)
    fig, axes = plt.subplots(1, 2, figsize=(W, panel_w / (top - bottom)), sharey=True)
    im = None
    for k, (F, name) in enumerate(((F_w, r"$F_w$"), (F_mu, r"$F_\mu$"))):
        im = _contour_panel(axes[k], F, name, 1.6, ylabel=(k == 0),
                            lim=2.25, ticks=(-2, 0, 2))
        axes[k].set_xlabel("")   # one shared label below, or the two collide
    axes[1].tick_params(labelleft=True)
    fig.supxlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", fontsize=AXIS_LABEL_PT, y=0.045)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)
    cb = matched_colorbar(fig, im, axes[-1], ticks=np.linspace(-1.6, 1.6, 5))
    cb.set_label(r"$F_w,\ F_\mu$", fontsize=7)
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
    images = {}
    for i, (left, right, cap) in enumerate(rows):
        for j, (F, name) in enumerate((left, right)):
            images[i] = _contour_panel(axes[i][j], F, name, cap, ylabel=(j == 0))
            if i == 0:
                axes[i][j].set_xlabel("")
    fig.subplots_adjust(left=0.105, right=0.87, bottom=0.10, top=0.97,
                        wspace=0.12, hspace=0.22)
    for i, (_, _, cap) in enumerate(rows):
        matched_colorbar(fig, images[i], axes[i][-1], ticks=np.linspace(-cap, cap, 5))
    return save(fig, "modulation_contours_all", style)


def figure_shift(style, bias=1.0):
    """What the prejudice field does to the trust sector, in one picture.

    The contour layout of :func:`figure_contours` applied to ``F_mu`` alone, for an
    out-group pair at the three values of the field the flow figure uses.
    ``F_mu`` is the right function to show, because the field enters through ``h_w``
    and the prefactor ``1 - 2*Phi(h_w)`` in ``F_mu`` is what carries it into the
    trust sector, so this is the panel on which the mechanism of the paper is a
    displacement of one line.  That line is the separatrix of the trust update,
    at ``h_w = b`` for an out-group pair, on one side of which trust grows and on
    the other of which it decays; it needs no drawing, being the zero contour of
    the function shown.  The middle panel is the unbiased case and the outer two
    are the same function seen through a shifted coordinate.
    """
    #: An *out-group* pair, for which kappa_r kappa_e = -1 and the receiver
    #: evaluates the update at h_w - b: titled by b, therefore, and shifted by -b.
    #: An in-group pair is the same three pictures with the order of b reversed,
    #: which the caption says rather than the figure repeating.
    biases = (-bias, 0.0, +bias)
    lim = 2.25
    cap = 1.6
    ticks = (-2, 0, 2)
    fig, axes = plt.subplots(1, len(biases), figsize=panel(1.0, 0.42), sharey=True)
    im = None
    for k, b in enumerate(biases):
        name = rf"$b = {b:+.0f}$" if b else r"$b = 0$"
        im = _contour_panel(axes[k], F_mu, None, cap, ylabel=(k == 0),
                            shift=-b, lim=lim, ticks=ticks)
        axes[k].set_xlabel("")
        axes[k].set_title(name, fontsize=7.5, pad=3)
    fig.supxlabel(r"disagree $\leftarrow h_w \rightarrow$ agree", fontsize=AXIS_LABEL_PT, y=0.065)
    fig.subplots_adjust(left=0.095, right=0.89, bottom=0.14, top=0.93, wspace=0.10)
    # Every panel is the same field, named in the caption; the bar carries only
    # its numbers, as in the matched pair of figures this layout comes from.
    matched_colorbar(fig, im, axes[-1], ticks=np.linspace(-cap, cap, 5))
    return save(fig, "modulation_shift", style)


def main():
    args, _ = setup(__doc__)
    figure_surfaces(args.style)
    figure_evidence(args.style)
    figure_landscape(args.style)
    figure_contours(args.style)
    figure_contours_all(args.style)
    figure_shift(args.style)


if __name__ == "__main__":
    main()
