"""The figure, and the two fits it rests on.

Drawn in the grammar of the paper's Figure 1 and of the sibling experiment's
Figure 2, so that the three can be read against one another: ``h_w`` runs
disagree-to-agree on the x axis, ``h_mu`` runs trust-to-distrust on the y axis,
and the green line is ``h_mu = h_w``, the separatrix across which blame for a
surprise passes from one sector to the other.  The style itself is imported from
``ednna.plotting`` rather than restated, for the same reason the equations are.

What the panels are for:

(a) is the whole measurement laid on the analytic surface.  Unlike the sibling's
    version, every point sits where the design put it rather than where the
    model happened to land, so the plane is covered rather than sampled in two
    bands.
(b) is the claim the experiment exists to test.  ``F_w`` changes sign with the
    *emitter*, not with the agreement, so the two arms -- messages the receiver
    already agreed with, and messages it did not -- must cross zero at the same
    place, and that place must be the uninformative track record.
(c) is the second thing ``F_w`` says and a plain trust-weighted rule does not.
    The size of the update is not symmetric in agreement: ``1/Z`` blows up where
    the message is dissonant, so for a distrusted emitter the larger move comes
    when the receiver *agreed*, and for a trusted one when it disagreed.  The two
    arms should therefore lean in opposite directions, and that mirror image is
    the signature -- a rule that merely scaled the update by trust would give two
    flat lines.

:func:`crossover_panel` is kept for the pair of properties no one has measured
on a language model -- the reflection symmetry between the sectors and the
crossover at ``h_mu = h_w`` -- and needs a trust prior loose enough for one
exchange to move it.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

__all__ = ["use_style", "panel", "save", "framed_axes", "pastel", "fit_scale",
           "plane_panel", "gate_panel", "conviction_panel", "crossover_panel",
           "figure_plane", "figure_gate", "figure_conviction",
           "figure_pair", "CAP_PCT",
           "PANEL_FIG", "PANEL_RECT", "CBAR_RECT", "AXIS_LABEL_PT"]

AXIS_LABEL_PT = 8
_STYLE = {"name": "iclr"}

#: Fraction of the measured updates a panel's colour scale is asked to span.
#: Each panel gets its own, from its own data, and so does each frame.  Sharing
#: them across the two sectors was tried and is worse: the two experiments reach
#: different parts of the plane and their updates are of different sizes, so one
#: frame wide enough for both is mostly empty theory and one clip large enough
#: for both washes the smaller of the two out.
CAP_PCT = 92


def use_style(name="iclr"):
    """Figure 1's style, taken from the package that draws Figure 1."""
    _STYLE["name"] = name
    from ednna.plotting import use_style as ednna_style
    ednna_style(name)
    plt.rcParams.update({"savefig.bbox": None, "figure.dpi": 150})


def panel(frac, aspect):
    w = (5.5 if _STYLE["name"] == "iclr" else 6.3) * frac
    return (w, w * aspect)


def save(fig, name, figure_dir, bbox=None, nest=True):
    """Write ``name`` as a PDF into ``figures/<style>/``.

    ``nest=False`` writes straight into ``figure_dir`` instead, for the one
    figure that belongs to the paper rather than to this experiment.

    PDF only: it is what LaTeX wants, it stays sharp at any size, and a parallel
    set of PNGs is just something else to keep in step.

    Untrimmed by default, and that is deliberate.  These figures are placed side
    by side at one width each -- including beside the sibling experiment's
    ``trust_plane`` -- and trimming each to its own ink would make the one with
    the longer axis label come out smaller than its neighbour.  The fixed
    geometry below is what keeps their frames the same size.
    """
    out = figure_dir / _STYLE["name"] if nest else figure_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path, **({} if bbox is None else
                         {"bbox_inches": bbox, "pad_inches": 0.03}))
    plt.close(fig)
    print(f"[figure] {path}")
    return path


#: One geometry for every panel, copied from the sibling experiment so that the
#: opinion plane and the trust plane are the same size on the page.  The axes
#: rectangle is identical in all of them and the colour bar lives in the strip
#: to its right, which the cuts leave empty.
PANEL_FIG = (2.75, 2.45)
PANEL_RECT = (0.215, 0.185, 0.60, 0.60 * 2.75 / 2.45)
CBAR_RECT = (0.215 + 0.60 + 0.028, 0.185, 0.032, 0.60 * 2.75 / 2.45)


def _standalone():
    fig = plt.figure(figsize=PANEL_FIG)
    return fig, fig.add_axes(PANEL_RECT)


def figure_plane(h_w, h_mu, delta, F, alpha, figure_dir, name="opinion_plane",
                 label=r"$F_w$", nest=True):
    """Panel: the measurement over the plane of Figure 1."""
    fig, ax = _standalone()
    im = plane_panel(ax, h_w, h_mu, delta, F, alpha, name=label)
    cb = fig.colorbar(im, cax=fig.add_axes(CBAR_RECT))
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])
    return save(fig, name, figure_dir, nest=nest)


def figure_gate(h_w, h_mu, delta, F, alpha, figure_dir, name="opinion_gate"):
    """Panel: the sign of the update against prior trust."""
    fig, ax = _standalone()
    gate_panel(ax, h_w, h_mu, delta, F, alpha,
               xlabel=r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust")
    return save(fig, name, figure_dir)


def figure_conviction(h_w, h_mu, delta, F, alpha, figure_dir,
                      name="opinion_conviction"):
    """Panel: the size of the update against conviction, per side of trust."""
    fig, ax = _standalone()
    conviction_panel(ax, h_w, h_mu, delta, F, alpha,
                     xlabel=r"disagree $\;\leftarrow\;h_w\;\rightarrow\;$ agree")
    return save(fig, name, figure_dir)


def pastel(cmap, amount=0.30, n=256):
    base = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    base[:, :3] = base[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"pastel_{cmap}", base)


def framed_axes(ax):
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.25")
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   color="0.25", width=0.6, length=3.0)
    ax.grid(False)
    return ax


def fit_scale(h_w, h_mu, delta, F, cap=None):
    """The one free constant, through the origin.

    The opinion update is ``F_w * C x / gamma_C`` and the opinion covariance is
    not observable here, so the theory carries an overall positive scale and
    nothing else -- the same single scale the sibling fit spends on ``V``.  The
    conversion from pieces of evidence into probit units is *not* fitted here;
    it is calibrated on the track records in :mod:`llmmod2.fields`.

    ``cap`` is what makes this different from the sibling's version, and it
    matters.  ``F_w`` carries a factor ``1/Z`` that diverges as the message
    becomes maximally dissonant, so over this grid it spans from ``0.003`` to
    ``1.64`` -- while the measured update cannot diverge, because a belief can
    only be pushed so far before the ladder runs out.  A plain least squares
    weights by ``F^2`` and therefore puts about two thirds of its weight on the
    9% of conditions in that corner: it fits the divergence and draws the whole
    bulk of the measurement five to twenty times too pale.

    Passing ``cap`` fits instead on the comparison the panel actually draws,
    with both sides clipped where the colour scale is clipped.  That is the
    quantity a reader judges by eye, it does not let one corner speak for the
    plane, and it changes no conclusion: every statistic quoted about the shape
    of the agreement is either a sign or a rank, and neither depends on the
    scale at all.
    """
    m = np.isfinite(h_w) & np.isfinite(h_mu) & np.isfinite(delta)
    f = F(h_w[m], h_mu[m])
    d = delta[m]
    if cap is None:
        return float(np.sum(d * f) / np.sum(f * f))
    lo = float(np.sum(d * f) / np.sum(f * f))
    grid = np.linspace(0.2 * lo, 6.0 * lo, 600)
    err = [np.mean((np.clip(d, -cap, cap) - np.clip(a * f, -cap, cap)) ** 2)
           for a in grid]
    return float(grid[int(np.argmin(err))])


def _decorate(ax, lim, xlabel=True, ticks=None):
    # Five ticks and no more.  The default locator puts nine on a range of this
    # size, which crowds a panel this small and reads as a grid.  Given
    # explicitly rather than by a locator so that the frame can be opened past
    # the last tick without the opening putting a tick of its own on the edge.
    if ticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True,
                                               steps=[1, 2, 5, 10]))
    else:
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], color="#5aa469", lw=0.9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if xlabel:
        ax.set_xlabel(r"disagree $\;\leftarrow\;h_w\;\rightarrow\;$ agree",
                      fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.set_ylabel(r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust",
                  fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.set_aspect("equal")
    framed_axes(ax)


def plane_panel(ax, h_w, h_mu, delta, F, alpha, lim=None, cap=None, name=None,
                ticks=None):
    """The measurement as points, over the analytic modulation function.

    Both the colour clip and the frame default to this panel's own measurement:
    the clip to a high percentile of the updates in it, the frame to whatever
    holds all of its points.  Neither is shared with the other sector, because
    the two experiments reach different parts of the plane and move by different
    amounts.
    """
    finite = delta[np.isfinite(delta)]
    cap = float(np.percentile(np.abs(finite), CAP_PCT) if cap is None else cap)
    if lim is None:
        both = np.concatenate([np.abs(h_w[np.isfinite(h_w)]),
                               np.abs(h_mu[np.isfinite(h_mu)])])
        lim = float(np.ceil((both.max() * 1.06) / 0.5) * 0.5)
    a = np.linspace(-lim, lim, 400)
    HW, HMU = np.meshgrid(a, a)
    V = np.clip(alpha * F(HW, HMU), -cap, cap)
    levels = np.linspace(-cap, cap, 25)
    ax.contourf(HW, HMU, V, levels=levels, cmap=pastel("coolwarm"))
    ax.contour(HW, HMU, V, levels=levels[::4], colors="k", linewidths=0.25,
               alpha=0.5)
    im = ax.scatter(h_w, h_mu, c=np.clip(delta, -cap, cap), cmap="coolwarm",
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap),
                    s=13, edgecolors="k", linewidths=0.3, zorder=3)
    _decorate(ax, lim, ticks=ticks)
    if name is not None:
        # Placed exactly as Figure 1 places it.  The only addition is the
        # z-order, which Figure 1 has no need of because it has no points for
        # the label to end up under.
        ax.text(0.05, 0.93, name, transform=ax.transAxes, color="#2f4f7f",
                ha="left", va="top", zorder=4)
    return im


def _binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 2:
            continue
        out.append((float(np.mean(x[m])), float(np.mean(y[m])),
                    float(np.std(y[m], ddof=1) / np.sqrt(m.sum())), int(m.sum())))
    return np.array(out).T if out else np.empty((4, 0))


def gate_panel(ax, h_w, h_mu, delta, F, alpha, bins=7, lim=2.6,
               xlabel=r"$h_\mu$"):
    """``Delta h_w`` against prior distrust, split by whether the receiver agreed.

    The prediction is not that the two arms lie on top of one another -- ``F_w``
    is larger where the message is dissonant, so they should not -- but that they
    cross zero together, at ``h_mu = 0``, because the sign of ``F_w`` is carried
    by ``1 - 2 Phi(h_mu)`` and by nothing else.  An update rule in which
    agreement decided the direction would put the two crossings on opposite
    sides.
    """
    edges = np.linspace(-lim, lim, bins + 1)
    grid = np.linspace(-lim, lim, 200)
    for agree, colour, label in ((True, "#B03A34", "receiver agreed"),
                                 (False, "#3B6FA8", "receiver disagreed")):
        m = (np.isfinite(h_w) & np.isfinite(h_mu) & np.isfinite(delta)
             & ((h_w > 0) if agree else (h_w < 0)))
        if m.sum() < 3:
            continue
        hw_typ = float(np.median(h_w[m]))
        ax.plot(grid, alpha * F(np.full_like(grid, hw_typ), grid),
                color=colour, lw=1.0, zorder=2)
        b = _binned(h_mu[m], delta[m], edges)
        if b.size:
            ax.errorbar(b[0], b[1], yerr=b[2], fmt="o", ms=4.0, mfc="white",
                        mec=colour, ecolor=colour, elinewidth=0.8, capsize=1.8,
                        mew=0.9, lw=0, label=label, zorder=3)
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.set_xlim(-lim, lim)
    # Just the symbol: the full "trust -> distrust" gloss is a third wider than
    # the panel and runs off the page, and panel (a) already orients both axes
    # for the whole figure.
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.set_ylabel(r"$\Delta h_w$", fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3,
              borderaxespad=0.15, fontsize=AXIS_LABEL_PT - 1.5,
              labelspacing=0.25)
    framed_axes(ax)


def conviction_panel(ax, h_w, h_mu, delta, F, alpha, bins=6, lim=3.0,
                     band=0.3, xlabel=r"$h_w$"):
    """``Delta h_w`` against conviction, one arm per side of the trust axis.

    Read the two arms against each other rather than each on its own.  Their
    separation is the sign gate of panel (b); their opposite *slopes* are the
    ``1/Z`` amplification, which is what distinguishes ``F_w`` from any rule that
    simply multiplies a learning rate by trust.
    """
    edges = np.linspace(-lim, lim, bins + 1)
    grid = np.linspace(-lim, lim, 200)
    for hot, colour, label in ((True, "#3B6FA8", "trusted emitter"),
                               (False, "#B03A34", "distrusted emitter")):
        m = (np.isfinite(h_w) & np.isfinite(delta)
             & ((h_mu < -band) if hot else (h_mu > band)))
        if m.sum() < 3:
            continue
        hmu_typ = float(np.median(h_mu[m]))
        curve = alpha * F(grid, np.full_like(grid, hmu_typ))
        ax.plot(grid, curve, color=colour, lw=1.0, zorder=2)
        b = _binned(h_w[m], delta[m], edges)
        if b.size:
            ax.errorbar(b[0], b[1], yerr=b[2], fmt="o", ms=4.0, mfc="white",
                        mec=colour, ecolor=colour, elinewidth=0.8, capsize=1.8,
                        mew=0.9, lw=0, zorder=3)
        ax.annotate(label, xy=(lim * 0.94, curve[-1]),
                    xytext=(0, 7 if hot else -7), textcoords="offset points",
                    ha="right", va="bottom" if hot else "top", color=colour,
                    fontsize=AXIS_LABEL_PT - 1.5)
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.set_ylabel(r"$\Delta h_w$", fontsize=AXIS_LABEL_PT, labelpad=1)
    framed_axes(ax)


def crossover_panel(ax, h_w, h_mu, d_w, d_mu, F_w, F_mu, alpha, bins=7, lim=3.0):
    """Which sector moves more, against which sector is the more certain.

    ``F_w(x, y) = F_mu(y, x)``, so the ratio of the two updates is one exactly on
    ``h_w = h_mu`` and departs from it in a direction set by which of the two the
    message was more surprising to.  Plotted as a log ratio so that the two
    sectors are treated alike and the prediction is a crossing of zero.
    """
    m = (np.isfinite(d_w) & np.isfinite(d_mu) & np.isfinite(h_w)
         & np.isfinite(h_mu) & (np.abs(d_w) > 1e-6) & (np.abs(d_mu) > 1e-6))
    x = (h_w - h_mu)[m]
    y = np.log(np.abs(d_w[m]) / np.abs(d_mu[m]))
    edges = np.linspace(-lim, lim, bins + 1)
    grid = np.linspace(-lim, lim, 200)
    hw_typ, hmu_typ = float(np.median(h_w[m])), float(np.median(h_mu[m]))
    theory = np.log(np.abs(F_w(hmu_typ + grid, hmu_typ))
                    / np.abs(F_mu(hmu_typ + grid, hmu_typ)))
    ax.plot(grid, theory, color="0.35", lw=1.0, zorder=2,
            label=r"$\log|F_w/F_\mu|$")
    b = _binned(x, y, edges)
    if b.size:
        ax.errorbar(b[0], b[1], yerr=b[2], fmt="o", ms=4.0, mfc="white",
                    mec="#5aa469", ecolor="#5aa469", elinewidth=0.8, capsize=1.8,
                    mew=0.9, lw=0, zorder=3, label="measured")
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="#5aa469", lw=0.9)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$h_w-h_\mu$", fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.set_ylabel(r"$\log\,|\Delta h_w| / |\Delta h_\mu|$",
                  fontsize=AXIS_LABEL_PT, labelpad=1)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.4,
              borderaxespad=0.2)
    framed_axes(ax)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def figure_pair(sectors, figure_dir, name="llm_modulation", nest=False,
                lim=None, cap=None, ticks=None):
    """Two sectors side by side, in the layout and conventions of Figure 1.

    Like Figure 1 the two panels share everything that is not the data: one
    frame, one colour scale, one colour bar, one x label, and the y axis of the
    left panel. Each still carries its own fitted scale, because the constant
    that Figure 1's panels do not need -- the unobservable variance in front of
    the update -- is a property of the sector and not of the drawing.

    ``cap`` defaults to the widest of the panels' own scales, rounded up, so that
    whichever sector moves more sets the range and the other is not stretched
    past its data. ``lim`` is a crop and not a rescaling: a frame smaller than a panel's
    measurement hides the part outside it, so the caller is told how many points
    each panel loses.
    """
    from ednna.plotting import matched_colorbar, text_width
    if cap is None:
        # rounded to a fifth, so the bar is labelled 3.2 / 1.6 / 0 and not
        # 3.14 / 1.57 / 0 -- the same tick values Figure 1 carries
        widest = max(float(np.percentile(np.abs(d[np.isfinite(d)]), CAP_PCT))
                     for _, _, _, _, d, _ in sectors)
        cap = float(np.ceil(widest / 0.2) * 0.2)
    left, right, bottom, top, wspace = 0.115, 0.87, 0.20, 0.98, 0.12
    W = 0.86 * text_width()
    panel_w = W * (right - left) / (2 + wspace)
    fig, axes = plt.subplots(1, 2, figsize=(W, panel_w / (top - bottom)),
                             sharey=True)
    im = None
    for ax, (F, label, h_w, h_mu, delta, alpha) in zip(axes, sectors):
        if lim is not None:
            outside = int(np.sum((np.abs(h_w) > lim) | (np.abs(h_mu) > lim)))
            if outside:
                print(f"  [{label}] {outside} of {len(h_w)} points "
                      f"({outside / len(h_w):.0%}) lie outside the "
                      f"+/-{lim:g} frame and are not drawn")
        im = plane_panel(ax, h_w, h_mu, delta, F, alpha, name=label, lim=lim,
                         cap=cap, ticks=ticks)
        ax.set_xlabel("")            # one shared label below, or the two collide
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.supxlabel(r"disagree $\leftarrow h_w \rightarrow$ agree",
                  fontsize=AXIS_LABEL_PT, y=0.045)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace)
    matched_colorbar(fig, im, axes[-1], ticks=np.linspace(-cap, cap, 5))
    return save(fig, name, figure_dir, nest=nest)
