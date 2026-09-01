#!/usr/bin/env python3
"""The four collective states, drawn as the agents rather than as averages.

The phase diagram says what the order parameters do; it does not show what the
society *looks like*.  This figure does, for one population in each of the four
states, by reducing the microscopic state to a plane and drawing one arrow per
agent, coloured by class.

    (I)   frustrated       trust locks onto the label the wrong way round, and
                           opinion does not organize at all
    (II)  neutral          two opposed bundles in both sectors, each a mixture
                           of both colours
    (III) discriminatory   the same two bundles, now one colour each
    (IV)  class-only       trust splits by colour while opinion stays mixed

Both rows are the *same* reduction applied in two different spaces: unit-normalize
one vector per agent, then project onto the leading right singular vectors of the
resulting matrix.  In the opinion sector the vector is the agent's own weights
``w_I`` in ``R^K``; in the trust sector it is the agent's outgoing trust profile
``eta_{.|I}`` in ``R^N``, the row of the trust matrix that says how much ``I``
trusts each of the others.  Two agents' arrows are close when they hold the same
opinion, or trust the same people, respectively.

Three properties of the reduction are worth stating, because the figure is read
geometrically and it would be easy to over-read it.

* It is *uncentered*.  The origin is the zero vector, not the population mean, so
  antipodal structure survives: two opposed camps come out as two bundles pointing
  opposite ways rather than as one blob split down the middle.
* Arrow *length* is the fraction of a unit vector that lies in the plane shown, so
  short arrows mean the sector has no low-dimensional structure to display, which
  is not a defect of the picture but the measurement: the opinion row of column (I)
  is short and diffuse because that state has no opinion structure to show.  The
  fraction of the total the plane captures is printed under each panel.  It is
  uninformative in the trust row, where a single axis carries almost everything in
  all four states --- what differs there is where that axis sits relative to the
  label, which is what the reference arrow below is for.
* The frame is fixed only up to the gauge stated in :func:`ednna.reduction.project`.
  In the opinion row the absolute orientation of a bundle means nothing; what means
  something is the angle between bundles and how the colours fall.

The trust row carries one further element, without which two of the four columns
would be indistinguishable.  States (I) and (III) both put trust into a single
class-pure axis, two opposed bundles with one colour each, and they differ only in
*which way round*: in (III) an agent trusts its own class, in (I) the out-group.
Nothing in a set of arrows fixed up to sign can say which.  So the trust panels are
drawn in an absolute frame: the class-indicator vector ``kappa`` lives in the same
``R^N`` as the trust rows, so it is projected onto the same plane, drawn as the grey
arrow, and rotated onto ``+x``, which makes the right half of every trust panel the
profiles that trust class $A$ and the left half those that trust $B$.  Those halves
are tinted in the class colours, so an agent that trusts its own class sits on the
half matching its own colour and one that trusts the out-group sits on the half that
clashes with it --- the difference between (III) and (I), visible without tracing an
arrow.  The grey arrow's own length is how much of ``kappa`` lies in the plane at
all, which is near zero exactly where the population has not organized around the
label.

Filled arrowheads mark the prejudiced agents and open ones the class-blind
majority they learn from, which is what makes ``f_p`` visible in a picture of a
single population: in (IV) the two groups point different ways.


A three-component version of the same reduction was tried and dropped: both
sectors put almost everything into one or two directions, so the third axis adds
perspective distortion and no structure.

Writes ``state_portraits``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from _cli import setup  # noqa: E402

from ednna.order_params import trust  # noqa: E402
from ednna.reduction import project  # noqa: E402
from ednna.plotting import HIST_BLUE, HIST_RED, panel, save  # noqa: E402
from ednna.society import SocietyBatch  # noqa: E402
from ednna.sweep import DATA_DIR  # noqa: E402

#: One population per state, and the four are taken at the extremes of their
#: regions rather than at the interior points where ``phase_diagram.py`` places its
#: labels.  A portrait is one realization, so it should be drawn where the state is
#: least equivocal: a corner of the plane says what the state *is*, and how far it
#: reaches back towards the neutral axis is the phase diagram's job to say, not this
#: figure's.  (II) is the extreme case of that argument --- it is drawn at
#: ``p = f_p = 0``, a population with no prejudiced agent in it at all, so the
#: column is a control rather than a sample of a region.
STATES = {
    "(I)": ("frustrated", -1.0, 1.0),
    "(II)": ("neutral", 0.0, 0.0),
    "(III)": ("discriminatory", 0.5, 1.0),
    "(IV)": ("class-only", 1.0, 0.5),
}

#: Darker of the two tones in each histogram pair: these are strokes, not fills.
CLASS_COLORS = {+1: HIST_RED[1], -1: HIST_BLUE[1]}
CLASS_LABELS = {+1: "class $A$", -1: "class $B$"}


def run(preset, P, use_cache=True):
    """Simulate the four representative populations and keep their microstates.

    The sweeps cache order parameters only, which is all a phase diagram needs and
    three orders of magnitude less data than the states they were measured from.
    This figure needs the states, so it re-runs the four points it draws --- four
    societies rather than the sweep's forty thousand, which is seconds of work.
    The four are run as one :class:`SocietyBatch`, so they share an interaction
    schedule and differ only in ``(p, f_p)`` and their own draws, exactly as
    neighbouring pixels of the sweep do.
    """
    model = preset.model.with_(n_issues=P)
    labels = list(STATES)
    p_values = np.array([STATES[k][1] for k in labels])
    fp_values = np.array([STATES[k][2] for k in labels])

    cache = (
        DATA_DIR / f"portraits_P{P}_N{model.n_agents}_K{model.n_dim}"
        f"_T{model.interactions_per_channel:g}.npz"
    )
    if use_cache and cache.exists():
        with np.load(cache) as z:
            print(f"[portraits] loaded cache {cache.name}")
            return {k: z[k] for k in z.files} | {"labels": labels}

    batch = SocietyBatch(
        n_agents=model.n_agents, n_dim=model.n_dim, n_issues=P,
        d=p_values, f_d=fp_values, case=model.case, seed=4321,
        literal_draft_sign=model.literal_draft_sign, dtype=model.numpy_dtype(),
        shared_schedule=model.shared_schedule,
    )
    steps = model.n_steps()
    print(f"[portraits] 4 societies of N={model.n_agents} "
          f"(P={P}, alpha={model.alpha:.3g}), {steps:,} interactions ...", flush=True)
    batch.run(steps)

    out = {
        "w": np.moveaxis(batch.w, 1, 0),      # (R, N, K), one matrix per state
        "eta": trust(batch),                  # (R, N, N)
        "kappa": batch.kappa,
        "prejudiced": batch.discriminates.T,  # (R, N)
        "p": p_values,
        "fp": fp_values,
        "alpha": np.asarray(model.alpha),
    }
    np.savez_compressed(cache, **out)
    print(f"[portraits] cached {cache.name}")
    return out | {"labels": labels}


def _sectors(data, r, n_comp):
    """The two reductions for state ``r``: opinion first, then trust.

    The trust rows and the class indicator are both vectors in ``R^N``, so the
    indicator can be carried through the same projection and drawn; opinions live
    in ``R^K``, where the class label is not a direction at all, so the opinion
    panels have no reference arrow and none is available to them.
    """
    kappa = np.asarray(data["kappa"], float)
    eta = np.array(data["eta"][r], dtype=float)
    np.fill_diagonal(eta, 0.0)  # the self-entry is +1 by convention, and carries
    return [                    # an agent's own index rather than its opinions
        project(data["w"][r], n_comp, positive_class=kappa == +1),
        project(eta, n_comp, reference=kappa),
    ]


ROW_LABELS = (r"opinion  $\hat{w}_I$", r"trust  $\eta_{\cdot|I}$")


def _column_title(data, r):
    label = data["labels"][r]
    return (f"{label} {STATES[label][0]}\n"
            f"$p={data['p'][r]:+.1f}$,  $f_p={data['fp'][r]:.1f}$")


def _legend(fig, y=-0.01):
    handles = [plt.Line2D([], [], color=CLASS_COLORS[k], lw=1.1, label=CLASS_LABELS[k])
               for k in (+1, -1)]
    handles += [
        plt.Line2D([], [], color="0.35", lw=0, marker="o", ms=3.0,
                   label="prejudiced"),
        plt.Line2D([], [], color="0.35", lw=0, marker="o", ms=3.0, mfc="white",
                   mew=0.5, label="class-blind"),
        plt.Line2D([], [], color="0.45", lw=0.9, ls=(0, (3, 2)),
                   label=r"$\kappa$: trusts every $A$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, y),
               handlelength=1.4, columnspacing=1.4, borderpad=0.2)


def interleave(kappa):
    """A draw order that alternates the classes.

    Agents are stored class-major, and in the trust panels the two bundles are
    nearly collinear, so drawing in storage order paints one class entirely over
    the other and a mixed bundle reads as a pure one.  Alternating the classes
    makes the two overdraw each other equally, which is the only honest option
    short of moving arrows that belong on top of each other.
    """
    rank = np.empty(kappa.size, dtype=int)
    for k in (+1, -1):
        mask = kappa == k
        rank[mask] = np.arange(mask.sum())
    return np.lexsort((kappa, rank))


def _draw_agents(ax, coords, kappa, prejudiced, lw=0.8, ms=3.2):
    """One line per agent from the origin, tip filled if the agent is prejudiced."""
    order = interleave(kappa)
    coords, kappa, prejudiced = coords[order], kappa[order], prejudiced[order]
    colors = np.array([CLASS_COLORS[int(np.sign(k))] for k in kappa])
    segs = [((0.0, 0.0), tuple(c)) for c in coords[:, :2]]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, alpha=0.85))
    for mask, kw in ((prejudiced, {}), (~prejudiced, {"facecolors": "white"})):
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=ms, edgecolors=colors[mask],
                       linewidths=0.45, zorder=3, **kw)


def _draw_reference(ax, ref, tint=0.055):
    """The class indicator, and the two halves of the plane it separates.

    The reference is rotated onto ``+x``, so the right half of a trust panel is the
    set of profiles that trust class $A$ and distrust class $B$, and the left half is
    the reverse.  Tinting the two halves in the class colours turns the distinction
    between (III) and (I) into one that can be seen without tracing an arrow: an
    agent that trusts its own class sits on the half that matches its own colour, and
    one that trusts the out-group sits on the half that clashes with it.  Without the
    tint the two states are the same picture with the colours exchanged, and nothing
    on the page says which exchange is which.
    """
    if ref is None or np.linalg.norm(ref[:2]) < 1e-3:
        return
    for sign, klass in ((+1, +1), (-1, -1)):
        ax.axvspan(0 if sign > 0 else -1.08, 1.08 if sign > 0 else 0,
                   color=CLASS_COLORS[klass], alpha=tint, lw=0, zorder=0)
    ax.annotate("", xy=(ref[0], ref[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.13,head_length=0.3",
                                color="0.45", lw=0.9, ls=(0, (3, 2)),
                                shrinkA=0, shrinkB=0))
    ax.text(ref[0] + 0.05, ref[1] - 0.15, r"$\kappa$", fontsize=7, color="0.35")


def figure(data, style, name="state_portraits"):
    """The 2D version: eight unit disks, one arrow per agent."""
    ncol = len(data["labels"])
    fig, axes = plt.subplots(
        2, ncol, figsize=panel(1.0, 0.60), squeeze=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.24},
    )
    circle = np.linspace(0, 2 * np.pi, 256)
    kappa = np.asarray(data["kappa"], float)

    for r in range(ncol):
        prejudiced = np.asarray(data["prejudiced"][r], bool)
        for row, (coords, captured, ref) in enumerate(_sectors(data, r, 2)):
            ax = axes[row][r]
            ax.plot(np.cos(circle), np.sin(circle), color="0.8", lw=0.5, ls=(0, (3, 3)))
            _draw_reference(ax, ref)
            _draw_agents(ax, coords, kappa, prejudiced)
            ax.set_xlim(-1.08, 1.08)
            ax.set_ylim(-1.08, 1.08)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "bottom", "left", "right"):
                ax.spines[side].set_linewidth(0.5)
                ax.spines[side].set_color("0.7")
            ax.text(0.5, -0.02, f"{100 * captured:.0f}% in plane",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                    color="0.35")
            if r == 0:
                ax.set_ylabel(ROW_LABELS[row], labelpad=4, fontsize=8.5)
            if row == 0:
                ax.set_title(_column_title(data, r), pad=4, fontsize=7.5)

    _legend(fig)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    data = run(preset, preset.p_small, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
