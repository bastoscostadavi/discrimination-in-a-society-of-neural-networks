#!/usr/bin/env python3
"""The phase plane of one field component: strength against prevalence.

The same sweep the main line of work runs over ``(p, f_p)``, run instead over the
strength and prevalence of any one of the four components of
:mod:`dirfield.fields` -- by default ``c``, the status field, in which one class
is believed more by everyone including its own members.

Which channel responds depends on which component is swept
(:data:`dirfield.order_params.CHANNEL_OF`), and the composite and the cut follow
it: ``--component b`` plots ``R_cred`` where ``--component c`` plots ``R_stat``.
Running ``--component a`` is a control rather than a phase diagram -- ``a`` refers
to no label, so every class channel is zero by construction and what moves is
whether the population polarizes at all (``R_wmu``, ``B_eta``).

Three figures:

``directional_channels``
    the four class-symmetry channels of the trust matrix over the plane, and
    beneath them the two within-class balances, their difference, and the
    ordinary opinion-trust alignment.  Read the top row first: one channel
    responds and the published one stays flat everywhere.

``directional_phase``
    the composite, with status as red, the asymmetry in internal cohesion as
    green, and ordinary polarization as blue.

``directional_cut``
    the status channel and the published trust-class correlation along the
    strength axis at several prevalences, which is the comparison an audit would
    have to make.

No regions are named on the composite.  The four states of the ``p`` plane were
identified from a sweep before they were labelled, and the same is owed to this
one.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from _cli import setup  # noqa: E402

from dirfield.config import default_s_range  # noqa: E402
from dirfield.order_params import CHANNEL_NAMES, CHANNEL_OF  # noqa: E402
from dirfield.plotting import (  # noqa: E402
    DESCRIPTIONS, LABELS, add_phase_axes, component, panel, phase_map, save,
    status_composite,
)
from dirfield.sweep import sweep  # noqa: E402

#: Bottom row: what the population has done with the field structurally.  The top
#: row is the four channels, ordered so the responding one comes first, which
#: :func:`_channel_row` builds from the component being swept.
BOTTOM = ("B_eta_A", "B_eta_B", "atomization", "R_wmu")

#: The channel panels, keyed so that R_muc is drawn on the channels' shared
#: diverging map rather than on the paper's red one.
_PANEL_KEY = {"T_mu": "T_mu", "R_cred": "R_cred", "R_stat": "R_stat",
              "R_muc": "R_muc_channel"}


def _channel_row(comp):
    """The four channel panels, the one this component drives first."""
    responding = CHANNEL_OF[comp]
    rest = [k for k in CHANNEL_NAMES if k != responding]
    return [_PANEL_KEY[k] for k in [responding] + rest]

#: The map and the cut print side by side at one width, so both are generated at
#: one size and one axes rectangle rather than each cropped to its own content.
PAIR_ASPECT = 1.0
PAIR_RECT = dict(left=0.235, right=0.975, bottom=0.165, top=0.975)

#: Prevalences the cut draws.  The lowest is one per cent, where a single society
#: cannot be told from one with no biased agent in it.
CUT_FRACTIONS = (0.01, 0.10, 0.50, 1.00)
CUT_HALFWIDTH = 0.02

#: Columns pooled into the running mean along the strength axis.  One pixel is
#: one realization, so an unsmoothed cut is unreadable at any grid size worth
#: plotting; the window shrinks automatically on a grid narrower than itself.
SMOOTH_WIDTH = 7


def atomization(data):
    """``(B_eta^AA - B_eta^BB)/2``: how differently the two classes cohere.

    Zero when the classes are internally alike, whether both are blocs or both
    are glasses; one when one is a perfectly balanced bloc and the other
    perfectly frustrated.  It is the quantity that says a hierarchy has *organized*
    the two halves differently, which neither class's balance says on its own.
    """
    return 0.5 * (data["B_eta_A"] - data["B_eta_B"])


def run(preset, use_cache=True):
    model = preset.model
    s_range = default_s_range(model.component)
    cfg = preset.sweep.with_(s_range=s_range)
    print(f"[phase] component '{model.component}' over {s_range}, "
          f"{cfg.n_s}x{cfg.n_f}, P={model.n_issues} (alpha={model.alpha:.3g})")
    return sweep(model, cfg, use_cache=use_cache)


def figure_channels(data, style, name="directional_channels"):
    """Two rows of four maps over the plane."""
    fig, axes = plt.subplots(2, 4, figsize=panel(1.0, 0.56), squeeze=False)
    grid = {**data, "atomization": atomization(data),
            "R_muc_channel": data["R_muc"]}
    for i, row in enumerate((_channel_row(component()), BOTTOM)):
        for j, key in enumerate(row):
            ax = axes[i][j]
            title = LABELS[key]
            if key in DESCRIPTIONS:
                title = f"{title}   {DESCRIPTIONS[key]}"
            phase_map(ax, grid[key], data["s"], data["f"], key,
                      ylabel=(j == 0), title=title, sparse_ticks=True)
    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def figure_map(data, style, name="directional_phase"):
    rgb = status_composite(data[CHANNEL_OF[component()]], atomization(data),
                           data["R_wmu"])
    s, f = data["s"], data["f"]
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    ax.imshow(rgb, origin="lower", extent=[s[0], s[-1], f[0], f[-1]], aspect="auto")
    ax.set_box_aspect(1)
    add_phase_axes(ax, xlim=(s[0], s[-1]))
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)


def _cut(ax, data, fractions=CUT_FRACTIONS, half=CUT_HALFWIDTH):
    """Two channels along the strength axis, one colour per prevalence.

    Solid is the channel this component drives, dotted the one the main line of
    work reports.  For ``b`` and ``c`` the dotted curves lying on zero at every
    prevalence and every strength is the finding, and it is easier to read here
    than off a colour.  For ``p`` the two coincide, and the panel degenerates to
    one family of curves; that is correct rather than broken.
    """
    s, f = data["s"], data["f"]
    responding = CHANNEL_OF[component()]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(fractions)))
    # The window has to shrink on a grid coarser than itself: with mode="same"
    # numpy returns max(len(a), len(kernel)) samples, so a kernel wider than the
    # axis lengthens the curve instead of erroring, and it is then plotted
    # against the shorter axis.  Keep it odd so the mean stays centred.
    width = min(SMOOTH_WIDTH, len(s) if len(s) % 2 else len(s) - 1) or 1
    k = np.ones(width) / width
    # mode="same" pads with zeros, which drags both ends of every curve towards
    # the axis; dividing by the same convolution of a ones-vector is the
    # edge-correct running mean
    norm = np.convolve(np.ones_like(s), k, mode="same")

    for frac, col in zip(fractions, colors):
        m = (f >= frac - half) & (f <= frac + half)
        if not m.any():
            continue
        keys = [(responding, "-")] + ([("R_muc", ":")] if responding != "R_muc" else [])
        for key, ls in keys:
            rows = data[key][m]
            mu = np.convolve(rows.mean(0), k, mode="same") / norm
            se = np.convolve(rows.std(0) / np.sqrt(rows.shape[0]), k,
                             mode="same") / norm
            ax.plot(s, mu, ls, color=col, lw=1.1,
                    label=f"{frac:.2f}" if key == responding else None)
            ax.fill_between(s, mu - se, mu + se, color=col, alpha=0.16, lw=0)

    ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_xlabel(rf"${component()}$")
    ax.set_ylabel(LABELS[responding] if responding == "R_muc"
                  else f"{LABELS[responding]},  {LABELS['R_muc']}")
    ax.set_box_aspect(1)

    first = ax.legend(title=rf"$f_{component()}$", fontsize=6, title_fontsize=6.5,
                      loc="upper left", frameon=False, handlelength=1.1,
                      labelspacing=0.22, borderpad=0.2)
    first._legend_box.align = "left"
    ax.add_artist(first)
    if responding != "R_muc":
        handles = [Line2D([], [], color="0.35", ls=ls, lw=1.1) for ls in ("-", ":")]
        ax.legend(handles, [LABELS[responding], LABELS["R_muc"]], fontsize=6,
                  loc="lower right", frameon=False, handlelength=1.4,
                  labelspacing=0.22, borderpad=0.2)


def figure_cut(data, style, name="directional_cut"):
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    _cut(ax, data)
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)


def report(data):
    """The numbers the figures are read for, printed."""
    s, f = data["s"], data["f"]
    print("\n[phase] along the strength axis at full prevalence:")
    top = np.argmax(f)
    resp = CHANNEL_OF[component()]
    print(f"    {'strength':>9}{resp:>9}{'R_muc':>9}{'R_cw':>9}"
          f"{'atomiz.':>9}{'R_wmu':>9}")
    atom = atomization(data)
    for j in np.linspace(0, len(s) - 1, 6).astype(int):
        print(f"    {s[j]:>9.2f}{data[resp][top, j]:>9.3f}"
              f"{data['R_muc'][top, j]:>9.3f}{data['R_cw'][top, j]:>9.3f}"
              f"{atom[top, j]:>9.3f}{data['R_wmu'][top, j]:>9.3f}")

    hi = np.argmax(s)
    print("\n[phase] up the prevalence axis at full strength:")
    print(f"    {'fraction':>9}{resp:>9}{'R_muc':>9}{'atomiz.':>9}{'R_wmu':>9}")
    for i in np.linspace(0, len(f) - 1, 6).astype(int):
        print(f"    {f[i]:>9.2f}{data[resp][i, hi]:>9.3f}"
              f"{data['R_muc'][i, hi]:>9.3f}{atom[i, hi]:>9.3f}"
              f"{data['R_wmu'][i, hi]:>9.3f}")

    strong = data["s"] > 0.6
    print(f"\n[phase] over the whole plane: max |R_muc| = "
          f"{np.abs(data['R_muc']).max():.3f}, "
          f"max {resp} = {data[resp].max():.3f}")
    print(f"[phase] where the field is strong (s > 0.6): mean {resp} = "
          f"{data[resp][:, strong].mean():.3f}, "
          f"mean R_muc = {data['R_muc'][:, strong].mean():.3f}")


def main():
    args, preset = setup(__doc__)
    data = run(preset, use_cache=not args.no_cache)
    report(data)
    figure_channels(data, args.style)
    figure_map(data, args.style)
    figure_cut(data, args.style)


if __name__ == "__main__":
    main()
