#!/usr/bin/env python3
"""The (b, f_b) plane: the strength and prevalence of a credulity asymmetry.

The same sweep the main line of work runs over ``(p, f_p)``, run instead over the
strength and prevalence of ``b``, the component of a class-dependent field that
depends only on who is *listening*.  A prejudiced agent of class A reads every
message as more agreeable than it is and a prejudiced agent of class B reads
every message as less so, whoever is speaking -- so one class comes to trust
everyone and the other to trust nobody, itself included.

``b`` is the mirror of the status field ``c`` swept in
``../directional-prejudice/``, and the mirror is exact in a way worth stating:
under a pure ``b`` and a pure ``c`` of the same strength the trust matrices are
*transposes* of each other.  Every order parameter of the
main line of work reads ``eta`` only through ``eta_{I|J} + eta_{J|I}``, which a
transpose leaves alone, so the published five cannot tell a credulity split from
a status hierarchy -- and read zero on both.  What responds here is ``R_cred``
(:data:`credfield.order_params.CHANNEL_OF`), and the composite and the cut follow
the swept component, so ``--component c`` in this directory plots ``R_stat``
instead and reproduces the sibling result.

Only the positive half of the strength axis is swept: negating ``b`` is exactly
the relabelling ``A <-> B``, which maps the ensemble to itself, so the other half
is the mirror image and costs half a sweep to learn nothing
(``tests/test_sweep.py`` checks it on eight societies rather than asserting it).

Three figures:

``cred_asymmetry_channels``
    the four class-symmetry channels of the trust matrix over the plane, and
    beneath them the two within-class balances, their difference, and the
    ordinary opinion-trust alignment.  Read the top row first: one channel
    responds and the published one stays flat everywhere.

``cred_asymmetry_phase``
    the composite, with the credulity split as red, the asymmetry in internal
    cohesion as green, and ordinary polarization as blue.

``cred_asymmetry_cut``
    the credulity channel and the published trust-class correlation along the
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

from credfield.config import default_s_range  # noqa: E402
from credfield.order_params import CHANNEL_NAMES, CHANNEL_OF  # noqa: E402
from credfield.plotting import (  # noqa: E402
    DESCRIPTIONS, LABELS, add_phase_axes, channel_composite, component, panel,
    phase_map, save,
)
from credfield.sweep import sweep  # noqa: E402

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
    perfectly frustrated.  It is the quantity that says the field has *organized*
    the two halves differently, which neither class's balance says on its own.
    Under a saturated ``b`` the credulous class is a bloc in which everyone
    trusts everyone, and the suspicious class is a dust in which every triple is
    three mutual distrusts and so frustrated.
    """
    return 0.5 * (data["B_eta_A"] - data["B_eta_B"])


def run(preset, use_cache=True, n_strips=1):
    """Sweep the plane, optionally in horizontal strips.

    ``sweep`` has no checkpointing: it runs every batch and then writes one
    ``.npz`` at the very end, so an interrupt at 3h55m of a four-hour run costs
    exactly as much as an interrupt at 30 seconds.  At the resolutions this
    directory is for that is a real risk and it cost one full run.

    ``n_strips > 1`` cuts the prevalence axis into contiguous bands and sweeps
    each as its own cached sweep, then concatenates.  Total work is unchanged --
    same grid points, same interactions -- but a kill loses at most one strip,
    and re-running afterwards reloads the finished strips from cache and picks up
    where it stopped.  The strips are what the committed figures are built from;
    ``n_strips=1`` is the single-shot path.
    """
    model = preset.model
    s_range = default_s_range(model.component)
    cfg = preset.sweep.with_(s_range=s_range)
    print(f"[phase] component '{model.component}' over {s_range}, "
          f"{cfg.n_s}x{cfg.n_f}, P={model.n_issues} (alpha={model.alpha:.3g})")
    if n_strips <= 1:
        return sweep(model, cfg, use_cache=use_cache)
    return _run_striped(model, cfg, n_strips, use_cache)


def _strips(cfg, n_strips):
    """Split the prevalence axis into contiguous bands.

    Yields ``(index, f_range, n_f, row_offset)`` per strip.  The band's range is
    read off the *full* axis rather than computed from the band's own endpoints,
    so that ``linspace`` inside each strip reproduces exactly the rows the
    unstripped sweep would have put there and the concatenation is the same grid
    -- not a grid with duplicated boundaries and a kinked spacing, which is what
    slicing ``(0, 0.2), (0.2, 0.4), ...`` off a unit interval gives.
    """
    f_all = np.linspace(*cfg.f_range, cfg.n_f)
    bounds = np.array_split(np.arange(cfg.n_f), min(n_strips, cfg.n_f))
    for i, rows in enumerate(bounds):
        lo, hi = int(rows[0]), int(rows[-1])
        yield i, (float(f_all[lo]), float(f_all[hi])), len(rows), lo


def _run_striped(model, cfg, n_strips, use_cache):
    bands = list(_strips(cfg, n_strips))
    print(f"[phase] in {len(bands)} strips over f, each cached separately")
    pieces = []
    for i, f_range, n_f, row_offset in bands:
        # Each strip needs its own seed, or every strip draws the same
        # realizations: sweep() seeds batch b as `seed + flat_offset_within_this
        # _sweep`, which restarts at 0 for each strip.  Offsetting by the strip's
        # position in the full grid makes the seed a function of where a society
        # sits in the plane, so no two strips overlap and adding a strip does not
        # renumber the others.
        strip = cfg.with_(n_f=n_f, f_range=f_range,
                          seed=cfg.seed + row_offset * cfg.n_s)
        print(f"[phase] strip {i+1}/{len(bands)}: "
              f"f in [{f_range[0]:.4f}, {f_range[1]:.4f}], {n_f} rows")
        pieces.append(sweep(model, strip, tag=f"{model.component}_P"
                            f"{model.n_issues}_strip{i+1}of{len(bands)}",
                            use_cache=use_cache))

    out = {"s": pieces[0]["s"],
           "f": np.concatenate([p["f"] for p in pieces])}
    for i, p in enumerate(pieces):
        np.testing.assert_allclose(p["s"], out["s"], err_msg=f"strip {i+1} s axis")
    for name in pieces[0]:
        if name in ("s", "f"):
            continue
        out[name] = np.concatenate([p[name] for p in pieces], axis=0)
    assert out["f"].size == cfg.n_f, (out["f"].size, cfg.n_f)
    np.testing.assert_allclose(out["f"], np.linspace(*cfg.f_range, cfg.n_f),
                               atol=1e-12)
    return out


def figure_channels(data, style, name="cred_asymmetry_channels"):
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


def figure_map(data, style, name="cred_asymmetry_phase"):
    rgb = channel_composite(data[CHANNEL_OF[component()]], atomization(data),
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
    than off a colour: a reader can see that the published parameter is flat
    while the solid curve saturates, which a dark region of a composite only
    implies.  For ``p`` the two coincide and the panel degenerates to one family
    of curves; that is correct rather than broken.
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


def figure_cut(data, style, name="cred_asymmetry_cut"):
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
    data = run(preset, use_cache=not args.no_cache, n_strips=args.strips)
    report(data)
    figure_channels(data, args.style)
    figure_map(data, args.style)
    figure_cut(data, args.style)


if __name__ == "__main__":
    main()
