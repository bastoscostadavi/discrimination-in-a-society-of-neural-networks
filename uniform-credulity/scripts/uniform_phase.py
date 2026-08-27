#!/usr/bin/env python3
"""The ``(a, f_a)`` plane: uniform credulity against how many agents carry it.

The same sweep the main line of work runs over ``(d, f_d)``, run instead over the
strength and prevalence of the one component of the prejudice field that names no
class: ``D[r, e] = a`` for a fraction ``f_a`` of receivers, zero for the rest.

Both signs are swept, and they are not two strengths of one thing.  ``a`` shifts
the sign test that decides whether a message reads as agreement, so ``a > 0`` is
a population disposed to agree with whatever it hears and ``a < 0`` one disposed
to disagree; agreement builds trust, a distrusted source is anti-learned from
rather than ignored, and the two halves therefore drive the opinion sector in
opposite directions as well as the trust one.

Three figures:

``uniform_channels``
    two rows of four maps.  The top row is the four class-symmetry channels of
    the directed trust matrix, ``T_mu`` first: it is the one a uniform field
    drives, and the other three are controls on a model whose dynamics never
    reads the class label.  The bottom row is the partition that does matter
    here -- biased against unbiased agents -- as the direct effect on the trust
    each group extends, the emergent effect on the trust each group receives, the
    consensus, and the ordinary opinion-trust alignment.

``uniform_phase``
    the composite: universal distrust as red, universal trust as green, and the
    ordinary opinion-trust alignment as blue.

``uniform_cut``
    ``T_mu`` along the strength axis at several prevalences, against the
    published ``R_muc`` rescaled by ``-(N-1)``.  The second is the control, and
    it is not zero: it is a copy of the first scaled by ``-1/(N-1)``, so undoing
    that factor should put the two curves on top of each other.

No regions are named on the composite.  The four states of the ``p`` plane were
identified from a sweep before they were labelled, and the same is owed to this
one.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from _cli import setup  # noqa: E402

from credulity.order_params import (  # noqa: E402
    BIAS_BLOCKS, GROUP_BALANCES, PAPER_NAMES,
)
from credulity.plotting import (  # noqa: E402
    DESCRIPTIONS, LABELS, add_phase_axes, credulity_composite, panel, phase_map,
    save,
)
from credulity.sweep import sweep_in_strips  # noqa: E402

#: Top row: the class channels, the responding one first.  ``R_muc`` is drawn on
#: the channels' shared diverging map rather than on the paper's red one, so the
#: row reads as one comparison.
TOP = ("T_mu", "R_cred", "R_stat", "R_muc_channel")

#: Bottom row: the bias partition, and what the population did overall.  The
#: emergent panel is the block comparison rather than the pooled margin: pooling
#: over receivers averages the biased agents' own rows in, and those rows are the
#: field acting rather than a response to it.
BOTTOM = ("give_gap", "emergent_gap", "rho_mean", "R_wmu")

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

#: The control curve gets a wider one.  Rescaling ``R_muc`` by ``-(N-1)``
#: multiplies its single-realization noise by the same factor, which is 39 at
#: N = 40 -- the identity it is drawn to demonstrate holds in the mean, not
#: pixel by pixel, so the curve has to be averaged over enough pixels to show it.
CONTROL_SMOOTH_WIDTH = 15


#: Which swept quantities each derived quantity is built from.
#:
#: Declared rather than inferred so that ``tests/test_outputs.py`` can check
#: that every quantity the sweep records reaches a reader somehow -- either
#: printed under its own name or folded into one of these.  A quantity that
#: reaches neither is measured, cached and described but invisible, which is
#: worse than not measuring it, because the prose around it implies the evidence
#: exists.
DERIVED_FROM = {
    "give_gap": ("T_give_b", "T_give_u"),
    "get_gap": ("T_get_b", "T_get_u"),
    "emergent_gap": ("T_ub", "T_uu"),
    "rho_gap": ("rho_bb", "rho_uu"),
    "R_muc_channel": ("R_muc",),
}


def derived(data):
    """The three differences the figures read, added to the sweep's own arrays.

    Each is a difference between the two groups of the bias partition, so each is
    ``nan`` on the rows where one of the groups is empty.  They are computed here
    rather than swept because they carry no information the margins do not, and
    keeping the cache to independent quantities means a change of mind about how
    to contrast them costs a re-plot instead of a re-simulation.
    """
    return {
        **data,
        "give_gap": data["T_give_b"] - data["T_give_u"],
        "get_gap": data["T_get_b"] - data["T_get_u"],
        # the same emergent comparison made only by the receivers the field
        # never touched, which is the one that cannot be the field restated
        "emergent_gap": data["T_ub"] - data["T_uu"],
        "rho_gap": data["rho_bb"] - data["rho_uu"],
        "R_muc_channel": data["R_muc"],
    }


def run(preset, use_cache=True, n_strips=None):
    model = preset.model
    cfg = preset.sweep
    n_strips = preset.n_strips if n_strips is None else n_strips
    print(f"[phase] uniform field 'a' over {cfg.a_range}, {cfg.n_a}x{cfg.n_f}, "
          f"N={model.n_agents}, P={model.n_issues} (alpha={model.alpha:.3g}), "
          f"Delta t={model.interactions_per_channel:g}"
          + (f", in {n_strips} strips" if n_strips > 1 else ""))
    return sweep_in_strips(model, cfg, n_strips=n_strips, use_cache=use_cache)


#: Title size for the eight-panel grid.  Body size is right for a figure with
#: one or two panels and much too wide for eight: a quarter of the text width,
#: less the colour bar, leaves about an inch of axis, and a label like
#: ``T_mu^{b->} - T_mu^{u->}`` set at 10pt overruns its neighbour.
CHANNEL_TITLE_PT = 7.0


def figure_channels(data, style, name="uniform_channels"):
    """Two rows of four maps over the plane."""
    grid = derived(data)
    with plt.rc_context({"axes.titlesize": CHANNEL_TITLE_PT}):
        fig, axes = plt.subplots(2, 4, figsize=panel(1.0, 0.60), squeeze=False)
        for i, row in enumerate((TOP, BOTTOM)):
            for j, key in enumerate(row):
                look = "R_muc" if key == "R_muc_channel" else key
                # symbol above, plain-language gloss below: side by side they
                # do not fit, and rotating them costs more height than the
                # second line does
                title = LABELS[key]
                if look in DESCRIPTIONS:
                    title = f"{title}\n{DESCRIPTIONS[look]}"
                phase_map(axes[i][j], grid[key], data["a"], data["f"], key,
                          ylabel=(j == 0), title=title, sparse_ticks=True)
        fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def figure_map(data, style, name="uniform_phase"):
    rgb = credulity_composite(data["T_mu"], data["R_wmu"])
    a, f = data["a"], data["f"]
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    ax.imshow(rgb, origin="lower", extent=[a[0], a[-1], f[0], f[-1]], aspect="auto")
    ax.set_box_aspect(1)
    add_phase_axes(ax, xlim=(a[0], a[-1]))
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)


def _smoother(a, width=SMOOTH_WIDTH):
    """A running mean along the strength axis, and its edge correction.

    ``mode="same"`` pads with zeros, which drags both ends of every curve towards
    the axis; dividing by the same convolution of a ones-vector is the
    edge-correct running mean.  The window has to shrink on a grid coarser than
    itself: with ``mode="same"`` numpy returns ``max(len(a), len(kernel))``
    samples, so a kernel wider than the axis lengthens the curve instead of
    erroring, and it is then plotted against the shorter axis.  Kept odd so the
    mean stays centred.
    """
    width = min(width, len(a) if len(a) % 2 else len(a) - 1) or 1
    k = np.ones(width) / width
    norm = np.convolve(np.ones_like(a), k, mode="same")
    return lambda y: np.convolve(y, k, mode="same") / norm


def _cut(ax, data, n_agents, fractions=CUT_FRACTIONS, half=CUT_HALFWIDTH):
    """``T_mu`` along the strength axis, against the rescaled control.

    Solid, one per prevalence, is the channel the uniform field drives.  Dotted
    is the published trust-class correlation multiplied by ``-(N-1)``, and the
    rescaling is the point: ``R_muc`` on this plane is not zero but
    ``-T_mu/(N-1)``, the leak of the uniform channel into the matching one that
    excluding the diagonal introduces, so undoing the factor should land the
    dotted curve on the solid one.  Plotted raw it would be a flat line at two
    per cent of the axis, indistinguishable from a flat line at zero, and a
    reader would take the published parameter to read nothing here for the wrong
    reason.

    Drawn once, at the highest prevalence, rather than once per curve.  The
    rescaling multiplies the noise as well as the signal, so four copies of it
    are four noisy curves making one statement; the top prevalence is where
    ``|T_mu|`` is largest and the identity has the most to say.
    """
    a, f = data["a"], data["f"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(fractions)))
    smooth = _smoother(a)

    for frac, col in zip(fractions, colors):
        m = (f >= frac - half) & (f <= frac + half)
        if not m.any():
            continue
        rows = data["T_mu"][m]
        mu = smooth(rows.mean(0))
        se = smooth(rows.std(0) / np.sqrt(rows.shape[0]))
        ax.plot(a, mu, "-", color=col, lw=1.1, label=f"{frac:.2f}")
        ax.fill_between(a, mu - se, mu + se, color=col, alpha=0.16, lw=0)

    top = f >= f.max() - half
    control = _smoother(a, CONTROL_SMOOTH_WIDTH)(
        -(n_agents - 1) * data["R_muc"][top].mean(0))
    ax.plot(a, control, ":", color="0.25", lw=1.2, zorder=5)

    ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(a[0], a[-1])
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$T_\mu$,  $-(N-1)R_{\mu,c}$")
    ax.set_box_aspect(1)

    first = ax.legend(title=r"$f_a$", fontsize=6, title_fontsize=6.5,
                      loc="upper left", frameon=False, handlelength=1.1,
                      labelspacing=0.22, borderpad=0.2)
    first._legend_box.align = "left"
    ax.add_artist(first)
    handles = [Line2D([], [], color="0.35", ls="-", lw=1.1),
               Line2D([], [], color="0.25", ls=":", lw=1.2)]
    ax.legend(handles, [r"$T_\mu$", r"$-(N-1)R_{\mu,c}$,  $f_a=1$"],
              fontsize=6, loc="lower right", frameon=False, handlelength=1.4,
              labelspacing=0.22, borderpad=0.2)


def figure_cut(data, style, n_agents, name="uniform_cut"):
    fig, ax = plt.subplots(figsize=panel(0.49, PAIR_ASPECT))
    _cut(ax, data, n_agents)
    fig.subplots_adjust(**PAIR_RECT)
    return save(fig, name, style, bbox=None)


def _crossing(x, y, level):
    """Where a curve first reaches ``level``, linearly interpolated.

    Returns ``nan`` if it never does.  The direction is not assumed: the first
    place the curve touches or straddles the level is taken, whichever way it is
    going, so the same routine reads the credulous branch rising to ``+level``
    and the suspicious branch falling to ``-level`` without a sign argument that
    could be passed wrongly -- and a level of exactly zero, where a sign test
    would be ambiguous, still resolves.

    The interpolation matters: on a 200-column axis the grid spacing is 0.01, so
    reading off the nearest sample would quantize a threshold to two decimals and
    make two planes look like they agree, or disagree, on the last digit for no
    reason.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(y, dtype=float) - level
    if d.size < 2:
        return np.nan
    exact = np.flatnonzero(d == 0.0)
    straddle = np.flatnonzero(d[:-1] * d[1:] < 0.0)
    first = [int(c[0]) for c in (exact, straddle) if c.size]
    if not first:
        return np.nan
    i = min(first)
    if d[i] == 0.0:
        return float(x[i])
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))


def _branch_axis(a, branch):
    """Row indices and strengths of one branch, ordered outward from ``a = 0``.

    Outward and not left-to-right: the crossing wanted is the first one *leaving*
    the unbiased state, and scanning the suspicious branch from ``a = -1``
    upwards would find the last one entering it instead.
    """
    if branch == "credulous":
        idx = np.flatnonzero(a >= 0)
    else:
        idx = np.flatnonzero(a <= 0)[::-1]
    return idx, a[idx]


def threshold_curve(data, key="T_mu", branch="credulous", level=None,
                    min_saturation=0.3, smooth=True):
    """Where ``key`` crosses its threshold, one strength per prevalence row.

    Two definitions, and they are not the same question:

    ``level=None``
        half of *that row's own* saturated value.  Answers **where a given
        population's transition happens**.
    ``level=x``
        a fixed absolute level of the channel.  Answers **how much bias it takes
        to reach a stated degree of order**.

    The two disagree systematically, and the reason is worth stating because
    otherwise the disagreement looks like an error in one of them.  At low
    prevalence the channel saturates lower, so half of its own saturation is a
    lower bar and is reached at much the same strength -- a threshold that
    barely moves with prevalence.  A fixed bar does not move, so it takes more
    strength as fewer agents carry the bias -- a strength-prevalence trade-off.
    One reports a vertical line, the other a hyperbola, and both are correct
    about their own question.

    Rows that never order at all (``|saturation| < min_saturation``) are ``nan``
    rather than being assigned the edge of the axis.
    """
    a, f = data["a"], data["f"]
    idx, xs = _branch_axis(a, branch)
    M = data[key][:, idx]
    if smooth:
        smoother = _smoother(xs)
        M = np.stack([smoother(row) for row in M])
    sat = M[:, -1]
    out = np.full(len(f), np.nan)
    for i in range(len(f)):
        s_i = sat[i]
        if abs(s_i) < min_saturation:
            continue
        lv = 0.5 * s_i if level is None else np.sign(s_i) * level
        if abs(lv) > abs(s_i):
            continue  # the row never reaches a fixed bar above its own ceiling
        out[i] = _crossing(xs, M[i], lv)
    return f, out


#: Levels the threshold table reports.  ``None`` is the relative definition;
#: the two numbers are absolute levels of the channel.
THRESHOLD_LEVELS = (None, 0.5, 0.6)

#: Rows too sparse to say anything about a trade-off are dropped from the
#: summary: at very low prevalence a row may cross once and by luck.
THRESHOLD_MIN_ROWS = 8

#: The surviving rows must span at least this much prevalence before the summary
#: will name a conserved quantity.
#:
#: This is a guard against a failure that produces a confident wrong answer
#: rather than a missing one.  Both filters above reject low-prevalence rows
#: preferentially -- a row that never orders has no threshold, and a fixed bar
#: above a row's ceiling is never reached -- so the surviving rows can end up
#: crowded into the top of the plane.  Over a narrow span of ``f``, ``s*`` and
#: ``f s*`` are nearly proportional to each other, so their relative spreads are
#: nearly equal and whichever comes out smaller is decided by noise.  The
#: comparison then reports a conserved quantity it has no power to distinguish.
MIN_F_SPAN = 0.3

#: And the better-conserved quantity must be better conserved by at least this
#: factor before it is named.
#:
#: "Smaller" is not a finding.  Two relative spreads within a few per cent of
#: each other name a winner that the next realization would reverse, and on a
#: synthetic plane whose transition is *known* to sit at a fixed strength, a
#: fixed bar at 0.6 leaves rows spanning 0.39 in prevalence -- past the span
#: guard -- and reports "product" on a margin of 1.4.  The span guard alone does
#: not catch that; this one does.
#:
#: Set at 2.0 rather than 1.5 because of the one case where the relative
#: definition itself inverts: on a synthetic plane whose transition is at fixed
#: ``f|a|``, a transition width of 0.20 gives the wrong verdict at a margin of
#: 1.78.  At 2.0 no wrong verdict survives at any width tested (0.03 to 0.20) on
#: either synthetic plane, at the cost of also declining a *correct* verdict at
#: width 0.15 (margin 1.61).  That is the right trade: declining a correct answer
#: costs nothing, naming a wrong one costs a paper.
MIN_RATIO = 2.0


def _locus_verdict(level, span, ratio, s_rel, fs_rel):
    """Which quantity the transition holds fixed, or ``None`` if unanswerable.

    **A fixed absolute level can never answer this question**, however healthy
    its span and margin look, so it is never allowed to name one.  The reason is
    not noise, it is that the definition manufactures the effect it is being
    asked about: a fixed bar is reached later where the channel saturates lower,
    which is at low prevalence, so it produces a strength-prevalence trade-off
    out of a plane that has none.  On a synthetic plane whose transition sits at
    a fixed strength by construction, a fixed bar at 0.6 reports "product" at
    every transition width from 0.06 upwards -- with a span of 0.39 and margins
    up to 3.9, clearing both guards comfortably.  A broad margin there is not
    reassurance; it is the artifact getting stronger.

    The relative definition -- half of each row's own saturation -- moves with
    the ceiling and so does not manufacture the trade-off.  It is correct on both
    synthetic planes over the whole range of widths tested, with one exception at
    the broadest, which :data:`MIN_RATIO` is set to exclude.

    A fixed level still answers a real and different question, which is how much
    bias buys a stated degree of order.  Its numbers are reported; only its
    verdict on the *locus* is withheld.
    """
    if level is not None:
        return None
    if span < MIN_F_SPAN or ratio < MIN_RATIO:
        return None
    return "strength" if s_rel < fs_rel else "product"


def _decline_reason(level, span, ratio):
    """Why no locus was named, for the report to print instead of a verdict."""
    if level is not None:
        return "fixed level: cannot locate"
    if span < MIN_F_SPAN:
        return f"-- span {span:.2f}"
    if ratio < MIN_RATIO:
        return f"-- margin {ratio:.1f}x"
    return ""


#: Above this, the transition is broad enough that every failure mode of the
#: threshold routine is in play and a locus verdict should be treated as weak.
WIDE_TRANSITION = 0.15


def transition_widths(data, key="T_mu", branch="credulous", min_saturation=0.4,
                      smooth=True):
    """Quarter-to-three-quarters width in strength, one per prevalence row.

    Per row and not once at full prevalence: the width varies up the prevalence
    axis, and a single number taken at the top hides how many rows are in the
    regime where the locus verdict stops being trustworthy.  Rows that do not
    order at all are ``nan``.

    For a logistic of scale ``w`` the analytic answer is ``2 ln 3 w``, which is
    what ``tests/test_thresholds.py`` checks this against.

    **This is context, not locus evidence.**  Do not read the way the width
    scales with prevalence as a second opinion on where the transition sits.
    Two planes with the *same* transition -- both at ``f|a| = 0.30`` -- give
    opposite answers to that comparison depending only on whether the sigmoid's
    width was written in the product or in the strength, at margins of 210x and
    14x respectively.  The comparison measures the parameterisation of the
    width; it is not about the location at all, and a large margin is therefore
    no protection.  ``test_the_transition_width_is_not_evidence_about_the_locus``
    holds that down.  The locus rests on
    :func:`threshold_summary` under the relative definition, alone.
    """
    a, f = data["a"], data["f"]
    idx, xs = _branch_axis(a, branch)
    M = data[key][:, idx]
    if smooth:
        smoother = _smoother(xs)
        M = np.stack([smoother(row) for row in M])
    out = np.full(len(f), np.nan)
    for i, row in enumerate(M):
        sat = row[-1]
        if abs(sat) < min_saturation:
            continue
        lo = _crossing(xs, row, 0.25 * sat)
        hi = _crossing(xs, row, 0.75 * sat)
        if np.isfinite(lo) and np.isfinite(hi):
            out[i] = abs(hi - lo)
    return out


def width_summary(data, key="T_mu", **kw):
    """Median, range and how much of the plane is in the broad-transition regime.

    The fraction above :data:`WIDE_TRANSITION` is the number that matters: a
    healthy median with a fifth of the rows past the warn level is not the same
    situation as a healthy median with none.
    """
    out = {}
    for branch in ("suspicious", "credulous"):
        w = transition_widths(data, key, branch, **kw)
        w = w[np.isfinite(w)]
        if w.size == 0:
            continue
        out[branch] = {"n": int(w.size), "median": float(np.median(w)),
                       "range": (float(w.min()), float(w.max())),
                       "frac_wide": float((w > WIDE_TRANSITION).mean())}
    return out


def threshold_summary(data, key="T_mu", levels=THRESHOLD_LEVELS):
    """Is the threshold a fixed strength, or a fixed product of strength and
    prevalence?

    For each definition and each branch, take the threshold strength row by row
    and ask which of ``s*`` and ``f_a s*`` is the better conserved: whichever has
    the smaller relative spread is the quantity the transition actually holds
    fixed.  Reporting the comparison rather than one number is the point --
    ``s*`` alone reads as a property of the plane when it is partly a property of
    the definition.
    """
    f = data["f"]
    out = {}
    for branch in ("suspicious", "credulous"):
        for level in levels:
            _, s_star = threshold_curve(data, key, branch, level)
            m = np.isfinite(s_star) & (f > 0)
            if m.sum() < THRESHOLD_MIN_ROWS:
                continue
            s_abs = np.abs(s_star[m])
            prod = f[m] * s_abs
            s_rel = s_abs.std() / max(s_abs.mean(), 1e-12)
            fs_rel = prod.std() / max(prod.mean(), 1e-12)
            span = float(f[m].max() - f[m].min())
            lo, hi = float(f[m].min()), float(f[m].max())
            # Which is better conserved, and by how much.  The ratio is reported
            # because "smaller" is not a finding on its own: two relative
            # spreads that differ by a few per cent name a winner and mean
            # nothing.
            ratio = (max(s_rel, fs_rel) / max(min(s_rel, fs_rel), 1e-12))
            out[(branch, level)] = {
                "n": int(m.sum()),
                "f_span": (lo, hi),
                "span": span,
                "s": (s_abs.mean(), s_abs.std()),
                "s_rel": s_rel,
                "fs": (prod.mean(), prod.std()),
                "fs_rel": fs_rel,
                "ratio": ratio,
                # `None` where the routine cannot tell the two apart, and
                # that is an outcome rather than a default to one of them.
                # Three ways of failing to tell, and the third is the important
                # one -- see `_locus_verdict`.
                "conserved": _locus_verdict(level, span, ratio, s_rel, fs_rel),
                "declined": _decline_reason(level, span, ratio),
                # What the comparison *would* have said if it were allowed to.
                # Kept because it is the honest structure: a fixed level's
                # numbers really do support a conclusion, and it is the
                # definition that disqualifies them, not the guards.  Printing
                # it stops a reader wondering whether something failed.
                "would_have_said": ("strength" if s_rel < fs_rel else "product"),
            }
    return out


def thresholds(data, key="T_mu", smooth=True):
    """Half-saturation of ``key`` at full prevalence, one value per sign.

    Reported once per sign because there is no reason to expect the two to
    agree: the modulation functions are not symmetric in ``h_w``, so credulity
    and suspicion are two different routes out of the unbiased state.  A single
    number here would be assuming the answer.
    """
    a, f = data["a"], data["f"]
    smoother = _smoother(a) if smooth else (lambda y: y)
    top = smoother(data[key][np.argmax(f)])
    out = {}
    for branch in ("suspicious", "credulous"):
        idx, xs = _branch_axis(a, branch)
        ys = top[idx]
        sat = ys[-1]
        out[branch] = {"saturated": float(sat),
                       "half": _crossing(xs, ys, 0.5 * sat)}
        j = idx[-1]
        col = data[key][:, j]
        out[branch]["half_f"] = _crossing(f, col, 0.5 * col[np.argmax(f)])
    return out


def _row(values, fmt="{:>9.3f}"):
    return "".join("      nan" if not np.isfinite(v) else fmt.format(v)
                   for v in values)


def report(data, n_agents):
    """The numbers the figures are read for, printed."""
    g = derived(data)
    a, f = data["a"], data["f"]
    top = np.argmax(f)
    # PAPER_NAMES rather than a hand-written tuple: a hand-written list of "the
    # paper's five" is how one of them comes to be swept, described in the
    # README, and never once printed.
    cols = ("T_mu", "rho_mean") + PAPER_NAMES

    print("\n[phase] along the strength axis at full prevalence:")
    print(f"    {'a':>9}" + "".join(f"{c:>9}" for c in cols) + f"{'-T/(N-1)':>10}")
    for j in np.linspace(0, len(a) - 1, 9).astype(int):
        print(f"    {a[j]:>9.2f}" + _row([g[c][top, j] for c in cols])
              + f"{-g['T_mu'][top, j] / (n_agents - 1):>10.3f}")

    print("\n[phase] up the prevalence axis, at the two ends of the strength axis:")
    # short display names: the keys are wider than the columns they head
    gaps = (("T_mu", "T_mu"), ("give_gap", "direct"),
            ("emergent_gap", "emergent"), ("rho_gap", "rho_gap"))
    for label, j in (("a = %+.2f" % a[0], 0), ("a = %+.2f" % a[-1], len(a) - 1)):
        print(f"  {label}")
        print(f"    {'f_a':>9}" + "".join(f"{n:>9}" for _, n in gaps))
        for i in np.linspace(0, len(f) - 1, 6).astype(int):
            print(f"    {f[i]:>9.2f}" + _row([g[c][i, j] for c, _ in gaps]))

    print("\n[phase] half-saturation of T_mu at full prevalence, one per sign "
          "(the\n        modulation functions are not symmetric in h_w, so "
          "these need not agree):")
    # Smoothed and unsmoothed side by side.  Extracting a crossing depends on
    # whether the row was smoothed first, and on these planes the two can differ
    # by as much as the difference between two planes -- so the sensitivity
    # belongs on the page next to the number, not in a footnote discovered after
    # someone has quoted three decimals.
    th = thresholds(data)
    raw = thresholds(data, smooth=False)
    print(f"    {'branch':>12}{'saturated':>11}{'half at |a|':>13}"
          f"{'unsmoothed':>12}{'half at f_a':>13}")
    for tag in ("suspicious", "credulous"):
        t, r = th[tag], raw[tag]
        def _c(v, w=13):
            return f"{'nan':>{w}}" if not np.isfinite(v) else f"{abs(v):>{w}.3f}"
        print(f"    {tag:>12}{t['saturated']:>11.3f}"
              + _c(t["half"]) + _c(r["half"], 12) + _c(t["half_f"]))
    drift = max((abs(abs(th[t]["half"]) - abs(raw[t]["half"]))
                 for t in th if np.isfinite(th[t]["half"])
                 and np.isfinite(raw[t]["half"])), default=float("nan"))
    if np.isfinite(drift):
        print(f"        smoothing moves the crossing by up to {drift:.3f}; "
              f"quote no more precision than that.")
    print("\n[phase] transition width (quarter to three quarters of "
          "saturation), per row.\n        Every failure mode of the threshold "
          "routine below gets worse as this\n        grows, so it is the "
          "context for the numbers that follow:")
    widths = width_summary(data)
    print(f"    {'branch':>12}{'rows':>6}{'median':>9}{'range':>16}"
          f"{'above ' + str(WIDE_TRANSITION):>12}")
    for branch, w in widths.items():
        lo, hi = w["range"]
        print(f"    {branch:>12}{w['n']:>6}{w['median']:>9.3f}"
              f"{lo:>8.3f}-{hi:<7.3f}{w['frac_wide'] * 100:>11.0f}%")
    if any(w["median"] > WIDE_TRANSITION for w in widths.values()):
        print("        NOTE: that median is broad. Treat any locus verdict "
              "below as weak.")
    if len(widths) == 2:
        wm = [w["median"] for w in widths.values()]
        if max(wm) > 1.3 * min(wm):
            print("        The two branches have different widths, which is a "
                  "result rather than\n        a nuisance: the modulation "
                  "functions are not symmetric in h_w, so\n        credulity "
                  "and suspicion need not leave the unbiased state alike.")

    print("\n[phase] and the threshold resolved by prevalence, under both "
          "definitions.\n        'own' is half of each row's own saturation; "
          "the numbers are fixed\n        absolute levels of T_mu.  Whichever "
          "of |a*| and f_a|a*| has the smaller\n        relative spread is what "
          "the transition holds fixed:")
    summary = threshold_summary(data)
    print(f"    {'branch':>12}{'level':>6}{'rows':>5}{'f span':>12}"
          f"{'|a*|':>8}{'rel':>7}{'f|a*|':>8}{'rel':>7}{'ratio':>7}  conserved")
    for (branch, level), v in summary.items():
        name = "own" if level is None else f"{level:g}"
        lo, hi = v["f_span"]
        why = v["declined"]
        ratio = ">999" if v["ratio"] > 999 else f"{v['ratio']:.1f}"
        verdict = v["conserved"] or why
        if v["conserved"] is None and level is not None:
            verdict += f" [would have said {v['would_have_said']}]"
        print(f"    {branch:>12}{name:>6}{v['n']:>5}"
              f"{lo:>6.2f}-{hi:<5.2f}"
              f"{v['s'][0]:>8.3f}{v['s_rel']:>7.3f}"
              f"{v['fs'][0]:>8.3f}{v['fs_rel']:>7.3f}"
              f"{ratio:>5}x  {verdict}")
    print(f"        Only the relative definition is allowed to name a locus. "
          f"A fixed bar is\n        reached later where the channel saturates "
          f"lower -- at low prevalence --\n        so it manufactures a "
          f"strength-prevalence trade-off out of a plane with\n        none: on "
          f"a synthetic plane whose transition is at a fixed strength by\n"
          f"        construction it reports 'product' at every transition width "
          f"from 0.06 up,\n        clearing both guards at margins to 3.9x. Its "
          f"numbers answer a different\n        and real question -- how much "
          f"bias buys a stated degree of order -- but\n        they are not "
          f"evidence about where the transition is. The relative rows are\n"
          f"        additionally blanked unless they span {MIN_F_SPAN} in "
          f"prevalence and win by\n        {MIN_RATIO}x.\n        Quote which definition a "
          "threshold uses before comparing it with another\n        plane, and "
          "use one definition for all of them: otherwise the comparison\n"
          "        measures conventions rather than physics.")

    print("\n[phase] the class controls over the whole plane "
          "(the dynamics never reads the class label):")
    for c in ("R_cred", "R_stat", "R_cw"):
        v = data[c]
        print(f"    {c:>8}: mean {v.mean():>+7.4f}  sd {v.std():>6.4f}  "
              f"max |.| {np.abs(v).max():>6.3f}")

    # R_muc is the one control with a predicted non-zero: the uniform channel
    # leaks into the matching one at -1/(N-1) once the diagonal is excluded.
    # Quoted twice.  Over the whole plane the agreement is diluted by the pixels
    # where T_mu is near zero and there is nothing to leak, so the residual there
    # is the single-realization noise of R_muc and not a failure of the
    # prediction; the second line restricts to where the leak is worth measuring.
    pred = -data["T_mu"] / (n_agents - 1)
    resid = data["R_muc"] - pred
    print(f"    {'R_muc':>8}: mean {data['R_muc'].mean():>+7.4f}  "
          f"sd {data['R_muc'].std():>6.4f}  max |.| {np.abs(data['R_muc']).max():>6.3f}")
    for label, m in (("everywhere", np.ones_like(pred, dtype=bool)),
                     ("where |T_mu| > 0.5", np.abs(data["T_mu"]) > 0.5)):
        if m.sum() < 2:
            continue
        corr = np.corrcoef(pred[m].ravel(), data["R_muc"][m].ravel())[0, 1]
        print(f"              vs the predicted -T_mu/(N-1), {label:<18}: "
              f"r = {corr:>6.3f}, residual sd {resid[m].std():.4f} "
              f"({m.sum()} px)")

    # The tail, not just the mean.  A plane-wide mean of |R_muc| is dominated by
    # the pixels where T_mu is small and hides whatever the worst pixel does; on
    # the b plane next door the largest excursion is 0.58 and sits entirely in
    # the transition band.  Here R_muc has a predicted non-zero everywhere, so
    # the quantity with a tail worth reporting is the *excess* over that
    # prediction.
    excess = np.abs(resid)
    order = np.argsort(excess.ravel())[::-1]
    worst = order[0]
    iy, ix = np.unravel_index(worst, excess.shape)
    n_big = int((excess > 0.3).sum())
    print(f"\n[phase] the tail of that control, not its mean:")
    print(f"    largest excess over the prediction: {excess.ravel()[worst]:.3f} "
          f"at a = {a[ix]:+.3f}, f_a = {f[iy]:.3f}")
    print(f"    where T_mu = {data['T_mu'][iy, ix]:+.3f} and "
          f"R_muc = {data['R_muc'][iy, ix]:+.3f}")
    print(f"    pixels with an excess above 0.3: {n_big} of {excess.size}"
          + (f"; their median |a| is {np.median(np.abs(a[np.unravel_index(order[:n_big], excess.shape)[1]])):.3f}"
             if n_big else ""))
    print(f"    a single population can therefore read as "
          f"{'substantially ' if excess.max() > 0.3 else 'weakly '}"
          f"discriminating on the\n    paper's own parameter while nothing "
          f"in its dynamics refers to the label.")

    # Everything else the sweep records, at three representative regions.  This
    # block exists because a quantity that is measured, cached and described but
    # never printed is indistinguishable from one that was never measured -- and
    # `test_every_swept_quantity_reaches_an_output` now fails if one is added
    # without a home here.
    mid = (f > 0.25) & (f < 0.75)
    print("\n[phase] the trust blocks and the within-group structure, at mid "
          "prevalence\n        (0.25 < f_a < 0.75).  Receiver first: T_ub is an "
          "unbiased receiver's\n        trust in a biased emitter:")
    regions = (("suspicious (a < -0.6)", a < -0.6),
               ("neutral (|a| < 0.1)", np.abs(a) < 0.1),
               ("credulous (a > 0.6)", a > 0.6))
    # Two tables rather than one: nine columns of real key names do not fit a
    # terminal, and the headers ran together into `B_rho_ufrac_biased`.  Headers
    # are the keys themselves and not friendlier labels, because
    # `tests/test_outputs.py` reads them as the evidence that each quantity
    # reaches a reader -- a prettier heading would silently orphan it.
    for group, width in ((BIAS_BLOCKS + ("rho_bu",), 10),
                         (GROUP_BALANCES + ("frac_biased",), 12)):
        print(f"    {'region':>22}" + "".join(f"{k:>{width}}" for k in group))
        for label, col in regions:
            if not col.any():
                continue
            box = np.ix_(mid, col)
            cells = []
            for k in group:
                v = g[k][box]
                v = v[np.isfinite(v)]
                cells.append(f"{'nan':>{width}}" if v.size == 0
                             else f"{v.mean():>{width}.3f}")
            print(f"    {label:>22}" + "".join(cells))
    print("        frac_biased is the realized fraction, binomial in f_a rather "
          "than equal\n        to it; it is here as a sanity check on the "
          "sampling, not as a result.")

    # The emergent margin.  Quoted over a region rather than as a plane-wide
    # maximum: at f_a near 0 or 1 one group holds a handful of agents, its block
    # mean is an average over a few pairs, and the extreme pixel of the plane is
    # always one of those.  The window keeps both groups populated.
    print("\n[phase] the emergent margin, over pixels with both groups populated")
    print("        (0.25 < f_a < 0.75) -- nothing in a uniform field says a "
          "biased agent\n        should be trusted differently:")
    mid = (f > 0.25) & (f < 0.75)
    # The mean is quoted with the standard error of the mean over the pixels in
    # the box, because one pixel is one realization and the margin is small.
    print(f"    {'region':>22}{'emergent':>18}{'pooled':>10}{'direct':>10}"
          f"{'rho_gap':>16}")
    for label, col in (("suspicious (a < -0.6)", a < -0.6),
                       ("neutral (|a| < 0.1)", np.abs(a) < 0.1),
                       ("credulous (a > 0.6)", a > 0.6)):
        if not col.any():
            continue
        box = np.ix_(mid, col)
        cells = []
        # 'emergent' is the unbiased receivers alone; 'pooled' is get_gap, the
        # same comparison averaged over all receivers.  Printed side by side
        # because the difference between them is the reason the narrower one is
        # the headline: pooling folds the biased agents' own rows in, and those
        # rows are the field rather than a response to it.
        for k, width in (("emergent_gap", 18), ("get_gap", 10),
                         ("give_gap", 10), ("rho_gap", 16)):
            v = g[k][box]
            v = v[np.isfinite(v)]
            if v.size == 0:
                cells.append(f"{'nan':>{width}}")
            elif width > 12:
                cells.append(f"{v.mean():>{width - 8}.3f} +/-{v.std() / np.sqrt(v.size):>5.3f}")
            else:
                cells.append(f"{v.mean():>{width}.3f}")
        print(f"    {label:>22}" + "".join(cells))


def main():
    args, preset = setup(__doc__)
    data = run(preset, use_cache=not args.no_cache, n_strips=args.strips)
    report(data, preset.model.n_agents)
    figure_channels(data, args.style)
    figure_map(data, args.style)
    figure_cut(data, args.style, preset.model.n_agents)


if __name__ == "__main__":
    main()
