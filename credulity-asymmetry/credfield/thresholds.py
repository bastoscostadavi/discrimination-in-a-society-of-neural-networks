"""Where a plane's transition is, and what it is a transition *in*.

Every threshold this directory quotes comes from here, so that the numbers in the
README are reproducible from committed code rather than from a script somebody
ran once.  It is also meant to be run on the sibling directories' planes: the
functions take a plain dict of arrays as :mod:`credfield.sweep` returns, so a
cached ``.npz`` from anywhere loads straight into them.

Two definitions, and why both are kept
--------------------------------------
A row of the plane is one prevalence.  Along it the responding channel rises from
zero to a saturated value, and "the threshold" can mean either of two things:

``level="relative"``
    the strength at which the channel reaches **half of that row's own saturated
    value**.  This asks where *this* population's transition happens.

``level=<float>``
    the strength at which the channel reaches a **fixed absolute value**.  This
    asks how much bias it takes to reach a stated degree of order.

They do not agree, and the disagreement is not a defect of either.  At low
prevalence the channel saturates lower, so half of its own saturation is a lower
bar and is reached at much the same strength; a fixed bar does not move, so
reaching it takes more strength as fewer agents carry the bias.  On the planes in
this repository the first gives a **vertical line** in ``(strength, prevalence)``
and the second a **hyperbola** ``f s = const``, and a comparison between planes
that mixes the two definitions measures the definitions instead of the physics.

:func:`summarise` therefore always reports the conserved quantity *and* the
definition that produced it, and :func:`compare` refuses to put two planes side by
side except under one definition.

It also declines to answer at all in the two regimes where the answer would be an
artifact: too narrow a span of prevalence for ``s`` and ``f s`` to be
distinguishable (:data:`MIN_F_SPAN`), and too small a margin between them to be
worth naming (:data:`MIN_MARGIN`).  Neither gate implies the other and both are
needed.  Declining a verdict that would have been correct costs nothing; naming
one that is wrong is how a definition gets mistaken for a property of the plane.

The gates are not sufficient, and it is worth being blunt about the limit rather
than trusting them.  A fixed absolute level manufactures a prevalence dependence
whenever saturation varies with prevalence, and on a broad enough transition it
inverts the verdict with a healthy span *and* a healthy margin.  So the locus
question -- is the threshold in the strength, or in strength times prevalence? --
is answerable only under ``level="relative"``, which recovers the right answer on
both synthetic planes at every transition width tried.  A fixed level answers a
different question well: how much bias it takes to reach a stated degree of order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["crossing", "row_saturation", "row_thresholds", "summarise",
           "regime_table", "compare", "profile", "smooth", "transition_width"]

#: Rows whose tail is not flat to this relative tolerance have not saturated
#: within the swept range, so half of their tail mean is not half of their
#: saturated value and the relative threshold would be measured against a moving
#: target.  Such rows are dropped rather than reported.
#:
#: The test is applied to the *smoothed* row.  One pixel is one realization, so
#: an unsmoothed tail is never flat to any useful tolerance -- applying this to
#: raw rows rejects almost all of them, and rejects the low-prevalence ones
#: preferentially, which silently narrows the prevalence range.  That is fatal
#: here rather than merely wasteful: telling "constant strength" from "constant
#: strength times prevalence" is only possible across a wide span of prevalence,
#: so a guard that narrows the span makes the two indistinguishable and the
#: comparison meaningless.
TAIL_FLATNESS = 0.06

#: How many columns at the high-strength end define "the tail".
TAIL_COLUMNS = 20

#: Columns pooled into the running mean along the strength axis before anything
#: is measured.  Matches the smoothing the cut figure uses.
SMOOTH_WIDTH = 7

#: The prevalence span the retained rows must cover for :func:`summarise` to name
#: a conserved quantity.  Over a narrow span ``s`` and ``f s`` are nearly
#: proportional, so both look equally conserved and whichever wins does so by
#: accident.
MIN_F_SPAN = 0.3

#: How much better conserved one quantity must be before it is named.  The span
#: gate is **not sufficient on its own**: a fixed bar partway up a plane can
#: retain a perfectly adequate span and still invert the verdict, and how badly
#: depends on how wide the transition is relative to the range swept.  On a
#: synthetic plane that is vertical by construction, a bar at 0.6 keeps a span of
#: 0.39 -- comfortably past the span gate -- and returns "product" once the
#: transition is broad enough (``tests/test_thresholds.py`` builds exactly that
#: plane).  With a narrower transition the same bar returns "strength" by a
#: margin of 1.4, so a verdict at that margin is luck either way and is worth
#: declining.
#:
#: The two gates catch different failures and neither implies the other: at a
#: high bar the margin can be comfortable while the span is short, and partway up
#: the span can be ample while the margin is marginal.
#:
#: Both guard the same *kind* of failure, though: a quantity that is about the
#: locus but poorly resolved.  Neither does anything about a quantity that is not
#: about the locus at all -- there the margin can run into the hundreds and mean
#: nothing (:func:`transition_width`).  A healthy margin is only as good as the
#: relevance of what it is a margin on.
#:
#: **Neither gate, nor both, makes a fixed-level reading safe.** On the same
#: vertical-by-construction plane with a transition of width 0.10, a bar at 0.6
#: keeps a span of 0.39 and prefers the product by 3.4x -- both gates pass -- and
#: is wrong.  The gates catch the marginal and the narrow; they cannot catch a
#: definition that manufactures the effect it is being asked about.  Use
#: ``level="relative"`` to ask what the threshold is a threshold *in*; a fixed
#: level answers a different and equally legitimate question, and its answer to
#: this one is not evidence.  ``tests/test_thresholds.py`` asserts both halves.
#:
#: Set to 2.0 rather than 1.5 to exclude an inversion of the *relative* definition
#: seen at transition width 0.20 on an independently built synthetic plane, at a
#: margin of 1.78.  It does not occur on the planes built here at any width up to
#: 0.40, which is the reason for the stricter value rather than against it: the
#: robustness of the relative definition is a property of the plane as well as of
#: the definition, so the gate is set for the worse case rather than the observed
#: one.  The cost is declining some correct verdicts near that width.
MIN_MARGIN = 2.0


def smooth(y, width=SMOOTH_WIDTH):
    """Edge-corrected running mean along the last axis.

    ``mode="same"`` pads with zeros, which drags both ends of the curve towards
    the axis and would move a crossing near the left edge; dividing by the same
    convolution of a ones-vector is the edge-correct mean.  The window shrinks on
    an axis narrower than itself and is kept odd so the mean stays centred.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    width = min(int(width), n if n % 2 else n - 1)
    if width <= 1:
        return y
    k = np.ones(width) / width
    norm = np.convolve(np.ones(n), k, mode="same")
    return np.convolve(y, k, mode="same") / norm


#: Above this transition width the locus verdict is weak and should be quoted
#: with the width beside it.  Every failure mode of both definitions is a function
#: of how broad the transition is relative to the range swept: a fixed level
#: inverts on a vertical plane from about 0.06 upwards, and the relative
#: definition, while far more robust, has been seen to invert near 0.20 on an
#: independently built synthetic plane (it does not on the ones here, at any width
#: up to 0.40, which is the point -- the robustness is a property of the plane as
#: much as of the definition, so the width belongs next to the verdict).
WIDE_TRANSITION = 0.15


def transition_width(x, row, lo=0.25, hi=0.75, smooth_width=SMOOTH_WIDTH, **kw):
    """Quarter-to-three-quarters width of one row's transition, in strength.

    ``nan`` if the row has no saturated value or does not cross both levels.  The
    width is what decides whether either threshold definition can be trusted on a
    plane, so it is measured rather than assumed.

    **This is a scale, not evidence about the locus, and no margin makes it one.**
    How flat the width is across prevalence -- whether ``width`` or ``f width``
    varies less -- measures how the transition's width is *parameterised*, not
    where the transition sits.  Two planes can put the transition at exactly the
    same place and give opposite answers to that comparison, both by margins in
    the hundreds: see
    ``tests/test_thresholds.py::test_the_transition_width_is_not_evidence_about_the_locus``.
    The temptation is worth naming because it is asymmetric -- a width in strength
    is only a meaningful single number *if* the transition is located in strength,
    so the converse looks as though it should follow.  It does not.
    """
    row = np.asarray(row, dtype=float)
    if smooth_width:
        row = smooth(row, smooth_width)
    sat = row_saturation(row, **kw)
    if not np.isfinite(sat):
        return float("nan")
    a = crossing(x, row, lo * sat)
    b = crossing(x, row, hi * sat)
    return float("nan") if not (np.isfinite(a) and np.isfinite(b)) else b - a


def crossing(x, y, level):
    """First ``x`` where ``y`` rises through ``level``, linearly interpolated.

    ``nan`` if it never does.  Note that the comparison is written so that the
    sign of ``level`` is irrelevant: dispatching on it (``y > level`` for positive
    levels, ``y < level`` for negative) is ambiguous at exactly zero and returns
    ``nan`` for a curve that plainly crosses it.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    below = y[:-1] < level
    above = y[1:] >= level
    idx = np.flatnonzero(below & above)
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    span = y[k + 1] - y[k]
    if span == 0:
        return float(x[k])
    t = (level - y[k]) / span
    return float(x[k] + t * (x[k + 1] - x[k]))


def row_saturation(row, tail=TAIL_COLUMNS, flatness=TAIL_FLATNESS,
                   smooth_width=None):
    """The saturated value of one row, or ``nan`` if it has not saturated.

    The tail mean, guarded by a flatness test: a row still climbing at the end of
    the swept range has no saturated value to take half of, and silently using
    its tail mean anyway is how a threshold gets measured against a target that
    moves with prevalence.  Pass ``smooth_width`` to smooth first, which is
    necessary on real rows -- see :data:`TAIL_FLATNESS`.
    """
    row = np.asarray(row, dtype=float)
    if smooth_width:
        row = smooth(row, smooth_width)
    end = row[-tail:] if row.size >= tail else row
    mu = float(end.mean())
    if not np.isfinite(mu) or abs(mu) < 1e-12:
        return float("nan")
    if float(end.std()) > flatness * abs(mu):
        return float("nan")
    return mu


def row_thresholds(data, channel, level="relative", sat_min=0.4,
                   smooth_width=SMOOTH_WIDTH, **kw):
    """Per-row thresholds. Returns ``(f, s_c, f * s_c)`` as an ``(n, 3)`` array.

    Rows with no crossing, or with no saturated value when ``level`` is
    ``"relative"``, are omitted rather than filled with ``nan``: they are rows
    where the plane has no transition inside the box, which is a fact about the
    range swept and not a threshold.
    """
    s, f = np.asarray(data["s"]), np.asarray(data["f"])
    ch = np.asarray(data[channel])
    rows = []
    for i in range(ch.shape[0]):
        row = smooth(ch[i], smooth_width) if smooth_width else ch[i]
        if level == "relative":
            sat = row_saturation(row, **kw)
            if not np.isfinite(sat) or sat < sat_min:
                continue
            target = 0.5 * sat
        else:
            target = float(level)
        s_c = crossing(s, row, target)
        if not np.isfinite(s_c):
            continue
        rows.append((f[i], s_c, f[i] * s_c))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def summarise(data, channel, level="relative", **kw):
    """Is the threshold in the strength, or in strength times prevalence?

    Returns a dict carrying both candidates with their relative spreads, and
    ``conserved``: the name of whichever varies less across prevalence.  The
    definition is carried in the result, because the answer depends on it.
    """
    P = row_thresholds(data, channel, level=level, **kw)
    if len(P) < 3:
        return {"definition": level, "n_rows": len(P), "conserved": None,
                "reason": "fewer than three rows have a transition in range"}
    out = {"definition": level, "n_rows": len(P),
           "f_min": float(P[:, 0].min()), "f_max": float(P[:, 0].max())}
    out["f_span"] = out["f_max"] - out["f_min"]
    for key, col in (("strength", 1), ("product", 2)):
        mu, sd = float(P[:, col].mean()), float(P[:, col].std())
        out[key] = mu
        out[f"{key}_sd"] = sd
        out[f"{key}_spread"] = sd / abs(mu) if mu else float("inf")
    out["margin"] = (max(out["strength_spread"], out["product_spread"])
                     / max(min(out["strength_spread"], out["product_spread"]), 1e-300))
    winner = ("strength" if out["strength_spread"] < out["product_spread"]
              else "product")
    if out["f_span"] < MIN_F_SPAN:
        # Over a narrow prevalence span s and f*s are nearly proportional, so
        # naming a winner would be reporting an accident.
        out["conserved"] = None
        out["reason"] = (f"prevalence span {out['f_span']:.2f} < {MIN_F_SPAN}: "
                         "s and f*s are not distinguishable over it")
        return out
    if out["margin"] < MIN_MARGIN:
        out["conserved"] = None
        out["reason"] = (f"margin {out['margin']:.2f}x < {MIN_MARGIN}x: "
                         f"'{winner}' is not better conserved by enough to name")
        return out
    if level != "relative":
        # A fixed level manufactures a prevalence dependence, so its verdict is
        # not evidence about the locus however healthy the span and margin.  The
        # numbers stay in the result; the verdict does not.
        out["conserved"] = None
        out["reason"] = ("a fixed absolute level cannot locate the transition; "
                         f"it would have said '{winner}'")
        out["would_have_said"] = winner
        return out
    out["conserved"] = winner
    return out


def profile(data, key, f_min=0.95):
    """One quantity along the strength axis, pooled over the top prevalences.

    A single row is a single realization per pixel and is unreadable; pooling the
    rows above ``f_min`` is what makes a crossing meaningful at all.
    """
    f = np.asarray(data["f"])
    m = f >= f_min
    if not m.any():
        m = f >= np.percentile(f, 95)
    return np.asarray(data[key])[m].mean(axis=0)


def regime_table(data, channel, param="R_muc", bands=((None, 0.2), (0.2, 0.9),
                                                      (0.9, None))):
    """``param`` resolved by how far the responding channel has got.

    A plane-wide mean or max of ``R_muc`` is the wrong summary: the phase where
    the field has done its work and the band where it is half-formed behave
    differently, and the aggregate hides both.  Returns one row per band with the
    pixel count, mean, standard deviation and largest magnitude.
    """
    ch, arr = np.asarray(data[channel]), np.asarray(data[param])
    out = []
    for lo, hi in bands:
        m = np.ones_like(ch, dtype=bool)
        if lo is not None:
            m &= ch > lo
        if hi is not None:
            m &= ch <= hi
        if not m.any():
            out.append((lo, hi, 0, float("nan"), float("nan"), float("nan")))
            continue
        v = arr[m]
        out.append((lo, hi, int(m.sum()), float(v.mean()), float(v.std()),
                    float(np.abs(v).max())))
    return out


def compare(planes, level="relative", **kw):
    """Summarise several planes under **one** definition.

    ``planes`` maps a label to ``(data, channel)``.  The shared ``level`` is the
    whole point: two planes summarised under two definitions differ by the
    definitions, and that difference is larger than the difference between the
    planes in this repository.
    """
    return {label: summarise(data, channel, level=level, **kw)
            for label, (data, channel) in planes.items()}
