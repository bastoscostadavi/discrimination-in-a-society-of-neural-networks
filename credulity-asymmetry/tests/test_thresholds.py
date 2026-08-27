"""The threshold routine, tested against planes whose answer is known.

Testing the extraction on real data can only check that it returns something
plausible.  These build synthetic planes with a transition put in by hand -- one
at a fixed strength, one at a fixed product of strength and prevalence -- and
require the routine to recover the number that was put in, and to *name* the
right conserved quantity.  The two planes are the two answers the real planes
could give, so a routine that cannot tell them apart cannot be trusted on either.
"""

import numpy as np
import pytest

from credfield.thresholds import (
    MIN_F_SPAN, MIN_MARGIN, WIDE_TRANSITION, compare, crossing, regime_table,
    row_saturation, row_thresholds, summarise, transition_width,
)

N_S = N_F = 120
WIDTH = 0.03  # sharpness of the synthetic transition


def _plane(s_star_of_f, sat_of_f=lambda f: f, n_s=N_S, n_f=N_F, width=WIDTH):
    """A synthetic plane: channel(s, f) = sat(f) * sigmoid((s - s*(f)) / width).

    At ``s = s*(f)`` the sigmoid is exactly 1/2, so the *relative* threshold --
    half of that row's own saturated value -- is ``s*(f)`` exactly, whatever
    ``sat`` does.  That is what makes the true answer known.
    """
    s = np.linspace(0.0, 1.0, n_s)
    f = np.linspace(0.0, 1.0, n_f)
    ch = np.empty((n_f, n_s))
    for i, fv in enumerate(f):
        star = s_star_of_f(fv)
        # clipped: on a hyperbolic plane s*(f) runs to huge values as f -> 0,
        # and an unclipped exponent overflows to a warning and a nan
        z = np.clip((s - star) / width, -500.0, 500.0)
        ch[i] = sat_of_f(fv) / (1.0 + np.exp(-z))
    return {"s": s, "f": f, "channel": ch,
            "R_muc": np.zeros((n_f, n_s))}


def vertical_plane(s_star=0.40):
    """Transition at one strength, whatever the prevalence."""
    return _plane(lambda f: s_star)


def hyperbolic_plane(product=0.30):
    """Transition where strength times prevalence is constant."""
    return _plane(lambda f: product / f if f > 1e-9 else 1e9)


# --- the crossing finder -------------------------------------------------

def test_crossing_interpolates_linearly():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 2.0])
    assert crossing(x, y, 1.0) == pytest.approx(1.5)


def test_crossing_is_nan_when_the_level_is_never_reached():
    assert np.isnan(crossing([0.0, 1.0], [0.0, 0.5], 0.9))


def test_crossing_handles_a_level_of_zero_and_of_either_sign():
    """A finder that dispatches on the sign of the level breaks exactly at zero.

    ``y > level`` for positive levels and ``y < level`` for negative ones is the
    natural-looking implementation and it is ambiguous at zero, silently
    returning nan for a curve that plainly crosses it.
    """
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    assert crossing(x, y, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert crossing(x, y, -0.5) == pytest.approx(-0.5, abs=1e-12)
    assert crossing(x, y, +0.5) == pytest.approx(+0.5, abs=1e-12)


def test_crossing_takes_the_first_upward_crossing_not_a_later_one():
    x = np.linspace(0.0, 1.0, 5)          # 0, .25, .5, .75, 1
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert crossing(x, y, 0.5) == pytest.approx(0.125)


# --- the saturation guard ------------------------------------------------

def test_a_row_still_climbing_has_no_saturated_value():
    """Otherwise the relative threshold is measured against a moving target.

    This is the failure the guard exists for: on a hyperbolic plane the
    low-prevalence rows have their transition at a strength outside the box, so
    their tail is the middle of a rise, and half of that tail mean is not half of
    anything.
    """
    climbing = np.linspace(0.0, 1.0, 60)
    assert np.isnan(row_saturation(climbing))
    flat = np.concatenate([np.linspace(0.0, 0.9, 40), np.full(20, 0.9)])
    assert row_saturation(flat) == pytest.approx(0.9, abs=1e-9)


def test_a_row_that_is_flat_at_zero_has_no_saturated_value():
    assert np.isnan(row_saturation(np.zeros(40)))


# --- recovering a known transition --------------------------------------

def test_the_relative_definition_recovers_a_vertical_transition():
    d = vertical_plane(0.40)
    P = row_thresholds(d, "channel", level="relative", sat_min=0.4)
    assert len(P) > 40
    assert P[:, 1].mean() == pytest.approx(0.40, abs=0.01)
    assert P[:, 1].std() < 0.01                      # flat in prevalence
    out = summarise(d, "channel")
    assert out["conserved"] == "strength"
    assert out["strength"] == pytest.approx(0.40, abs=0.01)


def test_the_relative_definition_recovers_a_hyperbolic_transition():
    d = hyperbolic_plane(0.30)
    P = row_thresholds(d, "channel", level="relative", sat_min=0.4)
    assert len(P) > 20
    assert P[:, 2].mean() == pytest.approx(0.30, abs=0.01)   # f * s_c
    assert P[:, 2].std() < 0.01
    out = summarise(d, "channel")
    assert out["conserved"] == "product"
    assert out["product"] == pytest.approx(0.30, abs=0.01)


def test_the_two_synthetic_planes_are_told_apart():
    """The whole point: one routine, two planes, two different answers."""
    v = summarise(vertical_plane(0.40), "channel")
    h = summarise(hyperbolic_plane(0.30), "channel")
    assert (v["conserved"], h["conserved"]) == ("strength", "product")
    assert v["margin"] > 3 and h["margin"] > 3


# --- the definitions genuinely disagree ---------------------------------

#: Widths at which the recovery is essentially exact, and the broader ones where
#: it is not.  Kept separate because one tolerance cannot cover both without being
#: so loose that it stops testing anything -- which is what an earlier version of
#: this file did, asserting ``abs=0.04`` on a value of ``0.40`` and so passing at
#: every width including those where the answer is 23% wrong.
SHARP_WIDTHS = (0.03, 0.06)
BROAD_WIDTHS = (0.10, 0.12, 0.15, 0.20, 0.30)

#: The recovered value is biased *low* at finite width, and by how much is worth
#: pinning rather than describing: this is what bounds how many digits of a
#: threshold may be quoted.  Relative error of the recovered locus, my planes.
EXPECTED_BIAS = {0.10: 0.006, 0.12: 0.013, 0.15: 0.030, 0.20: 0.077, 0.30: 0.234}


def _planes():
    """The two synthetic planes, with the truth each one encodes."""
    return (("vertical", lambda w: _plane(lambda f: 0.40, width=w),
             "strength", 0.40),
            ("hyperbolic", lambda w: _plane(
                lambda f: 0.30 / f if f > 1e-9 else 1e9, width=w), "product", 0.30))


@pytest.mark.parametrize("w", SHARP_WIDTHS)
def test_the_relative_definition_is_exact_on_a_sharp_transition(w):
    """Where the transition is sharp, the recovery is exact to a part in a thousand."""
    for name, build, truth, value in _planes():
        o = summarise(build(w), "channel")
        assert o["conserved"] == truth, (name, w)
        assert o[truth] == pytest.approx(value, rel=2e-3), (name, w)


@pytest.mark.parametrize("w", BROAD_WIDTHS)
def test_the_relative_definition_still_names_the_right_quantity_when_broad(w):
    """The *verdict* survives every width tried here; the *value* does not.

    Worth separating, because the margin is about which quantity is better
    conserved and says nothing about whether the recovered number is accurate.
    """
    for name, build, truth, value in _planes():
        assert summarise(build(w), "channel")["conserved"] == truth, (name, w)


def test_the_recovered_value_is_biased_low_and_the_bias_grows_with_width():
    """What bounds the digits: a systematic error, measured rather than described.

    The threshold comes out *below* the truth once the transition is broad, on
    both planes, and monotonically in the width.  At the width the real planes
    actually have (~0.12) the bias is about 1%, which is why a threshold from this
    routine is quotable to two significant figures and not three.  The bias is
    shared by any two planes of similar width, so a *difference* between planes
    survives it even where the absolute value does not.
    """
    errs = {}
    for w, expected in EXPECTED_BIAS.items():
        for name, build, truth, value in _planes():
            got = summarise(build(w), "channel")[truth]
            assert got < value, (name, w)                 # biased low, not scattered
            errs.setdefault(w, []).append(abs(got - value) / value)
        # the pinned magnitude, generous enough not to be brittle across planes
        assert max(errs[w]) == pytest.approx(expected, rel=0.5), (w, errs[w])

    worst = [max(errs[w]) for w in sorted(errs)]
    assert worst == sorted(worst)                          # monotone in width
    assert worst[0] < 0.01                                 # ~0.6% at width 0.10
    assert worst[-1] > 0.20                                # ~23% at width 0.30


@pytest.mark.parametrize("level", [0.5, 0.6, 0.7])
def test_a_fixed_level_manufactures_a_prevalence_dependence(level):
    """Why a fixed absolute bar cannot answer "in what?" at all.

    On a plane whose transition is at one strength by construction, asking where
    the channel reaches a *fixed* value gives an answer that climbs as prevalence
    falls -- not because the transition moves, but because a fixed bar is a harder
    bar for a row that saturates lower.  The apparent spread in the strength
    threshold is conjured out of nothing: 0.000 under the relative definition.
    """
    d = vertical_plane(0.40)
    rel = summarise(d, "channel", level="relative")
    assert rel["strength_spread"] < 0.01

    fixed = summarise(d, "channel", level=level)
    assert fixed["strength_spread"] > 5 * max(rel["strength_spread"], 1e-4)
    assert fixed["conserved"] is None            # never a locus verdict

    # and in a direction: the lower the prevalence, the higher the apparent bar
    P = row_thresholds(d, "channel", level=level)
    lo = P[P[:, 0] < np.percentile(P[:, 0], 25), 1].mean()
    hi = P[P[:, 0] > np.percentile(P[:, 0], 75), 1].mean()
    assert lo > hi + 0.05


def test_neither_guard_rescues_a_fixed_level_and_this_is_the_limitation():
    """The guards catch marginal and narrow cases; they do not make a bar safe.

    This is the one test here that asserts a *wrong* answer, deliberately, so the
    limitation is recorded rather than rediscovered.  On a vertical-by-
    construction plane with a transition of width 0.10, a bar at 0.6:

      * keeps a prevalence span of ~0.39, comfortably past ``MIN_F_SPAN``;
      * prefers the product by ~3.4x, comfortably past ``MIN_MARGIN``;
      * and is wrong.

    So a fixed-level reading must not be used to infer what the threshold is a
    threshold *in*, however healthy its span and margin look.  It remains a fine
    description of how much bias buys a stated degree of order.  Only the relative
    definition answers the locus question, which is what the test above shows.
    """
    d = _plane(lambda f: 0.40, width=0.10)          # truth: strength, at 0.40
    out = summarise(d, "channel", level=0.6)

    assert out["f_span"] > MIN_F_SPAN               # span gate passes
    assert out["margin"] > MIN_MARGIN               # margin gate passes
    # ...and the answer the numbers support is still wrong.  Which is why the
    # verdict is withheld on the definition rather than on the two gates: they
    # are what would have let it through.
    assert out["would_have_said"] == "product"
    assert out["conserved"] is None
    assert "cannot locate the transition" in out["reason"]

    # the relative definition, on the same plane, is right
    assert summarise(d, "channel")["conserved"] == "strength"


def test_the_guards_still_decline_the_marginal_and_the_narrow():
    """Both gates earn their keep, on the cases they do catch."""
    narrow = vertical_plane(0.40)
    # a high bar: only rows saturating above it survive, so the span collapses
    high = summarise(narrow, "channel", level=0.8)
    assert high["conserved"] is None and "prevalence span" in high["reason"]
    # a mid bar on a sharp transition: ample span, but too close to call
    mid = summarise(narrow, "channel", level=0.6)
    assert mid["f_span"] > MIN_F_SPAN
    assert mid["conserved"] is None and "margin" in mid["reason"]


def test_compare_uses_one_definition_for_every_plane():
    out = compare({"vertical": (vertical_plane(0.40), "channel"),
                   "hyperbolic": (hyperbolic_plane(0.30), "channel")},
                  level="relative")
    assert out["vertical"]["definition"] == out["hyperbolic"]["definition"]
    assert out["vertical"]["conserved"] == "strength"
    assert out["hyperbolic"]["conserved"] == "product"


# --- the regime table ----------------------------------------------------

def test_regime_table_partitions_the_plane():
    d = vertical_plane(0.40)
    d["R_muc"] = np.random.default_rng(0).normal(0, 0.05, d["channel"].shape)
    rows = regime_table(d, "channel")
    assert sum(r[2] for r in rows) == d["channel"].size
    assert all(r[2] > 0 for r in rows)


def test_summarise_reports_nothing_rather_than_guessing_on_a_flat_plane():
    """A plane with no transition has no threshold, and must not invent one."""
    s, f = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
    d = {"s": s, "f": f, "channel": np.zeros((20, 20)),
         "R_muc": np.zeros((20, 20))}
    out = summarise(d, "channel")
    assert out["conserved"] is None and out["n_rows"] == 0


# --- the transition width, which every failure mode depends on ----------

@pytest.mark.parametrize("w", [0.02, 0.05, 0.10])
def test_transition_width_recovers_the_logistic_width(w):
    """25-75% width of a logistic is ``2 ln(3) w``, about ``2.2 w``."""
    s = np.linspace(0.0, 1.0, 400)
    y = 0.9 / (1.0 + np.exp(-(s - 0.45) / w))
    assert transition_width(s, y, smooth_width=0) == pytest.approx(
        2 * np.log(3) * w, rel=0.05)


def test_transition_width_is_nan_without_a_saturated_value():
    s = np.linspace(0.0, 1.0, 100)
    assert np.isnan(transition_width(s, np.linspace(0.0, 1.0, 100)))


def test_wide_transition_brackets_a_narrow_and_a_broad_plane():
    """The guard constant sits between the two synthetic planes, as intended.

    Only that, despite the temptation to name this after the general claim: a
    fixed level inverts from about width 0.06 upwards, which is *below*
    :data:`WIDE_TRANSITION`, so this constant does not certify a fixed-level
    verdict and nothing here shows that it does.  What it marks is where even a
    relative verdict wants its width quoted beside it.
    """
    narrow = _plane(lambda f: 0.40, width=0.03)
    wide = _plane(lambda f: 0.40, width=0.20)
    s = np.asarray(narrow["s"])
    wn = np.nanmedian([transition_width(s, r) for r in narrow["channel"]])
    ww = np.nanmedian([transition_width(s, r) for r in wide["channel"]])
    assert wn < WIDE_TRANSITION < ww
    # the relative definition survives both on these planes; that is a property
    # of these planes and not a guarantee, which is why the width is reported
    assert summarise(narrow, "channel")["conserved"] == "strength"
    assert summarise(wide, "channel")["conserved"] == "strength"


# --- what the width can and cannot be used for --------------------------

def _plane_from_arg(arg_fn, n=140, w=0.05):
    """``channel = f * sigmoid(arg(s, f) / w)`` for an arbitrary argument.

    Lets the transition's *location* and the parameterisation of its *width* be
    varied independently, which is the distinction the test below turns on.
    """
    s = np.linspace(0.0, 1.0, n)
    f = np.linspace(0.0, 1.0, n)
    ch = np.empty((n, n))
    for i, fv in enumerate(f):
        z = np.clip(arg_fn(s, fv) / w, -500.0, 500.0)
        ch[i] = fv / (1.0 + np.exp(-z))
    return {"s": s, "f": f, "channel": ch, "R_muc": np.zeros((n, n))}


def _width_locus(d):
    """Is the width flatter in strength, or in strength times prevalence?"""
    s, f, ch = d["s"], d["f"], d["channel"]
    rows = []
    for i in range(ch.shape[0]):
        sat = row_saturation(ch[i], smooth_width=0)
        if not (np.isfinite(sat) and sat >= 0.4):
            continue
        wd = transition_width(s, ch[i], smooth_width=0)
        if np.isfinite(wd):
            rows.append((f[i], wd))
    P = np.asarray(rows)
    fv, wd = P[:, 0], P[:, 1]
    rw = wd.std() / wd.mean()
    rp = (fv * wd).std() / (fv * wd).mean()
    return ("width" if rw < rp else "f*width"), max(rw, rp) / min(rw, rp)


def test_the_transition_width_is_not_evidence_about_the_locus():
    """The width says how it was parameterised, not where the transition is.

    Both planes here put the transition at exactly ``f s = 0.30``: same locus, by
    construction.  They differ only in whether the width of the sigmoid is
    measured in strength or in the product.  The width diagnostic gives *opposite*
    answers on them, while the threshold definition correctly says "product" for
    both.

    So a flat width in strength is not evidence that the transition is located in
    strength.  It is a tempting inference -- a width in strength is only a
    meaningful single number if the transition is located in strength, so the
    converse looks like it should hold -- and it does not.  The width is worth
    reporting as the scale that decides whether either threshold definition can be
    trusted, and worth nothing as an argument about the locus.
    """
    same_locus = {
        "width in strength": _plane_from_arg(
            lambda s, f: s - (0.30 / f if f > 1e-9 else 1e9)),
        "width in the product": _plane_from_arg(lambda s, f: f * s - 0.30),
    }
    verdicts = {}
    for name, d in same_locus.items():
        assert summarise(d, "channel", smooth_width=0)["conserved"] == "product", name
        verdicts[name] = _width_locus(d)

    # opposite answers, both confident, on planes with one locus
    assert verdicts["width in strength"][0] == "width"
    assert verdicts["width in the product"][0] == "f*width"
    assert verdicts["width in strength"][1] > 3
    assert verdicts["width in the product"][1] > 3
