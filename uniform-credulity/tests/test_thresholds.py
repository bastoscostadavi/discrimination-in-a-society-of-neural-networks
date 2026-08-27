"""The threshold routine, on planes whose answer is known in advance.

A threshold read off a phase diagram is a number two directories will be
compared on, so the routine that extracts it has to be tested against something
other than the plane it was written for.  These build synthetic planes with an
analytic transition and check the routine recovers it -- and, more importantly,
check that the *diagnostic* works: given a plane whose transition really is at a
fixed strength, it must say so, and given one whose transition really is at a
fixed product of strength and prevalence, it must say that instead.

The two definitions of "threshold" disagree systematically on the first kind of
plane and agree on the second, which is not a bug in either and is exactly what
these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniform_phase import (
    MIN_F_SPAN, MIN_RATIO, _crossing, threshold_curve, threshold_summary, thresholds,
)

N_A, N_F = 201, 101
WIDTH = 0.05          # transition width of the synthetic sigmoids
S0 = 0.40             # the vertical line's strength
K = 0.30              # the hyperbola's conserved product


def _plane(kind, s0=S0, k=K, width=WIDTH):
    """A synthetic ``(a, f_a)`` plane with an analytic transition.

    ``sat(f) = f`` so that the ceiling falls with prevalence, which is the
    ingredient that makes a fixed absolute level and a relative one disagree.

    Three kinds, and the last two matter for one specific reason.  ``hyperbola``
    and ``hyperbola_ws`` have the **same locus** -- the transition is at
    ``f|a| = k`` in both -- and differ only in whether the sigmoid's *width* is
    parameterised in the product or in the strength.  Any diagnostic that gives
    them different answers is measuring the parameterisation, not the locus.
    """
    a = np.linspace(-1.0, 1.0, N_A)
    f = np.linspace(0.0, 1.0, N_F)
    A, F = np.meshgrid(a, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "vertical":
            arg = np.abs(A) - s0
        elif kind == "hyperbola":            # locus and width both in f|a|
            arg = F * np.abs(A) - k
        elif kind == "hyperbola_ws":         # same locus, width in |a|
            arg = np.abs(A) - np.divide(k, F, out=np.full_like(F, np.inf),
                                        where=F > 0)
        else:
            raise ValueError(kind)
    T = F / (1.0 + np.exp(-np.clip(arg / width, -50.0, 50.0)))
    return {"a": a, "f": f, "T_mu": np.sign(A) * T}


def test_crossing_interpolates_rather_than_snapping_to_a_sample():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert _crossing(x, y, 0.25) == pytest.approx(0.25)
    assert np.isnan(_crossing(x, y, 5.0))


def test_a_flat_segment_at_the_level_does_not_divide_by_zero():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0])
    assert np.isfinite(_crossing(x, y, 0.0))


#: Widths at which the relative definition recovers the transition to better
#: than a few per cent.  Kept separate from :data:`WIDTHS` because recovery
#: degrades outside it, which is the next test rather than a caveat in prose.
SHARP_WIDTHS = (0.03, 0.05, 0.06)

#: How far off the recovered value gets at the broadest width tested, per plane.
#: Recorded as a number rather than described, so that a change to the estimator
#: that quietly made it worse would fail here.
WORST_RECOVERY_ERROR = {"vertical": 0.06, "hyperbola": 0.20}


def _recovered(data, kind, branch="credulous"):
    """The quantity each plane holds fixed: ``|a*|``, or ``f|a*|``."""
    f, s = threshold_curve(data, "T_mu", branch, level=None, smooth=False)
    m = np.isfinite(s) & (f > 0)
    assert m.sum() > 20
    return (np.abs(s[m]) if kind == "vertical" else f[m] * np.abs(s[m]))


@pytest.mark.parametrize("width", SHARP_WIDTHS)
@pytest.mark.parametrize("branch", ["credulous", "suspicious"])
@pytest.mark.parametrize("kind, truth", [("vertical", S0), ("hyperbola", K)])
def test_the_relative_definition_recovers_a_sharp_transition(kind, truth, branch,
                                                             width):
    """Both planes, both branches, at widths where the estimator is accurate.

    Parametrized over width on purpose.  Running one width and naming the test
    for the general claim is how a test comes to assume its own conclusion --
    the failure this file already records once, at
    :func:`test_the_transition_width_is_not_evidence_about_the_locus`.
    """
    got = _recovered(_plane(kind, width=width), kind, branch)
    assert got.mean() == pytest.approx(truth, rel=0.03)
    assert got.std() < 0.03


@pytest.mark.parametrize("kind, truth", [("vertical", S0), ("hyperbola", K)])
def test_recovery_degrades_as_the_transition_broadens(kind, truth):
    """And here is where it stops being accurate, quantified rather than implied.

    The estimator does not fail loudly at broad widths, it drifts: on the
    hyperbola plane the recovered product falls from 0.300 to 0.247 between
    widths 0.03 and 0.20, an 18 per cent error with no warning attached to it.
    That is the reason the report prints the transition width beside every
    threshold, and the reason a locus verdict from a broad plane is worth less
    than the same verdict from a sharp one.
    """
    errors = [abs(_recovered(_plane(kind, width=w), kind).mean() - truth) / truth
              for w in WIDTHS]
    assert errors[0] < 0.02                       # sharp: essentially exact
    assert errors[-1] == pytest.approx(WORST_RECOVERY_ERROR[kind], abs=0.03)
    assert errors[-1] > errors[0]
    # monotone in width, which is what makes the width a usable warning sign
    assert all(b >= a - 1e-9 for a, b in zip(errors, errors[1:])), errors


def test_a_fixed_level_turns_a_vertical_line_into_a_trade_off():
    """The systematic disagreement between the two definitions, reproduced.

    At low prevalence the channel saturates lower, so a fixed bar takes more
    strength to reach while half of the row's own ceiling does not.  A reader
    comparing two planes under two definitions would read this as a discrepancy
    in the physics.
    """
    data = _plane("vertical")
    f, rel = threshold_curve(data, "T_mu", "credulous", level=None, smooth=False)
    _, absolute = threshold_curve(data, "T_mu", "credulous", level=0.6,
                                  smooth=False)
    m = np.isfinite(rel) & np.isfinite(absolute)
    assert m.sum() > 20
    r_abs, a_abs = np.abs(rel[m]), np.abs(absolute[m])
    assert r_abs.std() < 0.01
    # the fixed bar moves with prevalence where the relative one does not
    assert a_abs.std() > 5 * r_abs.std()
    assert np.ptp(a_abs) > 0.1
    # always later, never earlier, and monotonically so as the ceiling falls
    assert np.all(a_abs >= r_abs - 1e-9)
    order = np.argsort(f[m])
    assert np.all(np.diff(a_abs[order]) <= 1e-9)


def test_the_summary_names_the_conserved_quantity():
    """The diagnostic itself: strength for one plane, product for the other."""
    for kind, expected in (("vertical", "strength"), ("hyperbola", "product")):
        summary = threshold_summary(_plane(kind), levels=(None,))
        assert summary, kind
        for (branch, _), v in summary.items():
            assert v["conserved"] == expected, f"{kind} {branch}: {v}"
            assert v["ratio"] > 2.0, f"{kind} {branch}: margin too thin: {v}"


def test_the_summary_declines_when_the_prevalence_span_is_too_narrow():
    """The guard against a confident wrong answer.

    Both filters drop low-prevalence rows preferentially, so a high fixed bar
    leaves only the top of the plane.  Over a narrow span of ``f``, ``s*`` and
    ``f s*`` are nearly proportional and whichever spread comes out smaller is
    decided by noise -- so the routine must decline rather than name one.  On
    this plane the true answer is "strength", and an unguarded routine can
    report "product" here with no warning.
    """
    summary = threshold_summary(_plane("vertical"), levels=(0.8,))
    assert summary
    for (branch, _), v in summary.items():
        assert v["span"] < MIN_F_SPAN, v
        assert v["conserved"] is None, f"{branch} named a winner it cannot see"


def test_the_summary_declines_on_a_thin_margin_even_with_a_wide_span():
    """The second hole, which the span guard alone does not close.

    On the vertical plane a fixed bar at 0.6 leaves rows spanning 0.39 of the
    prevalence axis -- past the span guard -- and the wrong quantity comes out
    marginally better conserved.  Naming it would be a confident wrong answer on
    a plane whose true threshold is a fixed strength.
    """
    summary = threshold_summary(_plane("vertical"), levels=(0.6,))
    assert summary
    for (branch, _), v in summary.items():
        assert v["span"] >= MIN_F_SPAN, v      # the span guard passes it
        assert v["ratio"] < MIN_RATIO, v       # and the margin guard catches it
        assert v["conserved"] is None, branch


def test_the_summary_still_answers_where_it_can_see():
    """The guards must not simply refuse everything."""
    for kind, expected in (("vertical", "strength"), ("hyperbola", "product")):
        summary = threshold_summary(_plane(kind), levels=(None,))
        assert summary, kind
        for (branch, _), v in summary.items():
            assert v["span"] >= MIN_F_SPAN, (kind, branch, v)
            assert v["conserved"] == expected, (kind, branch, v)


def test_the_summary_reports_the_span_it_actually_used():
    """Not just the count: a count cannot tell you the rows were all crowded
    into the top of the plane."""
    summary = threshold_summary(_plane("vertical"), levels=(0.5,))
    for (_, _), v in summary.items():
        lo, hi = v["f_span"]
        assert lo >= 0.5 - 1e-9        # rows below the bar cannot cross it
        assert hi == pytest.approx(1.0, abs=0.02)
        assert v["span"] == pytest.approx(hi - lo)


def test_a_row_that_never_orders_is_nan_and_not_the_edge_of_the_axis():
    data = _plane("vertical")
    f, s = threshold_curve(data, "T_mu", "credulous", level=None,
                           min_saturation=0.3, smooth=False)
    assert np.all(np.isnan(s[f < 0.29]))
    assert np.isfinite(s[f > 0.5]).all()


def test_a_fixed_bar_above_a_row_ceiling_is_nan():
    """Not an extrapolation, and not the end of the axis."""
    data = _plane("vertical")
    f, s = threshold_curve(data, "T_mu", "credulous", level=0.8, smooth=False)
    assert np.all(np.isnan(s[f < 0.79]))


def test_thresholds_reports_both_signs_separately():
    data = _plane("vertical")
    th = thresholds(data, smooth=False)
    assert set(th) == {"suspicious", "credulous"}
    assert abs(th["credulous"]["half"]) == pytest.approx(S0, abs=0.02)
    assert abs(th["suspicious"]["half"]) == pytest.approx(S0, abs=0.02)
    assert th["credulous"]["saturated"] > 0 > th["suspicious"]["saturated"]


# --- the limitation, recorded rather than rediscovered ---------------------

WIDTHS = (0.03, 0.05, 0.06, 0.10, 0.15, 0.20)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("kind, truth", [("vertical", "strength"),
                                         ("hyperbola", "product")])
def test_the_relative_definition_never_names_the_wrong_locus(kind, truth, width):
    """The guarantee the guards are tuned to give.

    Over every transition width tested, on both synthetic planes, the relative
    definition either names the right quantity or names none.  It is allowed to
    decline -- at the broadest widths it does -- but it must never invert.
    """
    summary = threshold_summary(_plane(kind, width=width), levels=(None,))
    assert summary, (kind, width)
    for (branch, _), v in summary.items():
        assert v["conserved"] in (truth, None), (kind, branch, width, v)


# Widths at which BOTH guards are satisfied and the fixed-level answer is still
# wrong.  At 0.06 the margin happens to come out at 1.93 and the margin guard
# catches it -- which is luck rather than protection, and is why the guards are
# not what the caller is relying on.
UNRESCUED_WIDTHS = (0.10, 0.15, 0.20)


@pytest.mark.parametrize("width", UNRESCUED_WIDTHS)
def test_neither_guard_rescues_a_fixed_level_and_this_is_the_limitation(width):
    """A fixed absolute level cannot locate a transition, and the guards do not
    make it able to.

    Asserted with the *wrong* answer on purpose, so the limitation is recorded
    instead of being rediscovered.  On a plane whose transition sits at a fixed
    strength by construction, a fixed bar reports a strength-prevalence
    trade-off -- with a prevalence span past ``MIN_F_SPAN`` and a margin past
    ``MIN_RATIO``, both guards satisfied.  The margin even *grows* with the
    transition width, so a comfortable margin here is the artifact getting
    stronger rather than reassurance.

    What protects the caller is not a guard but :func:`_locus_verdict` refusing
    to draw a locus from a fixed level at all.
    """
    summary = threshold_summary(_plane("vertical", width=width), levels=(0.6,))
    assert summary
    for (branch, _), v in summary.items():
        # both guards are satisfied ...
        assert v["span"] >= MIN_F_SPAN, (branch, width, v)
        assert v["ratio"] >= MIN_RATIO, (branch, width, v)
        # ... and the reading the numbers support is still wrong, every time.
        # Asserted on `would_have_said` rather than on the verdict, which is the
        # honest structure: the numbers really do support a conclusion, and it
        # is the definition that disqualifies them, not the guards.
        assert v["would_have_said"] == "product", (branch, width, v)
        assert v["conserved"] is None
        assert "cannot locate" in v["declined"]


def test_a_fixed_level_still_reports_its_numbers():
    """Withholding the locus verdict must not throw away the trade-off answer,
    which is a real result about how much bias buys a stated degree of order."""
    summary = threshold_summary(_plane("vertical"), levels=(0.6,))
    for (_, _), v in summary.items():
        assert np.isfinite(v["s"][0]) and np.isfinite(v["fs"][0])
        assert v["n"] > 0 and v["span"] > 0


def test_the_fixed_level_artifact_is_strongest_at_moderate_widths():
    """The specific reason a healthy margin is not reassurance here.

    Named for what it does rather than for the tidier claim.  The confidence in
    the *wrong* answer grows sharply from a narrow transition to a moderate one
    and then eases off -- 1.9, 3.9, 3.6, 3.2 at widths 0.06, 0.10, 0.15, 0.20 --
    so it is not monotone in width, and a test asserting that it were would be
    asserting something false about the artifact in order to tell a simpler
    story about it.  What matters is that past the narrowest case it never
    returns to the marginal range where the margin guard would catch it.
    """
    widths = (0.06, 0.10, 0.15, 0.20)
    ratios = [threshold_summary(_plane("vertical", width=w),
                                levels=(0.6,))[("credulous", 0.6)]["ratio"]
              for w in widths]
    assert ratios[1] > 2 * ratios[0]           # sharply worse by 0.10
    assert not all(b >= a for a, b in zip(ratios, ratios[1:]))   # and not monotone
    assert min(ratios[1:]) > MIN_RATIO         # never back within the guard


def test_the_transition_width_is_recovered():
    """The context number: a quarter-to-three-quarters width of a logistic of
    scale ``w`` is ``2 ln 3 w``.

    On the vertical plane only.  On the hyperbola plane the transition is in
    ``f|a|``, so its width measured *in strength* is ``2 ln 3 w / f`` and varies
    up the prevalence axis -- which is the next test, and is a property of that
    plane rather than an error in the estimator.
    """
    from uniform_phase import width_summary
    for w in (0.05, 0.10):
        got = width_summary(_plane("vertical", width=w), smooth=False)
        for branch, v in got.items():
            assert v["median"] == pytest.approx(2 * np.log(3) * w, rel=0.35), \
                (branch, w, v)
            lo, hi = v["range"]
            assert hi - lo < 0.2 * v["median"]      # and it is flat in f


def _width_diagnostic(data):
    """Which of ``width`` and ``f * width`` is the flatter, and by how much.

    Not exported from the module under test, and deliberately so: see
    :func:`test_the_transition_width_is_not_evidence_about_the_locus`.  It lives
    here only so the test can demonstrate why it must not be used.
    """
    from uniform_phase import transition_widths
    w = transition_widths(data, smooth=False)
    f = data["f"]
    m = np.isfinite(w) & (f > 0)
    rel_w = w[m].std() / w[m].mean()
    rel_fw = (f[m] * w[m]).std() / (f[m] * w[m]).mean()
    flatter = "width" if rel_w < rel_fw else "f*width"
    return flatter, max(rel_w, rel_fw) / max(min(rel_w, rel_fw), 1e-12)


def test_the_transition_width_is_not_evidence_about_the_locus():
    """The trap, recorded so it is not re-derived.

    It is tempting to read the width's scaling as a second, independent line of
    evidence about where the transition sits: a width in strength is only a
    meaningful single number if the transition is located in strength, so the
    converse feels as though it should follow.  It does not.

    These two planes have the *same locus* -- ``f|a| = 0.30`` in both, by
    construction -- and differ only in whether the sigmoid's width is written in
    the product or in the strength.  The threshold routine gets both right.  The
    width comparison gives them **opposite** answers, both at enormous margins.
    It is measuring the parameterisation of the width, not the location of the
    transition, and a large margin is no protection because the quantity is not
    about the locus at all.
    """
    verdicts = {}
    for kind in ("hyperbola", "hyperbola_ws"):
        data = _plane(kind)
        summary = threshold_summary(data, levels=(None,))[("credulous", None)]
        # the threshold routine, which *is* locus evidence, agrees on both
        assert summary["conserved"] == "product", (kind, summary)
        assert summary["fs"][0] == pytest.approx(K, abs=0.02), kind
        verdicts[kind] = _width_diagnostic(data)

    # ... and the width diagnostic contradicts itself on the same locus
    assert verdicts["hyperbola"][0] == "f*width"
    assert verdicts["hyperbola_ws"][0] == "width"
    assert min(v[1] for v in verdicts.values()) > 5.0, verdicts


def test_the_width_estimator_is_still_correct_about_the_width():
    """Retiring the diagnostic does not retire the measurement.

    ``transition_widths`` remains the context number that says how much to trust
    a locus verdict -- it is only its *scaling with prevalence* that says nothing
    about the locus.  On the plane whose width really is written in the product,
    ``f * width`` really is the constant; that is a true statement about that
    plane and a false one about planes in general.
    """
    from uniform_phase import transition_widths
    w = 0.05
    data = _plane("hyperbola", width=w)
    widths = transition_widths(data, smooth=False)
    f = data["f"]
    m = np.isfinite(widths) & (f > 0)
    assert m.sum() > 20
    scaled = widths[m] * f[m]
    assert scaled.mean() == pytest.approx(2 * np.log(3) * w, rel=0.35)
    assert scaled.std() / scaled.mean() < 0.1
