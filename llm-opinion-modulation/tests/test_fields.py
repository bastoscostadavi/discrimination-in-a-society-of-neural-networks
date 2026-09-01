"""The conversion into the paper's fields, and the sign conventions it must meet.

The conventions are checked against ``ednna.modulation`` itself rather than
against a transcription of it, so that a change to the paper's sign convention
breaks these tests rather than silently disagreeing with them.
"""

import numpy as np
import pytest
from ednna.modulation import F_mu, F_w

from llmmod2.fields import (fit_lam, h_of, nominal_h_mu, opinion_fields,
                            trust_fields)


def test_h_of_is_odd_monotone_and_unbounded():
    lean = np.array([-40.0, -8.0, -1.0, 0.0, 1.0, 8.0, 40.0])
    h = h_of(lean, 0.9)
    assert np.all(np.diff(h) > 0)
    assert h[3] == pytest.approx(0.0, abs=1e-9)
    assert h_of(-lean, 0.9) == pytest.approx(-h)
    assert abs(h[0]) > 3.0, "the readout must not have a ceiling"


def test_nominal_h_mu_signs_follow_the_track_record():
    assert nominal_h_mu(20, 20) < 0.0, "a source always right is trusted"
    assert nominal_h_mu(0, 20) > 0.0, "a source always wrong is distrusted"
    assert nominal_h_mu(10, 20) == pytest.approx(0.0)


def test_fit_lam_recovers_the_scale_it_was_generated_with():
    """Weights generated at a known scale must calibrate back to that scale."""
    from scipy.special import logit, ndtr

    ks = np.array([0, 4, 10, 16, 20])
    lam = 0.8
    target = np.array([-nominal_h_mu(k, 20) for k in ks])
    weights = logit(ndtr(target)) / lam          # h_of inverted
    got, rss, r = fit_lam(weights, ks, 20)
    assert got == pytest.approx(lam, rel=0.05)
    assert rss < 1e-6


def test_opinion_fields_are_signed_by_the_message():
    """``h_w > 0`` is agreement and ``dh_w > 0`` is movement towards the message."""
    h_w, dh_w = opinion_fields(2.0, 3.0, +1, 1.0)
    assert h_w > 0 and dh_w > 0
    flipped_h, flipped_d = opinion_fields(2.0, 3.0, -1, 1.0)
    assert flipped_h == pytest.approx(-h_w)
    assert flipped_d == pytest.approx(-dh_w)


def test_trust_fields_call_a_heavier_word_less_distrusted():
    h_mu, dh_mu = trust_fields(1.0, 3.0, 1.0)
    assert h_mu < 0.0, "a colleague worth evidence is trusted"
    assert dh_mu < 0.0, "a colleague worth more evidence is trusted more"


def test_the_predictions_this_experiment_tests_are_the_papers():
    """The three properties the measurement is compared against, from the theory."""
    # the trust gate: the sign of F_w follows the emitter, not the agreement
    assert F_w(0.5, 2.0) < 0.0 and F_w(-0.5, 2.0) < 0.0
    assert F_w(0.5, -2.0) > 0.0 and F_w(-0.5, -2.0) > 0.0
    # conviction damping, at fixed trust
    assert abs(F_w(0.2, -1.0)) > abs(F_w(2.5, -1.0))
    # the reflection symmetry the two sectors are measured to test
    assert F_w(0.7, -1.3) == pytest.approx(F_mu(-1.3, 0.7))
