"""The modulation functions are log-derivatives of the evidence, and the two
sectors are mirror images of each other.  Both facts are checked numerically.
"""

import numpy as np
import pytest

from ednna.modulation import F_C, F_V, F_mu, F_w, Phi, evidence, g, modulation

GRID = np.linspace(-4.0, 4.0, 41)
HW, HMU = np.meshgrid(GRID, GRID)


def test_evidence_is_a_probability():
    Z = evidence(HW, HMU)
    assert np.all(Z > 0.0)
    assert np.all(Z < 1.0)


def test_evidence_is_low_in_dissonant_quadrants():
    """Agreeing with a distrusted emitter, or disagreeing with a trusted one,
    is surprising; the consonant quadrants are not."""
    consonant = evidence(3.0, -3.0)  # agree, trust
    dissonant = evidence(3.0, 3.0)  # agree, distrust
    assert dissonant < 0.02 < consonant
    assert evidence(-3.0, 3.0) > 0.9  # disagree, distrust: also consonant
    assert evidence(-3.0, -3.0) < 0.02  # disagree, trust: dissonant


def test_sector_symmetry():
    """F_w(x, y) = F_mu(y, x) and F_C(x, y) = F_V(y, x)."""
    np.testing.assert_allclose(F_w(HW, HMU), F_mu(HMU, HW), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F_C(HW, HMU), F_V(HMU, HW), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("h_mu", [-2.0, -0.5, 0.5, 2.0])
def test_F_w_is_dlogZ_dhw(h_mu):
    h = 1e-6
    for h_w in (-2.0, -0.5, 0.5, 2.0):
        fd = (np.log(evidence(h_w + h, h_mu)) - np.log(evidence(h_w - h, h_mu))) / (2 * h)
        assert F_w(h_w, h_mu) == pytest.approx(fd, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("h_w", [-2.0, -0.5, 0.5, 2.0])
def test_F_mu_is_dlogZ_dhmu(h_w):
    h = 1e-6
    for h_mu in (-2.0, -0.5, 0.5, 2.0):
        fd = (np.log(evidence(h_w, h_mu + h)) - np.log(evidence(h_w, h_mu - h))) / (2 * h)
        assert F_mu(h_w, h_mu) == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_second_derivatives():
    """F_C and F_V are the second log-derivatives, i.e. -F(F+h) as written."""
    h = 1e-4
    for h_w in (-1.5, 0.3, 2.0):
        for h_mu in (-1.5, 0.3, 2.0):
            fdw = (
                np.log(evidence(h_w + h, h_mu))
                - 2 * np.log(evidence(h_w, h_mu))
                + np.log(evidence(h_w - h, h_mu))
            ) / h**2
            assert F_C(h_w, h_mu) == pytest.approx(fdw, rel=1e-3, abs=1e-4)
            fdm = (
                np.log(evidence(h_w, h_mu + h))
                - 2 * np.log(evidence(h_w, h_mu))
                + np.log(evidence(h_w, h_mu - h))
            ) / h**2
            assert F_V(h_w, h_mu) == pytest.approx(fdm, rel=1e-3, abs=1e-4)


def test_agreement_builds_trust_and_disagreement_erodes_it():
    """The sign of F_mu is what fixes the discrimination-field convention:
    perceived agreement (h_w > 0) drives distrust mu down (F_mu < 0)."""
    assert F_mu(2.0, 0.5) < 0.0
    assert F_mu(-2.0, -0.5) > 0.0


def test_unlearning_from_distrusted_emitters():
    """A sufficiently distrusted emitter flips the sign of the Hebbian term, so
    the receiver learns the opposite of the incoming opinion."""
    assert F_w(0.5, 2.0) < 0.0  # distrusted emitter: unlearn
    assert F_w(0.5, -2.0) > 0.0  # trusted emitter: learn


def test_modulation_matches_individual_functions():
    fw, fc, fm, fv = modulation(HW, HMU)
    np.testing.assert_allclose(fw, F_w(HW, HMU), rtol=1e-13)
    np.testing.assert_allclose(fc, F_C(HW, HMU), rtol=1e-13)
    np.testing.assert_allclose(fm, F_mu(HW, HMU), rtol=1e-13)
    np.testing.assert_allclose(fv, F_V(HW, HMU), rtol=1e-13)


def test_modulation_finite_in_the_divergent_corners():
    """Z -> 0 deep in the dissonant corners; the floor keeps F finite there."""
    for point in [(8.0, 8.0), (-8.0, -8.0), (30.0, 30.0)]:
        assert np.all(np.isfinite(modulation(*point)))


def test_phi_and_g_are_consistent():
    h = 1e-6
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose((Phi(x + h) - Phi(x - h)) / (2 * h), g(x), rtol=1e-6)
