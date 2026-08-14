"""The local analysis, checked numerically.

Every claim the paper's mechanism section makes is asserted here, so a
proposition that stops being true cannot survive in the text.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from socsim.modulation import F_C, F_mu, F_V, F_w, evidence
from socsim.theory import (
    GAP_C,
    GAP_SIGMA,
    G,
    G_prime_at_zero,
    flow,
    gap_cdf,
    gap_pdf,
    predict_C_CT,
    restoring_force_vanishes,
    saddle_jacobian,
    separatrix_residual,
)

GRID = np.linspace(-4.0, 4.0, 41)


# -- Proposition 1 ----------------------------------------------------
def test_translation_is_exact():
    """X_D(u) = X_0(u + D e_w): the field translates the portrait rigidly."""
    for D in (-1.3, -0.4, 0.0, 0.7, 2.0):
        for h_w in GRID:
            for h_mu in (-2.0, -0.3, 0.0, 0.9, 3.0):
                a = flow(h_w, h_mu, D)
                b = flow(h_w + D, h_mu, 0.0)
                np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_evidence_is_invariant_under_the_point_reflection():
    HW, HMU = np.meshgrid(GRID, GRID)
    np.testing.assert_allclose(evidence(HW, HMU), evidence(-HW, -HMU), atol=1e-14)


def test_flow_is_equivariant_under_the_point_reflection():
    """The symmetry the discrimination field breaks.

    Both modulation functions are odd under u -> -u, because Phi(-z) = 1 - Phi(z)
    flips the (1 - 2 Phi) prefactors while leaving Z and g untouched.
    """
    HW, HMU = np.meshgrid(GRID, GRID)
    np.testing.assert_allclose(F_w(-HW, -HMU), -F_w(HW, HMU), atol=1e-12)
    np.testing.assert_allclose(F_mu(-HW, -HMU), -F_mu(HW, HMU), atol=1e-12)


def test_flow_is_equivariant_under_exchanging_the_sectors():
    HW, HMU = np.meshgrid(GRID, GRID)
    np.testing.assert_allclose(F_w(HW, HMU), F_mu(HMU, HW), atol=1e-14)
    np.testing.assert_allclose(F_C(HW, HMU), F_V(HMU, HW), atol=1e-14)


def test_the_symmetry_is_broken_exactly_when_the_field_is_present():
    for D in (-0.8, 0.8):
        u = np.array([0.6, -1.1])
        a = np.array(flow(u[0], u[1], D))
        b = -np.array(flow(-u[0], -u[1], D))
        assert not np.allclose(a, b, atol=1e-6)
    u = np.array([0.6, -1.1])
    np.testing.assert_allclose(
        flow(u[0], u[1], 0.0), -np.array(flow(-u[0], -u[1], 0.0)), atol=1e-12
    )


@pytest.mark.parametrize("D", [-1.5, -0.5, 0.0, 0.5, 1.5])
def test_separatrix_is_exactly_invariant(D):
    """On h_mu = h_w + D the flow is parallel to the line, to machine precision."""
    res = separatrix_residual(GRID, D)
    assert np.max(np.abs(res)) < 1e-12


@pytest.mark.parametrize("D", [-1.0, 0.0, 0.8])
def test_saddle_is_hyperbolic_with_the_predicted_eigenvalues(D):
    J = saddle_jacobian(D)
    np.testing.assert_allclose(J, [[0.0, -2 / np.pi], [-2 / np.pi, 0.0]], atol=1e-5)
    vals = np.sort(np.linalg.eigvals(J).real)
    np.testing.assert_allclose(vals, [-2 / np.pi, 2 / np.pi], atol=1e-5)


def test_fixed_point_sits_at_minus_D():
    for D in (-1.2, 0.0, 0.9):
        np.testing.assert_allclose(flow(-D, 0.0, D), (0.0, 0.0), atol=1e-14)


def test_antidiagonal_is_also_invariant():
    """On h_mu = -(h_w + D) the two components are equal and opposite."""
    for D in (-0.7, 0.0, 1.1):
        h_w = GRID
        fw, fmu = flow(h_w, -(h_w + D), D)
        np.testing.assert_allclose(fmu, -fw, atol=1e-12)


def test_basin_of_trust_grows_with_D():
    """Set inclusion, which is the whole mechanism in one line."""
    pts = np.stack(np.meshgrid(GRID, GRID), -1).reshape(-1, 2)
    sizes = []
    for D in (-1.0, -0.5, 0.0, 0.5, 1.0):
        sizes.append(int(np.count_nonzero(pts[:, 1] < pts[:, 0] + D)))
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


# -- the closed-form gap distribution ---------------------------------
def test_gap_cdf_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n = 2_000_000
    psi = rng.uniform(-GAP_C, GAP_C, n) - rng.normal(0.0, GAP_SIGMA, n)
    for t in (-1.5, -0.5, 0.0, 0.4, 1.2):
        assert gap_cdf(t) == pytest.approx((psi < t).mean(), abs=2e-3)


def test_gap_pdf_is_the_derivative_of_the_cdf():
    t = np.linspace(-3, 3, 61)
    eps = 1e-5
    num = (gap_cdf(t + eps) - gap_cdf(t - eps)) / (2 * eps)
    np.testing.assert_allclose(num, gap_pdf(t), atol=1e-8)


def test_gap_cdf_is_a_distribution():
    assert gap_cdf(-40.0) == pytest.approx(0.0, abs=1e-12)
    assert gap_cdf(40.0) == pytest.approx(1.0, abs=1e-12)
    t = np.linspace(-5, 5, 201)
    assert np.all(np.diff(gap_cdf(t)) >= -1e-15)


def test_G_is_odd_and_monotone_and_bounded():
    t = np.linspace(-6, 6, 121)
    np.testing.assert_allclose(G(t), -G(-t), atol=1e-12)
    assert np.all(np.diff(G(t)) > 0)
    assert G(0.0) == pytest.approx(0.0, abs=1e-14)
    assert G(30.0) == pytest.approx(1.0, abs=1e-9)
    assert G(-30.0) == pytest.approx(-1.0, abs=1e-9)


def test_initial_slope_is_the_quoted_number():
    assert G_prime_at_zero() == pytest.approx(0.9655, abs=5e-4)


# -- the prediction ---------------------------------------------------
def test_prediction_is_a_product_and_collapses():
    d = np.linspace(-1, 1, 21)
    for f_d in (0.25, 0.5, 1.0):
        np.testing.assert_allclose(predict_C_CT(d, f_d) / f_d, G(d), atol=1e-14)


def test_prediction_cannot_exceed_the_discriminator_fraction():
    for f_d in (0.1, 0.5, 0.9):
        assert abs(predict_C_CT(10.0, f_d)) <= f_d + 1e-12


def test_prediction_vanishes_without_discriminators_or_field():
    assert predict_C_CT(0.9, 0.0) == pytest.approx(0.0)
    assert predict_C_CT(0.0, 0.9) == pytest.approx(0.0)


# -- why there is no critical field strength --------------------------
def test_no_linear_restoring_force_at_the_class_symmetric_state():
    """The structural reason a mean-field boundary for C_CT cannot exist.

    F_mu is identically zero along h_w = 0, so its derivative along that line is
    zero too: the class-trust contrast has no linear restoring force at the
    symmetric state.  An earlier closure measured a largest eigenvalue that was
    identically zero across its entire grid, which this explains.
    """
    for h_mu, deriv in restoring_force_vanishes().items():
        assert abs(deriv) < 1e-9, h_mu


def test_F_mu_vanishes_identically_on_the_agreement_axis():
    np.testing.assert_allclose(F_mu(0.0, GRID), 0.0, atol=1e-15)
