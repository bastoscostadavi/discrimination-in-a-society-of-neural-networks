import numpy as np
import pytest

from controlledcv.modulation import F_C, F_V, F_mu, F_w, Phi, Z, modulation, phi


def test_evidence_uses_equivalent_formula():
    H = np.array([-2.0, 0.0, 2.0])
    mu = np.array([1.0, -1.0, 0.25])
    expected = Phi(H) + Phi(mu) - 2.0 * Phi(H) * Phi(mu)
    np.testing.assert_allclose(Z(H, mu), expected)


def test_modulation_formulas():
    H = 0.4
    mu = -0.7
    fw = (1.0 - 2.0 * Phi(mu)) * phi(H) / Z(H, mu)
    fmu = (1.0 - 2.0 * Phi(H)) * phi(mu) / Z(H, mu)
    assert F_w(H, mu) == pytest.approx(fw)
    assert F_mu(H, mu) == pytest.approx(fmu)
    assert F_C(H, mu) == pytest.approx(-fw * (fw + H))
    assert F_V(H, mu) == pytest.approx(-fmu * (fmu + mu))


def test_modulation_symmetry():
    H = np.linspace(-2.0, 2.0, 11)
    mu = np.linspace(1.5, -1.5, 11)
    np.testing.assert_allclose(F_w(H, mu), F_mu(mu, H))
    np.testing.assert_allclose(F_C(H, mu), F_V(mu, H))


def test_extreme_fields_are_finite_without_floor():
    H = np.array([-8.0, -6.0, 6.0, 8.0])
    mu = np.array([-7.0, 7.0, -7.0, 7.0])
    values = modulation(H, mu)
    for value in values:
        assert np.all(np.isfinite(value))
    assert np.all(Z(H, mu) > 0.0)
