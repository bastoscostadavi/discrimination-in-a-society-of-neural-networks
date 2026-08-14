import numpy as np
import pytest

from controlledcv.fields import field_density


def trapz(y, x):
    return np.trapezoid(y, x)


def test_field_density_normalizes_for_correlations():
    h = np.linspace(-10.0, 10.0, 20001)
    for rho in (-0.8, 0.0, 0.7):
        p = field_density(h, q_r=1.0, rho=rho)
        assert trapz(p, h) == pytest.approx(1.0, abs=1.0e-12)
        assert abs(trapz(h * p, h)) < 0.8


def test_uncorrelated_field_is_centered_gaussian():
    h = np.linspace(-4.0, 4.0, 101)
    p = field_density(h, q_r=1.7, rho=0.0)
    expected = np.exp(-0.5 * h * h / 1.7) / np.sqrt(2.0 * np.pi * 1.7)
    np.testing.assert_allclose(p, expected)


def test_perfect_overlap_limits_are_one_sided():
    h = np.array([-1.0, 0.0, 1.0])
    p_pos = field_density(h, q_r=1.0, rho=1.0)
    p_neg = field_density(h, q_r=1.0, rho=-1.0)
    assert p_pos[0] == 0.0
    assert p_pos[2] > 0.0
    assert p_neg[0] > 0.0
    assert p_neg[2] == 0.0
