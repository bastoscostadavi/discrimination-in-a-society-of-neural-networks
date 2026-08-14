import numpy as np
import pytest

from smallcv.modulation import F_C, F_V, F_mu, F_w, evidence, g, Phi


def test_evidence_formula():
    hw = np.array([-1.0, 0.0, 1.0])
    hm = np.array([0.5, -0.5, 0.0])
    expected = Phi(hw) + Phi(hm) - 2.0 * Phi(hw) * Phi(hm)
    np.testing.assert_allclose(evidence(hw, hm), expected)


def test_modulation_formulas():
    hw = 0.4
    hm = -0.7
    Z = evidence(hw, hm)
    fw = (1.0 - 2.0 * Phi(hm)) * g(hw) / Z
    fm = (1.0 - 2.0 * Phi(hw)) * g(hm) / Z
    assert F_w(hw, hm) == pytest.approx(fw)
    assert F_mu(hw, hm) == pytest.approx(fm)
    assert F_C(hw, hm) == pytest.approx(-fw * (fw + hw))
    assert F_V(hw, hm) == pytest.approx(-fm * (fm + hm))


def test_sector_symmetry():
    hw = np.linspace(-2.0, 2.0, 9)
    hm = np.linspace(1.5, -1.5, 9)
    np.testing.assert_allclose(F_w(hw, hm), F_mu(hm, hw))
    np.testing.assert_allclose(F_C(hw, hm), F_V(hm, hw))
