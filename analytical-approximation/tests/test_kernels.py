import numpy as np

from controlledcv.kernels import (
    K_C,
    affective_kernels,
    affective_kernels_mc,
    covariance_kernel_mc,
    field_density,
    ideological_coefficients,
    ideological_coefficients_mc,
    ideological_moments,
)


def test_affective_quadrature_matches_monte_carlo():
    quad = affective_kernels(q_r=1.2, rho=0.35, mu=-0.25, D=0.4, order=120)
    mc = affective_kernels_mc(q_r=1.2, q_e=0.9, rho=0.35, mu=-0.25, D=0.4, n=600_000, seed=1)
    assert abs(quad.M_mu - mc.M_mu) < 3.5e-3
    assert abs(quad.M_V - mc.M_V) < 5.0e-3


def test_ideological_moments_reduced_match_direct_quadrature():
    moments = ideological_moments(
        q_r=1.1, q_e=0.7, rho=-0.45, mu=0.2, D=-0.3, direct_order=220
    )
    assert abs(moments.m_e - moments.m_e_direct) < 3.0e-5


def test_ideological_coefficients_match_monte_carlo():
    quad = ideological_coefficients(q_r=1.0, q_e=1.4, rho=0.25, mu=0.15, D=-0.2)
    mc = ideological_coefficients_mc(q_r=1.0, q_e=1.4, rho=0.25, mu=0.15, D=-0.2, n=800_000, seed=2)
    assert abs(quad.A - mc.A) < 5.0e-3
    assert abs(quad.B - mc.B) < 5.0e-3


def test_covariance_kernel_matches_monte_carlo():
    quad = K_C(q_r=0.9, q_e=1.3, rho=-0.2, mu=-0.1, D=0.25, order=70)
    mc = covariance_kernel_mc(q_r=0.9, q_e=1.3, rho=-0.2, mu=-0.1, D=0.25, n=800_000, seed=3)
    np.testing.assert_allclose(quad.active, mc.active, atol=5.0e-3)
    assert abs(quad.orthogonal - mc.orthogonal) < 5.0e-3


def test_rho_zero_density_is_used_by_kernels():
    h = np.linspace(-7.0, 7.0, 3001)
    p = field_density(h, q_r=1.0, rho=0.0)
    assert abs(np.trapezoid(p, h) - 1.0) < 1.0e-10
