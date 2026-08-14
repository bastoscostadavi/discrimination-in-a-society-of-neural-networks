"""Gaussian-averaged kernels for the controlled small-C,V reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import RHO_TOL, field_density
from .modulation import F_C, F_V, F_mu, F_w, Phi, phi
from .quadrature import normal_expectation_1d, normal_expectation_2d


@dataclass(frozen=True)
class AffectiveKernels:
    M_mu: float
    M_V: float


@dataclass(frozen=True)
class IdeologicalMoments:
    m_r: float
    m_e: float
    m_e_direct: float


@dataclass(frozen=True)
class IdeologicalCoefficients:
    A: float
    B: float
    m_r: float
    m_e: float


@dataclass(frozen=True)
class CovarianceKernel:
    """Low-dimensional representation of K_C.

    `active` is the 2x2 block in an orthonormal basis where
    w_r = sqrt(q_r) e1 and
    w_e = sqrt(q_e) (rho e1 + sqrt(1-rho^2) e2).
    `orthogonal` is the coefficient multiplying identity directions outside
    span{w_r, w_e}.
    """

    active: np.ndarray
    orthogonal: float


def _check(q_r, q_e, rho):
    q_r = float(q_r)
    q_e = float(q_e)
    rho = float(rho)
    if q_r <= 0.0 or q_e <= 0.0:
        raise ValueError("q_r and q_e must be positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    return q_r, q_e, rho


def _skew_expectation(q_r, rho, fn, order=100):
    """Compute E[fn(h)] under p(h | q_r, rho)."""

    q_r = float(q_r)
    rho = float(rho)
    sqrt_qr = np.sqrt(q_r)
    if rho >= 1.0 - RHO_TOL:
        return normal_expectation_1d(
            lambda z: 2.0 * (z >= 0.0) * fn(sqrt_qr * z), order=order
        )
    if rho <= -1.0 + RHO_TOL:
        return normal_expectation_1d(
            lambda z: 2.0 * (z <= 0.0) * fn(sqrt_qr * z), order=order
        )
    c = rho / np.sqrt(1.0 - rho * rho)
    return normal_expectation_1d(
        lambda z: 2.0 * Phi(c * z) * fn(sqrt_qr * z), order=order
    )


def affective_kernels(q_r, rho, mu, D=0.0, order=100):
    """Quadrature kernels M_mu and M_V."""

    M_mu = _skew_expectation(q_r, rho, lambda h: F_mu(h + D, mu), order=order)
    M_V = _skew_expectation(q_r, rho, lambda h: F_V(h + D, mu), order=order)
    return AffectiveKernels(float(M_mu), float(M_V))


def m_r(q_r, rho, mu, D=0.0, order=100):
    """One-dimensional quadrature for m_r = E[h F_w(h + D, mu)]."""

    return float(_skew_expectation(q_r, rho, lambda h: h * F_w(h + D, mu), order=order))


def _cond_abs_ue_given_h(h, q_r, q_e, rho):
    """Analytic E[|u_e| | h = sign(u_e) u_r]."""

    h = np.asarray(h, dtype=float)
    q_r, q_e, rho = _check(q_r, q_e, rho)
    scale = np.sqrt(q_e / q_r)
    if rho >= 1.0 - RHO_TOL:
        return np.where(h >= 0.0, scale * h, 0.0)
    if rho <= -1.0 + RHO_TOL:
        return np.where(h <= 0.0, -scale * h, 0.0)
    sigma = np.sqrt(q_e * (1.0 - rho * rho))
    mean = rho * np.sqrt(q_e / q_r) * h
    alpha = mean / sigma
    # Mills-ratio form for E[Y | Y > 0].  For alpha << 0, Phi(alpha)
    # underflows and alpha + phi(alpha)/Phi(alpha) has a small positive
    # asymptotic value.  The leading term is enough for the far-tail branch
    # and avoids NaNs in wide kernel grids.
    out = np.empty_like(alpha, dtype=float)
    far_left = alpha < -10.0
    out[far_left] = sigma / np.maximum(-alpha[far_left], 1.0e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[~far_left] = mean[~far_left] + sigma * phi(alpha[~far_left]) / Phi(alpha[~far_left])
    return out


def m_e_reduced(q_r, q_e, rho, mu, D=0.0, order=100):
    """One-dimensional quadrature for m_e using E[|u_e| | h]."""

    q_r, q_e, rho = _check(q_r, q_e, rho)
    return float(
        _skew_expectation(
            q_r,
            rho,
            lambda h: _cond_abs_ue_given_h(h, q_r, q_e, rho) * F_w(h + D, mu),
            order=order,
        )
    )


def _correlated_fields(z1, z2, q_r, q_e, rho):
    sqrt_qr = np.sqrt(q_r)
    sqrt_qe = np.sqrt(q_e)
    if rho >= 1.0 - RHO_TOL:
        return sqrt_qr * z1, sqrt_qe * z1
    if rho <= -1.0 + RHO_TOL:
        return sqrt_qr * z1, -sqrt_qe * z1
    return sqrt_qr * z1, sqrt_qe * (rho * z1 + np.sqrt(1.0 - rho * rho) * z2)


def m_e_direct(q_r, q_e, rho, mu, D=0.0, order=60):
    """Direct two-dimensional Gaussian quadrature for m_e."""

    q_r, q_e, rho = _check(q_r, q_e, rho)

    def integrand(z1, z2):
        u_r, u_e = _correlated_fields(z1, z2, q_r, q_e, rho)
        s = np.where(u_e >= 0.0, 1.0, -1.0)
        h = s * u_r
        return u_e * s * F_w(h + D, mu)

    return float(normal_expectation_2d(integrand, order=order))


def ideological_moments(q_r, q_e, rho, mu, D=0.0, order=100, direct_order=60):
    mr = m_r(q_r, rho, mu, D=D, order=order)
    me = m_e_reduced(q_r, q_e, rho, mu, D=D, order=order)
    med = m_e_direct(q_r, q_e, rho, mu, D=D, order=direct_order)
    return IdeologicalMoments(mr, me, med)


def ideological_coefficients(q_r, q_e, rho, mu, D=0.0, order=100):
    """Return A and B in K_w = A w_r + B w_e."""

    q_r, q_e, rho = _check(q_r, q_e, rho)
    mr = m_r(q_r, rho, mu, D=D, order=order)
    me = m_e_reduced(q_r, q_e, rho, mu, D=D, order=order)
    q_re = rho * np.sqrt(q_r * q_e)
    det = q_r * q_e - q_re * q_re
    if det <= 1e-12:
        raise ValueError("A and B are singular at |rho| = 1; use moments directly")
    A = (q_e * mr - q_re * me) / det
    B = (q_r * me - q_re * mr) / det
    return IdeologicalCoefficients(float(A), float(B), mr, me)


def K_C(q_r, q_e, rho, mu, D=0.0, order=60):
    """Quadrature covariance kernel K_C = E[x x^T F_C(h + D, mu)]."""

    q_r, q_e, rho = _check(q_r, q_e, rho)

    def fc(z1, z2):
        u_r, u_e = _correlated_fields(z1, z2, q_r, q_e, rho)
        s = np.where(u_e >= 0.0, 1.0, -1.0)
        return F_C(s * u_r + D, mu)

    k00 = normal_expectation_2d(lambda z1, z2: z1 * z1 * fc(z1, z2), order=order)
    k01 = normal_expectation_2d(lambda z1, z2: z1 * z2 * fc(z1, z2), order=order)
    k11 = normal_expectation_2d(lambda z1, z2: z2 * z2 * fc(z1, z2), order=order)
    k_perp = normal_expectation_2d(fc, order=order)
    active = np.array([[k00, k01], [k01, k11]], dtype=float)
    return CovarianceKernel(active=active, orthogonal=float(k_perp))


def affective_kernels_mc(q_r, q_e, rho, mu, D=0.0, n=200_000, seed=0):
    """Monte Carlo reference for M_mu and M_V using explicit Gaussian fields."""

    h, _u_r, _u_e = _sample_h(q_r, q_e, rho, n=n, seed=seed)
    return AffectiveKernels(
        float(np.mean(F_mu(h + D, mu))),
        float(np.mean(F_V(h + D, mu))),
    )


def ideological_coefficients_mc(q_r, q_e, rho, mu, D=0.0, n=300_000, seed=0):
    """Monte Carlo reference for A and B from explicit Gaussian fields."""

    q_r, q_e, rho = _check(q_r, q_e, rho)
    h, u_r, u_e = _sample_h(q_r, q_e, rho, n=n, seed=seed)
    s = np.where(u_e >= 0.0, 1.0, -1.0)
    fw = F_w(h + D, mu)
    mr = float(np.mean(u_r * s * fw))
    me = float(np.mean(u_e * s * fw))
    q_re = rho * np.sqrt(q_r * q_e)
    det = q_r * q_e - q_re * q_re
    if det <= 1e-12:
        raise ValueError("A and B are singular at |rho| = 1; use moments directly")
    A = (q_e * mr - q_re * me) / det
    B = (q_r * me - q_re * mr) / det
    return IdeologicalCoefficients(float(A), float(B), mr, me)


def covariance_kernel_mc(q_r, q_e, rho, mu, D=0.0, n=300_000, seed=0):
    """Monte Carlo reference for the active-basis covariance kernel."""

    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u_r, u_e = _correlated_fields(z1, z2, q_r, q_e, rho)
    s = np.where(u_e >= 0.0, 1.0, -1.0)
    fc = F_C(s * u_r + D, mu)
    active = np.array(
        [
            [np.mean(z1 * z1 * fc), np.mean(z1 * z2 * fc)],
            [np.mean(z1 * z2 * fc), np.mean(z2 * z2 * fc)],
        ],
        dtype=float,
    )
    return CovarianceKernel(active=active, orthogonal=float(np.mean(fc)))


def _sample_h(q_r, q_e, rho, n, seed):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u_r, u_e = _correlated_fields(z1, z2, q_r, q_e, rho)
    s = np.where(u_e >= 0.0, 1.0, -1.0)
    return s * u_r, u_r, u_e


def density_grid(q_r, rho, width=8.0, n=2001):
    """Convenience helper for plotting or numerically checking p(h | q_r, rho)."""

    h = np.linspace(-width * np.sqrt(q_r), width * np.sqrt(q_r), int(n))
    return h, field_density(h, q_r, rho)
