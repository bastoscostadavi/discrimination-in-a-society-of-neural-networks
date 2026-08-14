"""The local analysis: what one interacting pair does, solved exactly.

The population results are simulations, but the *mechanism* is not: the local
flow can be analysed in closed form, and it yields a prediction for the
trust--class correlation with no fitted parameters.  This module implements that
analysis so every claim in the paper's mechanism section is executable.

The reduced flow
----------------
Held at a fixed issue and emitter opinion, a receiver's two fields move as

    dh_w/dt  = a [ F_w(H, h_mu)  - (a/2) h_w  F_C(H, h_mu) ]
    dh_mu/dt = b [ F_mu(H, h_mu) - (b/2) h_mu F_V(H, h_mu) ]

with ``H = h_w + D``, ``a = x.C_r x / gamma_C^2`` and ``b = V / gamma_V^2``, both
in ``(0, 1)``.  At the model's own initialisation ``C = I``, ``V = 1`` and issues
are unit vectors, so ``a = b = 1/2`` exactly: the isotropic case is the model's
starting point, not an assumption imposed on it.  The correction terms are odd
under the same reflection as the leading terms and are cubic near the saddle, so
they change neither the symmetry nor the linearisation --- which is why the
paper states the proposition for ``(F_w, F_mu)`` and relegates the algebra.

What is proved
--------------
1. **Rigid translation.**  ``X_D(h_w, h_mu) = X_0(h_w + D, h_mu)`` --- exactly, with
   no assumption.  The portrait at field ``D`` is the unbiased portrait shifted
   by ``-D`` along the agreement axis.
2. **Two symmetries at ``D = 0``**, both from ``Phi(-z) = 1 - Phi(z)``: the flow is
   equivariant under the point reflection ``u -> -u`` and under exchanging the two
   sectors.  The first is the symmetry the discrimination field breaks.
3. **The separatrix is exactly the line ``h_mu = h_w + D``**, globally invariant
   rather than merely tangent, with a hyperbolic saddle at ``(-D, 0)`` whose
   eigenvalues are ``-+2/pi``.
4. **A parameter-free prediction.**  A channel is captured by the trust attractor
   exactly when its initial dissonance gap ``psi = h_mu - h_w`` falls below ``D``,
   so the expected asymptotic trust is ``G(D) = 2 Psi(D) - 1``, and under the
   symmetric field with balanced classes

       C_CT = f_d * G(d)

   because non-discriminating receivers contribute nothing on average.  ``Psi``
   is available in closed form at the model's initialisation, so ``G`` has no free
   parameters at all.

What is *not* claimed
---------------------
There is no critical field strength for ``C_CT``.  ``F_mu`` vanishes identically on
the line ``h_w = 0``, so the linear restoring force on the class-trust contrast is
zero at the class-symmetric state; the sign change at ``d = 0`` is a forced
zero-crossing of an odd response, not a bifurcation.  See
:func:`restoring_force_vanishes` for the executable statement.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .modulation import F_C, F_mu, F_V, F_w, evidence, g

__all__ = [
    "gap_cdf",
    "gap_pdf",
    "G",
    "G_prime_at_zero",
    "predict_C_CT",
    "flow",
    "separatrix_residual",
    "saddle_jacobian",
    "restoring_force_vanishes",
    "GAP_C",
    "GAP_SIGMA",
]

#: Half-width of the uniform part of the initial gap, and the Gaussian scale.
#:
#: At initialisation ``C = I`` and issues are unit vectors, so
#: ``gamma_C = sqrt(2)`` and the agreement field ``h_w = sigma_e (w.x) / gamma_C``
#: is ``N(0, 1/2)``.  Likewise ``V = 1`` gives ``gamma_V = sqrt(2)`` and
#: ``h_mu = mu / gamma_V`` is uniform on ``+-1/sqrt(2)`` for ``mu ~ U(-1, 1)``.
#: The gap ``psi = h_mu - h_w`` is therefore the convolution of the two.
GAP_C = 1.0 / np.sqrt(2.0)
GAP_SIGMA = 1.0 / np.sqrt(2.0)


def _F(z):
    """``F(z) = z Phi(z) + phi(z)``, an antiderivative of ``Phi``."""
    return z * ndtr(z) + g(z)


def gap_cdf(t, c=GAP_C, sigma=GAP_SIGMA):
    """CDF of the initial dissonance gap ``psi = h_mu - h_w``.

    The convolution of ``U(-c, c)`` with ``N(0, sigma^2)`` integrates in closed
    form because ``Phi`` has the elementary antiderivative :func:`_F`.
    """
    t = np.asarray(t, dtype=float)
    return (sigma / (2.0 * c)) * (_F((t + c) / sigma) - _F((t - c) / sigma))


def gap_pdf(t, c=GAP_C, sigma=GAP_SIGMA):
    """Density of the initial dissonance gap."""
    t = np.asarray(t, dtype=float)
    return (ndtr((t + c) / sigma) - ndtr((t - c) / sigma)) / (2.0 * c)


def G(D, c=GAP_C, sigma=GAP_SIGMA):
    """Expected asymptotic trust of a channel carrying field ``D``.

    A channel ends in the trust attractor exactly when ``psi < D``, so the
    expected asymptotic trust is ``P(psi < D) - P(psi > D) = 2 Psi(D) - 1``.
    Odd, strictly increasing, and tending to ``+-1``.
    """
    return 2.0 * gap_cdf(D, c, sigma) - 1.0


def G_prime_at_zero(c=GAP_C, sigma=GAP_SIGMA):
    """Initial slope ``G'(0) = 2 p(0)``. About 0.9655 at the model's initialisation."""
    return 2.0 * float(gap_pdf(0.0, c, sigma))


def predict_C_CT(d, f_d, c=GAP_C, sigma=GAP_SIGMA):
    """The parameter-free prediction ``C_CT = f_d G(d)``.

    Under the symmetric field with balanced classes, a discriminating receiver
    sees ``+d`` towards its own class and ``-d`` towards the other, so its
    contribution to ``<kappa_r kappa_e t>`` is ``G(d)``.  A non-discriminating
    receiver sees no field, and ``G(0) = 0``, so it contributes nothing on
    average.  Hence the product form --- which also predicts that the measured
    map, divided by ``f_d``, collapses onto the single curve ``G``.

    The prediction is a *local* one: it treats each channel as deciding
    independently.  Measured values exceed it, and that excess is the collective
    amplification the population contributes.
    """
    d = np.asarray(d, dtype=float)
    f_d = np.asarray(f_d, dtype=float)
    return f_d * G(d, c, sigma)


# -- the reduced flow --------------------------------------------------
def flow(h_w, h_mu, D=0.0, a=None, b=None):
    """The reduced flow ``(dh_w/dt, dh_mu/dt)``.

    With ``a`` and ``b`` given, the uncertainty-correction terms are included;
    with both ``None`` the leading form ``(F_w, F_mu)`` is returned, which is what
    the paper plots and states the proposition for.
    """
    H = h_w + D
    fw, fmu = F_w(H, h_mu), F_mu(H, h_mu)
    if a is None and b is None:
        return fw, fmu
    a = 0.5 if a is None else a
    b = 0.5 if b is None else b
    return (
        a * (fw - 0.5 * a * h_w * F_C(H, h_mu)),
        b * (fmu - 0.5 * b * h_mu * F_V(H, h_mu)),
    )


def separatrix_residual(h_w, D=0.0):
    """How far the flow departs from tangency to ``h_mu = h_w + D``.

    On that line the two components are equal, so the flow is parallel to it and
    the line is invariant.  This returns ``F_mu - F_w`` there, which must be zero
    to machine precision --- the check that the separatrix is exact rather than
    merely a good approximation.
    """
    h_w = np.asarray(h_w, dtype=float)
    h_mu = h_w + D
    fw, fmu = flow(h_w, h_mu, D)
    return fmu - fw


def saddle_jacobian(D=0.0, eps=1e-6):
    """Jacobian of the leading flow at the fixed point ``(-D, 0)``.

    Analytically ``[[0, -2/pi], [-2/pi, 0]]``, with eigenvalues ``-+2/pi`` and
    eigenvectors along the diagonal (stable) and antidiagonal (unstable).
    """
    p = np.array([-D, 0.0])

    def f(u):
        return np.array(flow(u[0], u[1], D))

    J = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        J[:, j] = (f(p + e) - f(p - e)) / (2 * eps)
    return J


def restoring_force_vanishes(h_mu_values=(-2.0, -0.5, 0.5, 2.0)):
    """``dF_mu/dh_mu = 0`` everywhere on ``h_w = 0``: no linear restoring force.

    This is why no critical field strength exists for the trust--class
    correlation, and why an earlier attempt at a mean-field boundary measured a
    largest eigenvalue that was identically zero across its whole grid.  It is a
    structural property of the rule, not a shortcoming of that closure.

    ``F_mu = (1 - 2 Phi(h_w)) g(h_mu) / Z`` carries the factor ``(1 - 2 Phi(h_w))``,
    which vanishes at ``h_w = 0`` --- so ``F_mu`` is identically zero along that
    whole line, and hence so is its derivative along it.
    """
    out = {}
    for h_mu in h_mu_values:
        eps = 1e-6
        d = (F_mu(0.0, h_mu + eps) - F_mu(0.0, h_mu - eps)) / (2 * eps)
        out[h_mu] = float(d)
    return out
