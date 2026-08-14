"""Manuscript evidence and modulation functions.

The evidence is evaluated as

    Z(H, mu) = Phi(H) * SF(mu) + Phi(mu) * SF(H)

where SF(x) = 1 - Phi(x).  This is algebraically identical to
Phi(H) + Phi(mu) - 2 Phi(H) Phi(mu), but avoids the most obvious cancellation
when both CDF values are close to one.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_erfc = np.vectorize(math.erfc, otypes=[float])


def _asarray(x):
    return np.asarray(x, dtype=float)


def Phi(x):
    """Standard normal CDF."""

    x = _asarray(x)
    return 0.5 * _erfc(-x / _SQRT_2)


def survival(x):
    """Standard normal survival function."""

    x = _asarray(x)
    return 0.5 * _erfc(x / _SQRT_2)


def phi(x):
    """Standard normal PDF."""

    x = _asarray(x)
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def Z(H, mu):
    """Evidence denominator from the manuscript."""

    H = _asarray(H)
    mu = _asarray(mu)
    return Phi(H) * survival(mu) + Phi(mu) * survival(H)


def modulation(H, mu):
    """Return ``(F_w, F_C, F_mu, F_V)`` at the manuscript fields."""

    evidence = Z(H, mu)
    fw = (1.0 - 2.0 * Phi(mu)) * phi(H) / evidence
    fmu = (1.0 - 2.0 * Phi(H)) * phi(mu) / evidence
    return fw, -fw * (fw + H), fmu, -fmu * (fmu + mu)


def F_w(H, mu):
    return modulation(H, mu)[0]


def F_C(H, mu):
    return modulation(H, mu)[1]


def F_mu(H, mu):
    return modulation(H, mu)[2]


def F_V(H, mu):
    return modulation(H, mu)[3]
