"""Reduced distributions of microscopic issue fields."""

from __future__ import annotations

import numpy as np

from .modulation import Phi, phi

RHO_TOL = 1e-12


def _check_q_rho(q_r, rho):
    q_r = float(q_r)
    rho = float(rho)
    if q_r <= 0.0:
        raise ValueError("q_r must be positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    return q_r, rho


def field_density(h, q_r, rho):
    """Density of h = sign(u_e) u_r for jointly Gaussian receiver/emitter fields.

    For |rho| < 1 this is the skew-normal density

        2 / sqrt(q_r) * phi(h / sqrt(q_r))
        * Phi(rho h / sqrt(q_r (1 - rho^2))).

    The limiting cases rho = +/-1 are treated as one-sided half-normal
    densities.
    """

    q_r, rho = _check_q_rho(q_r, rho)
    h = np.asarray(h, dtype=float)
    scale = np.sqrt(q_r)
    base = phi(h / scale) / scale
    if rho >= 1.0 - RHO_TOL:
        return np.where(h >= 0.0, 2.0 * base, 0.0)
    if rho <= -1.0 + RHO_TOL:
        return np.where(h <= 0.0, 2.0 * base, 0.0)
    arg = rho * h / np.sqrt(q_r * (1.0 - rho * rho))
    return 2.0 * base * Phi(arg)
