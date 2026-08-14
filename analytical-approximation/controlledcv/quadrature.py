"""Small Gauss-Hermite quadrature helpers for Gaussian expectations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def normal_hermgauss(order):
    """Nodes and weights for E[f(Z)] with Z ~ N(0, 1)."""

    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def normal_expectation_1d(fn, order=80):
    z, w = normal_hermgauss(order)
    return np.sum(w * fn(z), axis=0)


def normal_expectation_2d(fn, order=50):
    z, w = normal_hermgauss(order)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    ww = w[:, None] * w[None, :]
    return np.sum(ww * fn(z1, z2), axis=(0, 1))
