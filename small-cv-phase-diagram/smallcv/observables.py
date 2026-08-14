"""Macroscopic observables measured after microscopic interactions."""

from __future__ import annotations

import numpy as np

from .modulation import Phi

ORDER_PARAM_NAMES = ("R_wmu", "R_muc", "R_cw", "B_I", "B_A")


def overlaps(society):
    w = society.w
    wn = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-300)
    return np.einsum("irk,jrk->rij", wn, wn)


def trust(society, small_v=True):
    denom = 1.0 if small_v else np.sqrt(1.0 + society.V)
    eta = 1.0 - 2.0 * Phi(society.mu / denom)
    eta = np.ascontiguousarray(np.moveaxis(eta, 2, 0))
    idx = np.arange(society.N)
    eta[:, idx, idx] = 1.0
    return eta


def class_matrix(society, class_indicator="pm1"):
    if class_indicator == "pm1":
        return np.outer(society.kappa, society.kappa)
    if class_indicator == "01":
        return (society.class_of[:, None] == society.class_of[None, :]).astype(float)
    raise ValueError("class_indicator must be 'pm1' or '01'")


def correlations(society, rho=None, eta=None, class_indicator="pm1", literal_norm=False):
    N = society.N
    rho = overlaps(society) if rho is None else rho
    eta = trust(society, small_v=society.dynamics == "small_cv") if eta is None else eta
    G = class_matrix(society, class_indicator)
    iu = np.triu_indices(N, 1)
    S = eta + np.swapaxes(eta, 1, 2)
    denom = N * (N - 1)
    R_wmu = (S[:, iu[0], iu[1]] * rho[:, iu[0], iu[1]]).sum(axis=1) / denom
    R_muc = (S[:, iu[0], iu[1]] * G[iu]).sum(axis=1) / denom
    R_cw = (rho[:, iu[0], iu[1]] * G[iu]).sum(axis=1) / denom
    if not literal_norm:
        R_cw *= 2.0
    return {"R_wmu": R_wmu, "R_muc": R_muc, "R_cw": R_cw}


def _mean_triple_product(M, N):
    M3_trace = np.einsum("rij,rji->r", M, np.einsum("rij,rjk->rik", M, M))
    pair_sym = np.einsum("rij,rji->r", M, M)
    distinct = M3_trace - 3.0 * (pair_sym - N) - N
    return distinct / (N * (N - 1) * (N - 2))


def balance(society, rho=None, eta=None):
    rho = overlaps(society) if rho is None else rho
    eta = trust(society, small_v=society.dynamics == "small_cv") if eta is None else eta
    return {
        "B_I": _mean_triple_product(rho, society.N),
        "B_A": _mean_triple_product(eta, society.N),
    }


def measure(society, class_indicator="pm1", literal_norm=False):
    rho = overlaps(society)
    eta = trust(society, small_v=society.dynamics == "small_cv")
    out = correlations(society, rho, eta, class_indicator, literal_norm)
    out.update(balance(society, rho, eta))
    return out
