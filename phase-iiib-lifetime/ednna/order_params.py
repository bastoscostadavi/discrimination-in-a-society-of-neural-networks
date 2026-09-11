"""Order parameters: correlations and social balance.

Three pair correlations characterize the macroscopic state of the society.
Writing ``rho_IJ = cos(w_I, w_J)`` for the ideological alignment of two agents,
``eta_{e|r} = 1 - 2 Phi(h_mu)`` for the trust a receiver ``r`` places in an
emitter ``e`` (+1 fully trusted, -1 fully distrusted), and ``G_IJ`` for the
class indicator:

    R_wmu = <(eta_{I|J} + eta_{J|I}) rho_IJ>      opinion-trust
    R_muc = <G_IJ (eta_{I|J} + eta_{J|I})>        trust-class
    R_cw  = <G_IJ rho_IJ>                         opinion-class

Social balance is measured on triples: a triple is ideologically balanced when
``b^I = rho_IJ rho_JK rho_KI > 0`` and balanced in trust when
``b^A = (eta_IJ eta_JK eta_KI + eta_JI eta_IK eta_KJ)/2 > 0``, and

    B_rho = <rho_IJK>,    B_eta = <eta_IJK>

average over all ``C(N, 3)`` triples.

Two conventions differ from the source draft, both documented in
``docs/model.md``:

* ``G_IJ`` is written as 1/0 in the draft, but the published trust-class maps
  span -1..1, which requires the +-1 form ``G_IJ = kappa_I kappa_J``.  That is
  the default here; ``class_indicator="01"`` selects the literal version.
* The draft normalizes all three correlations by ``N(N-1)``.  That is right for
  the two that sum ``eta_{I|J} + eta_{J|I}`` over unordered pairs, but it caps
  ``R_cw``, which has a single term per pair, at 1/2.  We use ``2/(N(N-1))``
  there so all three share the range [-1, 1]; ``literal_norm=True`` restores
  the draft's factor.

The triple sums are evaluated in closed form.  Enumerating triples costs
``O(N^3)`` per society in Python; for a symmetric or non-symmetric matrix ``M``
with unit diagonal,

    sum over distinct ordered (I,J,K) of M_IJ M_JK M_KI
        = tr(M^3) - 3 (sum_IJ M_IJ M_JI - N) - N

and each unordered triple appears six times, so ``<b> = that / (6 C(N,3))``.
This reduces to batched matrix products and reproduces enumeration exactly
(checked in ``tests/test_order_params.py``).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "overlaps",
    "sign_balance",
    "trust",
    "correlations",
    "balance",
    "measure",
    "class_trust_per_agent",
    "ORDER_PARAM_NAMES",
]

ORDER_PARAM_NAMES = ("R_wmu", "R_muc", "R_cw", "B_rho", "B_eta")


def overlaps(society):
    """Ideological alignment ``rho[run, I, J] = cos(w_I, w_J)``. Shape (R, N, N)."""
    w = society.w  # (N, R, K)
    wn = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-300)
    return np.einsum("irk,jrk->rij", wn, wn)


def trust(society):
    """Trust ``eta[run, r, e] = 1 - 2 Phi(mu_{e|r}/gamma_V)``. Shape (R, N, N).

    The diagonal is set to +1: an agent fully trusts itself (``eta_{I|I} = 1``
    in the draft), which is what the balance formulas assume.
    """
    h_mu = society.mu / np.sqrt(1.0 + society.V)  # (N, N, R)
    eta = 1.0 - 2.0 * ndtr(h_mu)
    eta = np.ascontiguousarray(np.moveaxis(eta, 2, 0))  # (R, N, N)
    idx = np.arange(society.N)
    eta[:, idx, idx] = 1.0
    return eta


def _class_matrix(society, class_indicator="pm1"):
    if class_indicator == "pm1":
        G = np.outer(society.kappa, society.kappa)
    elif class_indicator == "01":
        G = (society.class_of[:, None] == society.class_of[None, :]).astype(float)
    else:
        raise ValueError("class_indicator must be 'pm1' or '01'")
    return G


def correlations(society, rho=None, eta=None, class_indicator="pm1", literal_norm=False):
    """The three pair correlations. Returns a dict of arrays of shape (R,)."""
    N = society.N
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    G = _class_matrix(society, class_indicator)

    S = eta + np.swapaxes(eta, 1, 2)  # (R, N, N): eta_{I|J} + eta_{J|I}
    iu = np.triu_indices(N, 1)
    denom = N * (N - 1)

    rho_u = rho[:, iu[0], iu[1]]
    S_u = S[:, iu[0], iu[1]]
    G_u = G[iu]

    R_wmu = (S_u * rho_u).sum(axis=1) / denom
    R_muc = (S_u * G_u).sum(axis=1) / denom
    cw_factor = 1.0 if literal_norm else 2.0
    R_cw = cw_factor * (rho_u * G_u).sum(axis=1) / denom
    return {"R_wmu": R_wmu, "R_muc": R_muc, "R_cw": R_cw}


def _mean_triple_product(M, N):
    """``<M_IJ M_JK M_KI>`` over unordered triples, for M with unit diagonal.

    ``M`` has shape (R, N, N); returns shape (R,).
    """
    M3_trace = np.einsum("rij,rji->r", M, np.einsum("rij,rjk->rik", M, M))
    pair_sym = np.einsum("rij,rji->r", M, M)  # includes the N diagonal terms
    distinct = M3_trace - 3.0 * (pair_sym - N) - N
    n_triples = N * (N - 1) * (N - 2) / 6.0
    return distinct / (6.0 * n_triples)


def sign_balance(M):
    """Fraction of triples whose sign product is positive, for one matrix.

    ``B_rho`` and ``B_eta`` weight each triple by the magnitudes of its three overlaps,
    which conflates two questions: how *cleanly* the society has split, and how
    *strongly* each pair agrees.  This separates them by discarding magnitudes.

    It matters because of the structure theorem for signed complete graphs
    (Cartwright and Harary, 1956): a graph is balanced exactly when its vertices
    split into two mutually hostile cliques.  So a value of 1 says the society has
    exactly two factions, not merely that every pair has taken a side --- three
    equal factions give 3/4 and random signs give 1/2, both well clear of 1.

    Uses the same closed form as :func:`_mean_triple_product`, applied to
    ``sign(M)``; ``tests/test_order_params.py`` checks it against enumeration.
    """
    S = np.sign(np.asarray(M, dtype=float))
    N = S.shape[-1]
    if S.ndim == 2:
        S = S[None]
    S = S.copy()
    idx = np.arange(N)
    S[:, idx, idx] = 1.0
    mean_sign = _mean_triple_product(S, N)
    out = 0.5 * (1.0 + mean_sign)
    return out if out.size > 1 else float(out[0])


def balance(society, rho=None, eta=None):
    """Aggregate ideological and trust balance. Dict of arrays of shape (R,)."""
    N = society.N
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    return {"B_rho": _mean_triple_product(rho, N),
            "B_eta": _mean_triple_product(eta, N)}


def measure(society, class_indicator="pm1", literal_norm=False):
    """All five order parameters, sharing the overlap and trust matrices."""
    rho = overlaps(society)
    eta = trust(society)
    out = correlations(
        society, rho=rho, eta=eta, class_indicator=class_indicator, literal_norm=literal_norm
    )
    out.update(balance(society, rho=rho, eta=eta))
    return out


def class_trust_per_agent(society, eta=None):
    """``u_I = (1/N) kappa . eta_I``: how class-aligned each agent's trust is.

    An agent whose trust follows its own class has ``u_I = kappa_I``, so under
    prejudice the histogram of ``u`` over the society is bimodal at +-1,
    and under reverse discrimination the two modes swap places.  A society whose
    trust is uncorrelated with class has ``u`` concentrated at 0.  Multiply by
    ``kappa`` for a single-signed version.  Returns (R, N).
    """
    eta = trust(society) if eta is None else eta
    return (eta @ society.kappa) / society.N
