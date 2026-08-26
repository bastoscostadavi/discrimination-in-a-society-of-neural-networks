"""Order parameters, completed.

The main line of work measures five: three pair correlations and two balance
aggregates.  Writing ``rho_IJ = cos(w_I, w_J)`` for ideological alignment,
``eta_{e|r} = 1 - 2 Phi(h_mu)`` for the trust receiver ``r`` places in emitter
``e`` (+1 fully trusted, -1 fully distrusted), and ``G_IJ = kappa_I kappa_J`` for
the class indicator,

    R_wmu = <(eta_{I|J} + eta_{J|I}) rho_IJ / 2>      opinion-trust
    R_muc = <G_IJ (eta_{I|J} + eta_{J|I}) / 2>        trust-class
    R_cw  = <G_IJ rho_IJ>                             opinion-class

plus ``B_rho`` and ``B_eta``, the fraction of triples that are balanced in each
sector.

Why five is not enough
----------------------
``eta`` is a *directed* matrix: ``eta[r, e]`` is how far ``r`` trusts ``e``, and
it need not equal ``eta[e, r]``.  All three correlations above use it only
through the symmetric combination ``eta_{I|J} + eta_{J|I}``, which throws away
the antisymmetric part.  A field that writes the class label into precisely that
part is therefore invisible to all three.

Two of the four field components do exactly that.  The clean way to see it is
that the class structure of a directed matrix has four channels, orthogonal in
the same way and for the same reason as the four components of the field
(:mod:`dirfield.fields`):

    T_mu   = <eta>                        overall trust               (pairs with a)
    R_cred = <kappa_r eta[r, e]>          credulity of the listener   (pairs with b)
    R_stat = <kappa_e eta[r, e]>          status of the speaker       (pairs with c)
    R_muc  = <kappa_r kappa_e eta[r, e]>  matching                    (pairs with p)

averaged over ordered pairs ``r != e``.  The fourth is the paper's ``R_muc``
exactly: ``kappa_r kappa_e`` is symmetric under swapping the pair, so averaging
the directed product over ordered pairs is the same as averaging the symmetrized
trust over unordered ones (checked in ``tests/test_order_params.py``).  The other
two have no counterpart in the paper, and they are where a status asymmetry
lives.

The four weights are orthogonal over all ``N^2`` pairs but *not* quite over the
``N(N-1)`` that exclude the diagonal, and an agent's trust in itself is a
convention of :func:`trust` rather than a measurement, so excluding it is right
and the small non-orthogonality has to be lived with.  It is exactly known.  With
equal class sizes the only non-zero off-diagonal Gram entries are

    <1, kappa_r kappa_e> = <kappa_r, kappa_e> = -N,

against ``N(N-1)`` on the diagonal, so the uniform channel leaks into the
matching one and the credulity channel into the status one, each at ``-1/(N-1)``:
about ``-2.6%`` at ``N = 40``.  Two consequences, one for each direction.

A population with uniformly high trust and no class structure at all has
``R_muc = -T_mu/(N-1)``, not zero -- and that is a property of the paper's
parameter, not of this rewriting of it, since its weight has the same overlap
over unordered pairs.  It is small, it is a finite-size effect that vanishes as
``1/N``, and it is worth knowing about before reading a weak negative ``R_muc``
as reverse discrimination.

In the other direction, a pure status field leaves ``R_stat`` exact but puts a
spurious ``-R_stat/(N-1)`` into ``R_cred``.  Passing ``orthogonalize=True`` to
:func:`trust_channels` removes both by inverting the Gram matrix, which is a pair
of 2x2 systems in closed form.  The default is the raw moment, because that is
what makes ``R_muc`` the paper's number and comparable with it.

For equal class sizes the omission is not an approximation but an exact
cancellation.  Under a pure status field the trust matrix is ``eta[r, e] =
s(kappa_e)``, one value for each speaker class, so the symmetrized trust of a
pair is ``(s_I + s_J)/2``: ``+s`` on AA pairs, ``-s`` on BB pairs, ``0`` on AB
pairs, against a class indicator of ``+1, +1, -1``.  The AA and BB terms cancel
identically whenever there are as many of one as of the other, and ``R_muc``
vanishes however strong the hierarchy is.

``B_eta`` vanishes too, and for a prettier reason.  In the cycle
``eta_IJ eta_JK eta_KI`` each of the three agents appears exactly once as the
*emitter* -- the second index -- so under a pure status field the sign of the
product is ``(-1)^(number of class-B agents in the triple)``.  Balance is
therefore decided by the *parity* of how many stigmatized members a triple has,
and with equal class sizes there are exactly as many odd-parity triples as
even-parity ones, so the aggregate is exactly zero.
:func:`balance_by_composition` resolves the four cases the aggregate averages
away.

What replaces the aggregate
---------------------------
``B_eta`` computed inside each class separately (:func:`balance_within_classes`)
is cheap, needs no enumeration, and reports the asymmetry directly: under a pure
status field the credited class is a coherent trusting bloc (``+1``) while the
stigmatized class is atomized, every member distrusting every other member, so
every triple inside it is frustrated (``-1``).

The triple sums are evaluated in closed form.  For a matrix ``M`` with unit
diagonal, the sum over distinct ordered ``(I, J, K)`` of ``M_IJ M_JK M_KI`` is
``tr(M^3) - 3 (sum_IJ M_IJ M_JI - N) - N``, and each unordered triple appears
six times.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import ndtr

__all__ = [
    "overlaps",
    "trust",
    "correlations",
    "balance",
    "trust_channels",
    "balance_within_classes",
    "balance_by_composition",
    "class_block_trust",
    "status_per_agent",
    "credulity_per_agent",
    "measure",
    "ORDER_PARAM_NAMES",
    "PAPER_NAMES",
    "CHANNEL_NAMES",
    "CHANNEL_OF",
]

#: The five the main line of work reports.
PAPER_NAMES = ("R_wmu", "R_muc", "R_cw", "B_rho", "B_eta")

#: The four class-symmetry channels of the directed trust matrix.  ``R_muc``
#: appears in both lists and is the same number either way.
CHANNEL_NAMES = ("T_mu", "R_cred", "R_stat", "R_muc")

#: Which trust channel each field component writes into.  The pairing is the
#: content of this package: a field component and its channel are the same
#: function of the two class labels, one read on the field and one on the trust
#: matrix, so a sweep in one component should move one channel and leave the
#: other three flat.  Figures use it to decide which channel is the responding
#: one, so that a sweep in ``b`` plots ``R_cred`` where a sweep in ``c`` plots
#: ``R_stat`` rather than plotting an empty panel.
CHANNEL_OF = {"a": "T_mu", "b": "R_cred", "c": "R_stat", "p": "R_muc"}

#: Everything a sweep records.
ORDER_PARAM_NAMES = PAPER_NAMES + ("T_mu", "R_cred", "R_stat",
                                   "B_eta_A", "B_eta_B", "B_rho_A", "B_rho_B")


def overlaps(society):
    """Ideological alignment ``rho[run, I, J] = cos(w_I, w_J)``. Shape (R, N, N)."""
    w = society.w  # (N, R, K)
    wn = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-300)
    return np.einsum("irk,jrk->rij", wn, wn)


def trust(society):
    """Trust ``eta[run, r, e] = 1 - 2 Phi(mu_{e|r}/gamma_V)``. Shape (R, N, N).

    Directed: the first index is the receiver, the second the emitter.  The
    diagonal is set to +1, an agent fully trusting itself, which is what the
    balance formulas assume.
    """
    h_mu = society.mu / np.sqrt(1.0 + society.V)  # (N, N, R)
    eta = 1.0 - 2.0 * ndtr(h_mu)
    eta = np.ascontiguousarray(np.moveaxis(eta, 2, 0))  # (R, N, N)
    idx = np.arange(society.N)
    eta[:, idx, idx] = 1.0
    return eta


def _class_matrix(society):
    return np.outer(society.kappa, society.kappa)


def correlations(society, rho=None, eta=None):
    """The three pair correlations of the main line of work. Dict of ``(R,)``."""
    N = society.N
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    G = _class_matrix(society)

    S = 0.5 * (eta + np.swapaxes(eta, 1, 2))  # symmetrized trust
    iu = np.triu_indices(N, 1)
    n_pairs = N * (N - 1) / 2

    rho_u = rho[:, iu[0], iu[1]]
    S_u = S[:, iu[0], iu[1]]
    G_u = G[iu]

    return {
        "R_wmu": (S_u * rho_u).sum(axis=1) / n_pairs,
        "R_muc": (S_u * G_u).sum(axis=1) / n_pairs,
        "R_cw": (rho_u * G_u).sum(axis=1) / n_pairs,
    }


def trust_channels(society, eta=None, orthogonalize=False):
    """The four class-symmetry channels of the directed trust matrix.

    Averaged over ordered pairs with the diagonal excluded, since an agent's
    trust in itself is a convention of :func:`trust` rather than a measurement.
    Returns a dict of arrays of shape ``(R,)``.

    ``orthogonalize=True`` undoes the ``-1/(N-1)`` leakage between the two pairs
    of channels that the excluded diagonal introduces (see the module docstring),
    giving the exact coefficients of the class-block structure.  The default is
    off so that ``R_muc`` is the paper's number as published.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    kap = society.kappa
    off = ~np.eye(N, dtype=bool)
    n = off.sum()

    kr = np.repeat(kap[:, None], N, axis=1)   # receiver's class
    ke = np.repeat(kap[None, :], N, axis=0)   # emitter's class
    weights = {"T_mu": np.ones((N, N)), "R_cred": kr, "R_stat": ke, "R_muc": kr * ke}
    out = {name: (eta * np.where(off, w, 0.0)).sum(axis=(1, 2)) / n
           for name, w in weights.items()}
    if not orthogonalize:
        return out

    # The Gram matrix couples (T_mu, R_muc) and (R_cred, R_stat) and nothing
    # else, each pair with normalized overlap g, so the raw moments are
    # [[1, g], [g, 1]] times the coefficients wanted.  Inverting that 2x2 is
    # division by 1 - g^2 with the off-diagonal negated, so this needs no linear
    # algebra -- and the negation is the whole content: getting its sign wrong
    # scales both channels up instead of unmixing them.
    g = _leakage(society)
    if abs(g) >= 1.0:  # N = 2: the two channels are degenerate, nothing to undo
        return out
    scale = 1.0 / (1.0 - g * g)
    for u, v in (("T_mu", "R_muc"), ("R_cred", "R_stat")):
        mu_, mv_ = out[u], out[v]
        out[u] = scale * (mu_ - g * mv_)
        out[v] = scale * (mv_ - g * mu_)
    return out


def _leakage(society):
    """Normalized overlap between the two coupled channel pairs.

    ``sum_{r != e} kappa_r kappa_e / (N (N-1))``, which is ``-1/(N-1)`` for equal
    class sizes and picks up the class imbalance otherwise, so a society with
    unequal classes is handled by the same formula rather than by an assumption.
    """
    kap = society.kappa
    N = society.N
    return float((kap.sum() ** 2 - np.square(kap).sum()) / (N * (N - 1)))


def _mean_triple_product(M, N):
    """``<M_IJ M_JK M_KI>`` over unordered triples, for M with unit diagonal.

    ``M`` has shape (R, N, N); returns shape (R,).  Needs ``N >= 3``.
    """
    M3_trace = np.einsum("rij,rji->r", M, np.einsum("rij,rjk->rik", M, M))
    pair_sym = np.einsum("rij,rji->r", M, M)  # includes the N diagonal terms
    distinct = M3_trace - 3.0 * (pair_sym - N) - N
    n_triples = N * (N - 1) * (N - 2) / 6.0
    return distinct / (6.0 * n_triples)


def balance(society, rho=None, eta=None):
    """Aggregate ideological and trust balance. Dict of arrays of shape (R,)."""
    N = society.N
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    return {"B_rho": _mean_triple_product(rho, N),
            "B_eta": _mean_triple_product(eta, N)}


def balance_within_classes(society, rho=None, eta=None):
    """Balance computed inside each class separately.

    The aggregate mixes triples that straddle the class boundary with those that
    do not, and under a directional field the two halves of the population do
    quite different things: one can be a coherent bloc while the other is
    atomized.  Restricting the closed form to each diagonal block says so
    directly, at no extra cost.
    """
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    out = {}
    for tag, cls in (("A", 0), ("B", 1)):
        idx = np.flatnonzero(society.class_of == cls)
        n = idx.size
        if n < 3:
            nan = np.full(society.R, np.nan)
            out[f"B_rho_{tag}"], out[f"B_eta_{tag}"] = nan, nan
            continue
        sub = np.ix_(np.arange(society.R), idx, idx)
        out[f"B_rho_{tag}"] = _mean_triple_product(rho[sub], n)
        out[f"B_eta_{tag}"] = _mean_triple_product(eta[sub], n)
    return out


def balance_by_composition(society, eta=None, sector="eta"):
    """Trust balance resolved by how many class-B agents a triple contains.

    Returns ``(values, counts)``: ``values[run, k]`` is the mean balance over
    triples with ``k`` members of class B, for ``k = 0..3``, and ``counts[k]``
    how many such triples there are.  This is the breakdown the aggregate
    averages away, and under a pure status field it is the parity signature
    ``(-1)^k``.

    Enumerated rather than closed-form: it is meant for a handful of conditions,
    not for a sweep, and ``O(N^3)`` in Python is fine at that scale.  For a swept
    quantity that captures the same asymmetry use
    :func:`balance_within_classes`.
    """
    M = (trust(society) if eta is None else eta) if sector == "eta" \
        else overlaps(society)
    N, R = society.N, society.R
    triples = list(itertools.combinations(range(N), 3))
    if not triples:
        raise ValueError("need at least three agents")
    idx = np.array(triples)
    k = society.class_of[idx].sum(axis=1)  # class B is 1
    I, J, K = idx[:, 0], idx[:, 1], idx[:, 2]
    fwd = M[:, I, J] * M[:, J, K] * M[:, K, I]
    bwd = M[:, J, I] * M[:, K, J] * M[:, I, K]
    b = 0.5 * (fwd + bwd)  # (R, n_triples)
    values = np.stack([b[:, k == kk].mean(axis=1) if np.any(k == kk)
                       else np.full(R, np.nan) for kk in range(4)], axis=1)
    counts = np.array([int(np.sum(k == kk)) for kk in range(4)])
    return values, counts


def class_block_trust(society, eta=None):
    """Mean trust in each of the four class blocks. Shape ``(R, 2, 2)``.

    ``out[run, i, j]`` is the mean trust a receiver of class ``i`` places in an
    emitter of class ``j``, the diagonal of the trust matrix excluded.  This is
    the raw picture the channels are a rotation of, and the one to read when
    asking whether a class distrusts *itself*.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    out = np.empty((society.R, 2, 2))
    for i in range(2):
        for j in range(2):
            m = ((society.class_of[:, None] == i)
                 & (society.class_of[None, :] == j) & off)
            out[:, i, j] = eta[:, m].mean(axis=1) if m.any() else np.nan
    return out


def status_per_agent(society, eta=None):
    """``mean over r of eta[r, e]``: the trust each agent *receives*. ``(R, N)``.

    The per-agent quantity behind ``R_stat``.  Its histogram over a population is
    what a status hierarchy looks like directly: bimodal by class, with the two
    modes at opposite signs.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    return (eta * off).sum(axis=1) / (N - 1)


def credulity_per_agent(society, eta=None):
    """``mean over e of eta[r, e]``: the trust each agent *gives*. ``(R, N)``.

    The per-agent quantity behind ``R_cred``.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    return (eta * off).sum(axis=2) / (N - 1)


def measure(society):
    """Every swept order parameter, sharing the overlap and trust matrices."""
    rho = overlaps(society)
    eta = trust(society)
    out = correlations(society, rho=rho, eta=eta)
    out.update(balance(society, rho=rho, eta=eta))
    channels = trust_channels(society, eta=eta)
    # R_muc is in both; the channel form is the same number, and the test suite
    # holds it to that, so keep the correlations' copy and add the other three.
    out.update({k: v for k, v in channels.items() if k != "R_muc"})
    out.update(balance_within_classes(society, rho=rho, eta=eta))
    return out
