"""Order parameters for a field that names no class.

The main line of work measures five.  Writing ``rho_IJ = cos(w_I, w_J)`` for
ideological alignment, ``eta[r, e] = 1 - 2 Phi(h_mu)`` for the trust the receiver
``r`` places in the emitter ``e`` (+1 fully trusted, -1 fully distrusted), and
``G_IJ = kappa_I kappa_J`` for the class indicator,

    R_wmu = <(eta_{I|J} + eta_{J|I}) rho_IJ / 2>      opinion-trust
    R_muc = <G_IJ (eta_{I|J} + eta_{J|I}) / 2>        trust-class
    R_cw  = <G_IJ rho_IJ>                             opinion-class

plus ``B_rho`` and ``B_eta``, the fraction of triples balanced in each sector.
All five are measured here unchanged, and ``../directional-prejudice/`` adds the
four class-symmetry channels of the *directed* trust matrix,

    T_mu   = <eta>                        R_cred = <kappa_r eta[r, e]>
    R_stat = <kappa_e eta[r, e]>          R_muc  = <kappa_r kappa_e eta[r, e]>

which are also all measured here.  Under the uniform field every one of them
except ``T_mu`` is a **control** rather than a result: ``a`` refers to no label,
so the class variable is in the measurement and nowhere in the model, and a
non-zero reading would be a bug or a fluctuation.  One of them is not quite zero
for a reason that is known exactly, and is worth stating before it is read as
signal.

The one predictable non-zero
----------------------------
The four channel weights are orthogonal over all ``N^2`` pairs but not over the
``N(N-1)`` that exclude the diagonal, and excluding it is right, because an
agent's trust in itself is a convention of :func:`trust` rather than a
measurement.  With equal class sizes the only non-zero off-diagonal Gram entries
are ``<1, kappa_r kappa_e> = <kappa_r, kappa_e> = -N``, so the uniform channel
leaks into the matching one at ``-1/(N-1)``.  **A population with uniformly high
trust and no class structure at all therefore has ``R_muc = -T_mu/(N-1)``, not
zero** -- about ``-2.6%`` of ``T_mu`` at ``N = 40``.  That is a property of the
published parameter, not of this package, and the uniform field is the cleanest
place in the whole four-component basis to see it, because here the leak is the
*only* thing ``R_muc`` can be reading.

What the plane actually needs
-----------------------------
The interesting partition of a population under ``a`` is not into classes.  It is
into the fraction ``f_a`` that carries the field and the rest that does not, and
that partition is not symmetric in the way a class is: the field acts on the
*receiver*, so a biased agent trusts differently by construction, while whether
it is trusted differently back is something the dynamics has to decide.  Two
margins of the directed trust matrix separate those:

    T_give_b, T_give_u   mean trust a biased / unbiased agent extends
    T_get_b,  T_get_u    mean trust a biased / unbiased agent receives

``T_give_b - T_give_u`` is the direct effect, and its size at a given ``a`` is
close to a restatement of the field.  ``T_get_b - T_get_u`` is not: nothing in
``D[r, e] = a`` says a credulous agent should be believed any more or less than
anyone else, so whatever appears there is emergent, and it is the analogue of the
contagion question the status field answers in the negative.  The same
decomposition on the opinion side (``rho_bb``, ``rho_uu``, ``rho_bu``) says
whether the two groups end up holding the same opinions or drift apart, and
:func:`balance_within_groups` whether each is internally a bloc or a glass.

Group sizes are binomial in ``f_a`` rather than fixed, so every group quantity is
``nan`` where its group is empty -- at ``f_a = 0`` and ``f_a = 1`` by
construction, and for the triple sums wherever a group holds fewer than three
agents.  That is reported as ``nan`` rather than as zero: an empty group has no
mean trust, and drawing a zero there would put a boundary on the phase diagram
that is an artefact of the estimator.

The triple sums are evaluated in closed form.  For a matrix ``M`` with unit
diagonal, the sum over distinct ordered ``(I, J, K)`` of ``M_IJ M_JK M_KI`` is
``tr(M^3) - 3 (sum_IJ M_IJ M_JI - N) - N``, and each unordered triple appears six
times.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "overlaps",
    "trust",
    "correlations",
    "balance",
    "trust_channels",
    "bias_trust_margins",
    "bias_block_trust",
    "bias_blocks",
    "opinion_blocks",
    "balance_within_groups",
    "trust_given_per_agent",
    "trust_received_per_agent",
    "measure",
    "ORDER_PARAM_NAMES",
    "PAPER_NAMES",
    "CLASS_CHANNELS",
    "BIAS_MARGINS",
    "BIAS_BLOCKS",
    "OPINION_BLOCKS",
    "REALIZED",
    "GROUP_BALANCES",
]

#: The five the main line of work reports.
#:
#: Exactly two of them -- ``R_muc`` and ``R_cw`` -- reference the class label and
#: are therefore controls on this plane.  The other three reference no label and
#: are not controls: they move across the whole plane and are the result.  (Two
#: further controls, ``R_cred`` and ``R_stat``, come from :data:`CLASS_CHANNELS`
#: rather than from here, so the total is four drawn from two different sets --
#: which is not the same statement as "four of the five".)
PAPER_NAMES = ("R_wmu", "R_muc", "R_cw", "B_rho", "B_eta")

#: The four class-symmetry channels of the directed trust matrix.  ``T_mu`` is
#: the one the uniform field drives; the other three are controls, and ``R_muc``
#: appears in both this list and the previous one with the same value.
CLASS_CHANNELS = ("T_mu", "R_cred", "R_stat", "R_muc")

#: The directed trust matrix marginalized over the bias partition instead.
BIAS_MARGINS = ("T_give_b", "T_give_u", "T_get_b", "T_get_u")

#: The four blocks the margins are the marginalizations of, receiver first:
#: ``T_ub`` is the mean trust an *unbiased* receiver places in a *biased*
#: emitter.  Swept as well as the margins, and they are the sharper measurement:
#: a margin over receivers averages the biased agents' own rows in, and those
#: rows are the field rather than a response to it.  ``T_ub`` against ``T_uu``
#: is the same comparison made only by the agents the field never touched.
BIAS_BLOCKS = ("T_bb", "T_bu", "T_ub", "T_uu")

#: The fraction of agents actually drawn as biased.  Binomial in ``f_a`` rather
#: than equal to it, and recorded so that a margin can be reconstructed from the
#: blocks after the fact without re-simulating.
REALIZED = ("frac_biased",)

#: Mean ideological alignment overall and within each block of that partition.
OPINION_BLOCKS = ("rho_mean", "rho_bb", "rho_uu", "rho_bu")

#: Balance computed inside each group of the bias partition separately.
GROUP_BALANCES = ("B_eta_b", "B_eta_u", "B_rho_b", "B_rho_u")

#: Everything a sweep records.
ORDER_PARAM_NAMES = (PAPER_NAMES + ("T_mu", "R_cred", "R_stat")
                     + BIAS_MARGINS + BIAS_BLOCKS + OPINION_BLOCKS
                     + GROUP_BALANCES + REALIZED)


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


def trust_channels(society, eta=None):
    """The four class-symmetry channels of the directed trust matrix.

    Averaged over ordered pairs with the diagonal excluded.  Only ``T_mu``
    responds to a uniform field; the other three are the control, and the
    ``-1/(N-1)`` leakage of ``T_mu`` into ``R_muc`` described in the module
    docstring is left in, because the point of measuring it here is to check that
    it is exactly what shows up.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    kap = society.kappa
    off = ~np.eye(N, dtype=bool)
    n = off.sum()

    kr = np.repeat(kap[:, None], N, axis=1)   # receiver's class
    ke = np.repeat(kap[None, :], N, axis=0)   # emitter's class
    weights = {"T_mu": np.ones((N, N)), "R_cred": kr, "R_stat": ke, "R_muc": kr * ke}
    return {name: (eta * np.where(off, w, 0.0)).sum(axis=(1, 2)) / n
            for name, w in weights.items()}


# --- the bias partition ---------------------------------------------------

def _indicators(society):
    """``(biased, unbiased)`` membership as float ``(R, N)`` arrays."""
    b = society.biased.T.astype(np.float64)  # (R, N)
    return b, 1.0 - b


def _block_mean(M, Gi, Gj, off):
    """``<M[r, e]>`` over receivers in group ``i``, emitters in ``j``, ``r != e``.

    ``M`` is ``(R, N, N)`` and the indicators ``(R, N)``.  The denominator is the
    number of ordered pairs the numerator actually sums, ``n_i n_j`` less the
    ``|i and j|`` diagonal terms the mask drops, so the same expression is right
    whether the two groups are equal, disjoint, or neither.  Empty numerators
    give ``nan`` rather than zero.
    """
    num = np.einsum("rn,rnm,rm->r", Gi, M * off, Gj)
    den = Gi.sum(axis=1) * Gj.sum(axis=1) - np.einsum("rn,rn->r", Gi, Gj)
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


def bias_trust_margins(society, eta=None):
    """Mean trust each group extends and receives. Dict of ``(R,)``.

    ``T_give_*`` marginalizes over emitters, ``T_get_*`` over receivers.  The
    first pair is the field acting; the second is the population's response to
    it, and there is nothing in a uniform field that fixes it.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    b, u = _indicators(society)
    ones = np.ones_like(b)
    return {
        "T_give_b": _block_mean(eta, b, ones, off),
        "T_give_u": _block_mean(eta, u, ones, off),
        "T_get_b": _block_mean(eta, ones, b, off),
        "T_get_u": _block_mean(eta, ones, u, off),
    }


def bias_block_trust(society, eta=None):
    """Mean trust in each of the four blocks of the bias partition. ``(R, 2, 2)``.

    ``out[run, i, j]`` with index 0 for biased and 1 for unbiased: the mean trust
    a receiver from group ``i`` places in an emitter from group ``j``, the
    diagonal excluded.  This is the raw picture the margins are the two
    marginalizations of, and the one to read when asking whether the biased
    agents trust *each other* differently from how they trust everyone.
    """
    eta = trust(society) if eta is None else eta
    off = ~np.eye(society.N, dtype=bool)
    groups = _indicators(society)
    out = np.empty((society.R, 2, 2))
    for i in range(2):
        for j in range(2):
            out[:, i, j] = _block_mean(eta, groups[i], groups[j], off)
    return out


def bias_blocks(society, eta=None):
    """The four blocks of :func:`bias_block_trust` as named scalars.

    ``T_ub`` -- unbiased receiver, biased emitter -- is the one to read for an
    emergent effect: the receivers contributing to it carry no field at all, so
    ``T_ub - T_uu`` is a comparison made entirely by agents the field never
    touched, between speakers who differ only in whether *they* carry it.
    """
    eta = trust(society) if eta is None else eta
    blocks = bias_block_trust(society, eta=eta)
    return {"T_bb": blocks[:, 0, 0], "T_bu": blocks[:, 0, 1],
            "T_ub": blocks[:, 1, 0], "T_uu": blocks[:, 1, 1]}


def opinion_blocks(society, rho=None):
    """Mean ideological alignment overall and within the bias partition.

    ``rho_bb`` and ``rho_uu`` are within each group, ``rho_bu`` across.  ``rho``
    is symmetric, so averaging over ordered pairs and over unordered ones is the
    same number and the ordered form is used throughout for consistency with the
    trust blocks.
    """
    rho = overlaps(society) if rho is None else rho
    off = ~np.eye(society.N, dtype=bool)
    b, u = _indicators(society)
    ones = np.ones_like(b)
    return {
        "rho_mean": _block_mean(rho, ones, ones, off),
        "rho_bb": _block_mean(rho, b, b, off),
        "rho_uu": _block_mean(rho, u, u, off),
        "rho_bu": _block_mean(rho, b, u, off),
    }


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


def balance_within_groups(society, rho=None, eta=None):
    """Balance computed inside each group of the bias partition separately.

    The aggregate mixes triples that straddle the partition with those that do
    not, and the two groups need not do the same thing: one can be a coherent
    bloc while the other is a glass.  Restricting the closed form to each
    diagonal block says so directly.

    Which agents are biased is drawn per run, so unlike the class partition the
    index sets differ from run to run and the closed form cannot be applied to
    the whole batch at once.  The loop is over runs; each iteration is two
    ``n x n`` matrix products with ``n <= N``, which is negligible against the
    cost of having simulated the run in the first place.
    """
    rho = overlaps(society) if rho is None else rho
    eta = trust(society) if eta is None else eta
    out = {k: np.full(society.R, np.nan) for k in GROUP_BALANCES}
    for run in range(society.R):
        biased = society.biased[:, run]
        for tag, member in (("b", biased), ("u", ~biased)):
            idx = np.flatnonzero(member)
            n = idx.size
            if n < 3:
                continue
            sub = np.ix_([run], idx, idx)
            out[f"B_rho_{tag}"][run] = _mean_triple_product(rho[sub], n)[0]
            out[f"B_eta_{tag}"][run] = _mean_triple_product(eta[sub], n)[0]
    return out


def trust_given_per_agent(society, eta=None):
    """``mean over e of eta[r, e]``: the trust each agent extends. ``(R, N)``.

    The per-agent quantity behind ``T_give_*``.  Its histogram over a population
    with ``0 < f_a < 1`` is what the field does directly: bimodal, one mode per
    group.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    return (eta * off).sum(axis=2) / (N - 1)


def trust_received_per_agent(society, eta=None):
    """``mean over r of eta[r, e]``: the trust each agent receives. ``(R, N)``.

    The per-agent quantity behind ``T_get_*``, and the one to look at for an
    emergent hierarchy: the field says nothing about it, so whether this
    histogram separates by group is a result.
    """
    eta = trust(society) if eta is None else eta
    N = society.N
    off = ~np.eye(N, dtype=bool)
    return (eta * off).sum(axis=1) / (N - 1)


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
    out.update(bias_trust_margins(society, eta=eta))
    out.update(bias_blocks(society, eta=eta))
    out.update(opinion_blocks(society, rho=rho))
    out.update(balance_within_groups(society, rho=rho, eta=eta))
    out["frac_biased"] = society.biased.mean(axis=0).astype(float)
    return out
