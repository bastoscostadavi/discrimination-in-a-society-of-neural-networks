"""What we measure, and what each quantity can and cannot distinguish.

Two pairwise quantities carry everything.  The **ideological overlap** of two
agents is the cosine of their weight vectors, ``q_IJ``, which is ``+1`` when they
agree on every issue and ``-1`` when they disagree on every one.  The **directed
trust** a receiver ``r`` places in an emitter ``e`` is ``t = 1 - 2 Phi(h_mu)``,
which is ``+1`` for a fully trusted source and ``-1`` for a fully distrusted one.
With the signed class relation ``s_IJ = kappa_I kappa_J``:

    C_CT = mean over ordered pairs of  s_IJ t_{J|I}     trust follows class
    C_CO = mean over unordered pairs of  s_IJ q_IJ      opinion follows class
    C_TO = mean over unordered pairs of  tbar_IJ q_IJ   trust follows agreement

These are a **renaming** of the three correlations used previously, not a
correction: given the normalisations already in use they are numerically
identical quantity for quantity, and ``tests/test_observables.py`` pins that.
The paper says so explicitly, because a reader comparing against earlier figures
would otherwise assume the numbers had moved.

Why the class correlations are not enough
-----------------------------------------
A society can be strongly polarised along an axis that has nothing to do with
class; all three correlations above would then be near zero, and reporting only
them would describe a sharply divided population as unstructured.  So we also
measure **class-independent polarisation**

    P = lambda_1^2 / sum_a lambda_a^2

on the centred overlap matrix and, identically, on the centred symmetrised trust
matrix.  A perfect two-bloc (rank-one) structure gives ``P = (N-1)/N`` --- the
last ``1/N`` is the price of zeroing the diagonal, see :func:`p_ceiling` --- and
an isotropic one gives ``P ~ 1/rank``.  Together with ``A = (v_1 . kappa/sqrt(N))^2``
this gives every point a two-number description --- *how polarised* by *how
class-aligned* --- which is what separates a genuinely unstructured region from a
polarised but class-blind one.

The squared-eigenvalue fraction is used rather than the plain variance fraction
``lambda_1 / sum lambda`` because the centred *trust* matrix is symmetric but
indefinite, where a variance fraction is meaningless.  One formula for both
sectors is also what makes the first-passage comparison in
:mod:`socsim.firstpassage` a comparison of like with like.

Balance
-------
Pair correlations cannot tell organised disagreement from disorder: two coherent
mutually hostile blocs and a population where nobody can settle both have small
mean trust.  Balance over triples separates them.  Both conventions are computed
and named distinctly: ``B_*`` are the continuous scores, ``B_*_sign`` the
sign-based ones.  Only the latter may be described as a *fraction* of balanced
triples --- ``(1 + B_sign)/2`` --- and only they define a frustration
``(1 - B_sign)/2``.  Calling a continuous product a fraction of balanced triples
is a category error that the source material makes and we do not.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "OBS_NAMES",
    "overlaps",
    "trust",
    "correlations",
    "balance",
    "spectral_polarization",
    "p_ceiling",
    "permutation_null",
    "measure",
]

#: Observables bounded to [-1, 1] by construction.  The rest are diagnostics
#: (norms, counts) and are deliberately not bounded.
BOUNDED = (
    "C_CT", "C_CO", "C_TO", "P_O", "P_T", "A_O", "A_T",
    "B_O", "B_T", "B_O_sign", "B_T_sign", "mean_trust",
)

OBS_NAMES = (
    "C_CT",
    "C_CO",
    "C_TO",
    "P_O",
    "P_T",
    "A_O",
    "A_T",
    "B_O",
    "B_T",
    "B_O_sign",
    "B_T_sign",
    "C_CT_z",
    "C_CO_z",
    "C_CT_null_sd",
    "C_CO_null_sd",
    "mean_trust",
    "w_norm_mean",
)


def overlaps(society):
    """Ideological overlap ``q[run, I, J] = cos(w_I, w_J)``, shape ``(R, N, N)``."""
    w = society.w  # (N, R, K)
    wn = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-300)
    return np.einsum("irk,jrk->rij", wn, wn)


def trust(society):
    """Directed trust ``t[run, r, e] = 1 - 2 Phi(mu / sqrt(1 + V))``, ``(R, N, N)``.

    The diagonal is set to ``+1``: an agent fully trusts itself.  This is not
    cosmetic --- the closed-form triple identity below assumes a unit diagonal.
    """
    h_mu = society.mu / np.sqrt(1.0 + society.V)  # (N, N, R)
    t = 1.0 - 2.0 * ndtr(h_mu)
    t = np.ascontiguousarray(np.moveaxis(t, 2, 0))  # (R, N, N)
    idx = np.arange(society.N)
    t[:, idx, idx] = 1.0
    return t


def _class_relation(kappa):
    """``s[run, I, J] = kappa_I kappa_J``, shape ``(R, N, N)``.

    ``kappa`` is ``(N, R)`` because class assignment is drawn per society.
    """
    k = np.ascontiguousarray(kappa.T)  # (R, N)
    return k[:, :, None] * k[:, None, :]


def correlations(q, t, s):
    """The three pair correlations, each in ``[-1, 1]``. Arrays of shape ``(R,)``."""
    R, N, _ = q.shape
    iu = np.triu_indices(N, 1)
    tbar = 0.5 * (t + np.swapaxes(t, 1, 2))

    q_u = q[:, iu[0], iu[1]]
    tb_u = tbar[:, iu[0], iu[1]]
    s_u = s[:, iu[0], iu[1]]

    # C_CT averages the directed trust over ordered pairs, which for a
    # symmetric s is the same as averaging tbar over unordered pairs.
    return {
        "C_CT": (s_u * tb_u).mean(axis=1),
        "C_CO": (s_u * q_u).mean(axis=1),
        "C_TO": (tb_u * q_u).mean(axis=1),
    }


def _mean_triple_product(M, N):
    """``<M_IJ M_JK M_KI>`` over unordered triples, for ``M`` with unit diagonal.

    Enumerating triples costs ``O(N^3)`` per society, which is prohibitive across
    a replicated grid.  For any ``M`` with unit diagonal,

        sum over distinct ordered (I,J,K) of M_IJ M_JK M_KI
            = tr(M^3) - 3 (sum_IJ M_IJ M_JI - N) - N

    and each unordered triple appears six times --- three as the forward cycle
    and three as the reverse.  For a symmetric ``M`` this gives the ideological
    balance; for the asymmetric trust matrix the two cycle orientations are
    exactly the two terms of the affective balance, so one expression serves
    both.  Verified against enumeration in the tests.
    """
    M3_trace = np.einsum("rij,rji->r", M, np.einsum("rij,rjk->rik", M, M))
    pair_sym = np.einsum("rij,rji->r", M, M)  # includes the N diagonal terms
    distinct = M3_trace - 3.0 * (pair_sym - N) - N
    n_triples = N * (N - 1) * (N - 2) / 6.0
    return distinct / (6.0 * n_triples)


def balance(q, t):
    """Continuous and sign-based balance for both sectors."""
    R, N, _ = q.shape
    sq, st = np.sign(q), np.sign(t)
    idx = np.arange(N)
    sq[:, idx, idx] = 1.0
    st[:, idx, idx] = 1.0
    return {
        "B_O": _mean_triple_product(q, N),
        "B_T": _mean_triple_product(t, N),
        "B_O_sign": _mean_triple_product(sq, N),
        "B_T_sign": _mean_triple_product(st, N),
    }


def _centre(M):
    """Zero the diagonal, then double-centre. ``(R, N, N)`` in and out."""
    M = M.copy()
    R, N, _ = M.shape
    idx = np.arange(N)
    M[:, idx, idx] = 0.0
    row = M.mean(axis=2, keepdims=True)
    col = M.mean(axis=1, keepdims=True)
    tot = M.mean(axis=(1, 2), keepdims=True)
    return M - row - col + tot


def p_ceiling(N):
    """The largest ``P`` a society of ``N`` agents can attain.

    ``P`` does not reach 1 even for a perfectly split population, and the
    shortfall is a finite-size effect that must be divided out before comparing
    across ``N``.

    A perfect two-bloc overlap matrix is ``s s^T`` with ``s`` in ``{+-1}^N``,
    which is exactly rank one.  Zeroing the diagonal leaves ``s s^T - I``, and
    double-centring a balanced split (``sum s = 0``) adds ``J/N``.  In that basis
    the spectrum is

        N - 1   along s
        0       along the all-ones vector (which is orthogonal to s)
        -1      on the remaining N - 2 directions

    so the attainable maximum is

        P_max = (N-1)^2 / ((N-1)^2 + (N-2))

    which is 0.941 at ``N = 16`` and 0.995 at ``N = 200``.  Standardised
    polarisation therefore divides by ``p_ceiling(N) - P_null`` rather than by
    ``1 - P_null``; without that the finite-size panels would show a spurious
    drift in polarisation with ``N`` that is purely an artefact of the estimator.

    The diagonal is removed rather than kept because the trust matrix's unit
    diagonal is a convention (an agent trusting itself) rather than a
    measurement, and it would otherwise contribute a rank-one component of its
    own.
    """
    return (N - 1) ** 2 / ((N - 1) ** 2 + (N - 2))


def spectral_polarization(M):
    """``(P, v1)``: the squared-eigenvalue fraction and the leading eigenvector.

    ``M`` must be symmetric.  ``P`` is ``(N-1)/N`` for a perfect two-bloc
    structure --- see :func:`p_ceiling` --- and about ``1/rank`` for an isotropic
    one.
    """
    Mc = _centre(M)
    Mc = 0.5 * (Mc + np.swapaxes(Mc, 1, 2))
    vals, vecs = np.linalg.eigh(Mc)
    sq = vals**2
    total = sq.sum(axis=1)
    lead = np.argmax(sq, axis=1)
    r = np.arange(M.shape[0])
    P = np.where(total > 0, sq[r, lead] / np.maximum(total, 1e-300), 0.0)
    return P, vecs[r, :, lead]


def _alignment(v1, kappa):
    """``A = (v1 . kappa / sqrt(N))^2``: how much of the leading axis is class."""
    k = np.ascontiguousarray(kappa.T)  # (R, N)
    N = k.shape[1]
    return (np.einsum("rn,rn->r", v1, k) / np.sqrt(N)) ** 2


def permutation_null(q, tbar, kappa, n_perm=200, rng=None):
    """A per-society null for the two class correlations, at negligible cost.

    The overlap and trust matrices are already in memory, so each relabelling is
    one ``O(N^2)`` contraction.  Two hundred balanced relabellings therefore cost
    essentially nothing and give a significance value *at every grid point*,
    together with an empirical noise floor that the classification thresholds
    are calibrated against.

    What this tests, precisely: whether the structure that emerged is aligned
    with the true class partition more than with a random balanced partition of
    the same society.  It is **not** a claim that the dynamics would be unchanged
    under relabelling --- it would not be, since the field is built from the true
    labels.  That other experiment is the ``partition`` field kind, and the two
    must not be conflated.
    """
    rng = rng or np.random.default_rng(0)
    R, N, _ = q.shape
    iu = np.triu_indices(N, 1)
    q_u = q[:, iu[0], iu[1]]
    tb_u = tbar[:, iu[0], iu[1]]

    ct = np.empty((R, n_perm))
    co = np.empty((R, n_perm))
    for p in range(n_perm):
        perm = rng.permutation(N)
        k = np.ascontiguousarray(kappa.T)[:, perm]  # (R, N)
        s = (k[:, :, None] * k[:, None, :])[:, iu[0], iu[1]]
        ct[:, p] = (s * tb_u).mean(axis=1)
        co[:, p] = (s * q_u).mean(axis=1)
    return {
        "C_CT_null_mu": ct.mean(axis=1),
        "C_CT_null_sd": ct.std(axis=1, ddof=1),
        "C_CO_null_mu": co.mean(axis=1),
        "C_CO_null_sd": co.std(axis=1, ddof=1),
    }


def measure(society, n_perm=200, rng=None):
    """Every observable for every society in the batch. Dict of ``(R,)`` arrays."""
    q = overlaps(society)
    t = trust(society)
    s = _class_relation(society.kappa)
    tbar = 0.5 * (t + np.swapaxes(t, 1, 2))

    out = correlations(q, t, s)
    out.update(balance(q, t))

    P_O, v_O = spectral_polarization(q)
    P_T, v_T = spectral_polarization(tbar)
    out["P_O"], out["P_T"] = P_O, P_T
    out["A_O"] = _alignment(v_O, society.kappa)
    out["A_T"] = _alignment(v_T, society.kappa)

    N = society.N
    iu = np.triu_indices(N, 1)
    out["mean_trust"] = tbar[:, iu[0], iu[1]].mean(axis=1)
    out["w_norm_mean"] = np.linalg.norm(society.w, axis=2).mean(axis=0)

    if n_perm:
        null = permutation_null(q, tbar, society.kappa, n_perm=n_perm, rng=rng)
        out.update(null)
        for name in ("C_CT", "C_CO"):
            sd = np.maximum(null[f"{name}_null_sd"], 1e-12)
            out[f"{name}_z"] = (out[name] - null[f"{name}_null_mu"]) / sd
    return out
