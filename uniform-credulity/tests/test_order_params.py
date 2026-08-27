"""The order parameters, and the two things about them worth pinning down.

One is bookkeeping: the block means must be the means they claim to be, over the
pairs they claim, with the diagonal out and an empty group reported as ``nan``.
The other is the exact statement the uniform field is the cleanest place in the
project to make: a population with uniform trust and no class structure has
``R_muc = -T_mu/(N-1)``, not zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from credulity.order_params import (
    ORDER_PARAM_NAMES, balance_within_groups, bias_block_trust, bias_blocks,
    bias_trust_margins, correlations, measure, opinion_blocks, overlaps, trust,
    trust_channels, trust_given_per_agent, trust_received_per_agent,
)
from credulity.society import SocietyBatch

SMALL = dict(n_agents=12, n_dim=6, n_issues=3)


def make(a=1.0, f=0.5, seed=0, steps=300, **kw):
    b = SocietyBatch(**{**SMALL, **kw}, a=a, f=f, seed=seed)
    if steps:
        b.run(steps)
    return b


def test_measure_returns_exactly_the_swept_set():
    m = measure(make())
    assert sorted(m) == sorted(ORDER_PARAM_NAMES)
    assert len(set(ORDER_PARAM_NAMES)) == len(ORDER_PARAM_NAMES)


def test_every_swept_array_has_one_entry_per_run():
    b = SocietyBatch(**SMALL, a=np.linspace(-1, 1, 5), f=0.5, seed=1)
    b.run(200)
    for k, v in measure(b).items():
        assert np.shape(v) == (5,), k


# --- the channels ---------------------------------------------------------

def test_R_muc_is_the_same_number_as_a_correlation_and_as_a_channel():
    """``kappa_r kappa_e`` is symmetric, so the directed and symmetrized forms
    agree exactly.  The two are computed by different code paths."""
    b = make()
    eta = trust(b)
    assert correlations(b, eta=eta)["R_muc"] == pytest.approx(
        trust_channels(b, eta=eta)["R_muc"], abs=1e-12
    )


def test_T_mu_is_the_off_diagonal_mean_of_the_trust_matrix():
    b = make()
    eta = trust(b)
    N = b.N
    off = ~np.eye(N, dtype=bool)
    expected = np.array([eta[r][off].mean() for r in range(b.R)])
    assert trust_channels(b, eta=eta)["T_mu"] == pytest.approx(expected)


@pytest.mark.slow
def test_a_uniformly_trusting_population_has_R_muc_at_minus_T_mu_over_N_minus_1():
    """The one control that is not zero, and it is predicted exactly --
    *for equal class sizes*, which is the only case this package builds.

    The four channel weights are orthogonal over all ``N^2`` pairs but not over
    the ``N(N-1)`` that exclude the diagonal: with equal class sizes
    ``<1, kappa_r kappa_e> = -N``, so the uniform channel leaks into the matching
    one at ``-1/(N-1)``.  Checked at both signs, since the leak carries the sign
    of ``T_mu``.
    """
    reps = 4
    a = np.repeat([-1.0, 1.0], reps)
    b = SocietyBatch(n_agents=20, n_dim=10, n_issues=4, a=a, f=1.0, seed=8)
    b.run(int(200 * 20 * 19))
    m = measure(b)
    predicted = -m["T_mu"] / (b.N - 1)
    assert np.abs(m["T_mu"]).min() > 0.5           # the leak is worth measuring
    assert m["R_muc"] == pytest.approx(predicted, abs=0.02)
    # the identity is -1/(N-1) only because the classes are the same size; the
    # general Gram entry is what `_leakage` computes, and the two must agree
    assert b.N % 2 == 0 and (b.class_of == 0).sum() == (b.class_of == 1).sum()


@pytest.mark.slow
def test_the_class_channels_that_name_a_class_stay_at_zero():
    """``R_cred`` and ``R_stat`` are controls: the dynamics never reads the label."""
    reps = 4
    a = np.repeat([-1.0, 1.0], reps)
    b = SocietyBatch(n_agents=20, n_dim=10, n_issues=4, a=a, f=1.0, seed=9)
    b.run(int(200 * 20 * 19))
    m = measure(b)
    for k in ("R_cred", "R_stat"):
        assert np.abs(m[k].reshape(2, reps).mean(axis=1)).max() < 0.1, k


# --- the bias partition ---------------------------------------------------

def test_the_margins_are_the_means_they_claim_to_be():
    b = make(f=0.5, seed=6)
    eta = trust(b)
    m = bias_trust_margins(b, eta=eta)
    N = b.N
    off = ~np.eye(N, dtype=bool)
    for run in range(b.R):
        biased = b.biased[:, run]
        rows = eta[run][biased][:, :]           # every biased receiver
        mask = off[biased]
        assert m["T_give_b"][run] == pytest.approx(rows[mask].mean())
        cols = eta[run][:, biased]
        assert m["T_get_b"][run] == pytest.approx(cols[off[:, biased]].mean())


def test_the_margins_agree_with_the_per_agent_quantities():
    b = make(f=0.5, seed=6)
    eta = trust(b)
    m = bias_trust_margins(b, eta=eta)
    given = trust_given_per_agent(b, eta=eta)
    received = trust_received_per_agent(b, eta=eta)
    for run in range(b.R):
        biased = b.biased[:, run]
        assert m["T_give_b"][run] == pytest.approx(given[run][biased].mean())
        assert m["T_get_b"][run] == pytest.approx(received[run][biased].mean())


def test_the_blocks_reconstruct_the_margins():
    """A margin is the block means weighted by how many pairs each block holds."""
    b = make(f=0.5, seed=12)
    eta = trust(b)
    blocks = bias_block_trust(b, eta=eta)
    m = bias_trust_margins(b, eta=eta)
    for run in range(b.R):
        nb = int(b.biased[:, run].sum())
        nu = b.N - nb
        if min(nb, nu) < 2:
            continue
        # ordered off-diagonal pairs in each block of the biased row
        w = np.array([nb * (nb - 1), nb * nu], dtype=float)
        got = (blocks[run, 0] * w).sum() / w.sum()
        assert m["T_give_b"][run] == pytest.approx(got)


def test_the_whole_population_block_is_T_mu():
    """``rho_mean`` and ``T_mu`` use the same estimator on the two matrices."""
    b = make(f=0.5)
    eta = trust(b)
    assert opinion_blocks(b)["rho_mean"] == pytest.approx(
        np.array([overlaps(b)[r][~np.eye(b.N, dtype=bool)].mean()
                  for r in range(b.R)])
    )
    assert trust_channels(b, eta=eta)["T_mu"] == pytest.approx(
        bias_trust_margins(b, eta=eta)["T_give_b"] * 0
        + np.array([eta[r][~np.eye(b.N, dtype=bool)].mean() for r in range(b.R)])
    )


def test_the_diagonal_is_excluded_from_every_block():
    """Set the trust matrix to the identity: every off-diagonal mean is zero.

    If any block leaked the diagonal in, it would read positive instead.
    """
    b = make(f=0.5, steps=0)
    eta = np.stack([np.eye(b.N) for _ in range(b.R)])
    m = {**bias_trust_margins(b, eta=eta)}
    for k, v in m.items():
        assert np.allclose(v[np.isfinite(v)], 0.0), k


@pytest.mark.parametrize("f, empty", [(0.0, "b"), (1.0, "u")])
def test_an_empty_group_is_nan_and_not_zero(f, empty):
    """A group with no members has no mean trust, and drawing a zero there would
    put a boundary on the phase diagram that is an artefact of the estimator."""
    b = make(f=f, seed=3)
    m = {**bias_trust_margins(b), **opinion_blocks(b), **balance_within_groups(b)}
    present = "u" if empty == "b" else "b"
    for k, v in m.items():
        if k.endswith(f"_{empty}") or k.endswith(f"{empty}{empty}"):
            assert np.all(np.isnan(v)), k
        elif k.endswith(f"_{present}") or k.endswith(f"{present}{present}"):
            assert np.all(np.isfinite(v)), k
    # and the cross-block, which needs both, is nan either way
    assert np.all(np.isnan(opinion_blocks(b)["rho_bu"]))


def test_a_group_of_fewer_than_three_has_no_balance():
    """The triple sum needs three agents; two is ``nan``, not an extrapolation."""
    b = SocietyBatch(n_agents=12, n_dim=6, n_issues=3, a=1.0,
                     f=np.array([0.5]), seed=0)
    b.biased[:, 0] = False
    b.biased[:2, 0] = True
    out = balance_within_groups(b)
    assert np.isnan(out["B_eta_b"][0])
    assert np.isfinite(out["B_eta_u"][0])


def test_balance_within_a_group_matches_the_aggregate_when_the_group_is_everyone():
    b = make(f=1.0, seed=5)
    from credulity.order_params import balance
    assert balance_within_groups(b)["B_eta_b"] == pytest.approx(balance(b)["B_eta"])
    assert balance_within_groups(b)["B_rho_b"] == pytest.approx(balance(b)["B_rho"])


def test_a_perfectly_trusting_population_is_perfectly_balanced():
    b = make(f=1.0, steps=0)
    eta = np.ones((b.R, b.N, b.N))
    from credulity.order_params import balance
    assert balance(b, eta=eta)["B_eta"] == pytest.approx(1.0)


def test_the_named_blocks_are_the_matrix_the_right_way_round():
    """``T_ub`` must be the unbiased receiver's trust in a biased emitter.

    Getting this transposed would turn the emergent margin into the direct one
    and the result would look stronger, not broken -- so it is asserted against
    a hand-built trust matrix where the four blocks are four distinct constants.
    """
    b = make(f=0.5, seed=21, steps=0)
    eta = np.zeros((b.R, b.N, b.N))
    for run in range(b.R):
        biased = b.biased[:, run]
        for i, receiver in enumerate((biased, ~biased)):
            for j, emitter in enumerate((biased, ~biased)):
                eta[run][np.ix_(receiver, emitter)] = 0.1 * (2 * i + j) + 0.1
    named = bias_blocks(b, eta=eta)
    # receiver index first: bb=0.1, bu=0.2, ub=0.3, uu=0.4
    for key, want in (("T_bb", 0.1), ("T_bu", 0.2), ("T_ub", 0.3), ("T_uu", 0.4)):
        assert named[key] == pytest.approx(want), key


def test_the_named_blocks_are_the_block_matrix():
    b = make(f=0.5, seed=13)
    eta = trust(b)
    blocks = bias_block_trust(b, eta=eta)
    named = bias_blocks(b, eta=eta)
    assert named["T_bb"] == pytest.approx(blocks[:, 0, 0])
    assert named["T_ub"] == pytest.approx(blocks[:, 1, 0])


def test_the_realized_fraction_is_recorded_and_is_not_the_requested_one():
    """It is binomial in ``f_a``, so a margin cannot be rebuilt from the blocks
    without it."""
    b = SocietyBatch(n_agents=12, n_dim=6, n_issues=3, a=1.0,
                     f=np.full(40, 0.5), seed=4)
    b.run(50)
    got = measure(b)["frac_biased"]
    assert got == pytest.approx(b.biased.mean(axis=0))
    assert got.std() > 0.0            # it really does vary
    assert got.mean() == pytest.approx(0.5, abs=0.05)


@pytest.mark.slow
def test_only_the_class_referring_parameters_are_controls():
    """The distinction the README draws, asserted in both directions.

    Two of the paper's five reference the class label and must stay at zero here,
    because the dynamics never reads it.  The other three reference no label, and
    on this plane they are the *result*: they must move, and a great deal.
    Asserting only the first half would let "four of the five are controls" --
    which is false -- pass unnoticed.
    """
    reps = 4
    a = np.repeat([-1.0, 1.0], reps)
    b = SocietyBatch(n_agents=20, n_dim=10, n_issues=4, a=a, f=1.0, seed=31)
    b.run(int(200 * 20 * 19))
    m = measure(b)
    lo = {k: v.reshape(2, reps).mean(axis=1)[0] for k, v in m.items()}
    hi = {k: v.reshape(2, reps).mean(axis=1)[1] for k, v in m.items()}

    # controls: reference the label, so near zero at both signs.  R_muc is
    # excluded -- it has a predicted non-zero, checked by its own test above.
    for k in ("R_cw", "R_cred", "R_stat"):
        assert abs(lo[k]) < 0.1 and abs(hi[k]) < 0.1, (k, lo[k], hi[k])

    # not controls: reference no label, and the two signs must differ sharply
    assert hi["R_wmu"] - lo["R_wmu"] > 0.25, (lo["R_wmu"], hi["R_wmu"])
    assert hi["B_eta"] - lo["B_eta"] > 1.0, (lo["B_eta"], hi["B_eta"])
    assert lo["B_eta"] < 0 < hi["B_eta"]
