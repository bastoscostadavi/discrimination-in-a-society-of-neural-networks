"""The completed order parameters, and the exact cancellations they repair."""

import itertools

import numpy as np
import pytest

from dirfield.order_params import (
    _mean_triple_product,
    balance,
    balance_by_composition,
    balance_within_classes,
    class_block_trust,
    correlations,
    credulity_per_agent,
    measure,
    overlaps,
    status_per_agent,
    trust,
    trust_channels,
)
from dirfield.society import SocietyBatch


def bare(n_agents=12, n_dim=6, n_issues=3, runs=1, seed=0):
    """A batch with no field, not run: a carrier for a hand-built trust matrix."""
    return SocietyBatch(n_agents=n_agents, n_dim=n_dim, n_issues=n_issues,
                        f=np.zeros(runs), seed=seed)


def impose_trust(society, per_pair):
    """Force ``eta[r, e] = per_pair(r, e)`` by setting mu directly.

    With ``V = 0`` the scaled field is ``mu`` itself, so ``eta = 1 - 2 Phi(mu)``
    and any target is reached by inverting the probit.  Returns the eta actually
    obtained, so a test can assert against measurement rather than intent.
    """
    from scipy.special import ndtri
    N = society.N
    target = np.array([[per_pair(r, e) for e in range(N)] for r in range(N)])
    np.fill_diagonal(target, 0.0)
    society.V[...] = 0.0
    society.mu[...] = ndtri(0.5 * (1.0 - target))[:, :, None]
    return trust(society)


def brute_force_triple_mean(M, N):
    """The definition, enumerated: <(M_IJ M_JK M_KI + M_JI M_IK M_KJ)/2>."""
    vals = [
        0.5 * (M[i, j] * M[j, k] * M[k, i] + M[j, i] * M[i, k] * M[k, j])
        for i, j, k in itertools.combinations(range(N), 3)
    ]
    return float(np.mean(vals))


# --- the closed form, inherited unchanged --------------------------------

@pytest.mark.parametrize("N", [5, 8, 13])
def test_closed_form_triple_sum_matches_enumeration(N):
    rng = np.random.default_rng(N)
    M = rng.normal(size=(3, N, N))
    for r in range(3):
        np.fill_diagonal(M[r], 1.0)
    expected = np.array([brute_force_triple_mean(M[r], N) for r in range(3)])
    np.testing.assert_allclose(_mean_triple_product(M, N), expected, atol=1e-12)


# --- the four channels ---------------------------------------------------

def test_matching_channel_is_the_papers_correlation_exactly():
    """``<kappa_r kappa_e eta>`` over ordered pairs == ``R_muc`` over unordered ones.

    The claim that the four channels *extend* the paper's set rather than
    replacing it rests on this identity, so it is worth a test rather than a
    remark: kappa_r kappa_e is symmetric under swapping the pair, so averaging
    the directed product is the same as averaging the symmetrized trust.
    """
    soc = bare(n_agents=10, runs=3, seed=4)
    rng = np.random.default_rng(1)
    soc.mu[...] = rng.normal(size=soc.mu.shape)
    soc.V[...] = rng.uniform(0.1, 2.0, size=soc.V.shape)
    eta = trust(soc)
    np.testing.assert_allclose(
        trust_channels(soc, eta=eta)["R_muc"],
        correlations(soc, eta=eta)["R_muc"],
        atol=1e-12,
    )


def test_channels_reconstruct_the_class_blocks():
    """The four channels are a rotation of the 2x2 block means, so they invert.

    For equal class sizes the map is exactly the field basis read on the trust
    matrix instead of on the field: block[i, j] = T_mu + R_cred k_i + R_stat k_j
    + R_muc k_i k_j, up to the diagonal being excluded from both sides.
    """
    soc = bare(n_agents=12, runs=1, seed=5)
    eta = impose_trust(soc, lambda r, e: 0.3 * (-1) ** r + 0.5 * (-1) ** e)
    ch = trust_channels(soc, eta=eta, orthogonalize=True)
    blocks = class_block_trust(soc, eta=eta)[0]
    kap = np.array([+1.0, -1.0])
    rebuilt = (ch["T_mu"][0]
               + ch["R_cred"][0] * kap[:, None]
               + ch["R_stat"][0] * kap[None, :]
               + ch["R_muc"][0] * kap[:, None] * kap[None, :])
    np.testing.assert_allclose(rebuilt, blocks, atol=1e-12)


def test_uniform_trust_leaks_into_the_matching_channel_by_one_over_n_minus_one():
    """A population with no class structure at all has a non-zero ``R_muc``.

    Not a defect of this rewriting: the paper's ``R_muc`` has the same overlap
    over unordered pairs, so a uniformly trusting population reads as very
    slightly reverse-discriminating at any finite N.  It is exactly
    ``-T_mu/(N-1)``, and ``orthogonalize=True`` removes it.
    """
    soc = bare(n_agents=12, runs=1, seed=6)
    eta = impose_trust(soc, lambda r, e: 0.6)
    ch = trust_channels(soc, eta=eta)
    assert ch["T_mu"][0] == pytest.approx(0.6, abs=1e-6)
    assert ch["R_muc"][0] == pytest.approx(-0.6 / 11, abs=1e-6)
    for k in ("R_cred", "R_stat"):
        assert ch[k][0] == pytest.approx(0.0, abs=1e-9), k

    exact = trust_channels(soc, eta=eta, orthogonalize=True)
    assert exact["T_mu"][0] == pytest.approx(0.6, abs=1e-6)
    assert exact["R_muc"][0] == pytest.approx(0.0, abs=1e-9)


def test_status_field_shows_up_in_the_status_channel():
    """Trust depending on the speaker's class alone reads exactly as ``R_stat``.

    The status readout is exact; what the excluded diagonal costs is a spurious
    ``-R_stat/(N-1)`` in the credulity channel, which orthogonalizing removes.
    """
    soc = bare(n_agents=12, runs=1, seed=7)
    eta = impose_trust(soc, lambda r, e: 0.8 * soc.kappa[e])
    ch = trust_channels(soc, eta=eta)
    assert ch["R_stat"][0] == pytest.approx(0.8, abs=1e-6)
    assert ch["R_cred"][0] == pytest.approx(-0.8 / 11, abs=1e-6)
    for k in ("T_mu", "R_muc"):
        assert ch[k][0] == pytest.approx(0.0, abs=1e-9), k

    exact = trust_channels(soc, eta=eta, orthogonalize=True)
    assert exact["R_stat"][0] == pytest.approx(0.8, abs=1e-6)
    assert exact["R_cred"][0] == pytest.approx(0.0, abs=1e-9)


def test_credulity_field_shows_up_in_the_credulity_channel():
    """The mirror of the status case, with the leakage in the other direction."""
    soc = bare(n_agents=12, runs=1, seed=8)
    eta = impose_trust(soc, lambda r, e: 0.8 * soc.kappa[r])
    ch = trust_channels(soc, eta=eta)
    assert ch["R_cred"][0] == pytest.approx(0.8, abs=1e-6)
    assert ch["R_stat"][0] == pytest.approx(-0.8 / 11, abs=1e-6)
    for k in ("T_mu", "R_muc"):
        assert ch[k][0] == pytest.approx(0.0, abs=1e-9), k

    exact = trust_channels(soc, eta=eta, orthogonalize=True)
    assert exact["R_cred"][0] == pytest.approx(0.8, abs=1e-6)
    assert exact["R_stat"][0] == pytest.approx(0.0, abs=1e-9)


# --- the exact cancellations -------------------------------------------

def test_a_maximal_status_hierarchy_is_invisible_to_every_paper_parameter():
    """The result this package exists for.

    Trust depending on the speaker's class alone, at full strength: a population
    in which everyone trusts one class and distrusts the other, its own members
    included.  ``R_muc`` and ``B_eta`` both vanish *identically*, not merely
    approximately, because the AA and BB terms cancel term by term and the
    triples split evenly by parity.
    """
    soc = bare(n_agents=12, runs=1, seed=9)
    eta = impose_trust(soc, lambda r, e: 0.99 * soc.kappa[e])
    ch = trust_channels(soc, eta=eta)
    corr = correlations(soc, eta=eta)
    bal = balance(soc, eta=eta)

    assert ch["R_stat"][0] == pytest.approx(0.99, abs=1e-6)   # fully ordered
    assert corr["R_muc"][0] == pytest.approx(0.0, abs=1e-12)  # and invisible
    assert bal["B_eta"][0] == pytest.approx(0.0, abs=1e-12)


def test_the_cancellation_needs_the_classes_to_be_the_same_size():
    """It is exact, not generic: unbalanced classes leave a residue."""
    soc = bare(n_agents=13, runs=1, seed=10)  # 6 in A, 7 in B
    assert (soc.class_of == 0).sum() != (soc.class_of == 1).sum()
    eta = impose_trust(soc, lambda r, e: 0.99 * soc.kappa[e])
    assert abs(correlations(soc, eta=eta)["R_muc"][0]) > 0.01


def test_balance_by_composition_is_the_parity_of_the_stigmatized_count():
    """Each agent appears once as emitter in the cycle, so the sign is (-1)^k."""
    soc = bare(n_agents=12, runs=1, seed=11)
    eta = impose_trust(soc, lambda r, e: 0.99 * soc.kappa[e])
    values, counts = balance_by_composition(soc, eta=eta)
    signs = np.sign(values[0])
    np.testing.assert_allclose(signs, [+1, -1, +1, -1])
    for k in range(4):
        assert abs(values[0, k]) == pytest.approx(0.99 ** 3, abs=1e-3)
    # equal class sizes: as many odd-parity triples as even-parity ones, which is
    # why the aggregate is exactly zero
    assert counts[0] + counts[2] == counts[1] + counts[3]
    assert counts.sum() == 220  # C(12, 3)


def test_within_class_balance_separates_a_bloc_from_a_dust():
    """The cheap, sweepable signature of the same asymmetry.

    Under a status field the credited class trusts itself and is balanced; every
    member of the stigmatized class distrusts every other member, so every triple
    inside it is a product of three negatives and is frustrated.
    """
    soc = bare(n_agents=12, runs=1, seed=12)
    eta = impose_trust(soc, lambda r, e: 0.99 * soc.kappa[e])
    out = balance_within_classes(soc, eta=eta)
    assert out["B_eta_A"][0] == pytest.approx(+0.99 ** 3, abs=1e-3)
    assert out["B_eta_B"][0] == pytest.approx(-0.99 ** 3, abs=1e-3)


def test_within_class_balance_is_nan_when_a_class_is_too_small():
    soc = bare(n_agents=4, runs=1, seed=13)  # two per class, no triples inside one
    out = balance_within_classes(soc)
    assert np.isnan(out["B_eta_A"][0]) and np.isnan(out["B_eta_B"][0])


def test_a_matching_field_is_visible_to_the_paper_and_not_to_status():
    """The control: what the published parameter is for, it still does."""
    soc = bare(n_agents=12, runs=1, seed=14)
    eta = impose_trust(soc, lambda r, e: 0.99 * soc.kappa[r] * soc.kappa[e])
    assert correlations(soc, eta=eta)["R_muc"][0] == pytest.approx(0.99, abs=1e-6)
    assert trust_channels(soc, eta=eta)["R_stat"][0] == pytest.approx(0.0, abs=1e-9)


def test_orthogonalizing_is_a_no_op_where_the_channels_do_not_overlap():
    """It corrects the two coupled pairs and leaves an isolated signal alone."""
    soc = bare(n_agents=20, runs=1, seed=21)
    eta = impose_trust(soc, lambda r, e: 0.5 * soc.kappa[r] * soc.kappa[e])
    raw = trust_channels(soc, eta=eta)
    exact = trust_channels(soc, eta=eta, orthogonalize=True)
    assert raw["R_muc"][0] == pytest.approx(0.5, abs=1e-6)
    assert exact["R_muc"][0] == pytest.approx(0.5, abs=1e-6)
    assert exact["T_mu"][0] == pytest.approx(0.0, abs=1e-9)


# --- per-agent readouts -------------------------------------------------

def test_per_agent_status_and_credulity_are_bimodal_by_class():
    soc = bare(n_agents=12, runs=1, seed=15)
    eta = impose_trust(soc, lambda r, e: 0.9 * soc.kappa[e])
    recv = status_per_agent(soc, eta=eta)[0]
    np.testing.assert_allclose(recv, 0.9 * soc.kappa, atol=1e-6)  # exact: own column
    # Everyone gives alike -- except that nobody rates themselves, so an agent's
    # peer set holds one fewer of its own class, worth -kappa_I/(N-1).  A finite-size
    # fact about the measurement rather than an artefact to remove.
    given = credulity_per_agent(soc, eta=eta)[0]
    np.testing.assert_allclose(given, -0.9 * soc.kappa / (soc.N - 1), atol=1e-6)


def test_per_agent_means_average_to_the_channels():
    soc = bare(n_agents=10, runs=2, seed=16)
    rng = np.random.default_rng(2)
    soc.mu[...] = rng.normal(size=soc.mu.shape)
    eta = trust(soc)
    ch = trust_channels(soc, eta=eta)
    np.testing.assert_allclose(status_per_agent(soc, eta=eta).mean(axis=1),
                               ch["T_mu"], atol=1e-12)
    np.testing.assert_allclose(
        (status_per_agent(soc, eta=eta) * soc.kappa).mean(axis=1),
        ch["R_stat"], atol=1e-12)
    np.testing.assert_allclose(
        (credulity_per_agent(soc, eta=eta) * soc.kappa).mean(axis=1),
        ch["R_cred"], atol=1e-12)


# --- the sweep payload --------------------------------------------------

def test_measure_returns_every_swept_name_with_the_right_shape():
    from dirfield.order_params import ORDER_PARAM_NAMES
    soc = bare(n_agents=12, runs=3, seed=17)
    out = measure(soc)
    assert set(out) == set(ORDER_PARAM_NAMES)
    for k, v in out.items():
        assert np.shape(v) == (3,), k


def test_measure_agrees_with_the_individual_functions():
    soc = bare(n_agents=12, runs=2, seed=18)
    rng = np.random.default_rng(3)
    soc.mu[...] = rng.normal(size=soc.mu.shape)
    out = measure(soc)
    rho, eta = overlaps(soc), trust(soc)
    for k, v in correlations(soc, rho=rho, eta=eta).items():
        np.testing.assert_allclose(out[k], v, atol=1e-12)
    for k, v in trust_channels(soc, eta=eta).items():
        np.testing.assert_allclose(out[k], v, atol=1e-12)


def test_overlaps_and_trust_are_bounded():
    soc = bare(n_agents=10, runs=2, seed=19)
    rho, eta = overlaps(soc), trust(soc)
    assert rho.min() >= -1 - 1e-12 and rho.max() <= 1 + 1e-12
    assert eta.min() >= -1 and eta.max() <= 1
    idx = np.arange(soc.N)
    np.testing.assert_allclose(rho[:, idx, idx], 1.0, atol=1e-12)
    np.testing.assert_allclose(eta[:, idx, idx], 1.0)
