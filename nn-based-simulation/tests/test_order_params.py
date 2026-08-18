"""Order parameters: closed-form triple sums, and behaviour at known limits."""

import numpy as np
import pytest

from ednna.order_params import (
    _mean_triple_product,
    sign_balance,
    balance,
    class_trust_per_agent,
    correlations,
    measure,
    overlaps,
    trust,
)
from ednna.society import SocietyBatch


def brute_force_triple_mean(M, N):
    """The definition, enumerated: <(M_IJ M_JK M_KI + M_JI M_IK M_KJ)/2>."""
    vals = [
        0.5 * (M[i, j] * M[j, k] * M[k, i] + M[j, i] * M[i, k] * M[k, j])
        for i in range(N)
        for j in range(i + 1, N)
        for k in range(j + 1, N)
    ]
    return float(np.mean(vals))


@pytest.mark.parametrize("N", [5, 8, 13])
def test_closed_form_triple_sum_matches_enumeration(N):
    rng = np.random.default_rng(N)
    M = rng.normal(size=(3, N, N))
    for r in range(3):
        np.fill_diagonal(M[r], 1.0)
    expected = np.array([brute_force_triple_mean(M[r], N) for r in range(3)])
    np.testing.assert_allclose(_mean_triple_product(M, N), expected, atol=1e-12)


@pytest.mark.parametrize("N", [6, 10])
def test_closed_form_handles_symmetric_matrices(N):
    rng = np.random.default_rng(N + 100)
    A = rng.normal(size=(N, N))
    M = (A + A.T) / 2
    np.fill_diagonal(M, 1.0)
    got = _mean_triple_product(M[None], N)[0]
    assert got == pytest.approx(brute_force_triple_mean(M, N), abs=1e-12)


def test_two_faction_society_is_fully_balanced():
    """Two internally aligned, mutually opposed factions: every triple balanced."""
    N = 20
    s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
    M = np.outer(s, s)[None]
    assert _mean_triple_product(M, N)[0] == pytest.approx(1.0)


def test_uniformly_aligned_society_is_fully_balanced():
    N = 15
    M = np.ones((1, N, N))
    assert _mean_triple_product(M, N)[0] == pytest.approx(1.0)


def test_random_signs_are_unbalanced_on_average():
    N = 40
    rng = np.random.default_rng(3)
    M = np.sign(rng.normal(size=(1, N, N)))
    M = (M + np.swapaxes(M, 1, 2)) / 2
    np.fill_diagonal(M[0], 1.0)
    assert abs(_mean_triple_product(M, N)[0]) < 0.1


def _fixture_society(**kw):
    kw.setdefault("n_agents", 16)
    kw.setdefault("n_dim", 8)
    kw.setdefault("n_issues", 3)
    kw.setdefault("seed", 2)
    return SocietyBatch(**kw)


def test_order_parameters_are_in_range():
    b = _fixture_society(d=[0.0, 0.9], f_d=[0.5, 1.0])
    b.run(3000)
    for name, value in measure(b).items():
        assert np.all(value >= -1.0 - 1e-9), name
        assert np.all(value <= 1.0 + 1e-9), name


def test_correlations_at_the_perfectly_discriminating_limit():
    """Hand-built state: opinions and trust both perfectly split by class."""
    b = _fixture_society(d=0.0, f_d=0.0)
    # opinions: everyone in class A shares one weight vector, class B the opposite
    v = np.zeros(b.K)
    v[0] = 1.0
    b.w[:, 0, :] = np.outer(b.kappa, v)
    # trust: in-group fully trusted (mu very negative), out-group distrusted
    same = b.kappa[:, None] * b.kappa[None, :]
    b.mu[:, :, 0] = np.where(same > 0, -8.0, 8.0)
    b.V[:, :, 0] = 1e-6

    corr = correlations(b)
    bal = balance(b)
    assert corr["R_wmu"][0] == pytest.approx(1.0, abs=1e-3)
    assert corr["R_muc"][0] == pytest.approx(1.0, abs=1e-3)
    assert corr["R_cw"][0] == pytest.approx(1.0, abs=1e-3)
    assert bal["B_rho"][0] == pytest.approx(1.0, abs=1e-3)
    assert bal["B_eta"][0] == pytest.approx(1.0, abs=1e-3)
    # u_I = kappa_I when every agent's trust follows its own class, so the
    # histogram of u is bimodal at +-1; u_I * kappa_I = +1 for all agents.
    u = class_trust_per_agent(b)[0]
    assert u == pytest.approx(b.kappa, abs=1e-2)
    assert np.all(u * b.kappa > 0.99)


def test_reverse_discrimination_flips_the_class_correlations():
    b = _fixture_society(d=0.0, f_d=0.0)
    v = np.zeros(b.K)
    v[0] = 1.0
    b.w[:, 0, :] = np.outer(b.kappa, v)
    same = b.kappa[:, None] * b.kappa[None, :]
    b.mu[:, :, 0] = np.where(same > 0, 8.0, -8.0)  # trust the *other* class
    b.V[:, :, 0] = 1e-6
    corr = correlations(b)
    assert corr["R_muc"][0] == pytest.approx(-1.0, abs=1e-3)
    assert corr["R_cw"][0] == pytest.approx(1.0, abs=1e-3)  # opinions still class-split
    assert corr["R_wmu"][0] == pytest.approx(-1.0, abs=1e-3)


def test_class_indicator_conventions_differ_by_a_shift():
    """G in {0,1} cannot represent anti-correlation, which is why we use +-1."""
    b = _fixture_society(d=0.0, f_d=0.0)
    same = b.kappa[:, None] * b.kappa[None, :]
    b.mu[:, :, 0] = np.where(same > 0, 8.0, -8.0)
    b.V[:, :, 0] = 1e-6
    pm1 = correlations(b, class_indicator="pm1")["R_muc"][0]
    zero_one = correlations(b, class_indicator="01")["R_muc"][0]
    assert pm1 == pytest.approx(-1.0, abs=1e-3)
    # with G in {0,1} only in-group pairs contribute, and they are a fraction
    # (N/2 - 1)/(N - 1) of all pairs, so the same state reads as -7/15 at N=16
    assert zero_one == pytest.approx(-(b.N / 2 - 1) / (b.N - 1), abs=1e-3)


def test_literal_norm_halves_the_opinion_class_correlation():
    b = _fixture_society(d=0.0, f_d=0.0)
    v = np.zeros(b.K)
    v[0] = 1.0
    b.w[:, 0, :] = np.outer(b.kappa, v)
    default = correlations(b)["R_cw"][0]
    literal = correlations(b, literal_norm=True)["R_cw"][0]
    assert default == pytest.approx(1.0, abs=1e-6)
    assert literal == pytest.approx(0.5, abs=1e-6)


def test_random_initial_society_is_uncorrelated_and_frustrated():
    b = SocietyBatch(n_agents=60, n_dim=30, n_issues=5, d=0.0, f_d=0.0, seed=4)
    m = measure(b)
    for name in ("R_wmu", "R_muc", "R_cw", "B_rho", "B_eta"):
        assert abs(m[name][0]) < 0.1, name


def test_overlaps_and_trust_shapes_and_diagonals():
    b = _fixture_society(d=0.3, f_d=0.5)
    rho, eta = overlaps(b), trust(b)
    assert rho.shape == (b.R, b.N, b.N) == eta.shape
    idx = np.arange(b.N)
    np.testing.assert_allclose(rho[:, idx, idx], 1.0, atol=1e-10)
    np.testing.assert_allclose(eta[:, idx, idx], 1.0, atol=1e-12)
    np.testing.assert_allclose(rho, np.swapaxes(rho, 1, 2), atol=1e-12)


def brute_force_sign_balance(M):
    """The definition, enumerated: fraction of triples with a positive sign product."""
    S = np.sign(M)
    N = S.shape[0]
    trips = [
        S[i, j] * S[j, k] * S[k, i] > 0
        for i in range(N)
        for j in range(i + 1, N)
        for k in range(j + 1, N)
    ]
    return float(np.mean(trips))


@pytest.mark.parametrize("N", [7, 10, 13])
def test_sign_balance_matches_enumeration(N):
    rng = np.random.default_rng(N)
    A = rng.normal(size=(N, N))
    M = (A + A.T) / 2
    np.fill_diagonal(M, 1.0)
    assert sign_balance(M) == pytest.approx(brute_force_sign_balance(M), abs=1e-12)


def test_sign_balance_counts_factions():
    """This is why the measure is used: it separates two blocs from three.

    The structure theorem says a signed complete graph is balanced exactly when it
    splits into two mutually hostile cliques, so only a two-faction society reaches
    1. Anything else is visibly short of it.
    """
    def blocs(N, k):
        g = np.arange(N) % k
        M = np.where(g[:, None] == g[None, :], 1.0, -1.0)
        np.fill_diagonal(M, 1.0)
        return M

    assert sign_balance(blocs(12, 2)) == pytest.approx(1.0)
    assert sign_balance(blocs(12, 3)) < 0.8
    assert sign_balance(blocs(12, 4)) < 0.8
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 40))
    assert abs(sign_balance((A + A.T) / 2) - 0.5) < 0.1


def test_sign_balance_ignores_magnitudes():
    """Unlike B_rho, it depends only on the signs."""
    s = np.where(np.arange(14) < 6, 1.0, -1.0)
    clean = np.outer(s, s)
    faint = 0.05 * clean          # same signs, twentieth of the magnitude
    np.fill_diagonal(clean, 1.0)
    np.fill_diagonal(faint, 1.0)
    assert sign_balance(clean) == pytest.approx(sign_balance(faint))
    assert _mean_triple_product(clean[None], 14)[0] > 20 * _mean_triple_product(faint[None], 14)[0]
