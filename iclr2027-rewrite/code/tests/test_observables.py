"""Order parameters, validated on configurations whose answer is known.

Each test builds a society state by hand and asserts what the statistics must
report.  Several are the checks the plan document asks for; two of them it gets
wrong, and the corrections are noted where they arise.

The most important tests here are the ones that separate *polarised* from
*class-aligned*.  Reporting only the class correlations would describe a sharply
divided population, split along an axis unrelated to class, as unstructured ---
and that is exactly the region the source material called "neutral".
"""

import numpy as np
import pytest

from socsim import FieldSpec, ModelConfig, SocietyBatch
from socsim.observables import (
    BOUNDED,
    _class_relation,
    _mean_triple_product,
    balance,
    correlations,
    measure,
    overlaps,
    p_ceiling,
    permutation_null,
    spectral_polarization,
    trust,
)
from socsim.seeds import RunKey, point_id

N, K, P = 16, 8, 3


def _society(n_agents=N, seed=0):
    model = ModelConfig(
        n_agents=n_agents, n_dim=K, n_issues=P, interactions_per_channel=1
    )
    key = RunKey("obs", "obs", point_id({"d": 0.0, "s": seed}), 0, 0)
    return SocietyBatch.from_keys(
        model, [key], [FieldSpec(kind="none", d=0.0, f_d=0.0)], master=seed + 1
    )


def _impose(b, opinion_groups, trust_groups, sharp=8.0, invert_trust=False):
    """Force opinions and trust to follow the given +-1 group vectors.

    ``invert_trust`` makes same-group pairs *distrust* each other.  Note that
    negating ``trust_groups`` would not do this: the trust pattern depends on the
    outer product, which is invariant under a global sign flip.
    """
    v = np.zeros(b.K)
    v[0] = 1.0
    b.w[:, 0, :] = np.outer(opinion_groups, v)
    same = trust_groups[:, None] * trust_groups[None, :]
    trusted = -sharp if not invert_trust else sharp
    distrusted = sharp if not invert_trust else -sharp
    b.mu[:, :, 0] = np.where(same > 0, trusted, distrusted)
    b.V[:, :, 0] = 1e-6
    return b


def _orthogonal_split(b):
    """A balanced two-bloc split that carries no class information.

    Class labels are permuted per society, so an index-parity split is *not*
    orthogonal to class in general.  Taking half of each class by construction
    is.
    """
    kappa = b.kappa[:, 0]
    split = np.empty(b.N)
    for c in (+1.0, -1.0):
        idx = np.flatnonzero(kappa == c)
        split[idx[: idx.size // 2]] = +1.0
        split[idx[idx.size // 2 :]] = -1.0
    return split


def _obs(b):
    q, t = overlaps(b), trust(b)
    s = _class_relation(b.kappa)
    out = correlations(q, t, s)
    out.update(balance(q, t))
    P_O, v_O = spectral_polarization(q)
    tbar = 0.5 * (t + np.swapaxes(t, 1, 2))
    P_T, _ = spectral_polarization(tbar)
    out["P_O"], out["P_T"] = P_O, P_T
    return {k: float(np.asarray(v)[0]) for k, v in out.items()}


# -- the five checks of the plan's order-parameter list ---------------
def test_1_perfect_discrimination_saturates_every_correlation():
    b = _society()
    kappa = b.kappa[:, 0]
    _impose(b, kappa, kappa)
    o = _obs(b)
    assert o["C_CT"] == pytest.approx(1.0, abs=1e-3)
    assert o["C_CO"] == pytest.approx(1.0, abs=1e-3)
    assert o["C_TO"] == pytest.approx(1.0, abs=1e-3)
    assert o["B_O"] == pytest.approx(1.0, abs=1e-3)
    assert o["B_T"] == pytest.approx(1.0, abs=1e-3)


def test_2_class_unrelated_factions_are_polarised_but_uncorrelated():
    """The check that stops "neutral" being read as "nothing happening".

    Two coherent, mutually opposed blocs that cut across the class boundary: the
    class correlations vanish, but the population is maximally polarised. Only
    the spectral measures can tell this from an unstructured society.
    """
    b = _society()
    split = _orthogonal_split(b)
    _impose(b, split, split)
    o = _obs(b)
    ceiling = p_ceiling(b.N)
    assert abs(o["C_CT"]) < 0.08
    assert abs(o["C_CO"]) < 0.08
    assert o["P_O"] == pytest.approx(ceiling, abs=1e-3)
    assert o["P_T"] == pytest.approx(ceiling, abs=1e-3)
    assert o["C_TO"] == pytest.approx(1.0, abs=1e-3)  # trust still tracks agreement


def test_3_universal_trust_and_agreement():
    """The plan says the class correlations should be "near zero" here.

    At finite N that is not exactly right and the exact value is worth pinning:
    with every pair trusted and a balanced class split, averaging s_IJ over
    unordered pairs gives -1/(N-1), not 0.  A test written to the plan's wording
    would fail spuriously at small N.
    """
    b = _society()
    ones = np.ones(b.N)
    _impose(b, ones, ones)
    o = _obs(b)
    expected = -1.0 / (b.N - 1)
    assert o["C_CT"] == pytest.approx(expected, abs=1e-3)
    assert o["C_CO"] == pytest.approx(expected, abs=1e-3)
    assert o["C_TO"] == pytest.approx(1.0, abs=1e-3)
    assert o["B_O"] == pytest.approx(1.0, abs=1e-3)


def test_4_reverse_discrimination_flips_the_trust_class_correlation():
    b = _society()
    kappa = b.kappa[:, 0]
    _impose(b, kappa, kappa, invert_trust=True)
    o = _obs(b)
    assert o["C_CT"] == pytest.approx(-1.0, abs=1e-3)
    assert o["C_CO"] == pytest.approx(1.0, abs=1e-3)
    assert o["C_TO"] == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize("n_agents", [40, 100])
def test_5_random_state_is_uncorrelated_and_shrinks_with_N(n_agents):
    b = _society(n_agents=n_agents, seed=3)
    o = _obs(b)
    tol = 4.0 / np.sqrt(n_agents)
    for name in ("C_CT", "C_CO", "C_TO", "B_O", "B_T"):
        assert abs(o[name]) < tol, name


def test_5b_correlations_decay_as_N_grows():
    vals = []
    for n in (20, 80, 320):
        reps = [abs(_obs(_society(n_agents=n, seed=s))["C_CT"]) for s in range(5)]
        vals.append(np.mean(reps))
    assert vals[0] > vals[1] > vals[2]


# -- ranges and symmetries -------------------------------------------
def test_all_observables_stay_in_range():
    b = _society()
    b.run(2000)
    got = measure(b, n_perm=20)
    for name in BOUNDED:
        v = got[name]
        assert np.all(np.asarray(v) >= -1.0 - 1e-9), name
        assert np.all(np.asarray(v) <= 1.0 + 1e-9), name


def test_correlations_are_invariant_under_relabelling_the_classes():
    b = _society()
    kappa = b.kappa[:, 0]
    _impose(b, kappa, kappa)
    before = _obs(b)
    b.kappa[:, 0] *= -1
    after = _obs(b)
    for name in ("C_CT", "C_CO", "C_TO"):
        assert before[name] == pytest.approx(after[name], abs=1e-12)


def test_polarisation_is_invariant_under_rotating_every_opinion():
    """P measures structure, so a global rotation of the issue space cannot move it."""
    b = _society()
    b.run(500)
    P_before, _ = spectral_polarization(overlaps(b))
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(b.K, b.K)))
    b.w[:, 0, :] = b.w[:, 0, :] @ Q
    P_after, _ = spectral_polarization(overlaps(b))
    np.testing.assert_allclose(P_before, P_after, atol=1e-10)


def test_polarisation_is_one_for_two_blocs_and_small_for_noise():
    b = _society(n_agents=60)
    split = _orthogonal_split(b)
    _impose(b, split, split)
    assert spectral_polarization(overlaps(b))[0][0] == pytest.approx(
        p_ceiling(b.N), abs=1e-3
    )
    rand = _society(n_agents=60, seed=9)
    assert spectral_polarization(overlaps(rand))[0][0] < 0.35


# -- the closed-form triple identity ---------------------------------
def _brute_force(M, n):
    return float(
        np.mean(
            [
                0.5 * (M[i, j] * M[j, k] * M[k, i] + M[j, i] * M[i, k] * M[k, j])
                for i in range(n)
                for j in range(i + 1, n)
                for k in range(j + 1, n)
            ]
        )
    )


@pytest.mark.parametrize("n", [5, 9, 13])
def test_closed_form_matches_enumeration(n):
    rng = np.random.default_rng(n)
    M = rng.normal(size=(2, n, n))
    for r in range(2):
        np.fill_diagonal(M[r], 1.0)
    expected = np.array([_brute_force(M[r], n) for r in range(2)])
    np.testing.assert_allclose(_mean_triple_product(M, n), expected, atol=1e-12)


def test_closed_form_matches_enumeration_for_sign_matrices():
    """New: the identity was only ever exercised on continuous matrices.

    The sign-based balance is a different statistic and it is the only one that
    may be described as a fraction of balanced triples, so it needs its own
    check.
    """
    n = 11
    rng = np.random.default_rng(2)
    M = np.sign(rng.normal(size=(1, n, n)))
    np.fill_diagonal(M[0], 1.0)
    assert _mean_triple_product(M, n)[0] == pytest.approx(_brute_force(M[0], n), abs=1e-12)


def test_two_opposed_blocs_are_fully_balanced():
    n = 20
    s = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    assert _mean_triple_product(np.outer(s, s)[None], n)[0] == pytest.approx(1.0)


# -- the permutation null --------------------------------------------
def test_permutation_null_is_centred_when_there_is_no_class_signal():
    b = _society(n_agents=40)
    split = np.where(np.arange(b.N) % 2 == 0, 1.0, -1.0)
    _impose(b, split, split)
    q, t = overlaps(b), trust(b)
    tbar = 0.5 * (t + np.swapaxes(t, 1, 2))
    null = permutation_null(q, tbar, b.kappa, n_perm=300, rng=np.random.default_rng(0))
    got = correlations(q, t, _class_relation(b.kappa))["C_CT"][0]
    z = (got - null["C_CT_null_mu"][0]) / null["C_CT_null_sd"][0]
    assert abs(z) < 3.0


def test_permutation_null_detects_a_real_class_signal():
    b = _society(n_agents=40)
    kappa = b.kappa[:, 0]
    _impose(b, kappa, kappa)
    o = measure(b, n_perm=300, rng=np.random.default_rng(1))
    assert o["C_CT_z"][0] > 5.0
    assert o["C_CO_z"][0] > 5.0
