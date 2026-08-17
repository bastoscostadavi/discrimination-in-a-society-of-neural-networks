"""Invariants of the learning dynamics, and the discrimination-field sign."""

import numpy as np
import pytest

from ednna.discrimination import CASES, field_matrix
from ednna.order_params import measure
from ednna.society import SocietyBatch


def make(**kw):
    kw.setdefault("n_agents", 12)
    kw.setdefault("n_dim", 10)
    kw.setdefault("n_issues", 4)
    kw.setdefault("seed", 7)
    return SocietyBatch(**kw)


def test_covariance_stays_symmetric_positive_definite():
    b = make(d=0.5, f_d=0.5)
    b.run(4000)
    for i in range(b.N):
        for r in range(b.R):
            C = b.C[i, r]
            np.testing.assert_allclose(C, C.T, atol=1e-12)
            assert np.linalg.eigvalsh(C).min() >= -1e-10


def test_affective_variance_stays_positive():
    b = make(d=0.5, f_d=1.0)
    b.run(4000)
    assert np.all(b.V > 0.0)


def test_state_stays_finite():
    b = make(d=[-1.0, 0.0, 1.0], f_d=[1.0, 1.0, 1.0])
    b.run(6000)
    for arr in (b.w, b.C, b.mu, b.V):
        assert np.all(np.isfinite(arr))


def test_covariance_shrinks_as_the_agent_learns():
    """F_C < 0 anneals the learning rate: the agent grows more certain."""
    b = make(d=0.0, f_d=0.0)
    before = np.trace(b.C[:, 0], axis1=1, axis2=2).sum()
    b.run(4000)
    after = np.trace(b.C[:, 0], axis1=1, axis2=2).sum()
    assert after < before


def test_batch_independence():
    """Societies in a batch must not influence each other.

    Two batches with the same seed, size and f_d share initial conditions and
    schedule, so a column whose d is unchanged must evolve identically no
    matter what the *other* column's d is.
    """
    a = make(d=[0.0, 0.8], f_d=[0.5, 0.5], n_agents=10)
    b = make(d=[0.0, -0.5], f_d=[0.5, 0.5], n_agents=10)
    a.run(2000)
    b.run(2000)
    ma, mb = measure(a), measure(b)
    for k in ma:
        assert ma[k][0] == pytest.approx(mb[k][0], rel=1e-12, abs=1e-12)
    assert ma["R_muc"][1] != pytest.approx(mb["R_muc"][1], abs=1e-6)


def test_zero_field_is_class_blind():
    """With d = 0 no class information enters the dynamics at all."""
    b = make(d=0.0, f_d=1.0)
    assert np.all(b.D == 0.0)


def test_discriminating_fraction_matches_f_d():
    b = SocietyBatch(n_agents=200, n_dim=4, n_issues=2, d=0.5, f_d=0.3, seed=1)
    assert b.discriminates.mean() == pytest.approx(0.3, abs=0.05)


def test_only_discriminating_agents_carry_a_field():
    b = make(d=0.7, f_d=0.5)
    non_disc = ~b.discriminates[:, 0]
    assert np.all(b.D[non_disc, :, 0] == 0.0)
    assert np.any(b.D[b.discriminates[:, 0], :, 0] != 0.0)


@pytest.mark.parametrize("case", CASES)
def test_field_matrix_cases_are_antisymmetric_in_d(case):
    np.testing.assert_allclose(field_matrix(0.4, case), -field_matrix(-0.4, case))


def test_case_six_favours_in_group_for_positive_d():
    D = field_matrix(0.5, case=6)
    assert D[0, 0] > 0 and D[1, 1] > 0  # tolerant towards one's own class
    assert D[0, 1] < 0 and D[1, 0] < 0  # intolerant towards the other


def test_literal_draft_sign_is_the_mirror_image():
    np.testing.assert_allclose(
        field_matrix(0.5, case=6, literal_draft=True), field_matrix(-0.5, case=6)
    )


@pytest.mark.slow
def test_positive_field_produces_class_correlated_trust():
    """The central claim: with enough discriminating agents and d > 0, trust
    aligns with class; with d < 0 it anti-aligns."""
    b = SocietyBatch(n_agents=30, n_dim=30, n_issues=5, d=[-0.8, 0.8], f_d=[0.9, 0.9], seed=5)
    b.run(150 * 30 * 29)
    R_muc = measure(b)["R_muc"]
    assert R_muc[0] < -0.4
    assert R_muc[1] > 0.4


@pytest.mark.slow
def test_float32_agrees_with_float64():
    """float32 is offered for speed; it must not change the physics."""
    out = {}
    for dt in (np.float64, np.float32):
        b = SocietyBatch(
            n_agents=20, n_dim=30, n_issues=5, d=[0.0, 0.8], f_d=[0.5, 0.9], seed=11, dtype=dt
        )
        b.run(60 * 20 * 19)
        out[dt] = measure(b)
    for k in out[np.float64]:
        np.testing.assert_allclose(out[np.float64][k], out[np.float32][k], atol=0.1)


@pytest.mark.slow
def test_independent_schedule_agrees_statistically():
    """The shared schedule is an optimization, not a modelling choice."""
    res = {}
    for shared in (True, False):
        b = SocietyBatch(
            n_agents=20,
            n_dim=30,
            n_issues=5,
            d=np.full(8, 0.8),
            f_d=np.full(8, 0.9),
            seed=13,
            shared_schedule=shared,
        )
        b.run(80 * 20 * 19)
        res[shared] = measure(b)["R_muc"].mean()
    assert abs(res[True] - res[False]) < 0.15


def agenda_projector(batch, run=0):
    """Orthonormal projector onto the span of the agenda, for one run."""
    X = batch.X[:, run, :]
    Q, _ = np.linalg.qr(X.T)
    return Q @ Q.T


def test_component_orthogonal_to_the_agenda_is_conserved():
    """The agenda spans only P of K directions, and learning never leaves it.

    Every weight update is along ``C x``, and ``C x`` stays in the span of the
    agenda for all time (the covariance update only ever adds multiples of
    ``(Cx)(Cx)^T``).  So each agent keeps, untouched, whatever part of its initial
    weight vector lies outside that span.  This is what caps the opinion overlap
    below one when ``alpha = P/K < 1``, and hence caps B_I -- so it is load-bearing
    for the agenda-complexity result, not an incidental invariant.
    """
    b = SocietyBatch(n_agents=12, n_dim=20, n_issues=4, d=0.0, f_d=0.0, seed=3)
    P_par = agenda_projector(b)
    P_perp = np.eye(b.K) - P_par
    before = b.w[:, 0, :] @ P_perp
    b.run(20000)
    after = b.w[:, 0, :] @ P_perp
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-9)


def test_the_agenda_span_is_where_learning_happens():
    """The complementary half: inside the span, the weights move substantially.

    Without this the conservation test above would also pass on a society that
    simply never learns anything.  Measured growth is ~2.5x in norm over this run,
    so the bound is set well below that rather than at a guessed value.
    """
    b = SocietyBatch(n_agents=12, n_dim=20, n_issues=4, d=0.0, f_d=0.0, seed=3)
    P_par = agenda_projector(b)
    before = b.w[:, 0, :] @ P_par
    b.run(20000)
    after = b.w[:, 0, :] @ P_par
    assert np.linalg.norm(after - before) > np.linalg.norm(before)


def test_conservation_holds_with_discrimination_too():
    """The conservation is a property of the agenda, not of the field."""
    b = SocietyBatch(n_agents=12, n_dim=20, n_issues=4, d=0.8, f_d=0.9, seed=5)
    P_perp = np.eye(b.K) - agenda_projector(b)
    before = b.w[:, 0, :] @ P_perp
    b.run(20000)
    np.testing.assert_allclose(b.w[:, 0, :] @ P_perp, before, rtol=0, atol=1e-9)


def test_full_rank_agenda_leaves_no_frozen_component():
    """With P >= K the agenda spans everything and the ceiling disappears."""
    b = SocietyBatch(n_agents=10, n_dim=8, n_issues=40, d=0.0, f_d=0.0, seed=6)
    P_perp = np.eye(b.K) - agenda_projector(b)
    assert np.linalg.norm(P_perp) < 1e-8
