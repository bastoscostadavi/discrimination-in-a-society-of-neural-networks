"""The dynamics under a uniform field.

Two kinds of claim are checked here, and they need different tests.  Exact ones
-- that the field is applied to exactly the right agents, that ``a = 0``
reproduces the unbiased society bit for bit -- are checked pathwise on a single
society, because they hold realization by realization.  Statistical ones -- that
credulity raises trust, that suspicion destroys it -- are checked over an
ensemble, because a single society is a draw and not a statement.
"""

from __future__ import annotations

import numpy as np
import pytest

from credulity.order_params import bias_trust_margins, measure, trust
from credulity.society import SocietyBatch

SMALL = dict(n_agents=10, n_dim=6, n_issues=3)


def make(a=0.0, f=0.0, seed=0, steps=0, **kw):
    b = SocietyBatch(**{**SMALL, **kw}, a=a, f=f, seed=seed)
    if steps:
        b.run(steps)
    return b


# --- the field itself -----------------------------------------------------

def test_the_shift_is_indexed_by_the_receiver_alone():
    """``D`` has one entry per agent, not one per ordered pair.

    That is the whole content of the uniform component, and it is worth asserting
    on the shape rather than only on the values: a copy of this package that
    reintroduced an emitter index would still pass every value test if the
    matrix happened to be constant along it.
    """
    b = make(a=0.7, f=1.0)
    assert b.D.shape == (b.N, b.R)
    assert np.allclose(b.D, 0.7)


def test_only_the_biased_agents_carry_the_field():
    b = make(a=0.4, f=0.5, seed=3)
    assert np.allclose(b.D[b.biased], 0.4)
    assert np.allclose(b.D[~b.biased], 0.0)


def test_the_prevalence_sets_the_expected_number_of_biased_agents():
    """Binomial in ``f``, not exactly ``f N``: the group size fluctuates."""
    n = 4000
    b = SocietyBatch(**SMALL, a=1.0, f=np.full(n, 0.3), seed=11)
    assert b.biased.mean() == pytest.approx(0.3, abs=0.02)
    # and it really does vary run to run, which is why every group quantity has
    # to cope with an empty group
    assert b.biased.sum(axis=0).std() > 0.5


@pytest.mark.parametrize("f", [0.0, 0.3, 0.7, 1.0])
def test_zero_strength_is_the_unbiased_society_whatever_the_prevalence(f):
    """``a = 0`` must be an exact no-op, not merely a small one."""
    ref = make(a=0.0, f=0.0, seed=5, steps=400)
    other = make(a=0.0, f=f, seed=5, steps=400)
    assert np.allclose(ref.w, other.w)
    assert np.allclose(ref.mu, other.mu)


@pytest.mark.parametrize("a", [-1.0, -0.25, 0.25, 1.0])
def test_zero_prevalence_is_the_unbiased_society_whatever_the_strength(a):
    """Parametrized, because "whatever the strength" run at one strength is a
    test name claiming more than the test checks."""
    ref = make(a=0.0, f=0.0, seed=5, steps=400)
    other = make(a=a, f=0.0, seed=5, steps=400)
    assert np.allclose(ref.w, other.w)
    assert np.allclose(ref.mu, other.mu)


def test_the_class_labels_are_assigned_and_never_read():
    """The class split exists for the controls; the dynamics must not use it.

    Checked by permuting which agents are class A and confirming the trajectory
    is untouched -- the labels are metadata here, and a test that only read
    ``class_of`` would not notice if some future edit started using it.
    """
    b = make(a=0.8, f=0.6, seed=7)
    before = b.class_of.copy()
    b.class_of = b.class_of[::-1].copy()
    b.kappa = np.where(b.class_of == 0, 1.0, -1.0)
    b.run(400)
    ref = make(a=0.8, f=0.6, seed=7, steps=400)
    assert np.allclose(b.w, ref.w)
    assert np.allclose(b.mu, ref.mu)
    assert not np.array_equal(before, b.class_of)  # the permutation was real


def test_agent_assignment_matches_the_class_dependent_packages():
    """The rng stream is drawn in the same order and the same shapes.

    ``../directional-prejudice/`` draws w, mu, X, then the membership mask, and
    so does this package, so a run here and a run there at one seed bias the same
    agents and can be compared pixel for pixel.  Asserted against the sequence
    written out by hand rather than against the other package, which is not
    importable from here.
    """
    N, K, P, R, f = 10, 6, 3, 4, 0.5
    rng = np.random.default_rng(123)
    rng.normal(size=(N, R, K))          # w
    rng.uniform(-1.0, 1.0, size=(N, N, R))  # mu
    rng.normal(size=(P, R, K))          # X
    expected = rng.random((N, R)) < f

    b = SocietyBatch(n_agents=N, n_dim=K, n_issues=P, a=1.0,
                     f=np.full(R, f), seed=123)
    assert np.array_equal(b.biased, expected)


def test_mismatched_lengths_are_rejected_once():
    with pytest.raises(ValueError, match="one common length"):
        SocietyBatch(**SMALL, a=np.zeros(3), f=np.zeros(4))


# --- what the field does --------------------------------------------------

@pytest.mark.slow
def test_credulity_raises_trust_and_suspicion_destroys_it():
    """The sign of ``a`` decides the sign of the trust the population reaches.

    Over an ensemble: a single society at these sizes lands anywhere.
    """
    reps = 6
    a = np.repeat([-1.0, 0.0, 1.0], reps)
    b = SocietyBatch(n_agents=16, n_dim=10, n_issues=4, a=a, f=1.0, seed=42)
    b.run(int(200 * 16 * 15))
    T = measure(b)["T_mu"].reshape(3, reps).mean(axis=1)
    suspicious, neutral, credulous = T
    assert suspicious < -0.5
    assert credulous > 0.5
    assert suspicious < neutral < credulous


@pytest.mark.slow
def test_the_biased_group_extends_more_trust_than_the_unbiased_one():
    """The direct effect, which is the field acting on the receiver."""
    reps = 6
    b = SocietyBatch(n_agents=16, n_dim=10, n_issues=4,
                     a=np.full(reps, 1.0), f=np.full(reps, 0.5), seed=17)
    b.run(int(200 * 16 * 15))
    m = bias_trust_margins(b)
    assert np.nanmean(m["T_give_b"]) > np.nanmean(m["T_give_u"]) + 0.2


def test_the_diagonal_of_the_trust_matrix_is_the_convention_the_sums_assume():
    b = make(a=1.0, f=1.0, seed=2, steps=200)
    eta = trust(b)
    assert np.allclose(np.diagonal(eta, axis1=1, axis2=2), 1.0)


def test_the_covariance_projection_only_ever_shrinks_a_step():
    b = make(a=1.0, f=1.0, seed=4)
    xCx = np.array([2.0, 2.0])
    a_in = np.array([-10.0, 0.5])
    a_out = b._project_psd(a_in, xCx)
    assert a_out[1] == pytest.approx(0.5)          # untouched
    assert a_in[0] < a_out[0] < 0                  # shrunk towards zero
    assert 1.0 + a_out[0] * xCx[0] >= 0            # and back inside the PSD cone
    assert b.n_psd_clips == 1
