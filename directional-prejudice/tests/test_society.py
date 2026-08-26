"""The dynamics under a general field: wiring, invariants, and the headline."""

import numpy as np
import pytest

from dirfield.fields import field_matrix
from dirfield.order_params import (
    balance_within_classes,
    class_block_trust,
    correlations,
    measure,
    trust,
    trust_channels,
)
from dirfield.society import SocietyBatch


def small(**kw):
    kw.setdefault("n_agents", 12)
    kw.setdefault("n_dim", 8)
    kw.setdefault("n_issues", 3)
    kw.setdefault("seed", 0)
    return SocietyBatch(**kw)


# --- how the field reaches the agents -----------------------------------

def test_no_field_and_no_prejudice_means_no_shift():
    soc = small(f=0.0, c=1.0)
    assert np.all(soc.D == 0.0)  # f = 0: nobody applies it
    soc = small(f=1.0)
    assert np.all(soc.D == 0.0)  # all four components zero


def test_a_status_field_depends_on_the_emitter_only():
    """Every prejudiced receiver treats a given speaker the same way."""
    soc = small(c=1.0, f=1.0)
    assert soc.prejudiced.all()
    D = soc.D[:, :, 0]
    for e in range(soc.N):
        assert len(np.unique(np.round(D[:, e], 12))) == 1
    # and the value is set by that speaker's class
    for e in range(soc.N):
        assert D[0, e] == pytest.approx(soc.kappa[e])


def test_a_credulity_field_depends_on_the_receiver_only():
    soc = small(b=1.0, f=1.0)
    D = soc.D[:, :, 0]
    for r in range(soc.N):
        assert len(np.unique(np.round(D[r], 12))) == 1
        assert D[r, 0] == pytest.approx(soc.kappa[r])


def test_a_matching_field_is_the_outer_product_of_the_classes():
    soc = small(p=0.5, f=1.0)
    expected = 0.5 * np.outer(soc.kappa, soc.kappa)
    np.testing.assert_allclose(soc.D[:, :, 0], expected, atol=1e-12)


def test_only_prejudiced_agents_carry_a_field():
    soc = small(c=1.0, f=0.5, seed=3)
    D = soc.D[:, :, 0]
    for r in range(soc.N):
        if soc.prejudiced[r, 0]:
            assert np.any(D[r] != 0.0)
        else:
            assert np.all(D[r] == 0.0)


def test_the_field_matches_the_basis_it_was_asked_for():
    q = (0.1, -0.2, 0.3, 0.4)
    soc = small(a=q[0], b=q[1], c=q[2], p=q[3], f=1.0)
    M = field_matrix(*q)
    expected = M[np.ix_(soc.class_of, soc.class_of)]
    np.testing.assert_allclose(soc.D[:, :, 0], expected, atol=1e-12)


def test_per_run_parameters_broadcast_against_each_other():
    soc = SocietyBatch(n_agents=8, n_dim=4, n_issues=2,
                       c=np.array([0.0, 0.5, 1.0]), f=1.0, seed=1)
    assert soc.R == 3
    assert soc.D[:, :, 0].max() == 0.0
    assert soc.D[0, 0, 2] == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="one common length"):
        SocietyBatch(n_agents=8, n_dim=4, n_issues=2,
                     c=np.zeros(3), f=np.zeros(4))


def test_classes_are_split_in_half():
    soc = small(n_agents=10)
    assert (soc.class_of == 0).sum() == (soc.class_of == 1).sum() == 5
    np.testing.assert_allclose(soc.kappa.sum(), 0.0)


# --- invariants of the update -------------------------------------------

@pytest.mark.parametrize("kw", [{"c": 1.0, "f": 1.0}, {"b": 1.0, "f": 0.5},
                                {"p": 1.0, "f": 1.0}, {"a": 1.0, "f": 1.0}])
def test_covariance_stays_symmetric_and_positive_definite(kw):
    soc = small(**kw)
    soc.run(2000)
    C = soc.C[:, 0]
    np.testing.assert_allclose(C, np.swapaxes(C, 1, 2), atol=1e-10)
    for i in range(soc.N):
        assert np.linalg.eigvalsh(C[i]).min() > -1e-9


@pytest.mark.parametrize("kw", [{"c": 1.0, "f": 1.0}, {"p": 1.0, "f": 1.0}])
def test_trust_variance_stays_positive_and_trust_stays_bounded(kw):
    soc = small(**kw)
    soc.run(2000)
    assert soc.V.min() > 0.0
    eta = trust(soc)
    assert eta.min() >= -1.0 and eta.max() <= 1.0


def test_everything_measured_is_finite_after_a_run():
    soc = small(c=1.0, f=1.0)
    soc.run(2000)
    for k, v in measure(soc).items():
        assert np.all(np.isfinite(v)), k


def test_interaction_count_is_reported_per_channel():
    soc = small()
    soc.run(soc.N * (soc.N - 1) * 3)
    assert soc.n_interactions_per_channel == pytest.approx(3.0)


def test_float32_tracks_float64():
    a = small(c=1.0, f=1.0, dtype=np.float64)
    b = small(c=1.0, f=1.0, dtype=np.float32)
    a.run(1500)
    b.run(1500)
    for k in ("R_stat", "R_muc"):
        assert (trust_channels(a)[k][0] - trust_channels(b)[k][0]) == pytest.approx(0.0, abs=0.1)


# --- the result the package exists for -----------------------------------

@pytest.mark.slow
def test_a_status_field_orders_the_population_invisibly():
    """A pure status field, run for real: fully ordered, and unseen.

    The claim of the module docstring, on the dynamics rather than on a
    hand-built matrix.  Thresholds are loose because this is one small society;
    the point is the separation between the two numbers, which is an order of
    magnitude, not their precise values.
    """
    soc = SocietyBatch(n_agents=16, n_dim=10, n_issues=3, c=1.0, f=1.0, seed=5)
    soc.run(int(60 * 16 * 15))

    ch = trust_channels(soc)
    corr = correlations(soc)
    assert ch["R_stat"][0] > 0.9          # a maximal hierarchy
    assert abs(corr["R_muc"][0]) < 0.05   # invisible to the published parameter
    assert abs(corr["R_cw"][0]) < 0.1     # opinion has not followed the label

    # the credited class is a bloc, the other is dust
    bal = balance_within_classes(soc)
    assert bal["B_eta_A"][0] > 0.5
    assert bal["B_eta_B"][0] < -0.5

    # and the blocks say the stigmatized class distrusts itself
    blocks = class_block_trust(soc)[0]
    assert blocks[0, 0] > 0.5 and blocks[1, 0] > 0.5    # both trust A
    assert blocks[0, 1] < -0.5 and blocks[1, 1] < -0.5  # both distrust B


@pytest.mark.slow
def test_a_matching_field_is_seen_by_the_published_parameter():
    """The same run under the field the paper studies, as a control."""
    soc = SocietyBatch(n_agents=16, n_dim=10, n_issues=3, p=1.0, f=1.0, seed=5)
    soc.run(int(60 * 16 * 15))
    assert correlations(soc)["R_muc"][0] > 0.9
    assert abs(trust_channels(soc)["R_stat"][0]) < 0.15
