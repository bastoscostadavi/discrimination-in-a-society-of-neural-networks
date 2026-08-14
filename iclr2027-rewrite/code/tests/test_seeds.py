"""The seed design, which everything else is staked on.

A seed bug discovered after a twenty-hour campaign is unrecoverable, so these
properties are pinned before any long run starts.  The reference implementation
seeded one generator per *batch*, which meant the randomness a grid point
received depended on the batch size and on where the point fell in the flattened
array; nothing here may inherit that.
"""

import numpy as np
import pytest

from socsim import FieldSpec, ModelConfig, SocietyBatch
from socsim.seeds import RunKey, derived_seed, point_id, stream

N, K, P = 8, 6, 3
MODEL = ModelConfig(n_agents=N, n_dim=K, n_issues=P, interactions_per_channel=5)


def _key(d=0.5, f_d=0.5, experiment="e", crn="g", dis=0, init=0):
    return RunKey(experiment, crn, point_id({"d": d, "f_d": f_d}), dis, init)


def _build(keys, d=0.5, f_d=0.5, master=1):
    specs = [FieldSpec(kind="class", case=6, d=d, f_d=f_d) for _ in keys]
    return SocietyBatch.from_keys(MODEL, keys, specs, master=master)


# -- identity ---------------------------------------------------------
def test_point_id_is_stable_under_float_round_off():
    """linspace round-off must not split one point into two.

    Grid axes are regenerated whenever a run is resumed or refined, and the
    last bit of a float is not reproducible across those regenerations.  If the
    identifier moved with it, resumption would silently re-simulate everything.
    """
    a = point_id({"d": 0.1 + 0.2, "f_d": 0.5})
    b = point_id({"d": 0.30000000000000004, "f_d": 0.5})
    assert a == b


def test_point_id_normalises_negative_zero():
    assert point_id({"d": -0.0}) == point_id({"d": 0.0})


def test_point_id_separates_distinct_points():
    assert point_id({"d": 0.5, "f_d": 0.5}) != point_id({"d": 0.5, "f_d": 0.6})


def test_stream_is_deterministic():
    k = _key()
    assert stream(k, "init", 7).normal() == stream(k, "init", 7).normal()


def test_master_seed_changes_everything():
    k = _key()
    assert stream(k, "init", 1).normal() != stream(k, "init", 2).normal()


def test_roles_are_independent():
    k = _key()
    draws = {r: stream(k, r, 3).normal(size=5) for r in ("agenda", "mask", "field", "init")}
    for a in draws:
        for b in draws:
            if a != b:
                assert not np.allclose(draws[a], draws[b])


# -- batch invariance -------------------------------------------------
def test_society_is_independent_of_batch_composition():
    """The property the reference implementation did not have.

    The same key must give the same society whether it is simulated alone or
    alongside a hundred others, and regardless of its position in the batch.
    """
    target = _key(d=0.5)
    others = [_key(d=x) for x in (-0.9, -0.3, 0.1, 0.9)]

    alone = _build([target])
    with_others = _build([target] + others)
    at_end = _build(others + [target])

    for batch, idx in ((with_others, 0), (at_end, len(others))):
        np.testing.assert_array_equal(alone.w[:, 0, :], batch.w[:, idx, :])
        np.testing.assert_array_equal(alone.mu[:, :, 0], batch.mu[:, :, idx])
        np.testing.assert_array_equal(alone.X[:, 0, :], batch.X[:, idx, :])
        np.testing.assert_array_equal(alone.class_of[:, 0], batch.class_of[:, idx])
        np.testing.assert_array_equal(
            alone.discriminates[:, 0], batch.discriminates[:, idx]
        )


def test_trajectory_is_independent_of_batch_composition():
    target = _key(d=0.5)
    others = [_key(d=x) for x in (-0.4, 0.8)]
    alone = _build([target])
    together = _build([target] + others)
    n = MODEL.n_steps()
    alone.run(n)
    together.run(n)
    np.testing.assert_allclose(alone.w[:, 0, :], together.w[:, 0, :], rtol=0, atol=0)
    np.testing.assert_allclose(alone.mu[:, :, 0], together.mu[:, :, 0], rtol=0, atol=0)


# -- replicates -------------------------------------------------------
def test_replicates_differ_in_state_and_schedule():
    a = _build([_key(init=0)])
    b = _build([_key(init=1)])
    assert not np.allclose(a.w[:, 0, :], b.w[:, 0, :])
    assert derived_seed(_key(init=0), "schedule", 1) != derived_seed(
        _key(init=1), "schedule", 1
    )


def test_schedule_is_shared_within_a_replicate_across_points():
    """This is what makes the batched inner loop legal.

    Within one replicate every point draws the same interaction order, so the
    batch advances in lockstep.  Across replicates the order differs, which is
    what the seed-to-seed spread needs.
    """
    s1 = derived_seed(_key(d=0.1, init=0), "schedule", 1)
    s2 = derived_seed(_key(d=0.9, init=0), "schedule", 1)
    assert s1 == s2


def test_disorder_replicates_share_nothing_with_each_other():
    a = _build([_key(dis=0)])
    b = _build([_key(dis=1)])
    assert not np.allclose(a.X[:, 0, :], b.X[:, 0, :])


def test_disorder_and_init_are_separable():
    """Same environment, different initial state: the replica-overlap setup.

    Holding the agenda, the class labels and the discriminator mask fixed while
    varying the initial condition is what lets us ask whether the frustrated
    region is a glass or merely disordered, instead of asserting it.
    """
    a = _build([_key(dis=0, init=0)])
    b = _build([_key(dis=0, init=1)])
    np.testing.assert_array_equal(a.X[:, 0, :], b.X[:, 0, :])
    np.testing.assert_array_equal(a.class_of[:, 0], b.class_of[:, 0])
    np.testing.assert_array_equal(a.discriminates[:, 0], b.discriminates[:, 0])
    assert not np.allclose(a.w[:, 0, :], b.w[:, 0, :])


# -- common random numbers -------------------------------------------
def test_controls_are_paired_with_their_baseline():
    """The largest variance reduction available here, and it is free.

    A control sharing ``crn_group`` and point with its baseline starts from the
    same weights, the same distrust and the same interaction order.  Differences
    can then be reported paired, which removes the between-society variance that
    would otherwise swamp them.
    """
    base = RunKey("baseline", "main", point_id({"d": 0.5, "f_d": 0.5}), 0, 3)
    ctrl = RunKey("control", "main", point_id({"d": 0.5, "f_d": 0.5}), 0, 3)
    a, b = _build([base]), _build([ctrl])
    np.testing.assert_array_equal(a.w[:, 0, :], b.w[:, 0, :])
    np.testing.assert_array_equal(a.mu[:, :, 0], b.mu[:, :, 0])
    assert derived_seed(base, "schedule", 1) == derived_seed(ctrl, "schedule", 1)


def test_different_crn_groups_are_not_paired():
    a = RunKey("x", "one", point_id({"d": 0.5}), 0, 0)
    b = RunKey("x", "two", point_id({"d": 0.5}), 0, 0)
    assert not np.allclose(_build([a]).w[:, 0, :], _build([b]).w[:, 0, :])


def test_different_experiments_get_different_environments():
    a = RunKey("one", "main", point_id({"d": 0.5}), 0, 0)
    b = RunKey("two", "main", point_id({"d": 0.5}), 0, 0)
    assert not np.allclose(_build([a]).X[:, 0, :], _build([b]).X[:, 0, :])


# -- guard rails ------------------------------------------------------
def test_mixing_replicates_in_one_batch_is_refused():
    """Sharing a schedule across replicates would understate the uncertainty.

    It is the one way to silently bias the error bars, so it is an error rather
    than a convention.
    """
    with pytest.raises(ValueError, match="one \\(crn_group, init\\) pair"):
        _build([_key(init=0), _key(init=1)])


def test_unknown_role_is_refused():
    with pytest.raises(ValueError, match="role must be one of"):
        stream(_key(), "nonsense", 1)
