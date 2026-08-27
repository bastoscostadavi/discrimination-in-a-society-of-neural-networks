"""The sweep: caching, shape, and that the two halves of the plane differ.

The mirror symmetry that lets ``../directional-prejudice/`` sweep only half of
the ``b`` and ``c`` axes does not exist here.  Negating ``b`` or ``c`` is the
relabelling ``A <-> B``, which maps the ensemble to itself; negating ``a`` is not
a relabelling of anything, and credulity and suspicion are two different states.
That is asserted rather than assumed, because it is the reason this sweep is
twice the width of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from credulity.config import ModelConfig, SweepConfig, get_preset
from credulity.order_params import ORDER_PARAM_NAMES
from credulity.sweep import (cache_path, strip_configs, sweep,
                             sweep_in_strips)

TINY_MODEL = ModelConfig(n_agents=8, n_dim=5, n_issues=2,
                         interactions_per_channel=4.0)
TINY_SWEEP = SweepConfig(n_a=3, n_f=2, batch_size=8, n_workers=1, seed=1)


def test_the_grid_comes_back_with_the_axes_the_right_way_round(tmp_path, monkeypatch):
    monkeypatch.setattr("credulity.sweep.DATA_DIR", tmp_path)
    out = sweep(TINY_MODEL, TINY_SWEEP, use_cache=False, verbose=False)
    assert out["a"].shape == (3,)
    assert out["f"].shape == (2,)
    for name in ORDER_PARAM_NAMES:
        assert out[name].shape == (2, 3), name  # (n_f, n_a): rows are prevalence


def test_the_strength_axis_covers_both_signs():
    a, _ = SweepConfig().grids()
    assert a[0] < 0 < a[-1]
    assert a[0] == pytest.approx(-a[-1])


def test_the_cache_is_written_and_then_used(tmp_path, monkeypatch):
    monkeypatch.setattr("credulity.sweep.DATA_DIR", tmp_path)
    first = sweep(TINY_MODEL, TINY_SWEEP, use_cache=True, verbose=False)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    second = sweep(TINY_MODEL, TINY_SWEEP, use_cache=True, verbose=False)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    for name in ORDER_PARAM_NAMES:
        np.testing.assert_array_equal(first[name], second[name])


def test_the_cache_key_changes_when_the_measured_set_does(monkeypatch):
    """Adding an order parameter must invalidate every existing cache file.

    Without this the old file still looks valid and then fails on the missing
    array at read time -- or is silently re-plotted from a stale set.
    """
    before = cache_path(TINY_MODEL, TINY_SWEEP)
    monkeypatch.setattr("credulity.sweep.ORDER_PARAM_NAMES",
                        ORDER_PARAM_NAMES + ("something_new",))
    assert cache_path(TINY_MODEL, TINY_SWEEP) != before


def test_the_cache_key_separates_configurations_that_differ():
    a = cache_path(TINY_MODEL, TINY_SWEEP)
    b = cache_path(TINY_MODEL.with_(n_agents=9), TINY_SWEEP)
    c = cache_path(TINY_MODEL, TINY_SWEEP.with_(seed=2))
    assert len({a, b, c}) == 3
    assert "a_P2_N8" in a.name and "3x2" in a.name


def test_the_full_preset_is_the_resolution_the_paper_uses():
    """200x200 at N = 40 and the calibrated Delta t.  This is the number the
    whole directory exists to produce, so it is pinned rather than trusted."""
    p = get_preset("full")
    assert (p.sweep.n_a, p.sweep.n_f) == (200, 200)
    assert p.model.n_agents == 40
    assert p.model.n_dim == 30
    assert p.model.interactions_per_channel == 500.0
    assert p.sweep.a_range == (-1.0, 1.0)
    assert p.sweep.f_range == (0.0, 1.0)


def test_the_empty_group_rows_survive_the_repeat_average(tmp_path, monkeypatch):
    """``f = 0`` is a whole row with no biased agents in it, and the repeat
    average must leave it as ``nan`` rather than warn or fill it."""
    monkeypatch.setattr("credulity.sweep.DATA_DIR", tmp_path)
    cfg = TINY_SWEEP.with_(n_f=2, n_repeats=2)
    with np.errstate(all="raise"):
        out = sweep(TINY_MODEL, cfg, use_cache=False, verbose=False)
    assert np.all(np.isnan(out["T_give_b"][0]))     # f = 0: no biased agents
    assert np.all(np.isfinite(out["T_give_u"][0]))
    assert np.all(np.isnan(out["T_give_u"][-1]))    # f = 1: no unbiased agents


@pytest.mark.slow
def test_negating_a_is_not_a_symmetry_of_the_ensemble():
    """Credulity and suspicion are two states, not one state and its mirror.

    Compared as ensembles rather than pathwise: at a fixed seed the two runs are
    two different draws, so a pathwise comparison would be testing seed luck.
    Eight societies a side, and the separation is asserted against the standard
    error of the difference of the two means -- not against the per-realization
    spread, which at these sizes is a fifth of the range on a saturated quantity
    and would make a real effect look marginal.
    """
    from credulity.order_params import measure
    from credulity.society import SocietyBatch

    reps = 8
    a = np.repeat([-1.0, 1.0], reps)
    b = SocietyBatch(n_agents=16, n_dim=10, n_issues=4, a=a, f=1.0, seed=99)
    b.run(int(150 * 16 * 15))
    m = measure(b)
    for key in ("T_mu", "rho_mean"):
        halves = m[key].reshape(2, reps)
        lo, hi = halves.mean(axis=1)
        se = np.sqrt((halves.var(axis=1, ddof=1) / reps).sum())
        assert hi - lo > 4 * se, f"{key}: {lo:.3f} vs {hi:.3f} (se {se:.3f})"
    # and the asymmetry is not merely a sign flip of one quantity: the trust
    # sector reverses while the opinion sector does not mirror it
    t_lo, t_hi = m["T_mu"].reshape(2, reps).mean(axis=1)
    r_lo, r_hi = m["rho_mean"].reshape(2, reps).mean(axis=1)
    assert t_lo < 0 < t_hi
    assert abs(r_lo + r_hi) > 0.05 * max(abs(r_lo), abs(r_hi))


# --- striping -------------------------------------------------------------

@pytest.mark.parametrize("n_f, n_strips", [(200, 5), (64, 5), (7, 3), (10, 4),
                                           (5, 5), (3, 1), (6, 7)])
def test_strips_partition_the_prevalence_axis_exactly(n_f, n_strips):
    """Concatenating the strips' axes must reproduce the full axis.

    Parametrized over cases where the strips do not divide the rows evenly, and
    over one where there are more strips than rows, because the tempting
    implementation -- carving the unit interval into equal intervals and calling
    linspace inside each -- duplicates every boundary row and changes the row
    spacing within a band relative to between bands.  Both failures give a plane
    that looks right.
    """
    cfg = SweepConfig(n_a=4, n_f=n_f, f_range=(0.0, 1.0))
    strips = strip_configs(cfg, n_strips)
    rebuilt = np.concatenate([np.linspace(*s.f_range, s.n_f) for s in strips])
    np.testing.assert_allclose(rebuilt, np.linspace(*cfg.f_range, cfg.n_f),
                               atol=1e-12)
    assert sum(s.n_f for s in strips) == n_f


def test_strips_do_not_share_seeds():
    """Every band must draw different societies.

    ``sweep`` seeds batch ``b`` as ``seed + <flat offset within this sweep>``,
    and that offset restarts at zero in every band.  Left on a common seed the
    bands draw the *same* populations, and the plane comes out looking converged
    when it is one band repeated.
    """
    cfg = SweepConfig(n_a=200, n_f=200, seed=7)
    strips = strip_configs(cfg, 5)
    seeds = [s.seed for s in strips]
    assert len(set(seeds)) == len(seeds)
    # the offsets must be at least a whole strip's worth of grid points apart,
    # or two bands still overlap in the stream
    assert min(np.diff(sorted(seeds))) >= cfg.n_a * min(s.n_f for s in strips)


def test_a_strip_seed_depends_only_on_where_it_sits_in_the_plane():
    """Re-running one band must reproduce it and renumber no other.

    That is what makes a resume bit-identical rather than merely statistically
    equivalent, and it fails if the offsets are handed out by strip index.
    """
    cfg = SweepConfig(n_a=50, n_f=100, seed=3)
    five = {s.f_range[0]: s.seed for s in strip_configs(cfg, 5)}
    ten = {s.f_range[0]: s.seed for s in strip_configs(cfg, 10)}
    shared = set(five) & set(ten)
    assert len(shared) >= 5
    for f0 in shared:
        assert five[f0] == ten[f0]


def test_one_strip_is_an_ordinary_sweep():
    cfg = SweepConfig(n_a=8, n_f=8)
    assert strip_configs(cfg, 1) == [cfg]
    assert strip_configs(cfg, 0) == [cfg]


def test_striped_and_unstriped_planes_have_the_same_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("credulity.sweep.DATA_DIR", tmp_path)
    cfg = SweepConfig(n_a=3, n_f=6, batch_size=8, n_workers=1, seed=5)
    whole = sweep(TINY_MODEL, cfg, use_cache=False, verbose=False)
    striped = sweep_in_strips(TINY_MODEL, cfg, n_strips=3, use_cache=False,
                              verbose=False)
    np.testing.assert_allclose(striped["f"], whole["f"], atol=1e-12)
    np.testing.assert_array_equal(striped["a"], whole["a"])
    for name in ORDER_PARAM_NAMES:
        assert striped[name].shape == whole[name].shape, name


def test_a_killed_strip_is_the_only_work_lost(tmp_path, monkeypatch):
    """Each band caches on its own, so a re-run reloads the finished ones."""
    monkeypatch.setattr("credulity.sweep.DATA_DIR", tmp_path)
    cfg = SweepConfig(n_a=3, n_f=6, batch_size=8, n_workers=1, seed=5)
    first = sweep_in_strips(TINY_MODEL, cfg, n_strips=3, use_cache=True,
                            verbose=False)
    assert len(list(tmp_path.glob("*.npz"))) == 3

    # lose the middle band, keep the other two
    doomed = sorted(tmp_path.glob("*.npz"))[1]
    doomed.unlink()
    again = sweep_in_strips(TINY_MODEL, cfg, n_strips=3, use_cache=True,
                            verbose=False)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    for name in ORDER_PARAM_NAMES:
        np.testing.assert_allclose(again[name], first[name], equal_nan=True,
                                   err_msg=name)


def test_the_full_preset_asks_for_strips():
    """A four-hour sweep with no checkpointing loses everything to one kill."""
    assert get_preset("full").n_strips > 1
    assert get_preset("quick").n_strips == 1
