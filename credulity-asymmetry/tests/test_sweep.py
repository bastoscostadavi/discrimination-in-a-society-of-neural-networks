"""Sweep plumbing: the mirror symmetry that halves it, caching, and shapes."""

import pathlib

import numpy as np
import pytest

from credfield.config import ModelConfig, SweepConfig, default_s_range, get_preset
from credfield.order_params import ORDER_PARAM_NAMES
from credfield.sweep import cache_path, sweep


@pytest.fixture
def tiny():
    """A grid small enough to run in a test, in one process."""
    model = ModelConfig(n_agents=8, n_dim=5, n_issues=2,
                        component="b", interactions_per_channel=6.0)
    cfg = SweepConfig(n_s=3, n_f=2, s_range=(0.0, 1.0), batch_size=64, n_workers=1)
    return model, cfg


def test_sweep_returns_every_order_parameter_on_the_grid(tiny, tmp_path, monkeypatch):
    import credfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    model, cfg = tiny
    out = sweep(model, cfg, use_cache=False, verbose=False)
    assert set(out) == set(ORDER_PARAM_NAMES) | {"s", "f"}
    for name in ORDER_PARAM_NAMES:
        assert out[name].shape == (cfg.n_f, cfg.n_s), name
    np.testing.assert_allclose(out["s"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["f"], [0.0, 1.0])


def test_a_sweep_is_cached_and_reloaded(tiny, tmp_path, monkeypatch):
    import credfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    model, cfg = tiny
    first = sweep(model, cfg, use_cache=True, verbose=False)
    assert list(tmp_path.glob("*.npz"))
    second = sweep(model, cfg, use_cache=True, verbose=False)
    for k in first:
        np.testing.assert_allclose(first[k], second[k])


def test_the_cache_key_separates_configurations(tiny):
    model, cfg = tiny
    a = cache_path(model, cfg)
    assert a != cache_path(model.with_(component="c"), cfg)
    assert a != cache_path(model.with_(n_agents=9), cfg)
    assert a != cache_path(model, cfg.with_(n_s=4))
    assert a != cache_path(model, cfg, tag="other")
    assert "b_P2" in a.name  # the component is legible in the filename


def test_default_ranges_reflect_which_signs_are_distinct():
    for k in ("a", "p"):
        assert default_s_range(k) == (-1.0, 1.0)
    for k in ("b", "c"):
        assert default_s_range(k) == (0.0, 1.0)


@pytest.mark.slow
def test_negating_a_directional_field_is_relabelling_the_classes():
    """Why half the plane is not swept: the two signs are the same experiment.

    Negating ``b`` exchanges the roles of A and B -- which class is the credulous
    one and which the suspicious one -- and that maps the ensemble to itself,
    since the two classes are the same size and everything else about an agent is
    drawn without reference to its class.  So every class-odd quantity flips sign
    and every class-even one is unchanged, which is the argument for sweeping
    ``[0, 1]`` instead of ``[-1, 1]`` and so for this plane costing half of what
    the paper's ``p`` plane costs.

    The symmetry is one of the *ensemble*, not of a single trajectory: at a fixed
    seed the two runs are not each other's relabelling but two different draws,
    and at ``N = 16`` a single pair disagrees by up to ``0.2``.  So this compares
    means over eight independent societies against their spread, which is the
    claim actually being made.  (Asserting it pathwise at ``abs=0.02`` passes or
    fails on the seed.)
    """
    from credfield.order_params import measure, trust_channels
    from credfield.society import SocietyBatch

    R = 8
    kw = dict(n_agents=16, n_dim=8, n_issues=3, f=np.ones(R), seed=11)
    steps = int(40 * 16 * 15)
    out = {}
    for sign in (+1.0, -1.0):
        soc = SocietyBatch(b=sign * 0.8, **kw)
        soc.run(steps)
        out[sign] = (trust_channels(soc)["R_cred"], measure(soc))
    pos, neg = out[+1.0], out[-1.0]

    # class-odd: changes sign.  Both are saturated, so the test has teeth: the
    # two means are near +0.87 and -0.87 rather than both near zero.
    assert pos[0].mean() > 0.6 and neg[0].mean() < -0.6
    assert pos[0].mean() == pytest.approx(-neg[0].mean(), abs=0.12)

    # class-even: unchanged.  The two classes swap roles, so the balance of one
    # under +b is the balance of the other under -b.
    p, n = pos[1], neg[1]
    assert p["B_eta_A"].mean() == pytest.approx(n["B_eta_B"].mean(), abs=0.15)
    assert p["B_eta_B"].mean() == pytest.approx(n["B_eta_A"].mean(), abs=0.15)
    assert p["R_wmu"].mean() == pytest.approx(n["R_wmu"].mean(), abs=0.15)


def test_presets_are_ordered_by_cost():
    q, m, f = (get_preset(k) for k in ("quick", "medium", "full"))
    assert q.sweep.n_points < m.sweep.n_points < f.sweep.n_points
    assert q.model.n_steps() < m.model.n_steps()
    assert m.model.n_steps() == f.model.n_steps()  # same physics, finer grid


def test_an_unknown_component_is_rejected_at_construction():
    with pytest.raises(ValueError, match="component must be one of"):
        ModelConfig(component="q")


def test_field_kwargs_puts_the_strength_on_the_swept_component():
    model = ModelConfig(component="b", background=(0.2, 0.0, 0.0, 0.0))
    kw = model.field_kwargs(0.7)
    assert kw == {"a": 0.2, "b": 0.7, "c": 0.0, "p": 0.0}
    grid = model.field_kwargs(np.array([0.0, 1.0]))
    np.testing.assert_allclose(grid["b"], [0.0, 1.0])


def test_every_component_has_a_channel_and_they_are_distinct():
    """The pairing the package is about: one component, one channel, no overlap."""
    from credfield.fields import COMPONENTS
    from credfield.order_params import CHANNEL_NAMES, CHANNEL_OF
    assert set(CHANNEL_OF) == set(COMPONENTS)
    assert sorted(CHANNEL_OF.values()) == sorted(CHANNEL_NAMES)


@pytest.mark.parametrize("comp", ["a", "b", "c", "p"])
def test_the_figures_follow_the_swept_component(comp, tmp_path, monkeypatch):
    """A sweep in ``b`` must not plot ``R_stat`` and call it empty.

    This directory sweeps ``b``, so the failure mode is live rather than
    hypothetical: ``R_stat`` under a pure credulity field is not zero but
    ``-R_cred/(N-1)``, which would read as a faint real signal.

    The composite and the cut pick their channel from the component, so this
    walks all four through the plotting path on a grid too small to mean
    anything, checking only that each produces a figure naming the right channel.
    """
    import credfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    monkeypatch.setattr("credfield.plotting.FIGURE_DIR", tmp_path / "figures")

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import cred_asymmetry as dp
    from credfield.order_params import CHANNEL_OF
    from credfield.plotting import set_component, use_style

    use_style("iclr")
    set_component(comp)
    model = ModelConfig(n_agents=8, n_dim=5, n_issues=2, component=comp,
                        interactions_per_channel=4.0)
    cfg = SweepConfig(n_s=4, n_f=3, s_range=default_s_range(comp),
                      batch_size=64, n_workers=1)
    data = sweep(model, cfg, use_cache=False, verbose=False)

    assert dp._channel_row(comp)[0].startswith(CHANNEL_OF[comp])
    for fn, name in ((dp.figure_channels, "chan"), (dp.figure_map, "map"),
                     (dp.figure_cut, "cut")):
        out = fn(data, "iclr", name=f"{name}_{comp}")
        assert out.exists() and out.stat().st_size > 0


# --- striping ------------------------------------------------------------

def _phase_module():
    import pathlib as _p
    import sys
    sys.path.insert(0, str(_p.Path(__file__).resolve().parent.parent / "scripts"))
    import cred_asymmetry
    return cred_asymmetry


@pytest.mark.parametrize("n_f,n_strips", [(200, 5), (64, 5), (10, 3), (4, 4), (3, 7)])
def test_strips_partition_the_prevalence_axis_exactly(n_f, n_strips):
    """Concatenating the strips must give back the grid, not a kinked one.

    The tempting implementation slices ``(0, 0.2), (0.2, 0.4), ...`` off the unit
    interval and runs ``linspace`` inside each, which duplicates every boundary
    row and spaces the rows differently within a strip than between strips.  This
    reads each band's endpoints off the full axis instead, so the union is the
    full axis to floating point.
    """
    dp = _phase_module()
    cfg = SweepConfig(n_s=7, n_f=n_f, f_range=(0.0, 1.0))
    bands = list(dp._strips(cfg, n_strips))
    assert len(bands) == min(n_strips, n_f)

    rebuilt = np.concatenate([np.linspace(*f_range, n) for _, f_range, n, _ in bands])
    np.testing.assert_allclose(rebuilt, np.linspace(0.0, 1.0, n_f), atol=1e-12)

    # contiguous, in order, and covering every row exactly once
    assert sum(n for _, _, n, _ in bands) == n_f
    offsets = [off for _, _, _, off in bands]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0


def test_strips_do_not_share_seeds():
    """Otherwise every strip draws the same societies and the plane is banded.

    ``sweep`` seeds batch ``b`` as ``seed + offset_within_this_sweep``, and that
    offset restarts at zero for each strip, so strips at a common base seed are
    the same experiment repeated.  The offset makes the seed a function of a
    society's position in the plane; the check is that no strip's seed range can
    reach into the next strip's.
    """
    dp = _phase_module()
    cfg = SweepConfig(n_s=200, n_f=200, f_range=(0.0, 1.0), seed=1234)
    bands = list(dp._strips(cfg, 5))
    seeds = [cfg.seed + off * cfg.n_s for _, _, _, off in bands]
    assert len(set(seeds)) == len(bands)
    # a strip consumes at most (rows * n_s) consecutive seeds, so consecutive
    # bases must be at least that far apart
    for (base, (_, _, n, _)), nxt in zip(zip(seeds, bands), seeds[1:]):
        assert nxt - base >= n * cfg.n_s


def test_a_striped_sweep_is_the_same_grid_as_an_unstriped_one(tmp_path, monkeypatch):
    """End to end on a grid too small to mean anything: shapes and axes agree."""
    import credfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    dp = _phase_module()
    model = ModelConfig(n_agents=8, n_dim=5, n_issues=2, component="b",
                        interactions_per_channel=4.0)
    cfg = SweepConfig(n_s=3, n_f=6, s_range=(0.0, 1.0), batch_size=64, n_workers=1)

    whole = sweep(model, cfg, use_cache=False, verbose=False)
    striped = dp._run_striped(model, cfg, 3, use_cache=False)

    assert set(striped) == set(whole)
    np.testing.assert_allclose(striped["s"], whole["s"], atol=1e-12)
    np.testing.assert_allclose(striped["f"], whole["f"], atol=1e-12)
    for name in ORDER_PARAM_NAMES:
        assert striped[name].shape == whole[name].shape, name
        assert np.isfinite(striped[name]).all(), name
    # three strips, three separate cache files, and none of them the whole grid
    assert len(list(tmp_path.glob("*strip*.npz"))) == 3


def test_a_killed_strip_is_the_only_work_lost(tmp_path, monkeypatch):
    """The point of striping: finished strips reload, the rest re-runs.

    Simulated by running two strips, deleting one cache file, and checking that
    the surviving one is served from cache while the deleted one is recomputed.
    """
    import credfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    dp = _phase_module()
    model = ModelConfig(n_agents=8, n_dim=5, n_issues=2, component="b",
                        interactions_per_channel=4.0)
    cfg = SweepConfig(n_s=3, n_f=4, s_range=(0.0, 1.0), batch_size=64, n_workers=1)

    first = dp._run_striped(model, cfg, 2, use_cache=True)
    files = sorted(tmp_path.glob("*strip*.npz"))
    assert len(files) == 2
    kept = files[0].read_bytes()
    files[1].unlink()

    second = dp._run_striped(model, cfg, 2, use_cache=True)
    assert files[0].read_bytes() == kept          # untouched, served from cache
    assert files[1].exists()                      # and the lost one is back
    np.testing.assert_allclose(second["f"], first["f"], atol=1e-12)
    # the reloaded strip is bit-identical; the recomputed one is too, since the
    # seed is a function of position rather than of when it ran
    for name in ORDER_PARAM_NAMES:
        np.testing.assert_allclose(second[name], first[name], atol=1e-12)
