"""Sweep plumbing: the mirror symmetry that halves it, caching, and shapes."""

import pathlib

import numpy as np
import pytest

from dirfield.config import ModelConfig, SweepConfig, default_s_range, get_preset
from dirfield.order_params import ORDER_PARAM_NAMES
from dirfield.sweep import cache_path, sweep


@pytest.fixture
def tiny():
    """A grid small enough to run in a test, in one process."""
    model = ModelConfig(n_agents=8, n_dim=5, n_issues=2,
                        component="c", interactions_per_channel=6.0)
    cfg = SweepConfig(n_s=3, n_f=2, s_range=(0.0, 1.0), batch_size=64, n_workers=1)
    return model, cfg


def test_sweep_returns_every_order_parameter_on_the_grid(tiny, tmp_path, monkeypatch):
    import dirfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    model, cfg = tiny
    out = sweep(model, cfg, use_cache=False, verbose=False)
    assert set(out) == set(ORDER_PARAM_NAMES) | {"s", "f"}
    for name in ORDER_PARAM_NAMES:
        assert out[name].shape == (cfg.n_f, cfg.n_s), name
    np.testing.assert_allclose(out["s"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["f"], [0.0, 1.0])


def test_a_sweep_is_cached_and_reloaded(tiny, tmp_path, monkeypatch):
    import dirfield.sweep as sw
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
    assert a != cache_path(model.with_(component="b"), cfg)
    assert a != cache_path(model.with_(n_agents=9), cfg)
    assert a != cache_path(model, cfg.with_(n_s=4))
    assert a != cache_path(model, cfg, tag="other")
    assert "c_P2" in a.name  # the component is legible in the filename


def test_default_ranges_reflect_which_signs_are_distinct():
    for k in ("a", "p"):
        assert default_s_range(k) == (-1.0, 1.0)
    for k in ("b", "c"):
        assert default_s_range(k) == (0.0, 1.0)


@pytest.mark.slow
def test_negating_a_directional_field_is_relabelling_the_classes():
    """Why half the plane is not swept: the two signs are the same experiment.

    Negating ``c`` exchanges the roles of A and B, which maps the ensemble to
    itself.  So every class-odd quantity flips sign and every class-even one is
    unchanged -- checked here at one point rather than assumed, since it is the
    argument for sweeping ``[0, 1]`` instead of ``[-1, 1]``.
    """
    from dirfield.order_params import measure, trust_channels
    from dirfield.society import SocietyBatch

    kw = dict(n_agents=16, n_dim=8, n_issues=3, f=1.0, seed=11)
    pos = SocietyBatch(c=+0.8, **kw)
    neg = SocietyBatch(c=-0.8, **kw)
    steps = int(40 * 16 * 15)
    pos.run(steps)
    neg.run(steps)

    # class-odd: changes sign
    assert trust_channels(pos)["R_stat"][0] == pytest.approx(
        -trust_channels(neg)["R_stat"][0], abs=0.02)
    # class-even: does not.  The two classes swap roles, so the balance of one
    # is the balance of the other.
    p, n = measure(pos), measure(neg)
    assert p["B_eta_A"][0] == pytest.approx(n["B_eta_B"][0], abs=0.05)
    assert p["B_eta_B"][0] == pytest.approx(n["B_eta_A"][0], abs=0.05)
    assert p["R_wmu"][0] == pytest.approx(n["R_wmu"][0], abs=0.05)


def test_presets_are_ordered_by_cost():
    q, m, f = (get_preset(k) for k in ("quick", "medium", "full"))
    assert q.sweep.n_points < m.sweep.n_points < f.sweep.n_points
    assert q.model.n_steps() < m.model.n_steps()
    assert m.model.n_steps() == f.model.n_steps()  # same physics, finer grid


def test_an_unknown_component_is_rejected_at_construction():
    with pytest.raises(ValueError, match="component must be one of"):
        ModelConfig(component="q")


def test_field_kwargs_puts_the_strength_on_the_swept_component():
    model = ModelConfig(component="c", background=(0.2, 0.0, 0.0, 0.0))
    kw = model.field_kwargs(0.7)
    assert kw == {"a": 0.2, "b": 0.0, "c": 0.7, "p": 0.0}
    grid = model.field_kwargs(np.array([0.0, 1.0]))
    np.testing.assert_allclose(grid["c"], [0.0, 1.0])


def test_every_component_has_a_channel_and_they_are_distinct():
    """The pairing the package is about: one component, one channel, no overlap."""
    from dirfield.fields import COMPONENTS
    from dirfield.order_params import CHANNEL_NAMES, CHANNEL_OF
    assert set(CHANNEL_OF) == set(COMPONENTS)
    assert sorted(CHANNEL_OF.values()) == sorted(CHANNEL_NAMES)


@pytest.mark.parametrize("comp", ["a", "b", "c", "p"])
def test_the_figures_follow_the_swept_component(comp, tmp_path, monkeypatch):
    """A sweep in ``b`` must not plot ``R_stat`` and call it empty.

    The composite and the cut pick their channel from the component, so this
    walks all four through the plotting path on a grid too small to mean
    anything, checking only that each produces a figure naming the right channel.
    """
    import dirfield.sweep as sw
    monkeypatch.setattr(sw, "DATA_DIR", tmp_path)
    monkeypatch.setattr("dirfield.plotting.FIGURE_DIR", tmp_path / "figures")

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import directional_phase as dp
    from dirfield.order_params import CHANNEL_OF
    from dirfield.plotting import set_component, use_style

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
