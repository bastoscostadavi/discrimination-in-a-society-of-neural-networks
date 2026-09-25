import numpy as np
import pytest

from ednna import sweep as sweep_module
from ednna.config import ModelConfig, SweepConfig


def tiny_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "DATA_DIR", tmp_path)
    model = ModelConfig(
        n_agents=4,
        n_dim=3,
        n_issues=2,
        interactions_per_channel=0.0,
    )
    cfg = SweepConfig(
        n_d=3,
        n_fd=2,
        batch_size=2,
        n_workers=1,
        n_repeats=2,
        seed=17,
    )
    return model, cfg


def test_shards_merge_to_the_same_result_as_a_full_sweep(tmp_path, monkeypatch):
    model, cfg = tiny_run(tmp_path, monkeypatch)
    expected = sweep_module.sweep(
        model, cfg, tag="test", use_cache=False, verbose=False
    )
    sweep_module.cache_path(model, cfg, "test").unlink()

    for shard_index in range(3):
        sweep_module.sweep_shard(
            model,
            cfg,
            tag="test",
            shard_index=shard_index,
            n_shards=3,
            verbose=False,
        )
    actual = sweep_module.merge_sweep_shards(
        model, cfg, tag="test", n_shards=3, verbose=False
    )

    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


def test_merge_fails_loudly_when_a_shard_is_missing(tmp_path, monkeypatch):
    model, cfg = tiny_run(tmp_path, monkeypatch)
    sweep_module.sweep_shard(
        model, cfg, tag="test", shard_index=0, n_shards=2, verbose=False
    )

    with pytest.raises(FileNotFoundError, match="missing sweep shard"):
        sweep_module.merge_sweep_shards(
            model, cfg, tag="test", n_shards=2, verbose=False
        )


def test_lightweight_snapshots_are_chunked_and_resumable(tmp_path, monkeypatch):
    model, cfg = tiny_run(tmp_path, monkeypatch)
    ordinary = sweep_module.sweep(
        model, cfg, tag="plain", use_cache=False, verbose=False
    )
    expected = sweep_module.sweep(
        model, cfg, tag="snap", use_cache=False, verbose=False,
        save_snapshots=True,
    )
    for key in ordinary:
        np.testing.assert_allclose(expected[key], ordinary[key])

    directory = sweep_module.snapshot_cache_dir(model, cfg, "snap")
    chunks = sorted(directory.glob("batch-*.npz"))
    assert (directory / "manifest.json").exists()
    assert len(chunks) == 6

    with np.load(chunks[0]) as chunk:
        assert chunk["w"].shape == (2, model.n_agents, model.n_dim)
        assert chunk["eta"].shape == (2, model.n_agents, model.n_agents)
        assert chunk["X"].shape == (2, model.n_issues, model.n_dim)
        assert chunk["discriminates"].shape == (2, model.n_agents)
        assert chunk["w"].dtype == np.float32
        assert chunk["eta"].dtype == np.float32

    sweep_module.cache_path(model, cfg, "snap").unlink()

    def should_not_run(_job):
        raise AssertionError("completed snapshot chunks should be reused")

    monkeypatch.setattr(sweep_module, "_run_batch_with_snapshot", should_not_run)
    resumed = sweep_module.sweep(
        model, cfg, tag="snap", verbose=False, save_snapshots=True
    )
    for key in expected:
        np.testing.assert_allclose(resumed[key], expected[key])
