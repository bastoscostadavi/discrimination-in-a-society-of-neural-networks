"""Campaign execution: chunking, shard naming, and resumption.

The shard-naming test exists because the bug it pins actually happened: the
first production run finished 990 societies short after a pause and resume,
because shards were named by loop index and the resumed run's first shards
overwrote the originals.
"""

import numpy as np
import pytest

from socsim.config import GridSpec, ModelConfig, RunSpec
from socsim.runner import _shard_name, enumerate_keys, run_campaign
from socsim import store

SPEC = RunSpec(
    name="t",
    model=ModelConfig(n_agents=10, n_dim=6, n_issues=3, interactions_per_channel=2),
    grid=GridSpec(n_d=3, n_fd=3, n_init=3),
    n_permutations=0,
)


def _rows(pid, dis, init):
    return [{"point_id": pid, "replicate_dis": dis, "replicate_init": init}]


def test_shard_name_depends_on_contents_not_position():
    a = _shard_name(_rows("aa", 0, 0))
    b = _shard_name(_rows("bb", 0, 0))
    assert a != b
    assert _shard_name(_rows("aa", 0, 0)) == a  # deterministic


def test_shard_name_distinguishes_replicates():
    assert _shard_name(_rows("aa", 0, 0)) != _shard_name(_rows("aa", 0, 1))
    assert _shard_name(_rows("aa", 0, 0)) != _shard_name(_rows("aa", 1, 0))


def test_interrupted_run_loses_nothing_on_resume(tmp_path):
    """The regression. A partial run, then a resume, must total the full count."""
    keys, _ = enumerate_keys(SPEC)
    n_total = len(keys)

    # Simulate an interruption: run one chunk's worth, then resume.
    run_campaign(SPEC, n_workers=1, chunk_size=2, data_dir=tmp_path, verbose=False)
    first = len(list((tmp_path / "shards" / SPEC.name).glob("*.npz")))
    assert first > 1

    # Drop some shards to mimic work lost to a kill, then resume.
    shards = sorted((tmp_path / "shards" / SPEC.name).glob("*.npz"))
    for s in shards[:2]:
        s.unlink()
    path = run_campaign(SPEC, n_workers=1, chunk_size=2, data_dir=tmp_path, verbose=False)

    res = store.load(path)
    assert res["obs"].shape[0] == n_total

    counts = {}
    for pid in res["point_id"]:
        counts[pid] = counts.get(pid, 0) + 1
    assert set(counts.values()) == {SPEC.grid.n_init}


def test_rerunning_a_complete_campaign_is_a_noop(tmp_path):
    run_campaign(SPEC, n_workers=1, chunk_size=4, data_dir=tmp_path, verbose=False)
    n1 = len(list((tmp_path / "shards" / SPEC.name).glob("*.npz")))
    path = run_campaign(SPEC, n_workers=1, chunk_size=4, data_dir=tmp_path, verbose=False)
    n2 = len(list((tmp_path / "shards" / SPEC.name).glob("*.npz")))
    assert n1 == n2
    keys, _ = enumerate_keys(SPEC)
    assert store.load(path)["obs"].shape[0] == len(keys)
