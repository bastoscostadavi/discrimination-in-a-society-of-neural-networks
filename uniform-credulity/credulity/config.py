"""Model and run configuration.

Parameters are those of the main line of work, unchanged, so that a sweep here
and a sweep there differ only in the field.  Their provenance is recorded in the
paper's parameter appendix; the short version:

* ``K = 30`` is recovered rather than chosen, from the agenda-complexity figure
  of the companion manuscript.
* ``N = 40`` and the agenda size are choices; total work scales as ``N^3``.
* ``interactions_per_channel = 500`` is calibrated against the companion
  manuscript's published trajectory endpoints.  The dynamics anneals rather than
  reaching a stationary state, so the measurement time is a real parameter.

There is no ``component`` setting here, unlike in ``../directional-prejudice/``:
this package sweeps ``a`` and only ``a``.  The strength axis is swept over both
signs, because the two halves are physically distinct -- ``a > 0`` is credulity
and ``a < 0`` suspicion -- and neither is the relabelling of the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["ModelConfig", "SweepConfig", "PRESETS", "get_preset"]


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the society itself."""

    n_agents: int = 40
    n_dim: int = 30  # K, recovered from the companion manuscript
    n_issues: int = 5  # P, the "simple agenda"
    #: Total interactions per society, per ordered pair of agents so that the
    #: number is meaningful independently of N.
    interactions_per_channel: float = 500.0
    dtype: str = "float64"
    shared_schedule: bool = True

    @property
    def alpha(self):
        """Agenda complexity ``alpha = P / K``."""
        return self.n_issues / self.n_dim

    def n_steps(self):
        N = self.n_agents
        return int(round(self.interactions_per_channel * N * (N - 1)))

    def numpy_dtype(self):
        return np.float32 if self.dtype == "float32" else np.float64

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class SweepConfig:
    """Resolution and batching of an ``(a, f_a)`` sweep."""

    n_a: int = 96
    n_f: int = 96
    a_range: tuple = (-1.0, 1.0)
    f_range: tuple = (0.0, 1.0)
    #: societies per vectorized batch; memory scales as batch * N * K^2
    batch_size: int = 512
    n_workers: int = 10
    seed: int = 20260821
    #: repeats per grid point, averaged.  The published maps are visibly
    #: single-realization, so 1 is the faithful choice.
    n_repeats: int = 1

    @property
    def n_points(self):
        return self.n_a * self.n_f * self.n_repeats

    def grids(self):
        a = np.linspace(*self.a_range, self.n_a)
        f = np.linspace(*self.f_range, self.n_f)
        return a, f

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Preset:
    name: str
    model: ModelConfig
    sweep: SweepConfig
    #: society size and number of independent societies for the point table,
    #: which is a statement about blocks of a matrix and so wants a population
    #: big enough for the blocks to have small error bars
    demo_agents: int = 40
    demo_runs: int = 4
    #: How many horizontal bands to cut the plane into.  A sweep caches once, at
    #: the end, so this is the granularity at which an interrupted run loses
    #: work; only the long presets need it, and 1 is an ordinary single sweep.
    n_strips: int = 1
    #: prevalence at which the mixed-population table is taken.  Half is the
    #: value with the most pairs in the two off-diagonal blocks, which is where
    #: the emergent effect would show up if there is one.
    demo_fraction: float = 0.5


# Resolution only; the physics is the same in `medium` and `full`.  `quick` also
# shortens the runs, so its numbers are not comparable with the other two.
# Timings measured on 10 cores, and they scale with the number of pixels:
#   quick    ~1 min
#   medium   ~25 min
#   full     ~4 h.  `ModelConfig(dtype="float32")` roughly halves that.
#
# `full` is the resolution of the paper's own phase diagrams: 200x200 at N = 40
# and the calibrated Delta t = 500.  It is not the default, because a stray
# invocation of it costs an afternoon; ask for it explicitly.
PRESETS = {
    "quick": Preset(
        name="quick",
        model=ModelConfig(n_agents=24, interactions_per_channel=125.0),
        sweep=SweepConfig(n_a=32, n_f=32, batch_size=512, n_workers=8),
        demo_agents=24,
        demo_runs=2,
    ),
    "medium": Preset(
        name="medium",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_a=64, n_f=64, batch_size=512, n_workers=10),
        demo_agents=40,
        demo_runs=4,
    ),
    # Five strips of forty rows: about forty-five minutes each, which is the
    # most work a kill can cost.
    "full": Preset(
        name="full",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_a=200, n_f=200, batch_size=1024, n_workers=10),
        n_strips=5,
        demo_agents=40,
        demo_runs=8,
    ),
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
