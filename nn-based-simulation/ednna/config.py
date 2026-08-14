"""Model and run configuration.

The source draft states no simulation parameters, so every number here is a
choice.  Their provenance is recorded in ``docs/model.md``; the short version:

* ``K = 30`` is recovered from the draft, not chosen: its agenda-complexity
  figure uses ``alpha = P/K`` values {0.03, 0.17, 0.23, 0.33, 0.50, 0.67, 1.67,
  3.33, 333.33}, which are exactly ``P/30`` for ``P`` in {1, 5, 7, 10, 15, 20,
  50, 100, 10^4}.  The LLM protocol's "thirty issues" agrees.
* ``N``, the interaction count, and the small/large agenda sizes are fixed by
  ``scripts/calibrate.py`` against qualitative features of the draft's figures.
* The three presets trade resolution for time.  ``quick`` is for development,
  ``medium`` for review, ``full`` for the final figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["ModelConfig", "SweepConfig", "PRESETS", "get_preset"]


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the society itself."""

    n_agents: int = 40
    n_dim: int = 30  # K, recovered from the draft
    n_issues: int = 5  # P, the "small agenda"
    case: int = 6  # discrimination case of Table I
    #: Total interactions per society, expressed per ordered pair of agents so
    #: that the number is meaningful independently of N.
    #:
    #: The dynamics anneals rather than reaching a stationary state, so this is a
    #: real parameter and the draft omits it (its text has a literal
    #: ``Delta t = ????``).  500 is the value calibrated by
    #: ``scripts/calibrate.py`` against the draft's published trajectory
    #: endpoints; the residual minimum is flat-bottomed between 250 and 1000 and
    #: 500 wins whether or not the ambiguously digitised curves are included.
    interactions_per_channel: float = 500.0
    literal_draft_sign: bool = False
    class_indicator: str = "pm1"
    literal_norm: bool = False
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
    """Resolution and batching of a ``(d, f_d)`` phase-diagram sweep."""

    n_d: int = 96
    n_fd: int = 96
    d_range: tuple = (-1.0, 1.0)
    fd_range: tuple = (0.0, 1.0)
    #: societies per vectorised batch; memory scales as batch * N * K^2
    batch_size: int = 1024
    n_workers: int = 10
    seed: int = 20260812
    #: repeats per grid point, averaged.  The draft's maps are visibly
    #: single-realisation, so 1 is the faithful choice.
    n_repeats: int = 1

    @property
    def n_points(self):
        return self.n_d * self.n_fd * self.n_repeats

    def grids(self):
        d = np.linspace(*self.d_range, self.n_d)
        fd = np.linspace(*self.fd_range, self.n_fd)
        return d, fd

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Preset:
    name: str
    model: ModelConfig
    sweep: SweepConfig
    #: agenda sizes for the "small agenda" / "large agenda" panel rows
    p_small: int = 5
    p_large: int = 100
    #: society size and number of independent societies for the polarisation
    #: figure, which is a distribution over pairs and so wants many of them
    polarisation_agents: int = 400
    polarisation_runs: int = 8
    #: agenda sizes for the balance-trajectory figure (alpha = P/30)
    trajectory_issues: tuple = (1, 5, 7, 10, 15, 20, 50, 100, 10000)
    n_trajectory_repeats: int = 8
    n_trajectory_samples: int = 40


# The three presets differ only in resolution and in how many repeats the
# trajectory figure averages; the physics -- N, K, and the calibrated
# interaction count -- is the same in `medium` and `full`.  `quick` also shortens
# the runs, so its numbers are not comparable with the other two.
#
# Measured on 10 cores (14 available, ~0.2M agent-updates/s/core once the workers
# are competing for memory bandwidth):
#   quick    ~2 min per sweep
#   medium   ~45 min per sweep, so ~1.5 h for the two agenda sizes
#   full     200x200, matching the draft: ~2 h per sweep, so ~4.5 h for the two
#            agenda sizes. `ModelConfig(dtype="float32")` roughly halves that and
#            changes no order parameter by more than 0.1 (tested)
PRESETS = {
    "quick": Preset(
        name="quick",
        model=ModelConfig(n_agents=24, interactions_per_channel=125.0),
        sweep=SweepConfig(n_d=32, n_fd=32, batch_size=512, n_workers=8),
        polarisation_agents=60,
        polarisation_runs=4,
        n_trajectory_repeats=2,
        n_trajectory_samples=24,
    ),
    "medium": Preset(
        name="medium",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_d=64, n_fd=64, batch_size=512, n_workers=10),
        polarisation_agents=150,
        polarisation_runs=8,
        n_trajectory_repeats=4,
        n_trajectory_samples=32,
    ),
    # 200x200 matches the source draft: its most finely speckled panel resolves
    # ~170-200 constant-colour blocks per row in the 273-pixel raster embedded in
    # its PDF, which is the grid it sampled.
    "full": Preset(
        name="full",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_d=200, n_fd=200, batch_size=1024, n_workers=10),
        n_trajectory_repeats=8,
        n_trajectory_samples=40,
    ),
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
