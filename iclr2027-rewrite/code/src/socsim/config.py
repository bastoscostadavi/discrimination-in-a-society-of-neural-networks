"""Model and campaign configuration.

Every number here is either recovered from the source model, calibrated against
it, or an explicit choice; none is implicit.  The provenance of each is recorded
in the appendix table and reproduced in the comments below.

* ``n_dim = 30`` is **recovered**, not chosen.  The source reports balance
  trajectories at ``alpha = P/K`` in {0.03, 0.17, 0.23, 0.33, 0.50, 0.67, 1.67,
  3.33, 333.33}, which are exactly ``P/30`` for ``P`` in {1, 5, 7, 10, 15, 20,
  50, 100, 10^4}.
* ``interactions_per_channel = 500`` is **calibrated** against the source's
  published trajectory endpoints; the residual is flat-bottomed between 250 and
  1000.  Because the dynamics anneals rather than reaching a stationary state,
  this is a real parameter of the experiment and not a convergence threshold.
* ``n_agents = 40`` is a **choice**: large enough that the order parameters are
  not dominated by finite-size noise, small enough to afford a replicated grid.

Grid resolutions are deliberately **odd**.  ``np.linspace(-1, 1, 48)`` does not
contain ``d = 0``, so the no-discrimination baseline --- the column that
calibrates every classification threshold --- would not be sampled at all.  Odd
counts also make each coarse grid an exact stride-subgrid of the fine one, which
is what lets a control be paired point-by-point with its baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["ModelConfig", "GridSpec", "RunSpec", "PRESETS", "get_preset"]


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the society itself."""

    n_agents: int = 40
    n_dim: int = 30  # K, recovered from the source
    n_issues: int = 5  # P, the simple agenda
    interactions_per_channel: float = 500.0
    class_balance: float = 0.5
    rule: str = "entropic"
    freeze: tuple = ()
    dtype: str = "float64"
    step_size: float = None  # required for the non-entropic rules
    margin: float = 0.0

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
class GridSpec:
    """A grid over ``(d, f_d)``, with replication.

    ``n_d`` and ``n_fd`` should be odd (see the module docstring).  ``n_init`` is
    the number of independent initial conditions and schedules per point ---
    the replicates the uncertainty is estimated over --- and ``n_disorder`` the
    number of quenched environments, normally 1 except for the replica-overlap
    diagnostic.
    """

    n_d: int = 49
    n_fd: int = 49
    d_range: tuple = (-1.0, 1.0)
    fd_range: tuple = (0.0, 1.0)
    n_init: int = 24
    n_disorder: int = 1

    def __post_init__(self):
        for name, n in (("n_d", self.n_d), ("n_fd", self.n_fd)):
            if n % 2 == 0:
                raise ValueError(
                    f"{name}={n} is even, so d=0 is not on the grid; use an odd count"
                )

    def axes(self):
        return (
            np.linspace(*self.d_range, self.n_d),
            np.linspace(*self.fd_range, self.n_fd),
        )

    def subgrid(self, stride):
        """A coarser grid sharing every point with this one.

        Controls run on a subgrid so each of their societies pairs with a
        baseline society at the identical ``(d, f_d)``.
        """
        n_d = (self.n_d - 1) // stride + 1
        n_fd = (self.n_fd - 1) // stride + 1
        return replace(self, n_d=n_d, n_fd=n_fd)

    @property
    def n_points(self):
        return self.n_d * self.n_fd

    @property
    def n_societies(self):
        return self.n_points * self.n_init * self.n_disorder

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class RunSpec:
    """One experiment: what to simulate, and under what perturbation."""

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    grid: GridSpec = field(default_factory=GridSpec)
    field_kind: str = "class"
    case: int = 6
    convention: str = "algorithmic"
    #: Baseline and controls that share this string share initial conditions and
    #: interaction schedules, which is what makes their differences paired.
    crn_group: str = "main"
    master: int = 20260813
    n_permutations: int = 200

    def with_(self, **kw):
        return replace(self, **kw)


# Execution parameters (worker count, chunk size) are deliberately NOT part of
# RunSpec: they must never enter the identity of a result.  The reference
# implementation hashed them into its cache key, so changing the worker count
# invalidated tens of thousands of simulated societies.
PRESETS = {
    # Development: minutes, coarse, not comparable with the others.
    "quick": RunSpec(
        name="quick",
        model=ModelConfig(n_agents=24, interactions_per_channel=125.0),
        grid=GridSpec(n_d=9, n_fd=9, n_init=4),
    ),
    # Review resolution.
    "medium": RunSpec(
        name="medium",
        model=ModelConfig(n_agents=40),
        grid=GridSpec(n_d=25, n_fd=25, n_init=12),
    ),
    # The production campaign.
    #
    # float64 deliberately.  The reference implementation offered float32 as a
    # roughly 2x speedup, and it is not one here: measured on this machine with
    # the workers saturated, float64 runs at 2.99 societies/s against float32's
    # 2.39 -- float32 is 25% *slower*.  ``scipy.special.ndtr`` computes in double
    # regardless, so the single-precision path pays for conversions on every
    # call without ever saving the memory bandwidth it was supposed to.  Using
    # float64 is therefore both faster and one fewer numerical risk to defend.
    "full": RunSpec(
        name="full",
        model=ModelConfig(n_agents=40),
        grid=GridSpec(n_d=49, n_fd=49, n_init=24),
    ),
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
