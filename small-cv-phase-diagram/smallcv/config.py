"""Configuration for the small-C,V phase-diagram calculation."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of one microscopic society."""

    n_agents: int = 40
    n_dim: int = 30
    n_issues: int = 5
    case: int = 6
    initial_c: float = 0.05
    initial_v: float = 0.05
    interactions_per_channel_at_c1: float = 250.0
    dynamics: str = "small_cv"
    literal_draft_sign: bool = False
    class_indicator: str = "pm1"
    literal_norm: bool = False
    seed: int = 20260812

    @property
    def alpha(self):
        return self.n_issues / self.n_dim

    @property
    def ratio_v_over_c(self):
        return self.initial_v / self.initial_c

    def n_steps(self):
        """Scale run time as 1/c because the leading weight step is O(C)."""
        n_pairs = self.n_agents * (self.n_agents - 1)
        return int(round(self.interactions_per_channel_at_c1 * n_pairs / self.initial_c))

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class SweepConfig:
    """Grid and batching for the external control plane (d, f_d)."""

    n_d: int = 41
    n_fd: int = 41
    d_range: tuple = (-1.0, 1.0)
    fd_range: tuple = (0.0, 1.0)
    batch_size: int = 256
    n_repeats: int = 3
    seed: int = 20260812

    def grids(self):
        import numpy as np

        return (
            np.linspace(*self.d_range, self.n_d),
            np.linspace(*self.fd_range, self.n_fd),
        )

    def with_(self, **kw):
        return replace(self, **kw)
