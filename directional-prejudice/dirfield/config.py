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

What is new is :attr:`ModelConfig.component`: which of the four field components
of :mod:`dirfield.fields` the strength axis of a sweep drives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .fields import COMPONENTS

__all__ = ["ModelConfig", "SweepConfig", "PRESETS", "get_preset",
           "default_s_range"]


def default_s_range(component):
    """The strength range worth sweeping for one field component.

    ``a`` and ``p`` need both signs: the two halves are physically distinct.  For
    ``p`` the sign is the difference between favouring one's own class and
    favouring the other, which the main line of work finds to be a discriminatory
    phase against a frustrated one; for ``a`` it is credulity against suspicion.

    ``b`` and ``c`` need only one.  Negating either is exactly the relabelling
    ``A <-> B``, which maps the ensemble to itself since the two classes are the
    same size and are otherwise identical, so the negative half is the mirror
    image of the positive one and costs half a sweep to learn nothing.  Every
    class-odd quantity changes sign across it and every class-even one does not
    (``tests/test_sweep.py`` checks this on a small grid rather than asserting
    it).
    """
    return (-1.0, 1.0) if component in ("a", "p") else (0.0, 1.0)


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the society itself."""

    n_agents: int = 40
    n_dim: int = 30  # K, recovered from the companion manuscript
    n_issues: int = 5  # P, the "simple agenda"
    #: which field component the sweep's strength axis drives
    component: str = "c"
    #: the other three components, held fixed across the sweep
    background: tuple = (0.0, 0.0, 0.0, 0.0)
    #: Total interactions per society, per ordered pair of agents so that the
    #: number is meaningful independently of N.
    interactions_per_channel: float = 500.0
    dtype: str = "float64"
    shared_schedule: bool = True

    def __post_init__(self):
        if self.component not in COMPONENTS:
            raise ValueError(
                f"component must be one of {COMPONENTS}, got {self.component!r}"
            )

    @property
    def alpha(self):
        """Agenda complexity ``alpha = P / K``."""
        return self.n_issues / self.n_dim

    def n_steps(self):
        N = self.n_agents
        return int(round(self.interactions_per_channel * N * (N - 1)))

    def numpy_dtype(self):
        return np.float32 if self.dtype == "float32" else np.float64

    def field_kwargs(self, strength):
        """``dict(a=..., b=..., c=..., p=...)`` for a given strength.

        The swept component takes ``strength``; the others take their background
        value.  ``strength`` may be an array, in which case every entry of the
        returned dict broadcasts against it.
        """
        kw = dict(zip(COMPONENTS, self.background))
        kw[self.component] = strength
        return kw

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class SweepConfig:
    """Resolution and batching of a ``(strength, fraction)`` sweep."""

    n_s: int = 96
    n_f: int = 96
    s_range: tuple = (-1.0, 1.0)
    f_range: tuple = (0.0, 1.0)
    #: societies per vectorized batch; memory scales as batch * N * K^2
    batch_size: int = 512
    n_workers: int = 10
    seed: int = 20260821
    #: repeats per grid point, averaged
    n_repeats: int = 1

    @property
    def n_points(self):
        return self.n_s * self.n_f * self.n_repeats

    def grids(self):
        s = np.linspace(*self.s_range, self.n_s)
        f = np.linspace(*self.f_range, self.n_f)
        return s, f

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Preset:
    name: str
    model: ModelConfig
    sweep: SweepConfig
    #: agenda sizes either side of alpha = 1, where the polarization order reverses
    p_small: int = 5
    p_large: int = 100
    #: society size and number of independent societies for the demonstration
    #: table, which is a statement about blocks of a matrix and so wants a
    #: population big enough for the blocks to have small error bars
    demo_agents: int = 40
    demo_runs: int = 4


# Resolution only; the physics is the same in `medium` and `full`.  `quick` also
# shortens the runs, so its numbers are not comparable with the other two.
# Timings measured on 10 cores for the main line of work, which this matches.
PRESETS = {
    "quick": Preset(
        name="quick",
        model=ModelConfig(n_agents=24, interactions_per_channel=125.0),
        sweep=SweepConfig(n_s=32, n_f=32, batch_size=512, n_workers=8),
        demo_agents=24,
        demo_runs=2,
    ),
    "medium": Preset(
        name="medium",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_s=64, n_f=64, batch_size=512, n_workers=10),
        demo_agents=40,
        demo_runs=4,
    ),
    "full": Preset(
        name="full",
        model=ModelConfig(n_agents=40),
        sweep=SweepConfig(n_s=200, n_f=200, batch_size=1024, n_workers=10),
        demo_agents=40,
        demo_runs=8,
    ),
}


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]
