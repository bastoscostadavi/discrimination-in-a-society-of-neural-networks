"""Collective discrimination from class-correlated representation errors.

A population of neural agents, each a perceptron over a shared issue space, that
learn both *what to think* and *whom to trust* from one another.  A fraction of
them extend their representation of an incoming message with one coordinate that
carries no information about the issue and depends only on the sender's class.
This package simulates the population, measures it, and locates the collective
regimes that result.

Entry points::

    from socsim import ModelConfig, GridSpec, RunSpec, SocietyBatch, measure

Simulation and figures are deliberately separate: ``socsim run`` writes data,
and the figure scripts only ever read it.  A twenty-hour campaign cannot live
inside a plotting script.
"""

from .config import PRESETS, GridSpec, ModelConfig, RunSpec, get_preset
from .discrimination import CASES, FieldSpec, build_field, field_matrix
from .modulation import F_C, F_V, F_mu, F_w, evidence, modulation
from .observables import OBS_NAMES, balance, correlations, measure, spectral_polarization
from .seeds import RunKey, point_id, stream
from .society import SocietyBatch

__all__ = [
    "ModelConfig",
    "GridSpec",
    "RunSpec",
    "PRESETS",
    "get_preset",
    "FieldSpec",
    "build_field",
    "field_matrix",
    "CASES",
    "modulation",
    "evidence",
    "F_w",
    "F_C",
    "F_mu",
    "F_V",
    "measure",
    "correlations",
    "balance",
    "spectral_polarization",
    "OBS_NAMES",
    "RunKey",
    "point_id",
    "stream",
    "SocietyBatch",
]
