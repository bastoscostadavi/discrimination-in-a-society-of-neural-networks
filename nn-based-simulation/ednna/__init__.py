"""Societies of entropic-dynamics neural-network agents (EDNNA).

Agents are single-layer perceptrons that learn from each other's stated
opinions through an optimized on-line algorithm carrying both an ideological
sector (the weights) and an trust sector (the distrust each agent assigns
to each other agent).  When a fraction ``f_d`` of agents perturb the opinion
field by an amount ``d`` correlated with the class label of the emitter, the
society passes through several collective phases, which this package maps.

Modules
-------
``modulation``      the evidence Z and the four modulation functions
``society``         batched simulation of societies of agents
``discrimination``  the prejudice-field matrices and their sign convention
``order_params``    correlations and social balance
``sweep``           ``(d, f_d)`` phase-diagram sweeps with caching
``config``          model parameters and resolution presets
``plotting``        shared figure style
"""

from . import config, discrimination, modulation, order_params, plotting, society, sweep
from .config import PRESETS, ModelConfig, SweepConfig, get_preset
from .order_params import measure
from .society import SocietyBatch
from .sweep import sweep as run_sweep

__all__ = [
    "config",
    "discrimination",
    "modulation",
    "order_params",
    "plotting",
    "society",
    "sweep",
    "ModelConfig",
    "SweepConfig",
    "PRESETS",
    "get_preset",
    "SocietyBatch",
    "measure",
    "run_sweep",
]
