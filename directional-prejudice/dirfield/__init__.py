"""Directional prejudice fields: the components the phase diagram leaves out.

A class-dependent shift of an agent's opinion field has four independent
components (:mod:`dirfield.fields`)::

    D[r, e] = a + b kappa_r + c kappa_e + p kappa_r kappa_e

The main line of work studies ``p``, the one that depends on whether the classes
*match*: in-group favouritism and out-group hostility at once, symmetric between
the classes, producing two mutually distrustful camps aligned with the label.

This package studies the two that depend on one class alone.  ``c`` gives a
population in which one class is believed more by everyone, its own members
included: status, and its negative, stigma.  ``b`` gives one in which one class
believes everyone and the other believes nobody.  Both write the class label into
the *antisymmetric* part of the directed trust matrix, which every order
parameter of the main line of work averages away, so both are states that the
published phase diagram cannot see.

Modules
-------
``fields``        the four-component basis and the six tabulated cases in it
``modulation``    the evidence Z and the four modulation functions (unchanged)
``society``       batched dynamics under a general class-dependent field
``order_params``  the paper's five, plus the four trust channels and the
                  within-class balances
``sweep``         ``(strength, fraction)`` sweeps with caching
``config``        model parameters and resolution presets
``plotting``      shared figure style
"""

from . import config, fields, modulation, order_params, plotting, society, sweep
from .config import PRESETS, ModelConfig, SweepConfig, get_preset
from .fields import COMPONENTS, TABLE_I, decompose, field_matrix
from .order_params import measure, trust_channels
from .society import SocietyBatch
from .sweep import sweep as run_sweep

__all__ = [
    "config",
    "fields",
    "modulation",
    "order_params",
    "plotting",
    "society",
    "sweep",
    "ModelConfig",
    "SweepConfig",
    "PRESETS",
    "get_preset",
    "COMPONENTS",
    "TABLE_I",
    "field_matrix",
    "decompose",
    "SocietyBatch",
    "measure",
    "trust_channels",
    "run_sweep",
]
