"""Uniform credulity: the component of the prejudice field that names no class.

A class-dependent shift of an agent's opinion field has four independent
components (``../directional-prejudice/dirfield/fields.py``)::

    D[r, e] = a + b kappa_r + c kappa_e + p kappa_r kappa_e

The main line of work studies ``p``, the one that depends on whether the classes
*match*.  ``../directional-prejudice/`` studies ``c`` and ``b``, the two that
name one class.  This package studies ``a``, the one that names nobody: the same
shift whoever is speaking and whoever is listening.

That makes it the odd one out, and worth running for two separate reasons.

**As a phase diagram in its own right.**  ``a`` moves the trust separatrix.  A
receiver decides whether a message agrees with it by the sign of ``h_w``, and
adding ``a`` biases that decision uniformly: ``a > 0`` is a population disposed
to read everything as agreement, ``a < 0`` one disposed to read everything as
disagreement.  Since agreement builds trust and a distrusted source is
*anti*-learned from rather than ignored, the two halves of the plane are two
different states rather than two strengths of one, and neither is the ordinary
polarization the society reaches with no field at all.

**As the control the other three are read against.**  Every class order
parameter in the project is being applied here to a model in which the class
label does not appear.  Whatever they read is finite-size noise -- with one
exception, ``R_muc = -T_mu/(N-1)``, which is exactly predicted and which this is
the cleanest place in the basis to measure, because here the leak is the only
thing ``R_muc`` can be reading.

Modules
-------
``modulation``    the evidence Z and the four modulation functions (unchanged)
``society``       batched dynamics under a uniform receiver-side field
``order_params``  the paper's five, the class channels as controls, and the
                  trust and opinion blocks of the bias partition
``sweep``         ``(a, f_a)`` sweeps with caching
``config``        model parameters and resolution presets
``plotting``      shared figure style
"""

from . import config, modulation, order_params, plotting, society, sweep
from .config import PRESETS, ModelConfig, SweepConfig, get_preset
from .order_params import bias_trust_margins, measure, trust_channels
from .society import SocietyBatch
from .sweep import sweep as run_sweep

__all__ = [
    "config",
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
    "trust_channels",
    "bias_trust_margins",
    "run_sweep",
]
