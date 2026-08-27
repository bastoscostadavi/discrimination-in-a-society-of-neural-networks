"""The credulity asymmetry: the field component nothing in the paper can see.

A class-dependent shift of an agent's opinion field has four independent
components (:mod:`credfield.fields`)::

    D[r, e] = a + b kappa_r + c kappa_e + p kappa_r kappa_e

The main line of work studies ``p``, the one that depends on whether the classes
*match*: in-group favouritism and out-group hostility at once, symmetric between
the classes, producing two mutually distrustful camps aligned with the label.

This package studies ``b``, the one that depends on the *listener's* class alone.
A prejudiced agent of class A reads every message as more agreeable than it is
and a prejudiced agent of class B reads every message as less so, whoever is
speaking, so one class comes to trust everyone and the other to trust nobody --
itself included, which is what makes it a split in credulity rather than a
preference between groups.  ``b`` writes the class label into the *antisymmetric*
part of the directed trust matrix, and every order parameter of the main line of
work averages that part away, so this is a state the published phase diagram
cannot see.

Its mirror ``c``, in which one class is *believed* more by everyone, is swept in
``../directional-prejudice/``; the two are exact transposes of each other on the
trust matrix, so the published parameters cannot even tell them apart.  Both are
reachable from here with ``--component``, which is how that claim is checked
rather than asserted.

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
