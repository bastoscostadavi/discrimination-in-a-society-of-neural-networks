"""Modulation functions of the EDNNA learning algorithm.

Everything here is a direct transcription of the paper's equations, in the
notation of the paper:

    h_w   opinion field   = (w_r . x) sigma_e / gamma_C     (Eq. 6)
    h_mu  distrust field  = mu_{e|r} / gamma_V              (Eq. 6)

The evidence (Eq. 8) is

    Z = Phi(h_w) + Phi(h_mu) - 2 Phi(h_w) Phi(h_mu)

and the four modulation functions are its log-derivatives (Eqs. 21-24):

    F_w  = dlogZ/dh_w    = (1 - 2 Phi(h_mu)) g(h_w) / Z
    F_C  = d2logZ/dh_w2  = -F_w (F_w + h_w)
    F_mu = dlogZ/dh_mu   = (1 - 2 Phi(h_w)) g(h_mu) / Z
    F_V  = d2logZ/dh_mu2 = -F_mu (F_mu + h_mu)

with g the standard normal density and Phi its cumulative.

Note the symmetry between the ideological and trust sectors, which the
paper highlights (and which `tests/test_modulation.py` checks):

    F_w(x, y) = F_mu(y, x)        F_C(x, y) = F_V(y, x)

All functions are vectorised and broadcast over numpy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = ["Phi", "g", "evidence", "F_w", "F_C", "F_mu", "F_V", "modulation", "Z_FLOOR"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)

#: Floor applied to the evidence Z before dividing by it.
#:
#: The paper does not discuss this.  It is needed because Z -> 0 in the deep
#: dissonance corners (agreeing with a totally distrusted emitter, or
#: disagreeing with a totally trusted one), where the exact modulation
#: functions diverge.  The floor keeps |F| finite; with the default value the
#: cap on |F| is far above any value reached in practice, so the dynamics is
#: unaffected except in the measure-zero divergent corners.
Z_FLOOR = 1e-12


def Phi(x):
    """Cumulative distribution function of the standard Gaussian."""
    return ndtr(x)


def g(x):
    """Density of the standard Gaussian."""
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def evidence(h_w, h_mu):
    """Evidence Z of Eq. 8.

    Z is the probability the receiver assigns to the incoming message.  It is
    low in the two dissonant quadrants and high in the two consonant ones.
    """
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    return Pw + Pm - 2.0 * Pw * Pm


def F_w(h_w, h_mu, z_floor=Z_FLOOR):
    """Ideological modulation function (Eq. 21)."""
    Pm = Phi(h_mu)
    Z = np.maximum(evidence(h_w, h_mu), z_floor)
    return (1.0 - 2.0 * Pm) * g(h_w) / Z


def F_mu(h_w, h_mu, z_floor=Z_FLOOR):
    """Trust-sector modulation function (Eq. 23)."""
    Pw = Phi(h_w)
    Z = np.maximum(evidence(h_w, h_mu), z_floor)
    return (1.0 - 2.0 * Pw) * g(h_mu) / Z


def F_C(h_w, h_mu, z_floor=Z_FLOOR):
    """Modulation of the ideological annealing schedule (Eq. 22)."""
    fw = F_w(h_w, h_mu, z_floor)
    return -fw * (fw + h_w)


def F_V(h_w, h_mu, z_floor=Z_FLOOR):
    """Modulation of the trust-sector annealing schedule (Eq. 24)."""
    fm = F_mu(h_w, h_mu, z_floor)
    return -fm * (fm + h_mu)


def modulation(h_w, h_mu, z_floor=Z_FLOOR):
    """All four modulation functions at once, sharing the work.

    Returns ``(F_w, F_C, F_mu, F_V)``.  This is the entry point used by the
    simulation inner loop, where Phi/g/Z are each wanted only once.
    """
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    Z = np.maximum(Pw + Pm - 2.0 * Pw * Pm, z_floor)
    fw = (1.0 - 2.0 * Pm) * g(h_w) / Z
    fm = (1.0 - 2.0 * Pw) * g(h_mu) / Z
    return fw, -fw * (fw + h_w), fm, -fm * (fm + h_mu)
