"""Evidence and modulation functions from the microscopic EDNNA equations."""

from __future__ import annotations

import math

import numpy as np
try:
    from scipy.special import ndtr as _ndtr
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _ndtr = None

Z_FLOOR = 1e-12
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_SQRT_2 = np.sqrt(2.0)
_erf = np.vectorize(math.erf, otypes=[float])


def Phi(x):
    if _ndtr is not None:
        return _ndtr(x)
    return 0.5 * (1.0 + _erf(np.asarray(x) / _SQRT_2))


def g(x):
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def evidence(h_w, h_mu):
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    return Pw + Pm - 2.0 * Pw * Pm


def modulation(h_w, h_mu, z_floor=Z_FLOOR):
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    Z = np.maximum(Pw + Pm - 2.0 * Pw * Pm, z_floor)
    fw = (1.0 - 2.0 * Pm) * g(h_w) / Z
    fm = (1.0 - 2.0 * Pw) * g(h_mu) / Z
    return fw, -fw * (fw + h_w), fm, -fm * (fm + h_mu)


def F_w(h_w, h_mu, z_floor=Z_FLOOR):
    return modulation(h_w, h_mu, z_floor)[0]


def F_C(h_w, h_mu, z_floor=Z_FLOOR):
    return modulation(h_w, h_mu, z_floor)[1]


def F_mu(h_w, h_mu, z_floor=Z_FLOOR):
    return modulation(h_w, h_mu, z_floor)[2]


def F_V(h_w, h_mu, z_floor=Z_FLOOR):
    return modulation(h_w, h_mu, z_floor)[3]
