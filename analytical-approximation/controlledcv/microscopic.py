"""Full and leading small-C,V one-interaction increments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modulation import modulation


@dataclass(frozen=True)
class Increment:
    dw: np.ndarray
    dC: np.ndarray
    dmu: float
    dV: float
    h_w: float
    h_mu: float


def _sigma(w_e, x):
    s = float(np.sign(np.dot(w_e, x)))
    return 1.0 if s == 0.0 else s


def full_increment(w_r, w_e, C, mu, V, x, D=0.0):
    """Microscopic manuscript increment for one selected interaction."""

    w_r = np.asarray(w_r, dtype=float)
    w_e = np.asarray(w_e, dtype=float)
    C = np.asarray(C, dtype=float)
    x = np.asarray(x, dtype=float)
    sigma = _sigma(w_e, x)
    Cx = C @ x
    gamma_C = np.sqrt(1.0 + float(x @ Cx))
    gamma_V = np.sqrt(1.0 + float(V))
    h_w = sigma * float(w_r @ x) / gamma_C + float(D)
    h_mu = float(mu) / gamma_V
    fw, fC, fmu, fV = modulation(h_w, h_mu)
    return Increment(
        dw=(Cx * sigma * fw) / gamma_C,
        dC=np.outer(Cx, Cx) * fC / (gamma_C * gamma_C),
        dmu=float(V) * float(fmu) / gamma_V,
        dV=float(V) * float(V) * float(fV) / (gamma_V * gamma_V),
        h_w=float(h_w),
        h_mu=float(h_mu),
    )


def leading_increment(w_r, w_e, Cbar, mu, Vbar, x, D=0.0):
    """Leading slow-time increment dtheta/dtau for C=epsilon*Cbar, V=epsilon*Vbar."""

    w_r = np.asarray(w_r, dtype=float)
    w_e = np.asarray(w_e, dtype=float)
    Cbar = np.asarray(Cbar, dtype=float)
    x = np.asarray(x, dtype=float)
    sigma = _sigma(w_e, x)
    h = sigma * float(w_r @ x)
    H = h + float(D)
    fw, fC, fmu, fV = modulation(H, mu)
    Cx = Cbar @ x
    return Increment(
        dw=Cx * sigma * fw,
        dC=np.outer(Cx, Cx) * fC,
        dmu=float(Vbar) * float(fmu),
        dV=float(Vbar) * float(Vbar) * float(fV),
        h_w=float(H),
        h_mu=float(mu),
    )
