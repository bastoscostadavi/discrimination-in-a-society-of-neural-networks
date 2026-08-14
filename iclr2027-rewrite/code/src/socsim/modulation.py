"""Evidence and modulation functions of the learning rule.

An agent that receives a message forms two scaled fields --- the *agreement
field* ``h_w``, positive when it already agrees with what was said, and the
*distrust field* ``h_mu``, positive when it distrusts the source --- and the
evidence it assigns to the message is

    Z = Phi(h_w) + Phi(h_mu) - 2 Phi(h_w) Phi(h_mu)

which is small in the two *dissonant* quadrants (agreeing with a distrusted
source, disagreeing with a trusted one) and large in the two consonant ones.
The four modulation functions that scale the updates are its log-derivatives:

    F_w  = dlogZ/dh_w    = (1 - 2 Phi(h_mu)) g(h_w) / Z
    F_mu = dlogZ/dh_mu   = (1 - 2 Phi(h_w))  g(h_mu) / Z
    F_C  = d2logZ/dh_w2  = -F_w  (F_w  + h_w)
    F_V  = d2logZ/dh_mu2 = -F_mu (F_mu + h_mu)

Two structural facts are used throughout the paper and are checked in
``tests/test_modulation.py``:

* **Sector mirror symmetry.**  ``F_w(x, y) = F_mu(y, x)`` and
  ``F_C(x, y) = F_V(y, x)``.  The ideological and affective sectors are the same
  function with its arguments exchanged.
* **Point symmetry.**  ``Z(-h_w, -h_mu) = Z(h_w, h_mu)``, and the flow
  ``(F_w, F_mu)`` is equivariant under ``u -> -u``.  This is the symmetry the
  discrimination field breaks, and the proof is one line: ``Phi(-a) = 1 - Phi(a)``
  leaves ``Z`` invariant while flipping the sign of both ``(1 - 2 Phi)`` factors.

Numerical note
--------------
``Z -> 0`` in the deep dissonant corners, where the exact modulation functions
diverge.  Two independent guards are provided:

* :data:`Z_FLOOR`, a floor applied before dividing.  This is what the reference
  implementation used to produce every result we regress against, so it is the
  default and must not be changed casually.
* :func:`evidence_stable`, a cancellation-resistant form that needs no floor.
  ``tests/test_modulation.py`` checks the two agree to machine precision over
  the region the dynamics actually visits, so the floor is documented as inert
  rather than merely assumed to be.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

__all__ = [
    "Phi",
    "g",
    "evidence",
    "evidence_stable",
    "F_w",
    "F_C",
    "F_mu",
    "F_V",
    "modulation",
    "Z_FLOOR",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)

#: Floor applied to the evidence before dividing by it.
#:
#: The model does not prescribe it.  With this value the implied cap on |F| is
#: far above any value the dynamics reaches, so it is inert in practice; the
#: society tracks how often it binds (``SocietyBatch.n_zfloor_hits``) so that
#: "inert" is a measurement rather than a hope.
Z_FLOOR = 1e-12


def Phi(x):
    """Standard normal CDF."""
    return ndtr(x)


def g(x):
    """Standard normal density."""
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def evidence(h_w, h_mu):
    """Evidence ``Z`` assigned to the incoming message.

    This is the reference form, ``Phi + Phi - 2 Phi Phi``.  It loses precision
    when both fields are large and negative, which is what
    :func:`evidence_stable` exists to avoid.
    """
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    return Pw + Pm - 2.0 * Pw * Pm


def evidence_stable(h_w, h_mu):
    """Evidence in the cancellation-resistant form ``Phi(a)S(b) + Phi(b)S(a)``.

    With ``S(x) = 1 - Phi(x) = Phi(-x)`` the survival function, the identity

        Phi(a) + Phi(b) - 2 Phi(a) Phi(b) = Phi(a) S(b) + Phi(b) S(a)

    is exact, but the right-hand side is a sum of two positive terms and so
    never cancels.  It stays accurate where the reference form underflows, and
    needs no floor.
    """
    Pw, Pm = Phi(h_w), Phi(h_mu)
    Sw, Sm = Phi(-h_w), Phi(-h_mu)
    return Pw * Sm + Pm * Sw


def F_w(h_w, h_mu, z_floor=Z_FLOOR):
    """Ideological modulation: how far the opinion sector moves."""
    Pm = Phi(h_mu)
    Z = np.maximum(evidence(h_w, h_mu), z_floor)
    return (1.0 - 2.0 * Pm) * g(h_w) / Z


def F_mu(h_w, h_mu, z_floor=Z_FLOOR):
    """Affective modulation: how far the distrust sector moves."""
    Pw = Phi(h_w)
    Z = np.maximum(evidence(h_w, h_mu), z_floor)
    return (1.0 - 2.0 * Pw) * g(h_mu) / Z


def F_C(h_w, h_mu, z_floor=Z_FLOOR):
    """Modulation of the ideological annealing schedule."""
    fw = F_w(h_w, h_mu, z_floor)
    return -fw * (fw + h_w)


def F_V(h_w, h_mu, z_floor=Z_FLOOR):
    """Modulation of the affective annealing schedule."""
    fm = F_mu(h_w, h_mu, z_floor)
    return -fm * (fm + h_mu)


def modulation(h_w, h_mu, z_floor=Z_FLOOR, count_floor=False):
    """All four modulation functions at once, sharing ``Phi``, ``g`` and ``Z``.

    This is the entry point used by the simulation inner loop.  The arithmetic
    is deliberately identical to evaluating the four functions individually, so
    that the fast path and the readable path cannot drift apart.

    Returns ``(F_w, F_C, F_mu, F_V)``, or ``(F_w, F_C, F_mu, F_V, n_floored)``
    when ``count_floor`` is set.
    """
    Pw = Phi(h_w)
    Pm = Phi(h_mu)
    Z_raw = Pw + Pm - 2.0 * Pw * Pm
    Z = np.maximum(Z_raw, z_floor)
    fw = (1.0 - 2.0 * Pm) * g(h_w) / Z
    fm = (1.0 - 2.0 * Pw) * g(h_mu) / Z
    out = (fw, -fw * (fw + h_w), fm, -fm * (fm + h_mu))
    if count_floor:
        return (*out, int(np.count_nonzero(Z_raw < z_floor)))
    return out
