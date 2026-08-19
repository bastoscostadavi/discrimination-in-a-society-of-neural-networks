"""Stated probabilities in, the paper's two fields out.

Everything the trust-curve experiment asks for is a probability between 0 and
100, and everything the paper is written in is a scaled field.  This module is
the only place the two are connected, so a sign convention that is wrong here is
wrong everywhere and visibly so, rather than being wrong in one script and right
in another.

Three quantities, and what each question measures:

``reliability``
    "the chance this agent's judgement on this theme is sound".  The paper takes
    the probability that an emitter conveys the *wrong* label to be
    ``Phi(h_mu)``, so a sound judgement has probability ``Phi(-h_mu)`` and

        h_mu = -Phi^-1(r).

    It depends on the emitter alone, which is what makes it usable as an axis.

``agreement``
    "the chance their next statement is one you agree with".  The receiver
    agrees with the emitter when both are right or both are wrong, so with
    ``c = Phi(|h_w|)`` its own chance of being right and ``e = Phi(h_mu)`` the
    emitter's chance of being wrong,

        q = c(1-e) + (1-c)e = c + e - 2ce,

    which is exactly the evidence ``Z`` of Eq. 8 on the agreeing branch.  So this
    question measures ``Z`` and not trust: it is symmetric under
    ``(c, e) -> (1-c, 1-e)``, and a confident receiver facing an unreliable
    emitter returns the same number as an unsure receiver facing a reliable one.
    It is asked because, given ``e``, it inverts for the receiver's conviction.

``conviction``
    Not asked.  Recovered from the other two by inverting the line above,

        c = (q - e) / (1 - 2e),

    which is singular at ``e = 1/2``: a coin-flip emitter makes agreement a
    coin flip whatever the receiver believes, so conviction is genuinely
    unidentified at neutral trust rather than merely hard to estimate there.
    :func:`conviction` returns ``nan`` in that band instead of a large number
    divided by a small one.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["P_CLIP", "NEUTRAL_BAND", "to_p", "h_mu_from_reliability",
           "conviction", "h_w_from"]

#: Stated probabilities are clipped before the probit.  The model answers on a
#: 0-100 integer scale and does use the ends; ``Phi^-1(0)`` is ``-inf``, and a
#: single infinite point would take a whole bin with it.  The clip is recorded in
#: the rows so how many answers hit it can be counted rather than assumed small.
P_CLIP = (0.02, 0.98)

#: Half-width of the band around ``e = 1/2`` where the conviction inversion is
#: refused.  ``|1 - 2e| < NEUTRAL_BAND`` means the denominator is smaller than
#: the noise on ``q``, and the quotient is then arithmetic rather than
#: measurement.
NEUTRAL_BAND = 0.15


def to_p(chance):
    """A 0-100 answer as a clipped probability."""
    return float(np.clip(np.asarray(chance, dtype=float) / 100.0, *P_CLIP))


def h_mu_from_reliability(r):
    """The distrust field from a stated reliability.  Positive is distrust."""
    return -ndtri(r)


def conviction(q, r):
    """``c = Phi(|h_w|)`` from the agreement prediction and the reliability.

    Returns ``nan`` inside :data:`NEUTRAL_BAND`.  The result is *not* clipped
    into ``(0, 1)``: a value outside it means the two answers are jointly
    inconsistent with the model, and that is a finding about the agent rather
    than something for this function to hide.
    """
    e = 1.0 - r
    denom = 1.0 - 2.0 * e
    if abs(denom) < NEUTRAL_BAND:
        return float("nan")
    return (q - e) / denom


def h_w_from(c, sign):
    """The signed opinion field.  ``sign`` is +1 if the message agrees.

    The emitter supplies only the sign, as it does in the model, where the whole
    magnitude of ``h_w`` is the receiver's own confidence.
    """
    if not np.isfinite(c) or not 0.0 < c < 1.0:
        return float("nan")
    return sign * ndtri(c)
