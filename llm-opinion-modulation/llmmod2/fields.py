"""From pieces of evidence to the paper's fields.

The ladder returns beliefs in *evidence units*: how many pieces of evidence a
belief is worth.  The paper's fields are in probit units.  One number converts
between them -- ``lam``, the log-odds a single piece of evidence carries -- and
where it comes from matters, because a scale pulled out of the fit to ``F_w``
would be a second free parameter and the sibling experiment already spends the
one that is unavoidable.

``lam`` is not fitted to the update.  It is fixed on the *trust* axis, from
information the design already contains: a colleague who was right ``k`` of
``TRACK_TOTAL`` times has a stated flip probability of ``1 - k/TRACK_TOTAL``, so
the distrust field they ought to carry is known before any update is measured.
Requiring the measured testimony weights to reproduce those known values pins
``lam``, and the residual of that requirement is itself a result -- it says
whether the model weighs a track record the way a track record should be
weighed.  What is left over for the update is one positive scale, exactly as in
the sibling fit, standing for the unobservable variance in
``w += (F_w / gamma_C) sigma_e C x``.

Accumulating evidence is additive in log-odds and the fields are probits, so the
conversion is a logistic in and a probit out; the composition is monotone and
unbounded in both directions, which is the whole point.  Nothing here clips.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtri

__all__ = ["h_of", "nominal_h_mu", "fit_lam", "opinion_fields", "trust_fields"]


def h_of(leaning, lam):
    """Probit field of a belief worth ``leaning`` pieces of evidence.

    Evidence adds in log-odds; the field is the probit of the resulting
    confidence.  Odd in ``leaning``, unbounded, and strictly increasing.

    The magnitude is taken first and the sign put back afterwards.  Written the
    obvious way the two halves disagree in the last few digits far out in the
    tail, where the logistic has saturated against the floating-point grid; this
    form is odd by construction instead of odd to within rounding.
    """
    x = lam * np.asarray(leaning, float)
    return np.sign(x) * ndtri(expit(np.abs(x)))


def nominal_h_mu(k, total):
    """The distrust field a track record of ``k`` correct out of ``total`` states.

    The stated flip probability is ``1 - k/total`` and ``h_mu`` is its probit.  A
    perfect or perfectly wrong record would be infinite, so the count is read
    with the Jeffreys correction that a count of successes always wants; this
    touches only the two extreme rungs and does not bound anything measured.
    """
    k = np.asarray(k, float)
    eps = (total - k + 0.5) / (total + 1.0)
    return ndtri(eps)


def fit_lam(weights, ks, total, bounds=(1e-3, 10.0)):
    """Log-odds per piece of evidence, from the trust axis alone.

    ``weights`` are measured testimony weights in evidence units and ``ks`` the
    track records that produced them.  Returns ``(lam, rss, r)``: the scale, the
    residual sum of squares at it, and the correlation between the measured and
    stated distrust fields, which is the calibration's own goodness of fit.
    """
    w = np.asarray(weights, float)
    target = -nominal_h_mu(ks, total)          # h_of(weight) should equal this
    ok = np.isfinite(w) & np.isfinite(target)
    w, target = w[ok], target[ok]

    def rss(lam):
        return float(np.sum((h_of(w, lam) - target) ** 2))

    best = minimize_scalar(rss, bounds=bounds, method="bounded")
    lam = float(best.x)
    pred = h_of(w, lam)
    r = float(np.corrcoef(pred, target)[0, 1]) if len(w) > 2 else float("nan")
    return lam, float(best.fun), r


def opinion_fields(leaning_pre, leaning_post, sign, lam):
    """``(h_w, dh_w)`` from the two null points of one condition.

    ``sign`` is the direction the colleague asserted.  Both quantities are signed
    by it, so ``h_w > 0`` means the receiver already agreed with the message and
    ``dh_w > 0`` means it moved towards the message -- the paper's convention,
    in which ``dh_w`` is proportional to ``F_w``.
    """
    pre = h_of(leaning_pre, lam)
    post = h_of(leaning_post, lam)
    return sign * pre, sign * (post - pre)


def trust_fields(weight_pre, weight_post, lam):
    """``(h_mu, dh_mu)`` from the two testimony weights of one condition.

    Distrust is the negative of credibility: a colleague whose word is worth
    more evidence is one the receiver distrusts less.  ``dh_mu < 0`` therefore
    means trust rose, which is the sign ``F_mu`` carries after agreement.
    """
    pre = -h_of(weight_pre, lam)
    post = -h_of(weight_post, lam)
    return pre, post - pre
