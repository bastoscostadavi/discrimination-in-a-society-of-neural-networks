"""Uncertainty, and getting it right for map-level quantities.

Per-pixel means and standard errors are straightforward: replicates differ in
their initial condition *and* their interaction schedule, so the spread across
them estimates the full sampling variance.

Map-level quantities are not straightforward, and this is where an otherwise
careful analysis usually goes wrong.  Within one replicate every grid point
shares an interaction schedule (that is what makes the batched inner loop
affordable), so their errors are correlated.  Propagating per-pixel standard
errors into a contour location, a regime area, or a crossover point would
therefore understate the uncertainty, because it assumes an independence the
design deliberately does not have.

The fix is to resample whole *map sheets*: :func:`replicate_bootstrap` draws
replicates with replacement, recomputes the derived quantity on each bootstrap
mean map, and reports the spread of that.  Cross-pixel correlation is then
respected by construction rather than modelled.

For a control against its baseline, use :func:`paired_bootstrap`.  Controls share
``crn_group`` with their baseline, so each control society has a partner starting
from the same weights, the same distrust and the same schedule; differencing
within a pair removes the between-society variance, which here dominates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mean_sem",
    "bootstrap_ci",
    "replicate_bootstrap",
    "paired_bootstrap",
    "percentile_ci",
]


def mean_sem(x, axis=-1):
    """Replicate mean and standard error, ignoring missing entries."""
    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=axis)
    mean = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return mean, sd / np.sqrt(np.maximum(n, 1)), n


def percentile_ci(samples, level=0.95, axis=0):
    """A percentile interval.

    Preferred to mean +- k*sem for anything bounded or skewed --- balance
    measures near +-1, first-passage times, crossover locations --- where a
    symmetric interval would run outside the range the quantity can take.
    """
    lo = 100 * (1 - level) / 2
    return (
        np.nanpercentile(samples, lo, axis=axis),
        np.nanpercentile(samples, 100 - lo, axis=axis),
    )


def bootstrap_ci(x, statistic=np.nanmean, n_boot=2000, level=0.95, rng=None):
    """Percentile bootstrap interval for a statistic of a 1-D sample."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    draws = statistic(x[idx], axis=1)
    lo, hi = percentile_ci(draws, level)
    return float(statistic(x)), float(lo), float(hi)


def replicate_bootstrap(sheets, fn, n_boot=1000, level=0.95, rng=None):
    """Bootstrap a map-level quantity by resampling whole replicates.

    Parameters
    ----------
    sheets
        ``(S, ...)`` --- one complete map per replicate.
    fn
        Applied to a bootstrap **mean map**; may return a scalar or an array.

    Resampling sheets rather than pixels is the point: it preserves the
    within-replicate correlation between pixels that the shared schedule
    creates, so a boundary's confidence band reflects how much the boundary
    actually moves between replicates rather than an independence assumption
    that does not hold.
    """
    rng = rng or np.random.default_rng(0)
    sheets = np.asarray(sheets, dtype=float)
    S = sheets.shape[0]
    point = fn(np.nanmean(sheets, axis=0))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, S, size=S)
        draws.append(fn(np.nanmean(sheets[pick], axis=0)))
    draws = np.asarray(draws, dtype=float)
    lo, hi = percentile_ci(draws, level, axis=0)
    return point, lo, hi, draws


def paired_bootstrap(a, b, n_boot=2000, level=0.95, rng=None):
    """Bootstrap the mean difference of paired samples.

    ``a`` and ``b`` must be aligned so that ``a[i]`` and ``b[i]`` are the two
    members of one pair --- same point, same replicate index, same ``crn_group``,
    hence the same initial condition and schedule.  The variance of the paired
    difference is far smaller than that of the unpaired difference here, which is
    the entire reason the campaign shares common random numbers between a
    baseline and its controls.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must align: {a.shape} vs {b.shape}")
    diff = a - b
    ok = ~np.isnan(diff)
    return bootstrap_ci(diff[ok], n_boot=n_boot, level=level, rng=rng)


def cross_pixel_correlation(sheets):
    """Mean correlation between pixel errors within a replicate.

    A diagnostic for the choice above: if this is materially non-zero, then
    per-pixel standard errors must not be propagated into map-level quantities,
    and :func:`replicate_bootstrap` is required rather than merely preferable.
    """
    sheets = np.asarray(sheets, dtype=float)
    S = sheets.shape[0]
    flat = sheets.reshape(S, -1)
    resid = flat - np.nanmean(flat, axis=0, keepdims=True)
    good = ~np.any(np.isnan(resid), axis=0)
    resid = resid[:, good]
    if resid.shape[1] < 2 or S < 3:
        return np.nan
    c = np.corrcoef(resid.T)
    iu = np.triu_indices_from(c, 1)
    return float(np.nanmean(c[iu]))
