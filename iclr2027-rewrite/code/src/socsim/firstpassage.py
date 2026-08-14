"""Which polarisation arrives first, quantified.

The qualitative claim is that a narrow agenda makes affective structure form
before ideological structure, and a broad one reverses it.  Turning that into a
measurement means recording when each sector first crosses a threshold and
comparing the two times.

Three choices matter, and the obvious version of each is wrong.

**What to threshold.**  Not the two balance measures: they have different noise
floors and different dynamic ranges, so one common threshold is not one common
event and the *sign* of the comparison would partly reflect that mismatch.  We
threshold the baseline-standardised polarisations ``P_O_hat`` and ``P_T_hat``,
which are built to be comparable --- one formula, each divided by its own
attainable range at this ``(N, K)``.

**What to report.**  Not ``t_T - t_O``.  Crossing times span three decades across
the agenda sizes studied, so a raw difference is dominated by the largest
agendas and its interval says nothing about the small ones.  We report
``Delta = log10(t_T / t_O)``, keeping the sign convention: negative means
affective structure came first.

**Censoring.**  At a high threshold and a narrow agenda the crossing may never
occur within the run.  Those are right-censored, not missing at random.  We
return NaN, report the censored fraction, and use medians with rank-based
intervals --- a mean over a censored sample is simply wrong, and with the
annealing dynamics censoring definitely happens.
"""

from __future__ import annotations

import numpy as np

from .stats import percentile_ci

__all__ = [
    "sample_times",
    "first_passage",
    "delta_log",
    "crossover_alpha",
    "censoring_fraction",
]


def sample_times(total_steps, n_samples=48, first=None, power=3.0):
    """Measurement times, dense early where the crossings happen.

    A linear grid --- what the reference implementation used --- puts most of its
    samples in the late, slow part of the run and is sparsest exactly where the
    two sectors separate.  A power spacing fixes that at no extra cost, since the
    cost is in the interactions, not the measurements.
    """
    first = first or max(1, total_steps // 2000)
    frac = (np.arange(1, n_samples + 1) / n_samples) ** power
    t = np.unique((first + (total_steps - first) * frac).astype(np.int64))
    return t[t > 0]


def first_passage(times, values, tau):
    """First time ``values`` reaches ``tau``, interpolated in log time.

    ``values`` may be ``(..., T)``; the crossing is found along the last axis.
    Returns NaN where the threshold is never reached, which the caller must
    treat as censored rather than dropping.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    reached = values >= tau
    ever = reached.any(axis=-1)
    idx = np.argmax(reached, axis=-1)

    out = np.full(values.shape[:-1], np.nan)
    if not np.any(ever):
        return out

    flat_v = values.reshape(-1, values.shape[-1])
    flat_i = idx.reshape(-1)
    flat_e = ever.reshape(-1)
    res = np.full(flat_v.shape[0], np.nan)

    for k in np.flatnonzero(flat_e):
        i = flat_i[k]
        if i == 0:
            res[k] = times[0]
            continue
        v0, v1 = flat_v[k, i - 1], flat_v[k, i]
        t0, t1 = np.log(times[i - 1]), np.log(times[i])
        # Linear in log t between the bracketing samples.
        frac = 0.0 if v1 == v0 else (tau - v0) / (v1 - v0)
        res[k] = float(np.exp(t0 + frac * (t1 - t0)))
    return res.reshape(values.shape[:-1])


def censoring_fraction(t):
    """Share of replicates that never crossed."""
    t = np.asarray(t, dtype=float)
    return float(np.mean(np.isnan(t)))


def delta_log(t_opinion, t_trust):
    """``Delta = log10(t_T / t_O)``. Negative means affect came first."""
    t_o = np.asarray(t_opinion, dtype=float)
    t_t = np.asarray(t_trust, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(t_t / t_o)


def summarise_delta(delta, level=0.95):
    """Median and a rank-based interval, which censoring makes mandatory."""
    d = np.asarray(delta, dtype=float)
    ok = d[~np.isnan(d)]
    if ok.size == 0:
        return {"median": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    lo, hi = percentile_ci(ok, level)
    return {
        "median": float(np.median(ok)),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(ok.size),
    }


def crossover_alpha(alphas, deltas, n_boot=2000, rng=None):
    """Where ``Delta(alpha)`` changes sign, with a bootstrap interval.

    ``deltas`` is ``(n_alpha, n_replicates)``.  The crossover is found by
    monotone interpolation of the replicate-median curve in ``log alpha``; the
    interval comes from resampling replicates, which respects the fact that one
    replicate contributes a whole curve rather than independent points.
    """
    from scipy.interpolate import PchipInterpolator

    rng = rng or np.random.default_rng(0)
    alphas = np.asarray(alphas, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    la = np.log10(alphas)

    def root(curve):
        good = ~np.isnan(curve)
        if good.sum() < 3:
            return np.nan
        x, y = la[good], curve[good]
        if np.all(y > 0) or np.all(y < 0):
            return np.nan
        order = np.argsort(x)
        f = PchipInterpolator(x[order], y[order])
        grid = np.linspace(x.min(), x.max(), 2001)
        vals = f(grid)
        sign_change = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
        if sign_change.size == 0:
            return np.nan
        i = sign_change[0]
        # Linear interpolation between the bracketing grid points.
        w = vals[i] / (vals[i] - vals[i + 1])
        return float(10 ** (grid[i] + w * (grid[i + 1] - grid[i])))

    point = root(np.nanmedian(deltas, axis=1))
    S = deltas.shape[1]
    draws = np.array(
        [root(np.nanmedian(deltas[:, rng.integers(0, S, S)], axis=1)) for _ in range(n_boot)]
    )
    draws = draws[~np.isnan(draws)]
    if draws.size < 10:
        return point, np.nan, np.nan
    lo, hi = percentile_ci(draws, 0.95)
    return point, float(lo), float(hi)
