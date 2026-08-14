"""Naming the collective regimes, reproducibly.

The source material labels four regions by placing text at four hand-chosen
coordinates on a single-realisation heatmap.  That is the objection a reviewer
raises first, and rightly: nothing about it can be checked, reproduced, or
carried across to a different ``N``.

Here a regime is a function of measured order parameters against thresholds
calibrated from the no-discrimination null, and the whole map is recomputed at
several threshold multiples so its sensitivity is shown rather than asserted.

Why thresholds rather than clustering
-------------------------------------
Clustering the order-parameter vectors is the obvious alternative and it is
rejected deliberately.  Cluster identity is not stable across ``N``, across
agenda size, or between a baseline and its controls --- and comparing exactly
those things is the point of the finite-size and control figures.  A clustered
map also needs a chosen number of clusters and random restarts, so it is not
reproducible in the sense the objection demands.  Thresholds are legible in a
caption and their sensitivity is directly showable.

Two rules about application
---------------------------
* **Classify each replicate, then take the modal label.**  Classifying the
  replicate mean would render a genuinely bistable point --- half its replicates
  in one regime, half in another --- as some intermediate colour, hiding the most
  interesting thing about it.
* **Report the agreement fraction.**  ``agreement`` is the share of replicates
  landing in the modal regime; points below about 0.7 are drawn hatched.  That is
  the boundary-uncertainty display, and it costs nothing extra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["REGIMES", "REGIME_LABELS", "Thresholds", "calibrate", "classify", "classify_map"]

#: Regime codes.  Names describe measured behaviour, deliberately.  In
#: particular there is no "spin glass" here: that name requires overlap
#: distributions, metastability and initialisation dependence, none of which a
#: pair correlation establishes, so the frustrated region is named for what is
#: actually measured -- negative affective balance.
REGIMES = (
    "weakly_structured",
    "class_uncorrelated_polarized",
    "counter_aligned_frustrated",
    "discriminatory_ideological",
    "discriminatory_class_dominant",
)

REGIME_LABELS = {
    "weakly_structured": "weakly structured",
    "class_uncorrelated_polarized": "polarized, class-uncorrelated",
    "counter_aligned_frustrated": "counter-aligned, frustrated",
    "discriminatory_ideological": "discriminatory, ideological",
    "discriminatory_class_dominant": "discriminatory, class-dominant",
}


@dataclass(frozen=True)
class Thresholds:
    """Cut points, in units calibrated from the null.

    ``c`` bounds "no class correlation", ``p`` bounds "not polarised", ``b``
    bounds "affectively balanced".  All three are set from the spread of the
    corresponding statistic where no discrimination is present, so they carry
    the finite-size noise floor with them and do not have to be re-tuned by eye
    at each ``N``.
    """

    c: float = 0.10
    p: float = 0.10
    b: float = 0.05
    n_sigma: float = 3.0

    def scaled(self, factor):
        return replace(self, c=self.c * factor, p=self.p * factor, b=self.b * factor)


def calibrate(null_values, n_sigma=3.0, floor_c=0.05, floor_p=0.03):
    """Thresholds from the no-discrimination null.

    ``null_values`` maps statistic name to the sample of its values where no
    discrimination is present --- in practice the ``d = 0`` column of the main
    sweep, which carries every ``f_d`` at full replication.  Cutting at
    ``n_sigma`` times the null spread means "class-correlated" is calibrated
    against how large the correlation gets when there is provably no class
    signal at all, rather than against a number chosen to make the picture look
    right.

    Small floors keep a very tight null from producing a threshold so small that
    every point is classified as structured.
    """

    def spread(name, floor):
        v = np.asarray(null_values.get(name, []), dtype=float)
        v = v[~np.isnan(v)]
        return max(float(np.std(v, ddof=1)) * n_sigma, floor) if v.size > 2 else floor

    return Thresholds(
        c=spread("C_CT", floor_c),
        p=spread("P_O_hat", floor_p),
        b=spread("B_T_sign", 0.03),
        n_sigma=n_sigma,
    )


def classify(obs, thr):
    """The regime of one society, as an index into :data:`REGIMES`.

    ``obs`` maps statistic name to a scalar or an array; arrays are classified
    elementwise.  Required keys: ``C_CT``, ``C_CO``, ``P_O_hat``, ``P_T_hat``,
    ``B_T_sign``.

    The order of tests is the definition and is fixed in advance:

    1. no class correlation and no polarisation in either sector
       -> weakly structured;
    2. no class correlation but polarisation present
       -> polarized but class-uncorrelated (the region a class-only analysis
       would wrongly call neutral);
    3. class correlation of the *wrong* sign, with affective balance below the
       null -> counter-aligned and frustrated;
    4. class correlation with opinion following class too
       -> discriminatory, ideological;
    5. otherwise class-correlated distrust without ideological alignment
       -> discriminatory, class-dominant.
    """
    C_CT = np.asarray(obs["C_CT"], dtype=float)
    C_CO = np.asarray(obs["C_CO"], dtype=float)
    P_O = np.asarray(obs["P_O_hat"], dtype=float)
    P_T = np.asarray(obs["P_T_hat"], dtype=float)
    B_T = np.asarray(obs["B_T_sign"], dtype=float)

    out = np.full(C_CT.shape, REGIMES.index("discriminatory_class_dominant"), dtype=np.int8)
    quiet = np.abs(C_CT) < thr.c

    out[quiet & (P_O < thr.p) & (P_T < thr.p)] = REGIMES.index("weakly_structured")
    out[quiet & ~((P_O < thr.p) & (P_T < thr.p))] = REGIMES.index(
        "class_uncorrelated_polarized"
    )
    out[(~quiet) & (C_CT <= -thr.c)] = REGIMES.index("counter_aligned_frustrated")
    out[(~quiet) & (C_CT >= thr.c) & (C_CO >= thr.c)] = REGIMES.index(
        "discriminatory_ideological"
    )
    return out


def classify_map(per_replicate, thr):
    """Modal regime and agreement fraction over replicates.

    ``per_replicate`` maps statistic name to an array of shape
    ``(n_rows, n_cols, n_replicates)``.  Returns the modal label per point and
    the share of replicates agreeing with it.
    """
    labels = classify(per_replicate, thr)  # (rows, cols, reps)
    n = labels.shape[-1]
    counts = np.stack(
        [(labels == i).sum(axis=-1) for i in range(len(REGIMES))], axis=-1
    )
    modal = np.argmax(counts, axis=-1).astype(np.int8)
    agreement = counts.max(axis=-1) / max(n, 1)
    return modal, agreement


def sensitivity(per_replicate, thr, factors=(0.667, 1.0, 1.5)):
    """Regime areas as the thresholds are scaled.

    Reported as a small table beside the regime map: if a region's area moves
    sharply with the cut point, that is a fact about the result and belongs in
    the paper rather than in a footnote.
    """
    rows = []
    for f in factors:
        modal, agreement = classify_map(per_replicate, thr.scaled(f))
        total = modal.size
        rows.append(
            {
                "factor": f,
                "areas": {
                    name: float((modal == i).sum() / total)
                    for i, name in enumerate(REGIMES)
                },
                "mean_agreement": float(agreement.mean()),
            }
        )
    return rows
