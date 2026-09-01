"""Turn the measured null points into the paper's fields, and draw the figure.

Nothing is measured here; this is the whole of the analysis, and it is separate
from the stages that call the model so that it can be rerun on the committed
rows without touching the API.

The chain is short.  Stage 3's silent arm gives the weight the receiver puts on
a colleague with a known track record; requiring those weights to reproduce the
distrust the track records state fixes ``lam``, the probit value of one piece of
evidence, and the quality of that requirement is reported rather than assumed.
With ``lam`` in hand every null point becomes a field and every pair of null
points becomes an update.  One positive scale is then fitted between the
measured updates and ``F_w`` -- the unobservable covariance -- and that is the
only quantity taken from the comparison rather than from the measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import _cli
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from llmmod2 import plotting
from llmmod2.fields import fit_lam, h_of, nominal_h_mu, opinion_fields, trust_fields

ROOT = _cli.ROOT
ROWS = ROOT / "data" / "rows"
FIGURES = ROOT / "figures"

#: The trust sector's measurement, from the sibling experiment.  Figure 2 puts
#: the two sectors side by side, and two panels of one figure have to be drawn
#: by one piece of code or they drift apart; the sibling's rows are read here
#: rather than its finished figure being copied.
SIBLING_ROWS = (ROOT.parent / "llm-agent-modulation" / "data" / "trust"
                / "curve.rows.jsonl")

#: Where the merged two-sector figure goes.  It is the paper's Figure 2 and
#: nothing else -- half of it is the sibling's measurement -- so it is written to
#: the paper directly rather than kept here.  This experiment's own copy of the
#: opinion sector is ``figures/iclr/opinion_plane.pdf``.
PAPER_FIGURES = ROOT.parent / "paper" / "figures"

#: Ticks on both axes of Figure 2, given rather than derived: the frame is
#: deliberately wider than the last tick.
TICKS = (-2, -1, 0, 1, 2)


def sibling_trust():
    """``(h_w, h_mu, Delta h_mu)`` from the sibling experiment, or ``None``.

    The conviction is the direct reading rather than the inversion, which is the
    choice that experiment's appendix argues for: it is defined for every
    condition, while the inversion is singular at neutral trust.
    """
    if not SIBLING_ROWS.is_file():
        return None
    rows = [json.loads(line) for line in SIBLING_ROWS.open() if line.strip()]
    key = "h_w_direct" if "h_w_direct" in rows[0] else "h_w"

    def col(k):
        return np.array([r[k] if r[k] is not None else np.nan for r in rows],
                        dtype=float)

    hw, hmu, d = col(key), col("h_mu"), col("delta_mu")
    ok = np.isfinite(hw) & np.isfinite(hmu) & np.isfinite(d)
    return hw[ok], hmu[ok], d[ok]


def _load(path):
    if not path.is_file():
        raise SystemExit(f"missing {path}; run the stage that writes it")
    return [json.loads(line) for line in path.open() if line.strip()]


def calibrate(trust_rows):
    """``lam`` from the track records, plus how well they are honoured."""
    seen, weights, ks = set(), [], []
    for r in trust_rows:
        key = (r["world"], r["s"], r["k"])
        if key in seen or r["censored"]:
            continue
        seen.add(key)
        weights.append(r["weight_pre"])
        ks.append(r["k"])
    lam, rss, r = fit_lam(weights, ks, trust_rows[0]["track_total"])
    return lam, rss, r, np.array(weights), np.array(ks)


def measured_h_mu(trust_rows, lam):
    """Prior distrust per world and track record, from the silent arm."""
    acc = defaultdict(list)
    for r in trust_rows:
        if not r["censored"]:
            acc[(r["world"], r["k"])].append(r["weight_pre"])
    return {key: float(-h_of(np.mean(v), lam)) for key, v in acc.items()}


def build(opinion_rows, trust_rows, lam, h_mu_map, total):
    """One record per measured opinion update, with its trust partner if any."""
    d_mu = {}
    for r in trust_rows:
        if not r["censored"]:
            _, dm = trust_fields(r["weight_pre"], r["weight_post"], lam)
            d_mu[(r["world"], r["s"], r["k"], r["sign"])] = dm

    out = []
    for r in opinion_rows:
        if r["censored"]:
            continue
        h_w, dh_w = opinion_fields(r["lean_pre"], r["lean_post"], r["sign"], lam)
        key = (r["world"], r["k"])
        h_mu = h_mu_map.get(key)
        if h_mu is None:
            h_mu = float(nominal_h_mu(r["k"], total))
        out.append({**r, "h_w": h_w, "dh_w": dh_w, "h_mu": h_mu,
                    "h_mu_nominal": float(nominal_h_mu(r["k"], total)),
                    "dh_mu": d_mu.get((r["world"], r["s"], r["k"], r["sign"]))})
    return out


def report(rec, F_w, F_mu, alpha, lam, calib, alpha_plain=None):
    h_w = np.array([x["h_w"] for x in rec])
    h_mu = np.array([x["h_mu"] for x in rec])
    d_w = np.array([x["dh_w"] for x in rec])
    pred = alpha * F_w(h_w, h_mu)

    lam_v, rss, r_cal, _, _ = calib
    print(f"\n  rows                          {len(rec)}")
    print(f"  lam (probit per piece)        {lam:.3f}   "
          f"track-record calibration r = {r_cal:.2f}, rss = {rss:.2f}")
    print(f"  |h_w| reach                   "
          f"median {np.median(np.abs(h_w)):.2f}, max {np.abs(h_w).max():.2f}")
    print(f"  |h_mu| reach                  "
          f"median {np.median(np.abs(h_mu)):.2f}, max {np.abs(h_mu).max():.2f}")
    print(f"  alpha                         {alpha:.3f}"
          + (f"   (unclipped least squares would give {alpha_plain:.3f}, "
             f"fitting the divergence)" if alpha_plain else ""))
    print(f"  corr(measured, F_w)           {np.corrcoef(d_w, pred)[0, 1]:.3f}")
    print(f"  rank corr(measured, F_w)      {spearmanr(d_w, pred).statistic:.3f}"
          f"   <- scale-free")
    print(f"  sign agreement                "
          f"{np.mean(np.sign(d_w) == np.sign(pred)) * 100:.1f}%   <- scale-free")

    # P1: the sign is carried by trust, not by agreement
    print("\n  P1  trust gate")
    for label, m in (("distrusted emitter (h_mu > 0.3)", h_mu > 0.3),
                     ("trusted emitter   (h_mu < -0.3)", h_mu < -0.3)):
        want = -1 if "distrusted" in label else +1
        if m.sum():
            print(f"      {label}: mean dh_w = {d_w[m].mean():+.3f}, "
                  f"{np.mean(np.sign(d_w[m]) == want) * 100:.1f}% of "
                  f"{m.sum()} in the predicted direction")
    for label, m in (("agreeing messages", h_w > 0), ("disagreeing messages", h_w < 0)):
        if m.sum() > 3:
            frac = np.mean(np.sign(d_w[m]) == -np.sign(h_mu[m]))
            print(f"      {label}: sign follows -h_mu in {frac * 100:.1f}% of "
                  f"{m.sum()}")

    # P2: the update varies with conviction, and in opposite directions on the
    # two sides of the trust axis.  Pooling the two sides cancels the effect
    # exactly, which is why it is reported per side.
    print("\n  P2  conviction dependence, per side of the trust axis")
    for label, side in (("trusted   (h_mu < -0.3)", h_mu < -0.3),
                        ("distrusted(h_mu > +0.3)", h_mu > 0.3)):
        lo = side & (h_w < 0)
        hi = side & (h_w > 0)
        if lo.sum() and hi.sum():
            print(f"      {label}: dh_w = {d_w[lo].mean():+.3f} when it "
                  f"disagreed ({lo.sum()}), {d_w[hi].mean():+.3f} when it "
                  f"agreed ({hi.sum()})")
    pooled_lo = (np.abs(h_mu) > 0.3) & (np.abs(h_w) <= np.median(np.abs(h_w)))
    pooled_hi = (np.abs(h_mu) > 0.3) & (np.abs(h_w) > np.median(np.abs(h_w)))
    print(f"      pooled over both sides, |dh_w| is "
          f"{np.abs(d_w[pooled_lo]).mean():.3f} vs "
          f"{np.abs(d_w[pooled_hi]).mean():.3f} -- the two sides cancel")

    # P3: dissonance amplification
    print("\n  P3  dissonance amplification")
    dis = ((h_w > 0) & (h_mu > 0)) | ((h_w < 0) & (h_mu < 0))
    con = ((h_w > 0) & (h_mu < 0)) | ((h_w < 0) & (h_mu > 0))
    if dis.sum() and con.sum():
        print(f"      |dh_w| dissonant {np.abs(d_w[dis]).mean():.3f} ({dis.sum()}) "
              f"vs consonant {np.abs(d_w[con]).mean():.3f} ({con.sum()})  "
              f"ratio {np.abs(d_w[dis]).mean() / max(np.abs(d_w[con]).mean(), 1e-9):.2f}x")

    # P4/P5: the two sectors together.  This is where the design runs out, and
    # the reason is in the paper's own equation.  The trust update is
    # ``F_mu * V / gamma_V`` and ``V`` is the variance of the receiver's belief
    # about the emitter; a track record of TRACK_TOTAL settled questions makes
    # that belief very nearly certain, so one further exchange -- on an invented
    # question whose answer is never revealed -- has almost nothing to move.
    # The measured weights bear this out: their *level* tracks the stated
    # reliability almost exactly (the calibration r above), while their *change*
    # is an order of magnitude smaller than the opinion update and does not
    # recover the sign of F_mu.  A short-record probe (2 of 4 rather than 10 of
    # 20) does recover it, which says the limit is the settled prior and not the
    # readout.
    pair = [x for x in rec if x["dh_mu"] is not None]
    print(f"\n  P4/P5  both sectors on {len(pair)} conditions")
    if len(pair) > 5:
        hw = np.array([x["h_w"] for x in pair])
        hm = np.array([x["h_mu"] for x in pair])
        dw = np.array([x["dh_w"] for x in pair])
        dm = np.array([x["dh_mu"] for x in pair])
        a_mu = float(np.sum(dm * F_mu(hw, hm)) / np.sum(F_mu(hw, hm) ** 2))
        print(f"      alpha_w = {alpha:.3f}, alpha_mu = {a_mu:.3f}, "
              f"ratio {alpha / a_mu if a_mu else float('nan'):.2f}")
        print(f"      corr(dh_mu, F_mu)  {np.corrcoef(dm, F_mu(hw, hm))[0, 1]:.3f}")
        ok = (np.abs(dw) > 1e-6) & (np.abs(dm) > 1e-6)
        if ok.sum() > 3:
            ratio = np.log(np.abs(dw[ok]) / np.abs(dm[ok]))
            gap = (hw - hm)[ok]
            print(f"      corr(log |dh_w/dh_mu|, h_w - h_mu)  "
                  f"{np.corrcoef(gap, ratio)[0, 1]:.3f}   "
                  f"(the crossover: negative is the prediction)")
        print(f"      |dh_w| {np.abs(dw).mean():.3f} vs |dh_mu| "
              f"{np.abs(dm).mean():.3f} -- the trust sector barely moves, "
              f"which is what a settled prior predicts")
        print("      NOT SUPPORTED: the symmetry and the crossover need a "
              "looser trust prior than a 20-question record leaves.")
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", default="iclr", choices=("iclr", "paper"))
    ap.add_argument("--paper-dir", type=Path, default=PAPER_FIGURES,
                    help="where the merged Figure 2 is written")
    # Both panels of Figure 2 share one frame, so the two sectors can be read
    # against each other directly and one y axis serves both.  It is a crop, not
    # a rescaling: the opinion measurement reaches |h_w| = 3, and the count of
    # what falls outside is printed on every run rather than left to be
    # discovered.  The frame is opened half a unit past the last tick, which
    # cuts what is lost and costs nothing on the axis.  --lim 3 draws
    # essentially all of it; --lim 0 frames each panel on its own data.
    ap.add_argument("--lim", type=float, default=2.25,
                    help="shared axis range for both panels of Figure 2 "
                         "(default 2.25; 0 frames each panel on its own data)")
    ap.add_argument("--cuts", action="store_true",
                    help="also write the two one-dimensional cuts")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    F_w, F_mu = _cli.theory()
    if F_w is None:
        raise SystemExit("ednna not importable; the theory curves come from it")

    opinion = _load(ROWS / "opinion.jsonl")
    trust = _load(ROWS / "trust.jsonl")
    calib = calibrate(trust)
    lam = calib[0]
    h_mu_map = measured_h_mu(trust, lam)
    rec = build(opinion, trust, lam, h_mu_map, opinion[0]["track_total"])

    h_w = np.array([x["h_w"] for x in rec])
    h_mu = np.array([x["h_mu"] for x in rec])
    d_w = np.array([x["dh_w"] for x in rec])
    # One scale for the opinion sector, fitted at the clip its own panel is
    # drawn at, and used everywhere: in the report and in both figures it
    # appears in.  Two numbers for the same constant is one more than the paper
    # can quote.
    cap = float(np.percentile(np.abs(d_w[np.isfinite(d_w)]), plotting.CAP_PCT))
    alpha = plotting.fit_scale(h_w, h_mu, d_w, F_w, cap=cap)
    alpha_plain = plotting.fit_scale(h_w, h_mu, d_w, F_w)
    report(rec, F_w, F_mu, alpha, lam, calib, alpha_plain)

    (ROWS / "analysis.json").write_text(json.dumps(
        {"lam": lam, "alpha": alpha, "calibration_r": calib[2],
         "rows": [{k: v for k, v in x.items() if k not in ("pre", "post")}
                  for x in rec]}, indent=2))
    if args.stats_only:
        return

    plotting.use_style(args.style)
    plotting.figure_plane(h_w, h_mu, d_w, F_w, alpha, FIGURES)
    if args.cuts:
        plotting.figure_gate(h_w, h_mu, d_w, F_w, alpha, FIGURES)
        plotting.figure_conviction(h_w, h_mu, d_w, F_w, alpha, FIGURES)

    # The paper's Figure 2: the two sectors side by side.  Its right-hand panel
    # is the sibling's measurement, redrawn here rather than copied, because two
    # panels of one figure have to be drawn by one piece of code or they drift
    # apart.  It is written to the paper and not kept here -- half of what is in
    # it was not measured by this experiment.
    if not args.paper_dir.is_dir():
        print(f"\n  [figure 2] {args.paper_dir} not found; not written")
        return
    trust = sibling_trust()
    if trust is None:
        print(f"\n  [figure 2] {SIBLING_ROWS} not found; not written")
        return
    t_hw, t_hmu, t_d = trust
    t_cap = float(np.percentile(np.abs(t_d), plotting.CAP_PCT))
    alpha_mu = plotting.fit_scale(t_hw, t_hmu, t_d, F_mu, cap=t_cap)
    print(f"\n  Figure 2: alpha_w = {alpha:.3f}, alpha_mu = {alpha_mu:.3f}"
          f"   (each fitted at its own panel's clip)")
    plotting.figure_pair([(F_w, r"$F_w$", h_w, h_mu, d_w, alpha),
                          (F_mu, r"$F_\mu$", t_hw, t_hmu, t_d, alpha_mu)],
                         args.paper_dir, lim=args.lim or None,
                         ticks=TICKS)


if __name__ == "__main__":
    main()
