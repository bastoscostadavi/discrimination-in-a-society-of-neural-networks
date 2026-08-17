#!/usr/bin/env python3
"""Fix the parameters the source draft leaves unspecified.

The draft states no ``N``, no interaction count (its text has a literal
``Delta t = ????``), and no agenda sizes for the two rows of its correlation and
frustration maps.  ``K = 30`` is the one number that can be recovered from it
outright, from the alpha values of the trajectory figure.

Everything else is fixed here, by matching features of the draft's published
figures that do not depend on the missing numbers:

1.  **Trajectory endpoints.**  The draft's balance trajectories, digitized
    below, pin the interaction count: the whole family of curves is measured at
    one common ``Delta t``, and its value determines how far each curve has
    travelled.  We scan ``Delta t`` and report the residual against the
    digitized endpoints.
2.  **Above/below the diagonal.**  Simple agendas must finish above the
    diagonal (affective balance leads), complex ones below it, with the
    crossover near ``alpha ~ 1``.
3.  **Sign flip of the trust-class correlation** at ``d = 0``, sharp in ``d``.
4.  **The opinion-class wedge**: ``R_cw`` is large only in a triangle that opens
    towards large ``d`` and large ``f_d``, and is *smaller* for a simple agenda
    than for a complex one.

Run with ``--quick`` for a coarse version of the scan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ednna.config import ModelConfig, SweepConfig  # noqa: E402
from ednna.order_params import balance  # noqa: E402
from ednna.society import SocietyBatch  # noqa: E402
from ednna.sweep import sweep  # noqa: E402

#: Endpoints digitized from the source draft's balance-trajectory figure, as
#: ``alpha -> (B_I, B_A)``.  Read off the printed axes to about +-0.03; the
#: three curves that bunch near (0.5, 0.94) cannot be told apart reliably and
#: are given the same reading.
DRAFT_ENDPOINTS = {
    0.03: (0.02, 0.99),
    0.17: (0.24, 0.98),
    0.23: (0.54, 0.97),
    0.33: (0.55, 0.94),
    0.50: (0.55, 0.94),
    0.67: (0.62, 0.92),
    1.67: (0.80, 0.89),
    3.33: (0.90, 0.65),
    333.33: (0.97, 0.68),
}

ISSUES = (1, 5, 7, 10, 15, 20, 50, 100, 10000)
K = 30


def scan_interaction_count(n_agents, checkpoints, n_repeats, issues=ISSUES, seed=99):
    """B_I, B_A at each checkpoint for each agenda size. Returns a nested dict."""
    out = {}
    for i, P in enumerate(issues):
        times = [int(c * n_agents * (n_agents - 1)) for c in checkpoints]
        batch = SocietyBatch(
            n_agents=n_agents,
            n_dim=K,
            n_issues=P,
            d=np.zeros(n_repeats),
            f_d=np.zeros(n_repeats),
            seed=seed + i,
        )
        samples = batch.run(times[-1], measure_at=times, measure_fn=balance)
        out[P] = {
            c: (float(s["B_I"].mean()), float(s["B_A"].mean()))
            for c, s in zip(checkpoints, samples)
        }
        print(
            f"  P={P:>6d} (alpha={P/K:8.4g}): "
            + "  ".join(
                f"t={c}:({v[0]:+.2f},{v[1]:+.2f})" for c, v in out[P].items()
            ),
            flush=True,
        )
    return out


def draft_target(P):
    """The digitized endpoint for the agenda size ``P``, matched on alpha."""
    alpha = P / K
    key = min(DRAFT_ENDPOINTS, key=lambda a: abs(np.log(a) - np.log(alpha)))
    return DRAFT_ENDPOINTS[key]


def residual(scan, checkpoint):
    """RMS distance between simulated and digitized endpoints at one Delta t."""
    errs = []
    for P, per_t in scan.items():
        target = draft_target(P)
        got = per_t[checkpoint]
        errs.append((got[0] - target[0]) ** 2 + (got[1] - target[1]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def check_diagonal_ordering(scan, checkpoint):
    """Simple agendas above the diagonal, complex ones below."""
    rows = []
    for P, per_t in sorted(scan.items()):
        B_I, B_A = per_t[checkpoint]
        rows.append((P / K, B_I, B_A, "above" if B_A > B_I else "below"))
    ok_small = all(r[3] == "above" for r in rows if r[0] < 0.7)
    ok_large = all(r[3] == "below" for r in rows if r[0] > 2.0)
    return ok_small, ok_large, rows


def check_phase_signatures(model, n_grid=24, n_workers=8):
    """Sign flip of R_muc, and the R_cw wedge, on a coarse grid."""
    cfg = SweepConfig(n_d=n_grid, n_fd=n_grid, batch_size=256, n_workers=n_workers)
    out = {}
    for label, P in (("small", 5), ("large", 100)):
        data = sweep(model.with_(n_issues=P), cfg, tag=f"calib_P{P}", verbose=False)
        d, fd = data["d"], data["fd"]
        neg = data["R_muc"][:, d < -0.2].mean()
        pos = data["R_muc"][:, d > 0.2].mean()
        # the wedge: R_cw in the high-d, high-f_d corner vs the rest
        corner = data["R_cw"][np.ix_(fd > 0.6, d > 0.5)].mean()
        elsewhere = data["R_cw"][np.ix_(fd < 0.4, d < 0.0)].mean()
        out[label] = {
            "R_muc(d<0)": neg,
            "R_muc(d>0)": pos,
            "R_cw corner": corner,
            "R_cw elsewhere": elsewhere,
            "B_I(d<0)": data["B_I"][:, d < -0.2].mean(),
            "B_A(d<0)": data["B_A"][:, d < -0.2].mean(),
            "B_I(d>0)": data["B_I"][:, d > 0.2].mean(),
        }
        print(f"  {label} agenda (P={P}):")
        for k, v in out[label].items():
            print(f"    {k:<16s} {v:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-agents", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    # bracket the residual minimum: the trajectory family moves fast early and
    # crawls afterwards, so the informative range is well under 10^3
    checkpoints = [60, 125, 250] if args.quick else [60, 125, 250, 500, 1000]
    issues = (1, 5, 100, 10000) if args.quick else ISSUES

    print(f"1. interaction-count scan (N={args.n_agents}, K={K}, "
          f"{args.repeats} repeats, checkpoints in interactions per channel)")
    scan = scan_interaction_count(args.n_agents, checkpoints, args.repeats, issues)

    print("\n2. residual against the draft's digitized endpoints")
    best, best_res = None, np.inf
    for c in checkpoints:
        res = residual(scan, c)
        ok_small, ok_large, _ = check_diagonal_ordering(scan, c)
        flag = "" if (ok_small and ok_large) else "  (diagonal ordering violated)"
        print(f"  Delta t = {c:>5d} int/channel: rms = {res:.3f}{flag}")
        if res < best_res:
            best, best_res = c, res
    print(f"  -> best Delta t = {best} interactions per channel (rms {best_res:.3f})")

    print("\n3. diagonal ordering at the best Delta t")
    ok_small, ok_large, rows = check_diagonal_ordering(scan, best)
    for alpha, B_I, B_A, side in rows:
        print(f"  alpha={alpha:8.4g}: B_I={B_I:+.3f} B_A={B_A:+.3f}  {side}")
    print(f"  simple agendas above the diagonal: {ok_small}")
    print(f"  complex agendas below the diagonal: {ok_large}")

    print("\n4. phase-diagram signatures")
    model = ModelConfig(n_agents=args.n_agents, interactions_per_channel=float(best))
    check_phase_signatures(model, n_workers=args.workers)


if __name__ == "__main__":
    main()
