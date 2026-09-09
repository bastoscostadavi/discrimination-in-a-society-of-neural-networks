#!/usr/bin/env python3
"""Draw only this experiment's measured trust-sector modulation plane.

This deliberately does not read, overwrite, or restyle the sibling
``llm-agent-modulation`` F_mu experiment or the paper figure.  It reads the
paired trust rows measured in this directory and writes one standalone PDF.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

# The script is also run in headless terminals; set before importing pyplot via
# the local plotting module.
matplotlib.use("Agg", force=True)
import numpy as np

import _cli
from llmmod2 import plotting
from llmmod2.fields import fit_lam, nominal_h_mu, opinion_fields, trust_fields

ROOT = _cli.ROOT
ROWS = ROOT / "data" / "rows"
FIGURES = ROOT / "figures"


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", default="iclr", choices=("iclr", "paper"))
    args = ap.parse_args()

    F_w, F_mu = _cli.theory()
    if F_mu is None:
        raise SystemExit("ednna not importable; cannot evaluate analytic F_mu")

    opinion = load(ROWS / "opinion.jsonl")
    trust = load(ROWS / "trust.jsonl")
    seen, weights, ks = set(), [], []
    for row in trust:
        key = (row["world"], row["s"], row["k"])
        if key not in seen and not row["censored"]:
            seen.add(key); weights.append(row["weight_pre"]); ks.append(row["k"])
    lam, _, _ = fit_lam(weights, ks, trust[0]["track_total"])

    h_mu = {}
    for row in trust:
        if not row["censored"]:
            h_mu.setdefault((row["world"], row["k"]), []).append(row["weight_pre"])
    h_mu = {key: float(np.mean([trust_fields(value, value, lam)[0] for value in values]))
            for key, values in h_mu.items()}

    opinion_by_key = {}
    for row in opinion:
        if not row["censored"]:
            h_w, _ = opinion_fields(row["lean_pre"], row["lean_post"], row["sign"], lam)
            opinion_by_key[(row["world"], row["s"], row["k"], row["sign"])] = h_w

    measured = []
    for row in trust:
        key = (row["world"], row["s"], row["k"], row["sign"])
        if row["censored"] or key not in opinion_by_key:
            continue
        _, delta_mu = trust_fields(row["weight_pre"], row["weight_post"], lam)
        measured.append((opinion_by_key[key], h_mu.get((row["world"], row["k"]), float(nominal_h_mu(row["k"], row["track_total"]))), delta_mu))

    hw, hm, dm = map(np.asarray, zip(*measured))
    cap = float(np.percentile(np.abs(dm), plotting.CAP_PCT))
    alpha_mu = plotting.fit_scale(hw, hm, dm, F_mu, cap=cap)
    plotting.use_style(args.style)
    path = plotting.figure_plane(hw, hm, dm, F_mu, alpha_mu, FIGURES,
                                  name="fmu_plane", label=r"$F_\mu$")
    corr = float(np.corrcoef(dm, alpha_mu * F_mu(hw, hm))[0, 1])
    print(f"[F_mu] {len(dm)} rows; lam={lam:.3f}; alpha_mu={alpha_mu:.3f}; correlation={corr:.3f}; {path}")


if __name__ == "__main__":
    main()
