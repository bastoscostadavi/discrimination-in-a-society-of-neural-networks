#!/usr/bin/env python3
"""Draw the trust curve from a completed run.

Separate from the run so a restyle costs nothing: the measurement is on disk and
this reads it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _cli import ROOT, theory  # noqa: E402

from llmmod.plotting import figure_trust_summary, use_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="curve")
    ap.add_argument("--style", default="iclr")
    ap.add_argument("--name", default="trust_llm")
    args = ap.parse_args()
    use_style(args.style)

    path = ROOT / "data" / "trust" / f"{args.tag}.rows.jsonl"
    rows = [json.loads(line) for line in path.open()]
    get = lambda k: np.array([r[k] if r[k] is not None else np.nan  # noqa: E731
                              for r in rows], dtype=float)
    _, F_mu = theory()
    if F_mu is None:
        raise SystemExit("ednna not importable; the theory curves come from it")
    print(f"{len(rows)} rows; h_w identified for "
          f"{int(np.isfinite(get('h_w_direct') if 'h_w_direct' in rows[0] else get('h_w')).sum())} "
          f"of them")
    # the direct conviction read where it exists: it is defined everywhere the
    # answer is, while the inversion is singular at neutral trust and censored
    # wherever the agent expects to disagree more than c = 1 allows
    h_w = get("h_w_direct") if "h_w_direct" in rows[0] else get("h_w")
    figure_trust_summary(h_w, get("h_mu"), get("delta_mu"), get("sign"), F_mu,
                         name=args.name)


if __name__ == "__main__":
    main()
