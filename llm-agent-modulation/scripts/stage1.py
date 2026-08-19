#!/usr/bin/env python3
"""Stage 1: generate the opinions and measure where they land.

Writes one JSON per theme into ``data/themes/``, plus ``index.json`` and a
readable ``OPINIONS.md``.  Cached, so a rerun costs nothing and returns the same
stimulus material; delete ``data/cache/`` to draw fresh.
"""

from __future__ import annotations

import argparse

import _cli  # noqa: F401  (imported for its sys.path side effect)

from llmmod.generate import CROSS_THEME, PER_DEGREE, build_all  # noqa: E402
from llmmod.llm import MODEL, usage_total  # noqa: E402
from llmmod.themes import THEMES  # noqa: E402

PRICE_IN, PRICE_OUT = 0.20, 1.20  # $/M tokens, gpt-5.6-luna


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--themes", nargs="*", default=None, help="subset by key")
    ap.add_argument("--per-degree", type=int, default=PER_DEGREE)
    ap.add_argument("--cross", type=int, default=CROSS_THEME,
                    help="candidates borrowed from other themes, for the 0 column")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    themes = THEMES if not args.themes else tuple(
        t for t in THEMES if t.key in args.themes)
    build_all(themes, model=args.model, per_degree=args.per_degree,
              cross=args.cross, workers=args.workers)

    u = usage_total()
    cost = (u["prompt"] * PRICE_IN + u["completion"] * PRICE_OUT) / 1e6
    print(f"\n[usage] {u['calls']} calls ({u['cached']} served from cache), "
          f"{u['prompt']:,} in / {u['completion']:,} out "
          f"({u['reasoning']:,} reasoning) -> ${cost:.3f}")


if __name__ == "__main__":
    main()
