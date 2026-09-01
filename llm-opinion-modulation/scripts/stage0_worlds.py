"""Stage 0: invent the worlds and freeze them.

Run once.  The result is committed, because every later stage is built on this
material and a regenerated set would silently change the stimulus under a
half-finished analysis.  The call is cached, so a rerun with the same count and
prompt costs nothing and returns the same worlds.
"""

from __future__ import annotations

import argparse

import _cli  # noqa: F401  - path setup

from llmmod2 import worlds
from llmmod2.llm import usage_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen set")
    args = ap.parse_args()

    if worlds.INDEX.is_file() and not args.force:
        raise SystemExit(f"{worlds.INDEX} exists; pass --force to replace it")

    raw = worlds.generate(args.count)
    path = worlds.freeze(raw)
    print(f"wrote {len(raw)} worlds to {path}")
    for w in raw:
        pair = w["issues"][0]
        print(f"  {w['key']:12s} {w['predicate']:<28s} "
              f"{pair['a']} / {pair['b']}")
    u = usage_total()
    print(f"calls={u['calls']} cached={u['cached']} ${u['dollars']:.4f}")


if __name__ == "__main__":
    main()
