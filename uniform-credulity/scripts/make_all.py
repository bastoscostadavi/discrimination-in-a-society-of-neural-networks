#!/usr/bin/env python3
"""Regenerate every figure.

    python scripts/make_all.py --preset quick    # a minute, coarse grids
    python scripts/make_all.py --preset medium   # 64x64 at N=40
    python scripts/make_all.py --preset full     # 200x200 at N=40, the paper's
                                                 # own resolution; ~4 h on ten
                                                 # cores

Sweeps are cached in ``data/``, so re-running to restyle figures costs nothing.
Add ``--style iclr`` to render at the paper's column width instead.

The point table is printed as well as plotted: the figure carries the comparison
and the table carries the numbers an appendix would quote.
"""

from __future__ import annotations

import time

from _cli import setup  # noqa: E402

import bias_split  # noqa: E402
import uniform_phase  # noqa: E402


def main():
    args, preset = setup(__doc__)
    t0 = time.time()

    print("== what a uniform field does to the two groups it creates ==")
    batch, reps = bias_split.run(preset)
    bias_split.table(batch, reps)
    bias_split.figure(batch, reps, args.style)

    print("\n== the (a, f_a) plane ==")
    data = uniform_phase.run(preset, use_cache=not args.no_cache,
                             n_strips=args.strips)
    uniform_phase.report(data, preset.model.n_agents)
    uniform_phase.figure_channels(data, args.style)
    uniform_phase.figure_map(data, args.style)
    uniform_phase.figure_cut(data, args.style, preset.model.n_agents)

    print(f"\nall figures written in {(time.time()-t0)/60:.1f} min "
          f"(preset={preset.name}, style={args.style})")


if __name__ == "__main__":
    main()
