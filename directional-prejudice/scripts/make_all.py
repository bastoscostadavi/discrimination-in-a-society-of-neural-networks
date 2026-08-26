#!/usr/bin/env python3
"""Regenerate every figure.

    python scripts/make_all.py --preset quick    # minutes, coarse grids
    python scripts/make_all.py --preset medium   # the committed figures
    python scripts/make_all.py --preset full     # publication resolution

Sweeps are cached in ``data/``, so re-running to restyle figures costs nothing.
Add ``--style iclr`` to render at the paper's column width instead, and
``--component b`` to run the credulity field rather than the status one.

The demonstration table is printed as well as plotted: the figure carries the
comparison and the table carries the numbers the appendix would quote.
"""

from __future__ import annotations

import time

from _cli import setup  # noqa: E402

import invisibility  # noqa: E402
import directional_phase  # noqa: E402


def main():
    args, preset = setup(__doc__)
    t0 = time.time()

    print("== what each field component does, and what is measured ==")
    batch, reps = invisibility.run(preset)
    invisibility.table(batch, reps)
    invisibility.residual(preset)
    invisibility.figure(batch, reps, args.style)

    print(f"\n== the ({args.component}, f_{args.component}) plane ==")
    data = directional_phase.run(preset, use_cache=not args.no_cache)
    directional_phase.report(data)
    directional_phase.figure_channels(data, args.style)
    directional_phase.figure_map(data, args.style)
    directional_phase.figure_cut(data, args.style)

    print(
        f"\nall figures written in {(time.time()-t0)/60:.1f} min "
        f"(preset={preset.name}, style={args.style}, component={args.component})"
    )


if __name__ == "__main__":
    main()
