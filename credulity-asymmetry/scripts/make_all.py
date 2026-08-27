#!/usr/bin/env python3
"""Regenerate every figure.

    python scripts/make_all.py --preset quick    # minutes, coarse grids
    python scripts/make_all.py --preset medium   # 64x64, about half an hour
    python scripts/make_all.py --preset full     # the committed figures:
                                                 # 200x200 at N=40, ~4 hours

``full`` is the resolution of the paper's own phase diagram and is this
directory's default, since producing the plane at that resolution is what the
directory is for.  Sweeps are cached in ``data/``, so re-running to restyle
figures costs nothing.  Add ``--style paper`` to render at the source draft's
proportions instead of the ICLR column width, and ``--component c`` to run the
status field rather than the credulity one -- which reproduces the sibling
directory's result from this code, and is the check that the two are mirrors.

The demonstration table is printed as well as plotted: the figure carries the
comparison and the table carries the numbers the appendix would quote.
"""

from __future__ import annotations

import time

from _cli import setup  # noqa: E402

import invisibility  # noqa: E402
import cred_asymmetry  # noqa: E402


def main():
    args, preset = setup(__doc__)
    t0 = time.time()

    print("== what each field component does, and what is measured ==")
    batch, reps = invisibility.run(preset)
    invisibility.table(batch, reps)
    invisibility.residual(preset)
    invisibility.figure(batch, reps, args.style)

    print(f"\n== the ({args.component}, f_{args.component}) plane ==")
    data = cred_asymmetry.run(preset, use_cache=not args.no_cache,
                              n_strips=args.strips)
    cred_asymmetry.report(data)
    cred_asymmetry.figure_channels(data, args.style)
    cred_asymmetry.figure_map(data, args.style)
    cred_asymmetry.figure_cut(data, args.style)

    print(
        f"\nall figures written in {(time.time()-t0)/60:.1f} min "
        f"(preset={preset.name}, style={args.style}, component={args.component})"
    )


if __name__ == "__main__":
    main()
