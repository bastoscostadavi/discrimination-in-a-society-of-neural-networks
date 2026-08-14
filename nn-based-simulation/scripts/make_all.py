#!/usr/bin/env python3
"""Regenerate every figure.

    python scripts/make_all.py --preset quick    # minutes, coarse grids
    python scripts/make_all.py --preset medium   # the committed figures
    python scripts/make_all.py --preset full     # publication resolution

Sweeps are cached in ``data/``, so re-running to restyle figures costs nothing.
Add ``--style iclr`` to render at the paper's column width instead.
"""

from __future__ import annotations

import time

from _cli import setup  # noqa: E402

import agenda_trajectories  # noqa: E402
import correlation_maps  # noqa: E402
import frustration_maps  # noqa: E402
import learning_flows  # noqa: E402
import modulation_landscape  # noqa: E402
import order_parameter_maps  # noqa: E402
import phase_diagram  # noqa: E402
import polarisation  # noqa: E402


def main():
    args, preset = setup(__doc__)
    t0 = time.time()

    print("== analytic figures ==")
    modulation_landscape.figure_surfaces(args.style)
    modulation_landscape.figure_contours(args.style)
    modulation_landscape.figure_slices(args.style)
    learning_flows.figure(args.style)

    print("== the unbiased society ==")
    polarisation.figure(
        polarisation.run(preset, preset.p_small, use_cache=not args.no_cache), args.style
    )

    print("== phase-diagram sweeps ==")
    rows = correlation_maps.agenda_sweeps(preset, use_cache=not args.no_cache)
    correlation_maps.figure(rows, args.style)
    frustration_maps.figure(rows, args.style)
    order_parameter_maps.figure(rows, args.style)

    print("== composite phase diagram ==")
    by_issues = {P: data for _, P, _, data in rows}
    phase_diagram.figure(by_issues[preset.p_small], args.style)
    phase_diagram.figure(
        by_issues[preset.p_large], args.style,
        name="phase_diagram_large_agenda", regions=False,
    )

    print("== balance trajectories ==")
    data = agenda_trajectories.trajectories(preset, use_cache=not args.no_cache)
    agenda_trajectories.figure(data, args.style)

    print(
        f"\nall figures written in {(time.time()-t0)/60:.1f} min "
        f"(preset={preset.name}, style={args.style})"
    )


if __name__ == "__main__":
    main()
