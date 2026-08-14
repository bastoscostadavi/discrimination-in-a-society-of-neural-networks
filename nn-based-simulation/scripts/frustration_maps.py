#!/usr/bin/env python3
"""Social balance across the ``(d, f_d)`` plane.

``B_I`` and ``B_A`` measure how many triples of agents are ideologically and
affectively balanced.  They separate the collective states in a way the pair
correlations cannot: the region of reverse discrimination is a *frustrated*
state, with ``B_A < 0`` and ``B_I`` near zero -- no consistent faction structure
exists, because every agent is pulled towards the other class while its
neighbours are pulled towards theirs.  The discriminatory region, by contrast,
is almost perfectly balanced: two internally coherent, mutually hostile blocs.

Writes ``frustration_maps``.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from correlation_maps import agenda_sweeps  # noqa: E402

from ednna.plotting import panel, phase_map, save  # noqa: E402

KEYS = ("B_I", "B_A")


def figure(rows, style, name="frustration_maps"):
    fig, axes = plt.subplots(
        len(rows), len(KEYS), figsize=panel(0.75, 0.36 * len(rows) / 0.72), squeeze=False
    )
    for i, (label, P, alpha, data) in enumerate(rows):
        for j, key in enumerate(KEYS):
            ax = axes[i][j]
            phase_map(ax, data[key], data["d"], data["fd"], key, ylabel=(j == 0))
            if j == 0:
                ax.text(
                    -0.42, 0.5, f"{label}, " + rf"$\alpha={alpha:.3g}$",
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=7.5,
                )
    fig.tight_layout(pad=0.5)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    rows = agenda_sweeps(preset, use_cache=not args.no_cache)
    figure(rows, args.style)


if __name__ == "__main__":
    main()
