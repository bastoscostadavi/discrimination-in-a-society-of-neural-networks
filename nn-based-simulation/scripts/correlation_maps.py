#!/usr/bin/env python3
"""The three pair correlations across the ``(d, f_d)`` plane.

Each row is one agenda size: a simple agenda of a few issues on top, a complex
one below.  Reading the columns:

``R_wmu`` (opinion-trust)
    High wherever the society has polarized at all, with or without classes.

``R_muc`` (trust-class)
    The discrimination order parameter.  Positive when in-group agents trust
    each other and distrust the out-group; negative under reverse
    discrimination, where agents favour the other class.

``R_cw`` (opinion-class)
    Whether the ideological split coincides with the class split.  It is the
    parameter that separates the two discriminatory phases: high where ideology
    still matters, low where class alone drives distrust.

Writes ``correlation_maps``.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.plotting import panel, phase_map, save  # noqa: E402
from ednna.sweep import sweep  # noqa: E402

KEYS = ("R_wmu", "R_muc", "R_cw")


def agenda_sweeps(preset, use_cache=True):
    """Sweeps for the small and large agenda, as ``[(label, data), ...]``."""
    out = []
    for label, P in (("small agenda", preset.p_small), ("large agenda", preset.p_large)):
        model = preset.model.with_(n_issues=P)
        print(f"[correlations] {label}: P={P}, alpha={model.alpha:.3g}")
        data = sweep(model, preset.sweep, tag=f"P{P}", use_cache=use_cache)
        out.append((label, P, model.alpha, data))
    return out


def figure(rows, style, name="correlation_maps"):
    fig, axes = plt.subplots(
        len(rows), len(KEYS), figsize=panel(1.0, 0.34 * len(rows)), squeeze=False
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
