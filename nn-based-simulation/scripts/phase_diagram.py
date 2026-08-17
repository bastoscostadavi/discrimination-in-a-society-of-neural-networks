#!/usr/bin/env python3
"""The phase diagram: three order parameters composited into one picture.

The three pair correlations are mapped to the three colour channels -- red for
trust-class, green for opinion-class, blue for opinion-trust -- so that the
colour of a point names the collective state of the society:

(I)   dark          reverse discrimination.  A frustrated, spin-glass-like state:
                    agents favour the out-group, no coherent faction can form,
                    and no correlation survives.
(II)  blue          neutral.  The society polarizes, but along a split that has
                    nothing to do with class.
(III) pale          discriminatory, ideological.  Distrust follows class *and*
                    opinions follow class: two coherent, opposed camps.
(IV)  magenta       discriminatory, class-only.  Distrust follows class while
                    opinions do not: hostility needs no disagreement to sustain
                    it.

Writes ``phase_diagram``.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.plotting import add_phase_axes, panel, rgb_composite, save  # noqa: E402
from ednna.sweep import sweep  # noqa: E402

#: label -> (d, f_d) placement.  The draft puts (IV) at small ``f_d``, but there the
#: signal is weak -- R_muc is only 0.27 at (d, f_d) = (0.62, 0.26) and B_A has fallen
#: to 0.56, so the point is a sub-quorum crossover rather than the class-only state.
#: The class-only character is what strengthens with ``d``: R_cw/R_muc falls
#: monotonically in ``d`` at every ``f_d``, reaching 0.34 at (1.0, 0.9) where R_muc is
#: 0.86 and the trust network is still organized (B_A = 0.90).  (III) and (IV) are
#: therefore placed on the same high-``f_d`` traverse the text argues along.
REGIONS = {
    "(I)": (-0.55, 0.78),
    "(II)": (-0.06, 0.42),
    "(III)": (0.45, 0.86),
    "(IV)": (0.88, 0.86),
}


def figure(data, style, name="phase_diagram", regions=True):
    rgb = rgb_composite(data["R_muc"], data["R_cw"], data["R_wmu"])
    d, fd = data["d"], data["fd"]
    fig, ax = plt.subplots(figsize=panel(0.55, 0.52/0.62))
    ax.imshow(rgb, origin="upper", extent=[d[0], d[-1], fd[-1], fd[0]], aspect="auto")
    add_phase_axes(ax)
    if regions:
        for label, (x, y) in REGIONS.items():
            ax.text(
                x, y, label, color="white", fontsize=8, ha="center", va="center",
                path_effects=None,
            )
    # channel key, so the colours are readable without the caption
    for i, (txt, colour) in enumerate(
        ((r"$R_{\mu,c}$", "#ff5555"), (r"$R_{c,w}$", "#55dd55"), (r"$R_{w,\mu}$", "#7777ff"))
    ):
        ax.text(
            0.015, 0.03 + 0.075 * i, txt, transform=ax.transAxes, color=colour,
            fontsize=6.5, ha="left", va="bottom",
        )
    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    model = preset.model.with_(n_issues=preset.p_small)
    data = sweep(model, preset.sweep, tag=f"P{preset.p_small}", use_cache=not args.no_cache)
    figure(data, args.style)
    # the same composite for the complex agenda, for comparison
    model_large = preset.model.with_(n_issues=preset.p_large)
    data_large = sweep(
        model_large, preset.sweep, tag=f"P{preset.p_large}", use_cache=not args.no_cache
    )
    figure(data_large, args.style, name="phase_diagram_large_agenda", regions=False)


if __name__ == "__main__":
    main()
