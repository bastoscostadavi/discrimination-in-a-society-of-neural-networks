"""Figure style and the shared plot elements.

Two things here are not merely cosmetic.

**Uncertainty is shown, not stored and forgotten.**  Every map has a companion
standard-error map, and the regime map is drawn with low-agreement points
hatched, so a reader can see where the boundary is genuinely uncertain rather
than being handed a crisp categorical picture that the data does not support.

**Maps put ``f_d = 0`` at the top**, matching the orientation the source material
uses, so the two can be compared directly.  This is done through ``extent``
rather than by inverting the axis; adding an ``invert_yaxis`` on top would flip
them back, which is exactly the kind of silent error a test should catch, and
one does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm

from .phases import REGIME_LABELS, REGIMES

__all__ = [
    "FIGURE_DIR",
    "use_style",
    "text_width",
    "save",
    "phase_map",
    "regime_map",
    "add_axes_labels",
    "LABELS",
    "CMAPS",
    "RANGES",
]

FIGURE_DIR = Path(__file__).resolve().parents[3] / "figures"

LABELS = {
    "C_CT": r"$C_{\mathrm{CT}}$",
    "C_CO": r"$C_{\mathrm{CO}}$",
    "C_TO": r"$C_{\mathrm{TO}}$",
    "P_O": r"$\hat{P}_{O}$",
    "P_T": r"$\hat{P}_{T}$",
    "P_O_hat": r"$\hat{P}_{O}$",
    "P_T_hat": r"$\hat{P}_{T}$",
    "B_O": r"$B_{O}$",
    "B_T": r"$B_{T}$",
    "B_O_sign": r"$B_{O}^{\mathrm{sgn}}$",
    "B_T_sign": r"$B_{T}^{\mathrm{sgn}}$",
    "A_O": r"$A_{O}$",
    "A_T": r"$A_{T}$",
}

CMAPS = {
    "C_CT": "RdBu_r",
    "C_CO": "Greens",
    "C_TO": "Blues",
    "P_O_hat": "Purples",
    "P_T_hat": "Purples",
    "B_O": "Purples",
    "B_T": "OrRd",
    "B_O_sign": "Purples",
    "B_T_sign": "OrRd",
}

RANGES = {
    "C_CT": (-1.0, 1.0),
    "C_CO": (-1.0, 1.0),
    "C_TO": (-1.0, 1.0),
    "P_O_hat": (0.0, 1.0),
    "P_T_hat": (0.0, 1.0),
    "B_O": (-1.0, 1.0),
    "B_T": (-1.0, 1.0),
    "B_O_sign": (-1.0, 1.0),
    "B_T_sign": (-1.0, 1.0),
}

#: One colour per regime.  Chosen to stay distinguishable in greyscale and for
#: the common forms of colour blindness: the ordering is light -> saturated as
#: structure increases, so the map reads even without hue.
REGIME_COLOURS = {
    "weakly_structured": "#f0f0f0",
    "class_uncorrelated_polarized": "#9ecae1",
    "counter_aligned_frustrated": "#54278f",
    "discriminatory_ideological": "#fdae6b",
    "discriminatory_class_dominant": "#d7301f",
}

_STYLE = {"name": "paper"}

_BASE_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
}


def use_style(style="paper"):
    """Activate a figure style. ``iclr`` targets the submission column width."""
    _STYLE["name"] = style
    mpl.rcParams.update(_BASE_RC)


def text_width():
    return 5.5 if _STYLE["name"] == "iclr" else 6.3


def save(fig, name, style=None):
    style = style or _STYLE["name"]
    out = FIGURE_DIR / style
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path.relative_to(FIGURE_DIR.parent)}")
    return path


def add_axes_labels(ax, d, fd, ylabel=True, sparse=False):
    ax.set_xlabel(r"$d$")
    if ylabel:
        ax.set_ylabel(r"$f_d$")
    ticks = (-1, 0, 1) if sparse else (-1, -0.5, 0, 0.5, 1)
    ax.set_xticks(ticks)
    ax.set_yticks((0, 0.5, 1) if sparse else (0, 0.25, 0.5, 0.75, 1))


def phase_map(ax, data, d, fd, key, vmin=None, vmax=None, cmap=None, colorbar=True,
              ylabel=True, sparse=False, title=None):
    """One order parameter over the ``(d, f_d)`` plane."""
    lo, hi = RANGES.get(key, (None, None))
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    norm = None
    if vmin is not None and vmax is not None and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        vmin = vmax = None
    im = ax.imshow(
        data,
        origin="upper",
        extent=[d[0], d[-1], fd[-1], fd[0]],
        aspect="auto",
        cmap=cmap or CMAPS.get(key, "viridis"),
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    add_axes_labels(ax, d, fd, ylabel=ylabel, sparse=sparse)
    ax.set_title(title if title is not None else LABELS.get(key, key), fontsize=8, pad=3)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
    return im


def regime_map(ax, modal, agreement, d, fd, hatch_below=0.7, legend=True):
    """The categorical map, with contested points marked.

    Points where fewer than ``hatch_below`` of replicates agree on the label are
    overlaid with hatching.  Those are where the boundary genuinely sits, and a
    flat categorical map would hide them.
    """
    cmap = ListedColormap([REGIME_COLOURS[r] for r in REGIMES])
    extent = [d[0], d[-1], fd[-1], fd[0]]
    ax.imshow(
        modal, origin="upper", extent=extent, aspect="auto",
        cmap=cmap, vmin=-0.5, vmax=len(REGIMES) - 0.5,
    )
    contested = np.where(agreement < hatch_below, 1.0, np.nan)
    ax.contourf(
        contested,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        extent=extent,
        origin="upper",
    )
    for c in ax.collections:
        c.set_edgecolor("0.25")
        c.set_linewidth(0.0)
    add_axes_labels(ax, d, fd)
    if legend:
        handles = [
            mpl.patches.Patch(facecolor=REGIME_COLOURS[r], edgecolor="0.4",
                              linewidth=0.4, label=REGIME_LABELS[r])
            for r in REGIMES
        ]
        handles.append(
            mpl.patches.Patch(facecolor="white", edgecolor="0.25", hatch="////",
                              linewidth=0.4, label="contested")
        )
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
            ncol=2, fontsize=6, frameon=False,
        )
    return ax
