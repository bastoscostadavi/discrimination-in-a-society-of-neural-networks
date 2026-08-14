"""Shared figure style, colour maps and output helpers.

Two output styles are supported.  ``--style paper`` matches the proportions of
the source draft (RevTeX, narrow single-column figures); ``--style iclr``
targets the ICLR single-column text width of 5.5 inches so that figures can be
included at ``width=\\linewidth`` with no rescaling.  Both write PDF (for
LaTeX) and PNG (for quick viewing) of every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "FIGURE_DIR",
    "use_style",
    "panel",
    "BODY_PT",
    "text_width",
    "save",
    "phase_map",
    "framed_axes",
    "HIST_BLUE",
    "HIST_RED",
    "add_phase_axes",
    "CMAPS",
    "PASTEL_CMAPS",
    "pastel",
    "rgb_composite",
]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

def pastel(cmap, amount=0.26, n=256):
    """A softened version of a colour map, blended towards white.

    ``amount`` is the fraction of white mixed in at every level, so the hue and the
    ordering are preserved while the saturation comes down.  Kept moderate on
    purpose: the small-agenda ``B_I`` map tops out near $0.2$ of its range, and past
    about a third of white it stops being legible at all.  Used so that the
    figures share one palette temperature rather than mixing ColorBrewer's
    saturated ends with the pastel histograms.
    """
    base = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    base[:, :3] = base[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"pastel_{cmap}", base)


#: One colour map per order parameter, following the draft: the two trust-side
#: quantities in blue/red, the class-opinion correlation in green, and the two
#: balance measures in purple/orange.  Diverging quantities get a symmetric
#: white-centred map so that the sign of the correlation is legible at a glance.
CMAPS = {
    "R_wmu": "Blues",
    "R_muc": "RdBu_r",
    "R_cw": "Greens",
    "B_I": "Purples",
    "B_A": "OrRd",
}

#: The same maps, softened.  ``phase_map`` uses these; the raw names above are kept
#: so a caller can ask for the saturated version.
PASTEL_CMAPS = {k: pastel(v) for k, v in CMAPS.items()}

#: Ranges used for each order parameter.  R_wmu and R_cw are non-negative in
#: practice; R_muc, B_A and B_I are signed.
RANGES = {
    "R_wmu": (0.0, 1.0),
    "R_muc": (-1.0, 1.0),
    "R_cw": (0.0, 1.0),
    "B_I": (0.0, 1.0),
    "B_A": (-1.0, 1.0),
}

LABELS = {
    "R_wmu": r"$R_{w,\mu}$",
    "R_muc": r"$R_{\mu,c}$",
    "R_cw": r"$R_{c,w}$",
    "B_I": r"$B_I$",
    "B_A": r"$B_A$",
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
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "image.interpolation": "nearest",
}


#: Body-text size of the target document, in points.  Figures are generated at the
#: size they are printed at (see :func:`panel`), so text set at this size in a
#: figure comes out matching the surrounding prose rather than scaled up or down by
#: whatever ``\includegraphics`` width happens to be used.
BODY_PT = {"paper": 10.0, "iclr": 10.0}


def use_style(style="paper"):
    """Activate one of the two output styles."""
    if style not in ("paper", "iclr"):
        raise ValueError("style must be 'paper' or 'iclr'")
    _STYLE["name"] = style
    pt = BODY_PT[style]
    rc = dict(_BASE_RC)
    rc.update({
        "font.size": pt,
        "axes.labelsize": pt,
        "axes.titlesize": pt,
        # axis labels sit at body size; the numbers do not need to, and at body
        # size they crowd the panels
        "xtick.labelsize": pt - 2.5,
        "ytick.labelsize": pt - 2.5,
        "legend.fontsize": pt - 1.5,
    })
    mpl.rcParams.update(rc)
    return style


def panel(frac, aspect):
    r"""Figure size, in inches, for a figure printed at ``frac`` of the text width.

    Generating at the printed size is what keeps figure text at body-text size:
    ``\includegraphics[width=frac\linewidth]`` then scales by exactly 1, so 10pt
    in the figure is 10pt on the page.  ``aspect`` is height/width.
    """
    w = text_width() * frac
    return (w, w * aspect)


def text_width():
    """Full text width in inches for the active style."""
    return 5.5 if _STYLE["name"] == "iclr" else 6.3


def save(fig, name, style=None):
    """Write ``name`` as a PDF into ``figures/<style>/``.

    PDF only: it is what LaTeX wants, it stays sharp at any size, and a parallel
    set of PNGs is just something else to keep in step.  To look at one, render
    it on demand (``pdftoppm -r 150 -png fig.pdf out``).
    """
    style = style or _STYLE["name"]
    out = FIGURE_DIR / style
    if not FIGURE_DIR.parent.is_dir():
        raise FileNotFoundError(
            f"{FIGURE_DIR.parent} no longer exists: the project directory was "
            f"probably moved or renamed after this run started."
        )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path.relative_to(FIGURE_DIR.parent)}")
    return path


#: Soft fills in the manner of Mathematica's default histogram styling: a pastel
#: body with a thin darker edge, light enough that two series can overlap and the
#: overlap still reads as a third tone.
HIST_BLUE = ("#7BA7D7", "#3B6FA8")
HIST_RED = ("#E8918C", "#B03A34")


def framed_axes(ax, minor=True):
    """Put an axis in the frame-and-inward-ticks style of a Mathematica plot.

    A closed box rather than two spines, ticks pointing inwards on all four sides,
    minor ticks between the majors, and no grid.  Matplotlib's defaults are the
    opposite of all four, so this is worth centralising rather than repeating.
    """
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.25")
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   color="0.25", width=0.6)
    ax.tick_params(which="major", length=3.2)
    ax.tick_params(which="minor", length=1.8)
    if minor:
        ax.minorticks_on()
    ax.grid(False)
    return ax


def phase_map(
    ax, data, d, fd, key, vmin=None, vmax=None, cmap=None, colorbar=True, title=None,
    ylabel=True, sparse_ticks=False,
):
    """Draw one order-parameter map in the draft's orientation.

    ``f_d`` increases downwards from 0 at the top; ``d`` runs from -1 to +1.
    """
    lo, hi = RANGES.get(key, (None, None))
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    cmap = cmap or PASTEL_CMAPS.get(key, pastel("viridis"))
    norm = None
    if vmin is not None and vmax is not None and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(
            data, origin="upper", cmap=cmap, norm=norm,
            extent=[d[0], d[-1], fd[-1], fd[0]], aspect="auto",
        )
    else:
        im = ax.imshow(
            data, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[d[0], d[-1], fd[-1], fd[0]], aspect="auto",
        )
    add_phase_axes(ax, ylabel=ylabel, sparse_ticks=sparse_ticks)
    framed_axes(ax, minor=False)
    ax.set_title(title if title is not None else LABELS.get(key, key), pad=3)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    return im


def add_phase_axes(ax, ylabel=True, sparse_ticks=False):
    """Label a phase-diagram axis.

    Note that no axis inversion happens here: the ``extent`` passed to
    ``imshow`` by :func:`phase_map` already puts ``f_d = 0`` at the top, which is
    the draft's layout.  Inverting again would flip the maps upside down.

    ``ylabel=False`` drops the redundant ``f_d`` label on inner columns of a
    grid, which also frees the margin for a row label.  ``sparse_ticks`` keeps
    only the endpoints and midpoint, for grids too narrow for five labels.
    """
    ax.set_xlabel(r"$d$", labelpad=1)
    if ylabel:
        ax.set_ylabel(r"$f_d$", labelpad=1)
    if sparse_ticks:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])


def rgb_composite(r_channel, g_channel, b_channel):
    """Compose three order-parameter maps into one RGB image.

    Each channel is clipped to [0, 1] after mapping its own range, so the
    resulting colour names the phase: black where nothing is correlated, blue
    where only opinion-trust is, magenta where opinion-trust and trust-class
    both are, and pale where all three are.
    """
    def norm(a, key):
        lo, hi = RANGES[key]
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    return np.stack(
        [norm(r_channel, "R_muc"), norm(g_channel, "R_cw"), norm(b_channel, "R_wmu")],
        axis=-1,
    )


def sequential_from(color, name="seq"):
    """A white-to-``color`` colour map, for one-off channels."""
    return LinearSegmentedColormap.from_list(name, ["white", color])
