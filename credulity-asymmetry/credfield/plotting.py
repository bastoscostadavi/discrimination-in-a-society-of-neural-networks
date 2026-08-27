"""Shared figure style, colour maps and output helpers.

The style machinery is that of the main line of work, so a figure from here and a
figure from there print at the same size with the same fonts.  ``--style paper``
matches the proportions of the companion manuscript; ``--style iclr`` targets the
ICLR single-column text width of 5.5 inches, so a figure can be included at
``width=\\linewidth`` with no rescaling.  PDF only.

What is added here is the vocabulary of the four field components: which symbol
the axes of a phase diagram carry is a property of the sweep, so
:func:`set_component` sets it once and :func:`add_phase_axes` reads it.  This
directory sweeps ``b``, so that is the default.
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
    "set_component",
    "component",
    "channel_composite",
    "DESCRIPTIONS",
    "LABELS",
    "RANGES",
    "CMAPS",
    "PASTEL_CMAPS",
    "DIVERGING",
    "pastel",
    "rgb_composite",
]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"

def pastel(cmap, amount=0.26, n=256):
    """A softened version of a colour map, blended towards white.

    ``amount`` is the fraction of white mixed in at every level, so the hue and the
    ordering are preserved while the saturation comes down.  Kept moderate on
    purpose: the small-agenda ``B_rho`` map tops out near $0.2$ of its range, and past
    about a third of white it stops being legible at all.  Used so that the
    figures share one palette temperature rather than mixing ColorBrewer's
    saturated ends with the pastel histograms.
    """
    base = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    base[:, :3] = base[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"pastel_{cmap}", base)


#: One colour map per order parameter.  All three correlations run white to
#: saturated across the whole of their range in :data:`RANGES`, ``R_muc`` included
#: even though it is signed: these three panels are also the three colour channels
#: of the phase diagram, where red is exactly a white-to-red ramp over [-1, 1], and
#: a panel that disagreed with the composite it feeds would be worse than one that
#: puts the sign change at half brightness.
CMAPS = {
    "R_wmu": "Blues",
    "R_muc": "Reds",
    "R_cw": "Greens",
    "B_rho": "Purples",
    "B_eta": "OrRd",
    # The four trust channels all share one *diverging* map, and both choices
    # matter.  Diverging because every channel is signed and centred, and a
    # signed quantity on a sequential ramp distinguishes its two halves only by
    # lightness -- which is exactly the reading that would be needed to see that
    # a channel is flat at zero rather than weakly positive.  Shared because the
    # point of the row is that one of the four responds and the others do not,
    # which is a comparison between panels: a different hue per panel would make
    # four unrelated pictures out of one statement.  It does mean R_muc is not
    # the red it is in the main line of work's figures; that comparison is served
    # by the cut, which is a line plot.
    "R_stat": "PuOr_r",
    "R_cred": "PuOr_r",
    "T_mu": "PuOr_r",
    "R_muc_channel": "PuOr_r",
    # the balances are signed too, and read against a different question, so they
    # get their own diverging map rather than sharing the channels'
    "B_eta_A": "RdBu_r",
    "B_eta_B": "RdBu_r",
    "B_rho_A": "RdBu_r",
    "B_rho_B": "RdBu_r",
    "atomization": "BrBG",
}

#: Which maps are diverging.  These are used as they come: a diverging map is
#: already light in the middle, so blending it towards white flattens the ends and
#: turns a saturated panel into a pale one.
DIVERGING = ("PuOr_r", "RdBu_r", "BrBG")

#: The sequential maps, softened; the diverging ones untouched.  ``phase_map``
#: uses these; the raw names above are kept so a caller can ask for either.
PASTEL_CMAPS = {k: (plt.get_cmap(v) if v in DIVERGING else pastel(v))
                for k, v in CMAPS.items()}

#: Ranges used for each order parameter.  R_wmu and R_cw are non-negative in
#: practice; R_muc, B_eta and B_rho are signed.
RANGES = {
    "R_wmu": (0.0, 1.0),
    "R_muc": (-1.0, 1.0),
    "R_cw": (0.0, 1.0),
    "B_rho": (0.0, 1.0),
    "B_eta": (-1.0, 1.0),
    # every trust channel is an average of eta and so lives in [-1, 1]; all three
    # are signed and centred, unlike R_wmu and R_cw
    "R_stat": (-1.0, 1.0),
    "R_cred": (-1.0, 1.0),
    "T_mu": (-1.0, 1.0),
    "B_eta_A": (-1.0, 1.0),
    "B_eta_B": (-1.0, 1.0),
    "B_rho_A": (0.0, 1.0),
    "B_rho_B": (0.0, 1.0),
    "atomization": (-1.0, 1.0),
    "R_muc_channel": (-1.0, 1.0),
}

LABELS = {
    "R_wmu": r"$R_{w,\mu}$",
    "R_muc": r"$R_{\mu,c}$",
    "R_cw": r"$R_{c,w}$",
    "B_rho": r"$B_\rho$",
    "B_eta": r"$B_\eta$",
    "R_stat": r"$R_{\mathrm{stat}}$",
    "R_cred": r"$R_{\mathrm{cred}}$",
    "T_mu": r"$T_\mu$",
    "B_eta_A": r"$B_\eta^{AA}$",
    "B_eta_B": r"$B_\eta^{BB}$",
    "B_rho_A": r"$B_\rho^{AA}$",
    "B_rho_B": r"$B_\rho^{BB}$",
    "atomization": r"$(B_\eta^{AA}-B_\eta^{BB})/2$",
    "R_muc_channel": r"$R_{\mu,c}$",
}

#: Longer names, for panel titles where the symbol alone does not say enough.
DESCRIPTIONS = {
    "R_stat": "status",
    "R_cred": "credulity",
    "T_mu": "mean trust",
    "R_muc": "matching",
    "atomization": "atomization",
}

_STYLE = {"name": "paper", "component": "b"}

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


def save(fig, name, style=None, bbox="tight"):
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
    # bbox=None keeps the figure at exactly the requested figsize.  Two panels
    # meant to print side by side at the same width must not be cropped to
    # their own content, or the one with the longer axis label comes out
    # shorter than the other.  Passing bbox_inches=None to savefig is not enough:
    # matplotlib reads it as "unset" and falls back to the rcParam, which is
    # "tight" here, so the setting has to be overridden for the call.
    with plt.rc_context({"savefig.bbox": bbox}):
        fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path.relative_to(FIGURE_DIR.parent)}")
    return path


#: Soft fills in the manner of Mathematica's default histogram styling: a pastel
#: body with a thin darker edge, light enough that two series can overlap and the
#: overlap still reads as a third tone.
HIST_BLUE = ("#7BA7D7", "#3B6FA8")
HIST_RED = ("#E8918C", "#B03A34")


def matched_colorbar(fig, im, ax, width=0.018, pad=0.018, **kw):
    """A colour bar exactly as tall as the panel it belongs to.

    ``fig.colorbar(ax=...)`` sizes the bar against the axes' *allocated* rectangle.
    For a panel with a fixed aspect the drawn box is shorter than its allocation, so
    the bar overhangs it top and bottom.  Measuring the drawn box after a draw and
    placing the bar against that makes the two agree.
    """
    fig.canvas.draw()
    box = ax.get_window_extent().transformed(fig.transFigure.inverted())
    cax = fig.add_axes([box.x1 + pad, box.y0, width, box.height])
    cb = fig.colorbar(im, cax=cax, **kw)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return cb


def framed_axes(ax, minor=True):
    """Put an axis in the frame-and-inward-ticks style of a Mathematica plot.

    A closed box rather than two spines, ticks pointing inwards on all four sides,
    minor ticks between the majors, and no grid.  Matplotlib's defaults are the
    opposite of all four, so this is worth centralizing rather than repeating.
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
    ax, data, s, f, key, vmin=None, vmax=None, cmap=None, colorbar=True, title=None,
    ylabel=True, sparse_ticks=False,
):
    """Draw one order-parameter map in the draft's orientation.

    ``f`` increases upwards from 0 at the bottom; the strength axis runs over
    whatever range the sweep used.
    """
    lo, hi = RANGES.get(key, (None, None))
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    cmap = cmap or PASTEL_CMAPS.get(key, pastel("viridis"))
    norm = None
    if vmin is not None and vmax is not None and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(
            data, origin="lower", cmap=cmap, norm=norm,
            extent=[s[0], s[-1], f[0], f[-1]], aspect="auto",
        )
    else:
        im = ax.imshow(
            data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[s[0], s[-1], f[0], f[-1]], aspect="auto",
        )
    add_phase_axes(ax, ylabel=ylabel, sparse_ticks=sparse_ticks,
                   xlim=(s[0], s[-1]))
    framed_axes(ax, minor=False)
    ax.set_title(title if title is not None else LABELS.get(key, key), pad=3)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    return im


def set_component(component):
    """Name the field component the horizontal axis of a phase diagram carries.

    A sweep drives one of ``a``, ``b``, ``c``, ``p`` (:mod:`credfield.fields`), and
    the axes should say which.  Set once per figure; :func:`add_phase_axes` reads
    it, so no panel has to be told twice and no two panels of one figure can
    disagree.
    """
    from .fields import COMPONENTS
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    _STYLE["component"] = component
    return component


def component():
    """The component the axes currently name."""
    return _STYLE["component"]


def add_phase_axes(ax, ylabel=True, sparse_ticks=False, xlim=(-1.0, 1.0)):
    """Label a phase-diagram axis.

    Note that no axis inversion happens here.  :func:`phase_map` uses
    ``origin="lower"`` so that ``f_d`` increases upwards, the reading a control
    parameter usually gets; the source draft prints these maps the other way up,
    which is ``imshow``'s row-major default rather than a choice.  Inverting here
    would flip them back.

    ``ylabel=False`` drops the redundant ``f_d`` label on inner columns of a
    grid, which also frees the margin for a row label.  ``sparse_ticks`` keeps
    only the endpoints and midpoint, for grids too narrow for five labels.
    """
    k = _STYLE["component"]
    ax.set_xlabel(rf"${k}$", labelpad=1)
    if ylabel:
        ax.set_ylabel(rf"$f_{k}$", labelpad=1)
    lo, hi = xlim
    if sparse_ticks:
        ax.set_xticks([lo, 0.5 * (lo + hi), hi])
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_xticks(np.linspace(lo, hi, 5))
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


def channel_composite(responding, atomization, r_wmu):
    """Compose the three maps that name a state under a directional field.

    Red is whichever trust channel the swept component drives -- ``R_cred`` here,
    the credulity split -- green the asymmetry between the two classes' internal
    cohesion, and blue the ordinary opinion-trust alignment ``R_wmu`` that a
    population reaches with no field at all.  So blue is a population that has
    polarized without reference to the label, red one split along it, and the
    green channel says whether the two classes are differently *organized*
    internally rather than merely differently trusting.

    Both signed channels use ``|value|``: which class is the credulous one is a
    matter of the sign of the field, and mirroring the colour with it would make
    half of every map read as empty when it is fully ordered.  The sign is read
    off the individual maps, which keep it.
    """
    def norm(a, key):
        lo, hi = RANGES[key]
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    return np.stack([norm(np.abs(responding), "R_wmu"),
                     norm(np.abs(atomization), "R_wmu"),
                     norm(r_wmu, "R_wmu")], axis=-1)


def sequential_from(color, name="seq"):
    """A white-to-``color`` colour map, for one-off channels."""
    return LinearSegmentedColormap.from_list(name, ["white", color])
