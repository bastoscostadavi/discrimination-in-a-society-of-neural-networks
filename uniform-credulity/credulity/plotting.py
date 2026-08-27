r"""Shared figure style, colour maps and output helpers.

The style machinery is that of the main line of work, so a figure from here and
a figure from there print at the same size with the same fonts.  ``--style
paper`` matches the proportions of the companion manuscript; ``--style iclr``
targets the ICLR single-column text width of 5.5 inches, so a figure can be
included at ``width=\linewidth`` with no rescaling.  PDF only.

What is added here is the vocabulary of the bias partition.  A quantity carrying
a ``b`` or a ``u`` is restricted to the agents that do or do not carry the field,
and superscript arrows say which way the trust runs: ``T_\mu^{b\to}`` is the mean
trust a biased agent *extends*, ``T_\mu^{\to b}`` the mean trust it *receives*.
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
    "credulity_composite",
    "DESCRIPTIONS",
    "LABELS",
    "RANGES",
    "CMAPS",
    "PASTEL_CMAPS",
    "DIVERGING",
    "pastel",
]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def pastel(cmap, amount=0.26, n=256):
    """A softened version of a colour map, blended towards white.

    ``amount`` is the fraction of white mixed in at every level, so the hue and
    the ordering are preserved while the saturation comes down.  Used so that
    the figures share one palette temperature with the rest of the project
    rather than mixing ColorBrewer's saturated ends with the pastel histograms.
    """
    base = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    base[:, :3] = base[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"pastel_{cmap}", base)


#: One colour map per order parameter.
#:
#: Everything this package measures except ``B_rho`` is *signed and centred*, and
#: on this plane the sign is the whole point: ``a > 0`` and ``a < 0`` are
#: credulity and suspicion, two different states rather than two strengths of
#: one, and a map that separated them only by lightness would be unreadable at
#: the one place it matters.  So the default here is diverging where the main
#: line of work's default is sequential.  The paper's three correlations keep
#: their published hues, since those three are also the colour channels of its
#: phase diagram.
CMAPS = {
    "R_wmu": "Blues",
    "R_muc": "Reds",
    "R_cw": "Greens",
    "B_rho": "Purples",
    "B_eta": "OrRd",
    # the class channels: one responds, three are controls, and they are read
    # against each other, so they share one map
    "T_mu": "PuOr_r",
    "R_cred": "PuOr_r",
    "R_stat": "PuOr_r",
    "R_muc_channel": "PuOr_r",
    # the bias partition: trust in one family, opinion in another, so that a
    # glance separates the two sectors
    "T_give_b": "PuOr_r",
    "T_give_u": "PuOr_r",
    "T_get_b": "PuOr_r",
    "T_get_u": "PuOr_r",
    "T_bb": "PuOr_r",
    "T_bu": "PuOr_r",
    "T_ub": "PuOr_r",
    "T_uu": "PuOr_r",
    "frac_biased": "Greys",
    "give_gap": "BrBG",
    "get_gap": "BrBG",
    "emergent_gap": "BrBG",
    "rho_mean": "RdBu_r",
    "rho_bb": "RdBu_r",
    "rho_uu": "RdBu_r",
    "rho_bu": "RdBu_r",
    "rho_gap": "BrBG",
    "B_eta_b": "RdBu_r",
    "B_eta_u": "RdBu_r",
    "B_rho_b": "RdBu_r",
    "B_rho_u": "RdBu_r",
}

#: Which maps are diverging.  These are used as they come: a diverging map is
#: already light in the middle, so blending it towards white flattens the ends
#: and turns a saturated panel into a pale one.
DIVERGING = ("PuOr_r", "RdBu_r", "BrBG")

#: The sequential maps, softened; the diverging ones untouched.
PASTEL_CMAPS = {k: (plt.get_cmap(v) if v in DIVERGING else pastel(v))
                for k, v in CMAPS.items()}

#: Ranges used for each order parameter.  Everything derived from ``eta`` or from
#: ``rho`` is a mean of a quantity in ``[-1, 1]`` and gets that range; the two
#: gaps are differences of two such means and so live in ``[-2, 2]``, but are
#: drawn on ``[-1, 1]`` because nothing on this plane comes near the wider
#: bound and the shared range makes them comparable with the panels above them.
RANGES = {
    "R_wmu": (0.0, 1.0),
    "R_muc": (-1.0, 1.0),
    "R_cw": (0.0, 1.0),
    "B_rho": (0.0, 1.0),
    "B_eta": (-1.0, 1.0),
    "T_mu": (-1.0, 1.0),
    "R_cred": (-1.0, 1.0),
    "R_stat": (-1.0, 1.0),
    "R_muc_channel": (-1.0, 1.0),
    "T_give_b": (-1.0, 1.0),
    "T_give_u": (-1.0, 1.0),
    "T_get_b": (-1.0, 1.0),
    "T_get_u": (-1.0, 1.0),
    "T_bb": (-1.0, 1.0),
    "T_bu": (-1.0, 1.0),
    "T_ub": (-1.0, 1.0),
    "T_uu": (-1.0, 1.0),
    "frac_biased": (0.0, 1.0),
    "give_gap": (-1.0, 1.0),
    "get_gap": (-1.0, 1.0),
    "emergent_gap": (-1.0, 1.0),
    "rho_mean": (-1.0, 1.0),
    "rho_bb": (-1.0, 1.0),
    "rho_uu": (-1.0, 1.0),
    "rho_bu": (-1.0, 1.0),
    "rho_gap": (-1.0, 1.0),
    "B_eta_b": (-1.0, 1.0),
    "B_eta_u": (-1.0, 1.0),
    "B_rho_b": (0.0, 1.0),
    "B_rho_u": (0.0, 1.0),
}

LABELS = {
    "R_wmu": r"$R_{w,\mu}$",
    "R_muc": r"$R_{\mu,c}$",
    "R_cw": r"$R_{c,w}$",
    "B_rho": r"$B_\rho$",
    "B_eta": r"$B_\eta$",
    "T_mu": r"$T_\mu$",
    "R_cred": r"$R_{\mathrm{cred}}$",
    "R_stat": r"$R_{\mathrm{stat}}$",
    "R_muc_channel": r"$R_{\mu,c}$",
    "T_give_b": r"$T_\mu^{b\rightarrow}$",
    "T_give_u": r"$T_\mu^{u\rightarrow}$",
    "T_get_b": r"$T_\mu^{\rightarrow b}$",
    "T_get_u": r"$T_\mu^{\rightarrow u}$",
    "T_bb": r"$T_\mu^{b\leftarrow b}$",
    "T_bu": r"$T_\mu^{b\leftarrow u}$",
    "T_ub": r"$T_\mu^{u\leftarrow b}$",
    "T_uu": r"$T_\mu^{u\leftarrow u}$",
    "frac_biased": r"realized $f_a$",
    "give_gap": r"$T_\mu^{b\rightarrow}-T_\mu^{u\rightarrow}$",
    "get_gap": r"$T_\mu^{\rightarrow b}-T_\mu^{\rightarrow u}$",
    "emergent_gap": r"$T_\mu^{u\leftarrow b}-T_\mu^{u\leftarrow u}$",
    "rho_mean": r"$\bar\rho$",
    "rho_bb": r"$\rho_{bb}$",
    "rho_uu": r"$\rho_{uu}$",
    "rho_bu": r"$\rho_{bu}$",
    "rho_gap": r"$\rho_{bb}-\rho_{uu}$",
    "B_eta_b": r"$B_\eta^{bb}$",
    "B_eta_u": r"$B_\eta^{uu}$",
    "B_rho_b": r"$B_\rho^{bb}$",
    "B_rho_u": r"$B_\rho^{uu}$",
}

#: Longer names, for panel titles where the symbol alone does not say enough.
DESCRIPTIONS = {
    "T_mu": "mean trust",
    "R_cred": "credulity (control)",
    "R_stat": "status (control)",
    "R_muc": "matching (control)",
    "rho_mean": "consensus",
    "give_gap": "direct",
    "get_gap": "emergent, pooled",
    "emergent_gap": "emergent",
    "rho_gap": "opinion split",
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

#: Body-text size of the target document, in points.  Figures are generated at
#: the size they are printed at (see :func:`panel`), so text set at this size in
#: a figure comes out matching the surrounding prose.
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
    # meant to print side by side at the same width must not be cropped to their
    # own content, or the one with the longer axis label comes out shorter than
    # the other.  Passing bbox_inches=None to savefig is not enough: matplotlib
    # reads it as "unset" and falls back to the rcParam, which is "tight" here.
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


def framed_axes(ax, minor=True):
    """Put an axis in the frame-and-inward-ticks style of a Mathematica plot."""
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
    ax, data, a, f, key, vmin=None, vmax=None, cmap=None, colorbar=True, title=None,
    ylabel=True, sparse_ticks=False,
):
    """Draw one order-parameter map.

    ``f`` increases upwards from 0 at the bottom; the strength axis runs over
    whatever range the sweep used.  Values that are ``nan`` -- a group quantity
    where its group is empty -- are left as the axes background rather than
    coloured, so an empty group reads as absent rather than as zero.
    """
    lo, hi = RANGES.get(key, (None, None))
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    cmap = cmap or PASTEL_CMAPS.get(key, pastel("viridis"))
    cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap.set_bad("0.92")
    kw = dict(origin="lower", cmap=cmap, extent=[a[0], a[-1], f[0], f[-1]],
              aspect="auto")
    if vmin is not None and vmax is not None and vmin < 0 < vmax:
        im = ax.imshow(data, norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), **kw)
    else:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, **kw)
    add_phase_axes(ax, ylabel=ylabel, sparse_ticks=sparse_ticks, xlim=(a[0], a[-1]))
    framed_axes(ax, minor=False)
    ax.set_title(title if title is not None else LABELS.get(key, key), pad=3)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    return im


def add_phase_axes(ax, ylabel=True, sparse_ticks=False, xlim=(-1.0, 1.0)):
    """Label a phase-diagram axis as ``(a, f_a)``.

    No axis inversion happens here.  :func:`phase_map` uses ``origin="lower"`` so
    that ``f_a`` increases upwards, the reading a control parameter usually gets;
    the source draft prints these maps the other way up, which is ``imshow``'s
    row-major default rather than a choice.

    ``ylabel=False`` drops the redundant label on inner columns of a grid, which
    also frees the margin for a row label.  ``sparse_ticks`` keeps only the
    endpoints and midpoint, for grids too narrow for five labels.
    """
    ax.set_xlabel(r"$a$", labelpad=1)
    if ylabel:
        ax.set_ylabel(r"$f_a$", labelpad=1)
    lo, hi = xlim
    if sparse_ticks:
        ax.set_xticks([lo, 0.5 * (lo + hi), hi])
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_xticks(np.linspace(lo, hi, 5))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])


def credulity_composite(t_mu, r_wmu):
    """Compose the two maps that name a state of the uniform field.

    Three colour channels, but only two quantities, because the first of them is
    signed and its sign is the whole plane:

    * red   ``max(-T_mu, 0)``  how far the population has been driven into
      universal *distrust*
    * green ``max(+T_mu, 0)``  how far into universal *trust*
    * blue  ``R_{w,mu}``       the opinion-trust alignment, which is what a
      population with no field at all settles into

    So red is a suspicious society, green a credulous one, blue an ordinary
    polarized one, and the mixtures are the transition bands between them.

    Splitting a signed channel across two colours rather than taking ``|T_mu|``
    is deliberate, and it is where this composite departs from the one in
    ``../directional-prejudice/``.  There, ``|.|`` is right because the sign of
    the field is the relabelling ``A <-> B``, which maps the ensemble to itself,
    so mirroring the colour with it would make half of a fully ordered map read
    as empty.  Here the sign is not a relabelling of anything: ``a > 0`` and
    ``a < 0`` are credulity and suspicion, two different states that the
    population reaches by different routes, and collapsing them onto one colour
    would discard the one distinction the plane exists to draw.

    Any pixel with a non-finite value in any channel comes out mid grey.
    """
    t_mu = np.asarray(t_mu, dtype=float)
    rgb = np.stack([
        np.clip(-t_mu, 0.0, 1.0),
        np.clip(+t_mu, 0.0, 1.0),
        np.clip(np.asarray(r_wmu, dtype=float), 0.0, 1.0),
    ], axis=-1)
    bad = ~np.isfinite(rgb).all(axis=-1)
    rgb[bad] = 0.5
    return rgb
