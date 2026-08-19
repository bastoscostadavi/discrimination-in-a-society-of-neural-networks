"""The measured landscape, drawn in the layout of the paper's Figure 1.

Three conventions are copied deliberately and must not drift, or the measured
map cannot be laid beside the analytic one: the x axis runs disagree-to-agree in
``h_w``, the y axis runs trust-to-distrust in ``h_mu``, and the green line is
``h_mu = h_w``, the separatrix across which blame for a surprise passes from one
sector to the other.

Two honesty constraints on how it is drawn.

``imshow`` with nearest-neighbour interpolation, never filled contours.  The
grid is five by five; smoothing it would draw structure that was never measured.
Every cell carries its own number for the same reason.

The theory row is evaluated on the same square, under a stated assumption: that
rung ``k`` of the ordinal scale corresponds to field value ``k``.  The model's
``h_w`` and ``h_mu`` are continuous scaled fields and the experiment's axes are
five named rungs, so *some* correspondence has to be assumed to compare them at
all.  This is the simplest one, it is a proxy rather than a calibration, and the
figure says so on its face rather than in a caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

__all__ = ["FIGURE_DIR", "use_style", "panel", "save", "framed_axes", "pastel",
           "fit_scale", "fit_curve", "plane_panel", "cut_panel",
           "figure_trust_summary"]

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"
_STYLE = {"name": "iclr"}


def use_style(name="iclr"):
    _STYLE["name"] = name
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.linewidth": 0.6,
    })


def panel(frac, aspect):
    w = (5.5 if _STYLE["name"] == "iclr" else 6.3) * frac
    return (w, w * aspect)


def save(fig, name):
    """Write ``name`` as a PDF into ``figures/<style>/``.

    PDF only: it is what LaTeX wants, it stays sharp at any size, and a parallel
    set of PNGs is just something else to keep in step.  To look at one, render
    it on demand (``pdftoppm -r 150 -png fig.pdf out``).
    """
    out = FIGURE_DIR / _STYLE["name"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path.relative_to(FIGURE_DIR.parent)}")
    return path


def framed_axes(ax):
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("0.25")
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   color="0.25", width=0.6, length=3.0)
    ax.grid(False)
    return ax


def _decorate(ax, lim, levels=None):
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], color="#5aa469", lw=0.9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"disagree $\;\leftarrow\;h_w\;\rightarrow\;$ agree")
    ax.set_ylabel(r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust")
    ax.set_aspect("equal")
    if levels is not None:
        ax.set_xticks(levels)
        ax.set_yticks(levels)
    framed_axes(ax)


# ---------------------------------------------------------------------------
# The trust curve
# ---------------------------------------------------------------------------

def pastel(cmap, amount=0.30, n=256):
    """A softened colour map, matching the paper's figures."""
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(cmap)(np.linspace(0, 1, n))
    base[:, :3] = base[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"pastel_{cmap}", base)


def fit_scale(h_w, h_mu, d_mu, F_mu):
    """The one free constant, by least squares through the origin.

    The update is ``F_mu * V/gamma_V`` and the trust variance ``V`` is not
    observable here, so the theory carries an overall positive scale and nothing
    else.  Fitted on the rows, not on the binned means, so that a bin holding
    few points does not weigh as much as a full one.
    """
    m = np.isfinite(h_w) & np.isfinite(h_mu) & np.isfinite(d_mu)
    F = F_mu(h_w[m], h_mu[m])
    return float(np.sum(d_mu[m] * F) / np.sum(F * F))


def fit_curve(h_mu, d_mu, sign, F_mu, grid=None):
    """The scale and the conviction together, by profiling over ``|h_w|``.

    ``|h_w|`` is not taken from the conviction inversion here.  Where the peak of
    ``|F_mu|`` sits along ``h_mu`` is set by ``|h_w|`` and by nothing else -- it
    moves outward as conviction rises, and in the limit ``|h_w| -> inf`` the
    function becomes the Gaussian hazard rate, which increases without bound and
    has no peak at all.  So the shape of the measured curve *is* a measurement of
    conviction, and a better one than the inversion: it uses every row rather
    than only those outside the neutral band.

    Returns ``(alpha, h_w, rmse, profile)``.  The profile is the residual at each
    trial ``|h_w|`` and is what says whether the fitted value is identified: past
    the point where it flattens, the data cannot distinguish one conviction from
    another, and the fitted number should be quoted as a lower bound.
    """
    grid = np.linspace(0.5, 10.0, 40) if grid is None else grid
    m = np.isfinite(h_mu) & np.isfinite(d_mu)
    profile = []
    for hw in grid:
        F = F_mu(sign[m] * hw, h_mu[m])
        a = float(np.sum(d_mu[m] * F) / np.sum(F * F))
        profile.append((hw, a, float(np.sqrt(np.mean((d_mu[m] - a * F) ** 2)))))
    hw, a, rmse = min(profile, key=lambda t: t[2])
    # quote the smallest conviction within 1% of the best residual: past it the
    # profile is flat and the fit is a lower bound, not a value
    floor = min(t[0] for t in profile if t[2] <= 1.01 * rmse)
    F = F_mu(sign[m] * floor, h_mu[m])
    a = float(np.sum(d_mu[m] * F) / np.sum(F * F))
    return a, floor, rmse, profile


def plane_panel(ax, h_w, h_mu, d_mu, F_mu, alpha, lim=None, cap=None,
                fill_label=r"$F_\mu$"):
    """Measured ``Delta h_mu`` as points, over the analytic ``F_mu``.

    Both coordinates of every point are measured -- ``h_mu`` from the emitter's
    stated reliability, ``h_w`` from the conviction inversion with the sign the
    message carries -- so the cloud is not confined to a grid of asserted rungs.
    """
    finite = d_mu[np.isfinite(d_mu)]
    cap = float(cap if cap is not None else np.percentile(np.abs(finite), 92))
    if lim is None:
        # framed on the data, not on a round number: ``h_mu`` is hard-bounded by
        # the probability clip and ``h_w`` by the inversion, so a wider frame is
        # theory with nothing measured in it.  Any point outside is counted in
        # the corner rather than silently dropped.
        lim = float(np.ceil(np.percentile(np.abs(h_w[np.isfinite(h_w)]), 95)
                            / 0.2) * 0.2)
    a = np.linspace(-lim, lim, 400)
    HW, HMU = np.meshgrid(a, a)
    V = np.clip(alpha * F_mu(HW, HMU), -cap, cap)
    levels = np.linspace(-cap, cap, 25)
    ax.contourf(HW, HMU, V, levels=levels, cmap=pastel("coolwarm"))
    ax.contour(HW, HMU, V, levels=levels[::4], colors="k", linewidths=0.25,
               alpha=0.5)
    im = ax.scatter(h_w, h_mu, c=np.clip(d_mu, -cap, cap), cmap="coolwarm",
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap),
                    s=11, edgecolors="k", linewidths=0.3, zorder=3)
    _decorate(ax, lim, np.arange(-int(lim), int(lim) + 1))
    # named exactly as the paper's Figure 1 names it: the function alone, in the
    # same colour and the same corner, so the two can be laid side by side and
    # read as one statement rather than two conventions.  What is fill and what
    # is measurement belongs in the caption.
    ax.text(0.05, 0.93, fill_label, transform=ax.transAxes, color="#2f4f7f",
            ha="left", va="top", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.0))
    return im


def cut_panel(ax, h_w, h_mu, d_mu, sign, F_mu, alpha, hw_fit, window,
              lim=3.4, bins=9):
    """``Delta h_mu`` against prior trust, one curve per message sign.

    The binned points are means over rows with the standard error of that mean.
    The theory curve is drawn at the fitted conviction from :func:`fit_curve`,
    and it is drawn *past* the edge of what the readout can reach.  That matters:
    a stated probability is clipped before the probit, so ``h_mu`` cannot be
    observed beyond ``+-window``, and the peak of the theory curve at this
    conviction lies outside it.  Inside the window the model predicts a
    monotonic rise, which is what is measured; the turnover it also predicts is
    simply not observable here, and the greyed margins say so.
    """
    edges = np.linspace(-lim + 0.2, lim - 0.2, bins + 1)
    grid = np.linspace(-lim, lim, 300)
    # named by the arm alone: which curve is theory and which is measurement is
    # carried by line-versus-marker, not by the words, and the caption says what
    # an arm is
    for s, colour, label in ((+1, "#B03A34", "agree"),
                             (-1, "#3B6FA8", "disagree")):
        m = (sign == s) & np.isfinite(d_mu)
        ax.plot(grid, alpha * F_mu(s * hw_fit, grid), color=colour, lw=1.1,
                label=label)
        idx = np.digitize(h_mu[m], edges)
        for b in range(1, len(edges)):
            sel = idx == b
            if sel.sum() < 4:
                continue
            v = d_mu[m][sel]
            ax.errorbar(h_mu[m][sel].mean(), v.mean(),
                        yerr=v.std(ddof=1) / np.sqrt(v.size), fmt="o",
                        color=colour, ms=3.4, lw=0.9, capsize=1.6,
                        mfc="white", mew=0.9, zorder=3)
    ax.axhline(0.0, color="0.55", lw=0.5)
    ax.axvline(0.0, color="0.55", lw=0.5)
    finite = d_mu[np.isfinite(d_mu)]
    pad = 1.15 * np.percentile(np.abs(finite), 99)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-pad, pad)
    ax.set_xlabel(r"trust $\;\leftarrow\;h_\mu\;\rightarrow\;$ distrust")
    ax.set_ylabel(r"measured $\Delta h_\mu$")
    ax.legend(fontsize=6, frameon=False, loc="lower left", handlelength=1.2,
              borderpad=0.1, labelspacing=0.25)
    framed_axes(ax)


def figure_trust_summary(h_w, h_mu, d_mu, sign, F_mu, name="trust_llm"):
    """The trust sector in two square panels: the plane, then the cut through it.

    The panels are drawn at equal box size rather than equal data aspect.  The
    left one is a plane and is square in its own units; the right one has a field
    on one axis and an update on the other, where an equal aspect would be
    meaningless, so its box is squared explicitly and the two read as a pair.
    """
    alpha_plane = fit_scale(h_w, h_mu, d_mu, F_mu)
    alpha, hw_fit, rmse, _ = fit_curve(h_mu, d_mu, sign, F_mu)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6),
                             gridspec_kw={"wspace": 0.62})
    im = plane_panel(axes[0], h_w, h_mu, d_mu, F_mu, alpha_plane)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cut_panel(axes[1], h_w, h_mu, d_mu, sign, F_mu, alpha, hw_fit, window=2.05)
    axes[1].set_box_aspect(1.0)
    for ax, tag in zip(axes, "ab"):
        ax.text(-0.26, 1.10, f"({tag})", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
    m = np.isfinite(h_w) & np.isfinite(d_mu)
    F = F_mu(h_w[m], h_mu[m])
    print(f"[plane] n={m.sum()}  alpha={alpha_plane:.3f}  "
          f"r={np.corrcoef(F, d_mu[m])[0, 1]:.3f}  "
          f"sign={np.mean(np.sign(F) == np.sign(d_mu[m])):.1%}")
    print(f"[cut]   alpha={alpha:.3f}  |h_w| >= {hw_fit:.2f}  RMSE={rmse:.3f}")
    return save(fig, name)
