#!/usr/bin/env python3
"""How learning moves a receiver's state, and how the discrimination field bends it.

One interaction changes the receiver's opinion field by an amount proportional
to ``F_w`` and its distrust field by an amount proportional to ``F_mu``, so the
pair ``(F_w, F_mu)`` evaluated on the ``(h_w, h_mu)`` plane is the flow that
learning induces.  The flow points away from the two dissonant quadrants
(agreeing with a distrusted emitter, disagreeing with a trusted one) and into
the two consonant ones.

A discriminating receiver evaluates the modulation at ``h_w + D`` while sitting
at ``h_w``, which tilts the flow: the diagonal separatrix moves to
``h_mu = h_w + D``, so one consonant basin grows at the other's expense.  With
``D < 0`` -- how a discriminating agent treats an out-group emitter when
``d > 0`` -- the "distrust and disagree" basin grows, and a society with enough
such agents ends up with distrust aligned to class.  With ``D > 0`` the
"trust and agree" basin grows instead.

Writes ``learning_flows``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.modulation import F_mu, F_w, evidence  # noqa: E402
from ednna.plotting import framed_axes, panel, pastel, save  # noqa: E402

LIM = 4.0


def figure(style, d=1.5, n_arrows=17, n_field=400):
    ax_field = np.linspace(-LIM, LIM, n_field)
    HW, HMU = np.meshgrid(ax_field, ax_field)
    ax_q = np.linspace(-LIM * 0.94, LIM * 0.94, n_arrows)
    QW, QMU = np.meshgrid(ax_q, ax_q)

    # shared colour scale across the three panels so they are comparable
    mags = []
    for D in (-d, 0.0, d):
        mags.append(np.hypot(F_w(HW + D, HMU), F_mu(HW + D, HMU)))
    vmax = float(np.percentile(np.concatenate([m.ravel() for m in mags]), 99.0))

    fig, axes = plt.subplots(1, 3, figsize=panel(1.0, 0.40), sharey=True)
    titles = (
        rf"out-group emitter ($D = {-d:+.1f}$)",
        r"no discrimination ($D = 0$)",
        rf"in-group emitter ($D = {d:+.1f}$)",
    )
    for ax, D, title in zip(axes, (-d, 0.0, d), titles):
        mag = np.hypot(F_w(HW + D, HMU), F_mu(HW + D, HMU))
        im = ax.imshow(
            np.clip(mag, 0, vmax),
            origin="lower",
            extent=[-LIM, LIM, -LIM, LIM],
            cmap=pastel("BuPu", 0.35),
            vmin=0.0,
            vmax=vmax,
            aspect="equal",
        )
        # contours of the evidence: the dissonance landscape the flow escapes
        ax.contour(
            HW, HMU, evidence(HW + D, HMU),
            levels=[0.05, 0.2, 0.5, 0.8, 0.95],
            colors="0.35", linewidths=0.35, alpha=0.9,
        )
        u = F_w(QW + D, QMU)
        v = F_mu(QW + D, QMU)
        ax.quiver(
            QW, QMU, u, v,
            np.hypot(u, v), cmap=pastel("autumn_r", 0.25), clim=(0, vmax),
            angles="xy", scale=26.0, width=0.006, headwidth=3.2, headlength=3.6,
            pivot="middle",
        )
        # the separatrix, displaced by the discrimination field
        ax.plot([-LIM, LIM], [-LIM + D, LIM + D], color="#5aa469", lw=0.9)
        framed_axes(ax, minor=False)
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.axvline(0.0, color="0.5", lw=0.4)
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel(r"disagree $\leftarrow h_w \rightarrow$ agree")
    axes[0].set_ylabel(r"trust $\leftarrow h_\mu \rightarrow$ distrust")
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.012)
    cb.set_label(r"$|(F_w, F_\mu)|$", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return save(fig, "learning_flows", style)


def main():
    args, _ = setup(__doc__)
    figure(args.style)


if __name__ == "__main__":
    main()
