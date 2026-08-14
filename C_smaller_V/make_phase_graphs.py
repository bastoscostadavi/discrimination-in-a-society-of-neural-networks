#!/usr/bin/env python3
"""Make leading-order C << V phase-diagram graphs.

The approximation freezes ideological weights and lets the directed trust layer
relax according to the sign of the issue-averaged fast drift.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.special import erf
except ImportError:  # pragma: no cover
    erf = np.vectorize(math.erf, otypes=[float])


def case6_matrix(d: float) -> np.ndarray:
    return d * np.array([[1.0, -1.0], [-1.0, 1.0]])


def frozen_observables(
    *,
    d_values: np.ndarray,
    fd_values: np.ndarray,
    n_repeats: int = 12,
    n_agents: int = 40,
    n_dim: int = 30,
    n_issues: int = 5,
    seed: int = 20260814,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_d = len(d_values)
    n_fd = len(fd_values)
    R_muc = np.zeros((n_fd, n_d))
    R_cw = np.zeros((n_fd, n_d))
    frac_class_locked = np.zeros((n_fd, n_d))

    class_of = np.zeros(n_agents, dtype=int)
    class_of[n_agents // 2 :] = 1
    kappa = np.where(class_of == 0, 1.0, -1.0)
    G = np.outer(kappa, kappa)
    iu = np.triu_indices(n_agents, 1)
    denom = n_agents * (n_agents - 1)

    for rep in range(n_repeats):
        w = rng.normal(size=(n_agents, n_dim))
        w_unit = w / np.linalg.norm(w, axis=1, keepdims=True)
        rho = w_unit @ w_unit.T
        X = rng.normal(size=(n_issues, n_dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sigma = np.sign(w @ X.T)
        sigma[sigma == 0] = 1.0

        # h0[r,e,p] = sigma_e,p * w_r.x_p
        h0 = np.einsum("rp,ep->rep", w @ X.T, sigma)

        for i_fd, fd in enumerate(fd_values):
            discriminates = rng.random(n_agents) < fd
            for i_d, d in enumerate(d_values):
                Dmat = case6_matrix(float(d))
                D = Dmat[np.ix_(class_of, class_of)]
                D = D * discriminates[:, None]
                H = h0 + D[:, :, None]

                # At mu=0, the fast drift sign is the sign of
                # mean_p[1 - 2 Phi(H_p)].  The saturated trust sign eta is the
                # opposite of the drift direction.
                score_pair = erf(H / math.sqrt(2.0)).mean(axis=2)
                eta = np.sign(score_pair)
                eta[eta == 0.0] = 0.0
                np.fill_diagonal(eta, 1.0)

                S = eta + eta.T
                R_muc[i_fd, i_d] += (S[iu] * G[iu]).sum() / denom
                R_cw[i_fd, i_d] += (rho[iu] * G[iu]).sum() * 2.0 / denom
                same_trust = eta[iu] == np.sign(G[iu])
                frac_class_locked[i_fd, i_d] += same_trust.mean()

    scale = 1.0 / n_repeats
    return {
        "R_muc": R_muc * scale,
        "R_cw": R_cw * scale,
        "frac_class_locked": frac_class_locked * scale,
    }


def save_phase_figure(d_values: np.ndarray, fd_values: np.ndarray, data: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), constrained_layout=True)
    panels = [
        ("Fast trust-class order\n$R_{\\mu c}$", data["R_muc"], "RdBu_r", -1, 1),
        ("Frozen opinion-class order\n$R_{cw}$", data["R_cw"], "RdBu_r", -0.2, 0.2),
        ("Class-locked directed trust\nfraction", data["frac_class_locked"], "viridis", 0, 1),
    ]
    extent = [d_values.min(), d_values.max(), fd_values.min(), fd_values.max()]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("discrimination field d")
        ax.set_ylabel("fraction discriminatory $f_d$")
        ax.axvline(0, color="black", lw=0.7, alpha=0.45)
        fig.colorbar(im, ax=ax, shrink=0.88)
    fig.suptitle("Leading-order C << V prediction: trust orders before ideology", fontsize=12)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_timescale_figure(out: Path) -> None:
    eps_values = [1.0, 0.1, 0.01]
    t = np.linspace(0, 6, 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    r_inf = 0.85
    for eps in eps_values:
        Rmuc = r_inf * (1.0 - np.exp(-t))
        Rcw = r_inf * (1.0 - np.exp(-eps * t))
        ax.plot(t, Rmuc, color="black", lw=1.8, alpha=0.85 if eps == 1.0 else 0.18)
        ax.plot(t, Rcw, lw=2.0, label=f"$R_{{cw}}$, C/V={eps:g}")
    ax.text(3.8, 0.78, "$R_{\\mu c}$ fast", fontsize=10)
    ax.set_xlabel("time in units of the fast trust scale")
    ax.set_ylabel("order parameter")
    ax.set_ylim(-0.02, 0.92)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("As C/V decreases, phase III is delayed relative to phase IV")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    d_values = np.linspace(-1.0, 1.0, 81)
    fd_values = np.linspace(0.0, 1.0, 61)
    data = frozen_observables(d_values=d_values, fd_values=fd_values, n_issues=5)
    save_phase_figure(d_values, fd_values, data, root / "c_smaller_v_phase_prediction.png")
    save_timescale_figure(root / "c_smaller_v_timescale_delay.png")
    print(root / "c_smaller_v_phase_prediction.png")
    print(root / "c_smaller_v_timescale_delay.png")


if __name__ == "__main__":
    main()
