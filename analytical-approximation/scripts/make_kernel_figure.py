"""Generate semi-analytical figures from the Gaussian-averaged kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".matplotlib"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controlledcv.fields import field_density
from controlledcv.kernels import affective_kernels, ideological_coefficients


FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def grid_kernels(rhos, mus, D, order):
    M_mu = np.empty((mus.size, rhos.size))
    M_V = np.empty_like(M_mu)
    A = np.empty_like(M_mu)
    B = np.empty_like(M_mu)
    for i, mu in enumerate(mus):
        for j, rho in enumerate(rhos):
            affective = affective_kernels(q_r=1.0, rho=rho, mu=mu, D=D, order=order)
            ideological = ideological_coefficients(
                q_r=1.0, q_e=1.0, rho=rho, mu=mu, D=D, order=order
            )
            M_mu[i, j] = affective.M_mu
            M_V[i, j] = affective.M_V
            A[i, j] = ideological.A
            B[i, j] = ideological.B
    return M_mu, M_V, A, B


def add_heatmap(ax, x, y, z, title, cmap="coolwarm", vlim=None, contour_zero=False):
    if vlim is None:
        vlim = float(np.nanmax(np.abs(z)))
    mesh = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
    if contour_zero and np.nanmin(z) < 0.0 < np.nanmax(z):
        ax.contour(x, y, z, levels=[0.0], colors="black", linewidths=1.1)
    ax.set_title(title)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\mu$")
    return mesh


def main():
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    rhos = np.linspace(-0.95, 0.95, 81)
    mus = np.linspace(-3.0, 3.0, 91)
    order = 90
    D0 = 0.0
    D1 = 0.75

    M_mu_0, M_V_0, A_0, B_0 = grid_kernels(rhos, mus, D0, order)
    M_mu_D, M_V_D, A_D, B_D = grid_kernels(rhos, mus, D1, order)

    out_npz = DATA / "semi_analytical_pairwise_kernels_q1.npz"
    np.savez_compressed(
        out_npz,
        rhos=rhos,
        mus=mus,
        D0=D0,
        D1=D1,
        M_mu_D0=M_mu_0,
        M_V_D0=M_V_0,
        A_D0=A_0,
        B_D0=B_0,
        M_mu_D1=M_mu_D,
        M_V_D1=M_V_D,
        A_D1=A_D,
        B_D1=B_D,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)

    h = np.linspace(-4.0, 4.0, 600)
    for rho, color in [(-0.75, "#4477aa"), (0.0, "#666666"), (0.75, "#cc6677")]:
        axes[0, 0].plot(h, field_density(h, q_r=1.0, rho=rho), label=rf"$\rho={rho:g}$", color=color)
    axes[0, 0].set_title(r"reduced field density $p(h\mid\rho)$")
    axes[0, 0].set_xlabel(r"$h$")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False)

    m = add_heatmap(axes[0, 1], rhos, mus, M_mu_0, rf"$\mathcal{{M}}_\mu$, $D={D0:g}$", contour_zero=True)
    fig.colorbar(m, ax=axes[0, 1])
    m = add_heatmap(axes[0, 2], rhos, mus, M_mu_D, rf"$\mathcal{{M}}_\mu$, $D={D1:g}$", contour_zero=True)
    fig.colorbar(m, ax=axes[0, 2])

    common = max(float(np.max(np.abs(A_D))), float(np.max(np.abs(B_D))))
    m = add_heatmap(axes[1, 0], rhos, mus, A_D, rf"$A$ in $K_w=A w_r+B w_e$, $D={D1:g}$", vlim=common)
    fig.colorbar(m, ax=axes[1, 0])
    m = add_heatmap(axes[1, 1], rhos, mus, B_D, rf"$B$ in $K_w=A w_r+B w_e$, $D={D1:g}$", vlim=common)
    fig.colorbar(m, ax=axes[1, 1])
    m = add_heatmap(axes[1, 2], rhos, mus, M_V_D, rf"$\mathcal{{M}}_V$, $D={D1:g}$", contour_zero=True)
    fig.colorbar(m, ax=axes[1, 2])

    fig.suptitle("Controlled small-(C,V) pairwise Gaussian kernels", fontsize=15)
    out_png = FIGURES / "semi_analytical_pairwise_kernels_q1.png"
    fig.savefig(out_png, dpi=180)
    print(out_png)
    print(out_npz)


if __name__ == "__main__":
    main()
