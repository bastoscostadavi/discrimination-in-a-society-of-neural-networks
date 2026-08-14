"""Stage 4: neutral-state stability of the four-population closure.

The discriminatory field appears directly in the pairwise kernels.  Therefore,
for d != 0 the class-symmetric neutral state is generally forced rather than an
unforced fixed point.  This script separates:

1. the largest homogeneous Jacobian eigenvalue of the affective subsystem; and
2. the class-contrast forcing in d(mu_out - mu_in)/dt at the neutral state.

If the largest eigenvalue crosses zero, the closure has a genuine linear
instability.  If it stays negative while the forcing is nonzero, the phase
diagram in this closure is a driven class-trust response rather than a
spontaneous neutral-state bifurcation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".matplotlib"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from run_population_phase_diagram import (
    DATA,
    FIGURES,
    field_matrix,
    group_classes,
    group_discriminates,
    kernels,
    proportions,
)


def affective_rhs(mu, V, d, fd, rho=0.98, order=40):
    """Affective right-hand side for the four-population closure."""

    mu = np.asarray(mu, dtype=float).reshape(4, 4)
    V = np.asarray(V, dtype=float).reshape(4, 4)
    p = proportions(fd)
    cls = group_classes()
    disc = group_discriminates()
    D_class = field_matrix(d)
    dmu = np.zeros((4, 4), dtype=float)
    dV = np.zeros((4, 4), dtype=float)
    for r in range(4):
        for e in range(4):
            if p[e] == 0.0:
                continue
            D = D_class[cls[r], cls[e]] if disc[r] else 0.0
            M_mu, M_V, _A, _B = kernels(rho, mu[r, e], D, order)
            dmu[r, e] = p[e] * V[r, e] * M_mu
            dV[r, e] = p[e] * V[r, e] * V[r, e] * M_V
    return np.concatenate([dmu.ravel(), dV.ravel()])


def neutral_state(vbar=1.0):
    mu = np.zeros((4, 4), dtype=float)
    V = np.full((4, 4), float(vbar))
    return np.concatenate([mu.ravel(), V.ravel()])


def jacobian(d, fd, rho=0.98, vbar=1.0, order=40, eps=1.0e-5):
    y0 = neutral_state(vbar)
    f0 = affective_rhs(y0[:16], y0[16:], d, fd, rho=rho, order=order)
    J = np.empty((y0.size, y0.size), dtype=float)
    for k in range(y0.size):
        yp = y0.copy()
        ym = y0.copy()
        step = eps * max(1.0, abs(y0[k]))
        yp[k] += step
        ym[k] -= step
        fp = affective_rhs(yp[:16], yp[16:], d, fd, rho=rho, order=order)
        fm = affective_rhs(ym[:16], ym[16:], d, fd, rho=rho, order=order)
        J[:, k] = (fp - fm) / (2.0 * step)
    return J, f0


def class_contrast_forcing(f0, fd):
    """Weighted out-minus-in forcing for mu at the neutral state."""

    dmu = f0[:16].reshape(4, 4)
    p = proportions(fd)
    cls = group_classes()
    weight = p[:, None] * p[None, :]
    same = cls[:, None] == cls[None, :]
    out = np.sum(weight[~same] * dmu[~same]) / max(np.sum(weight[~same]), 1.0e-300)
    inn = np.sum(weight[same] * dmu[same]) / max(np.sum(weight[same]), 1.0e-300)
    return float(out - inn)


def scan(n_d=41, n_fd=41, rho=0.98, vbar=1.0, order=36):
    d_axis = np.linspace(-1.0, 1.0, n_d)
    fd_axis = np.linspace(0.0, 1.0, n_fd)
    lambda_max = np.empty((n_fd, n_d), dtype=float)
    lambda_mu = np.empty_like(lambda_max)
    lambda_V = np.empty_like(lambda_max)
    forcing = np.empty_like(lambda_max)
    for i, fd in enumerate(fd_axis):
        for j, d in enumerate(d_axis):
            J, f0 = jacobian(d, fd, rho=rho, vbar=vbar, order=order)
            eig = np.linalg.eigvals(J)
            lambda_max[i, j] = float(np.max(np.real(eig)))
            lambda_mu[i, j] = float(np.max(np.real(np.linalg.eigvals(J[:16, :16]))))
            lambda_V[i, j] = float(np.max(np.real(np.linalg.eigvals(J[16:, 16:]))))
            forcing[i, j] = class_contrast_forcing(f0, fd)
        print(f"[stage4] row {i + 1}/{n_fd}")
    return {
        "d": d_axis,
        "fd": fd_axis,
        "lambda_max": lambda_max,
        "lambda_mu": lambda_mu,
        "lambda_V": lambda_V,
        "forcing": forcing,
    }


def phase_map(ax, data, d, fd, title, cmap="coolwarm", vmin=None, vmax=None):
    im = ax.imshow(
        data,
        origin="upper",
        extent=[d[0], d[-1], fd[-1], fd[0]],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$f_d$")
    ax.set_title(title)
    return im


def save_outputs(data, label):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    npz = DATA / f"stage4_neutral_stability_{label}.npz"
    np.savez_compressed(npz, **data)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35), constrained_layout=True)
    lam_abs = max(abs(float(np.min(data["lambda_mu"]))), abs(float(np.max(data["lambda_mu"]))))
    if lam_abs == 0.0:
        lam_abs = 1.0
    im = phase_map(
        axes[0],
        data["lambda_mu"],
        data["d"],
        data["fd"],
        r"$\max \operatorname{Re}\lambda(J_\mu)$",
        vmin=-lam_abs,
        vmax=lam_abs,
    )
    if float(np.min(data["lambda_mu"])) < 0.0 < float(np.max(data["lambda_mu"])):
        axes[0].contour(data["d"], data["fd"], data["lambda_mu"], levels=[0.0], colors="black", linewidths=1.0)
    fig.colorbar(im, ax=axes[0], shrink=0.82)

    force_abs = max(abs(float(np.min(data["forcing"]))), abs(float(np.max(data["forcing"]))))
    im = phase_map(
        axes[1],
        data["forcing"],
        data["d"],
        data["fd"],
        r"neutral forcing of $\mu_\mathrm{out}-\mu_\mathrm{in}$",
        vmin=-force_abs,
        vmax=force_abs,
    )
    fig.colorbar(im, ax=axes[1], shrink=0.82)
    fig.suptitle("Stage 4 stability diagnostic for the four-population closure")
    png = FIGURES / f"stage4_neutral_stability_{label}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png, npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-d", type=int, default=31)
    p.add_argument("--n-fd", type=int, default=31)
    p.add_argument("--rho", type=float, default=0.98)
    p.add_argument("--vbar", type=float, default=1.0)
    p.add_argument("--order", type=int, default=32)
    args = p.parse_args()
    data = scan(args.n_d, args.n_fd, rho=args.rho, vbar=args.vbar, order=args.order)
    label = f"{args.n_fd}x{args.n_d}_rho{args.rho:g}_v{args.vbar:g}"
    for path in save_outputs(data, label):
        print(path)
    print("lambda_max range", float(np.min(data["lambda_max"])), float(np.max(data["lambda_max"])))
    print("lambda_mu range", float(np.min(data["lambda_mu"])), float(np.max(data["lambda_mu"])))
    print("lambda_V range", float(np.min(data["lambda_V"])), float(np.max(data["lambda_V"])))
    print("forcing range", float(np.min(data["forcing"])), float(np.max(data["forcing"])))


if __name__ == "__main__":
    main()
