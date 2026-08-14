"""Population-level phase diagram from the Gaussian pairwise kernel closure.

This is a semi-analytical closure, not an agent-based simulation.  The four
populations are class A/B crossed with nondiscriminator/discriminator receiver
status.  Pairwise drifts are evaluated by Gaussian quadrature kernels.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".matplotlib"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from controlledcv.kernels import affective_kernels, ideological_coefficients
from controlledcv.modulation import Phi


FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def field_matrix(d):
    """Case 6: rows receiver class, columns emitter class."""

    return np.array([[d, -d], [-d, d]], dtype=float)


@lru_cache(maxsize=400_000)
def cached_kernels(rho_key, mu_key, D_key, order):
    rho = float(rho_key) / 1000.0
    mu = float(mu_key) / 1000.0
    D = float(D_key) / 1000.0
    aff = affective_kernels(1.0, rho, mu, D=D, order=order)
    rho_coeff = float(np.clip(rho, -0.94, 0.94))
    ideol = ideological_coefficients(1.0, 1.0, rho_coeff, mu, D=D, order=order)
    return aff.M_mu, aff.M_V, ideol.A, ideol.B


def kernels(rho, mu, D, order):
    rho = float(np.clip(rho, -0.98, 0.98))
    mu = float(np.clip(mu, -5.0, 5.0))
    D = float(np.clip(D, -2.0, 2.0))
    return cached_kernels(round(1000 * rho), round(1000 * mu), round(1000 * D), int(order))


def initial_vectors(dim=6, eps=0.08):
    """Small deterministic perturbations around one neutral ideology."""

    w = np.zeros((4, dim), dtype=float)
    w[:, 0] = 1.0
    # groups: A non-disc, A disc, B non-disc, B disc
    w[0, 1] += eps
    w[1, 1] += eps
    w[2, 1] -= eps
    w[3, 1] -= eps
    w[0, 2] -= eps
    w[1, 2] += eps
    w[2, 2] -= eps
    w[3, 2] += eps
    return normalize_rows(w)


def normalize_rows(w):
    return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1.0e-300)


def proportions(fd):
    return np.array([0.5 * (1.0 - fd), 0.5 * fd, 0.5 * (1.0 - fd), 0.5 * fd], dtype=float)


def group_classes():
    return np.array([0, 0, 1, 1], dtype=int)


def group_discriminates():
    return np.array([False, True, False, True])


def integrate_point(d, fd, steps=500, dt=0.04, vbar0=1.0, order=28):
    p = proportions(fd)
    cls = group_classes()
    disc = group_discriminates()
    D_class = field_matrix(d)
    w = initial_vectors()
    mu = np.zeros((4, 4), dtype=float)
    V = np.full((4, 4), float(vbar0))

    for _ in range(int(steps)):
        rho = np.clip(w @ w.T, -0.999, 0.999)
        dw = np.zeros_like(w)
        dmu = np.zeros_like(mu)
        dV = np.zeros_like(V)
        for r in range(4):
            if p[r] == 0.0:
                continue
            for e in range(4):
                if p[e] == 0.0:
                    continue
                D = D_class[cls[r], cls[e]] if disc[r] else 0.0
                M_mu, M_V, A, B = kernels(rho[r, e], mu[r, e], D, order)
                dmu[r, e] += p[e] * V[r, e] * M_mu
                dV[r, e] += p[e] * V[r, e] * V[r, e] * M_V
                if r != e and abs(rho[r, e]) < 0.97:
                    drift = A * w[r] + B * w[e]
                    dw[r] += p[e] * (drift - np.dot(drift, w[r]) * w[r])
        w = normalize_rows(w + dt * dw)
        mu += dt * dmu
        V = np.clip(V + dt * dV, 0.02, 5.0)
        mu = np.clip(mu, -5.0, 5.0)

    return observables(w, mu, p)


def observables(w, mu, p):
    cls = group_classes()
    kappa = np.where(cls == 0, 1.0, -1.0)
    rho = np.clip(w @ w.T, -1.0, 1.0)
    eta = 1.0 - 2.0 * Phi(mu)
    pair_weight = p[:, None] * p[None, :]
    G = kappa[:, None] * kappa[None, :]
    return {
        "R_wmu": float(np.sum(pair_weight * eta * rho)),
        "R_muc": float(np.sum(pair_weight * eta * G)),
        "R_cw": float(np.sum(pair_weight * rho * G)),
        "mean_eta": float(np.sum(pair_weight * eta)),
        "mean_rho": float(np.sum(pair_weight * rho)),
    }


def sweep(n_d, n_fd, steps, dt, vbar0, order):
    d_axis = np.linspace(-1.0, 1.0, n_d)
    fd_axis = np.linspace(0.0, 1.0, n_fd)
    result = {"d": d_axis, "fd": fd_axis}
    keys = ("R_wmu", "R_muc", "R_cw", "mean_eta", "mean_rho")
    for key in keys:
        result[key] = np.empty((n_fd, n_d), dtype=float)

    total = n_d * n_fd
    done = 0
    for i, fd in enumerate(fd_axis):
        for j, d in enumerate(d_axis):
            obs = integrate_point(d, fd, steps=steps, dt=dt, vbar0=vbar0, order=order)
            for key in keys:
                result[key][i, j] = obs[key]
            done += 1
        print(f"[closure] row {i + 1}/{n_fd}, {done}/{total}")
    return result


def phase_map(ax, data, d, fd, title, vmin=-1.0, vmax=1.0):
    im = ax.imshow(
        data,
        origin="upper",
        extent=[d[0], d[-1], fd[-1], fd[0]],
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$f_d$")
    ax.set_title(title)
    return im


def rgb_composite(R_muc, R_cw, R_wmu):
    def norm(x):
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    return np.dstack([norm(R_muc), norm(R_cw), norm(R_wmu)])


def save_outputs(data, label):
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    npz = DATA / f"population_closure_phase_{label}.npz"
    np.savez_compressed(npz, **data)

    keys = ("R_wmu", "R_muc")
    scales = {"R_wmu": (0.0, 0.7), "R_muc": (-0.6, 0.6)}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), constrained_layout=True)
    for ax, key in zip(axes, keys):
        vmin, vmax = scales[key]
        im = phase_map(ax, data[key], data["d"], data["fd"], key, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("Population closure from Gaussian pairwise kernels")
    heatmaps = FIGURES / f"population_closure_heatmaps_{label}.png"
    fig.savefig(heatmaps, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return heatmaps, npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-d", type=int, default=21)
    p.add_argument("--n-fd", type=int, default=21)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--vbar", type=float, default=1.0)
    p.add_argument("--order", type=int, default=28)
    args = p.parse_args()
    data = sweep(args.n_d, args.n_fd, args.steps, args.dt, args.vbar, args.order)
    label = f"{args.n_fd}x{args.n_d}_s{args.steps}_dt{args.dt:g}_v{args.vbar:g}"
    for path in save_outputs(data, label):
        print(path)


if __name__ == "__main__":
    main()
