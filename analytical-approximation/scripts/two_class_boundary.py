"""Two-class pairwise boundary from the Gaussian kernels.

This script studies a simpler closure than the four-population prototype model.
For two normalized class prototypes with cross-class overlap rho, the tangent
part of the ideological drift has the form

    d rho / d tau = c_eff * B(rho, mu_out, D_out) * (1 - rho^2),

up to a positive interaction-rate/learning-scale prefactor.  Thus the sign of
the Gaussian-kernel coefficient B controls whether cross-class ideology is
locally attractive or repulsive.

We close the affective variable by the stationary pairwise relation

    M_mu(rho, mu_out, D_out) = 0,

and use D_out = -d for the symmetric case-6 discrimination field.  The curve

    B(rho, mu_out^*(rho,d), -d) = 0

is a candidate analytical attraction/repulsion boundary for the onset of
ideological separation in this simplified regime.
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

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from controlledcv.kernels import _skew_expectation, ideological_coefficients
from controlledcv.modulation import F_mu

FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def M_mu(rho, mu, D, order):
    return float(_skew_expectation(1.0, rho, lambda h: F_mu(h + D, mu), order=order))


def B_coeff(rho, mu, D, order):
    rho_eval = float(np.clip(rho, -0.94, 0.94))
    return ideological_coefficients(1.0, 1.0, rho_eval, mu, D=D, order=order).B


def stationary_mu(rho, D, order=80, lo=-10.0, hi=10.0, n_scan=121):
    """Find a stationary root M_mu(rho, mu, D)=0, if one exists."""

    xs = np.linspace(lo, hi, n_scan)
    vals = np.array([M_mu(rho, x, D, order) for x in xs])
    idx = np.where(vals[:-1] * vals[1:] <= 0.0)[0]
    if idx.size == 0:
        # No finite root in the scan window.  Return the point closest to zero
        # and mark it as not bracketed.
        k = int(np.argmin(np.abs(vals)))
        return float(xs[k]), False, float(vals[k])

    a = float(xs[int(idx[0])])
    b = float(xs[int(idx[0]) + 1])
    fa = M_mu(rho, a, D, order)
    fb = M_mu(rho, b, D, order)
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = M_mu(rho, m, D, order)
        if abs(fm) < 1.0e-10 or (b - a) < 1.0e-8:
            return float(m), True, float(fm)
        if fa * fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    m = 0.5 * (a + b)
    return float(m), True, float(M_mu(rho, m, D, order))


def scan(rhos, ds, order=60):
    mu_star = np.empty((rhos.size, ds.size), dtype=float)
    has_root = np.empty_like(mu_star, dtype=bool)
    residual = np.empty_like(mu_star)
    B = np.empty_like(mu_star)
    for i, rho in enumerate(rhos):
        for j, d in enumerate(ds):
            D_out = -float(d)
            mu, ok, res = stationary_mu(rho, D_out, order=order)
            mu_star[i, j] = mu
            has_root[i, j] = ok
            residual[i, j] = res
            B[i, j] = B_coeff(rho, mu, D_out, order=order)
        print(f"[two-class] rho row {i + 1}/{rhos.size}")
    return {
        "rho": rhos,
        "d": ds,
        "mu_star": mu_star,
        "has_root": has_root,
        "residual": residual,
        "B": B,
    }


def save_outputs(data, label):
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    npz = DATA / f"two_class_boundary_{label}.npz"
    np.savez_compressed(npz, **data)

    rho = data["rho"]
    d = data["d"]
    B = np.ma.array(data["B"], mask=~data["has_root"])
    mu = np.ma.array(data["mu_star"], mask=~data["has_root"])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(B.filled(np.nan))))
    im = axes[0].imshow(
        B,
        origin="lower",
        extent=[d[0], d[-1], rho[0], rho[-1]],
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    if np.nanmin(B.filled(np.nan)) < 0.0 < np.nanmax(B.filled(np.nan)):
        axes[0].contour(d, rho, B, levels=[0.0], colors="black", linewidths=1.4)
    axes[0].set_title(r"$B(\rho,\mu^*_{\rm out},-d)$")
    axes[0].set_xlabel(r"$d$")
    axes[0].set_ylabel(r"$\rho$")
    fig.colorbar(im, ax=axes[0], shrink=0.84)

    im = axes[1].imshow(
        mu,
        origin="lower",
        extent=[d[0], d[-1], rho[0], rho[-1]],
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_title(r"stationary $\mu^*_{\rm out}$")
    axes[1].set_xlabel(r"$d$")
    axes[1].set_ylabel(r"$\rho$")
    fig.colorbar(im, ax=axes[1], shrink=0.84)

    fig.suptitle("Two-class analytical boundary from pairwise Gaussian kernels")
    png = FIGURES / f"two_class_boundary_{label}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png, npz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rho", type=int, default=81)
    p.add_argument("--n-d", type=int, default=101)
    p.add_argument("--order", type=int, default=60)
    args = p.parse_args()
    rhos = np.linspace(-0.9, 0.9, args.n_rho)
    ds = np.linspace(-1.5, 1.5, args.n_d)
    data = scan(rhos, ds, order=args.order)
    label = f"{args.n_rho}x{args.n_d}"
    for path in save_outputs(data, label):
        print(path)
    B = np.ma.array(data["B"], mask=~data["has_root"])
    print("root coverage", int(data["has_root"].sum()), "/", data["has_root"].size)
    print("B range", float(B.min()), float(B.max()))


if __name__ == "__main__":
    main()
