"""Plot helpers for small-C,V sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


def phase_map(ax, data, d, fd, title, vmin=-1.0, vmax=1.0, cmap="coolwarm"):
    im = ax.imshow(data, origin="upper", extent=[d[0], d[-1], fd[-1], fd[0]], aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$f_d$")
    ax.set_title(title)
    return im


def rgb_composite(R_muc, R_cw, R_wmu):
    def norm(x):
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    return np.dstack([norm(R_muc), norm(R_cw), norm(R_wmu)])


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
