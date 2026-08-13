#!/usr/bin/env python3
"""Generate the stub figures that hold the language-agent section's layout.

The point is that the paper compiles, paginates and reviews at its true length
before the study is run, so that adding the real panels changes only the image
files.  Each stub carries the axes and, where the simulation already predicts a
shape, the predicted curve -- and says plainly that the measurements are missing.

    python make_placeholders.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}


def stamp(ax, text):
    if text:
        ax.text(
            0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.5, color="purple", alpha=0.55, rotation=12,
        )


def correlations_stub():
    """Order parameters against the instructed field, with room for the data."""
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    d = np.linspace(-1, 1, 200)
    ax.plot(d, np.tanh(6.0 * d), color="tab:red", lw=1.0, ls="--",
            label=r"$R_{\mu,c}$ predicted")
    ax.plot(d, 0.9 * np.sqrt(np.clip(d, 0, None)), color="tab:green", lw=1.0, ls="--",
            label=r"$R_{c,w}$ predicted")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"instructed field $d$")
    ax.set_ylabel("order parameter")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left")
    stamp(ax, "LLM measurements to be overlaid")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "placeholder_llm_correlations.pdf")
    plt.close(fig)


def balance_stub():
    """The (B_I, B_A) plane with the diagonal and the two predicted regimes."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    t = np.linspace(0, 1, 60)
    ax.plot(0.25 * t**2, np.sqrt(t), color="tab:olive", lw=0.9, label="simple agenda")
    ax.plot(np.sqrt(t), 0.45 * t**1.5, color="tab:purple", lw=0.9, label="complex agenda")
    ax.set_xlabel(r"$B_I$")
    ax.set_ylabel(r"$B_A$")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "placeholder_llm_balance.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        correlations_stub()
        balance_stub()
    print(f"wrote placeholder figures into {FIG}")


if __name__ == "__main__":
    main()
