#!/usr/bin/env python3
"""How do we know the society polarises?

The phase diagrams answer questions about the *discriminating* society. This
figure answers the prior one: with no discrimination field at all, what does the
society do? It splits into two mutually distrustful camps whose membership has
nothing to do with any label, and the evidence is direct --- the distribution of
pairwise overlaps starts unimodal at zero and ends bimodal at plus and minus one,
and the overlap matrix, with agents sorted by faction, shows two blocks.

Everything here comes from the existing machinery: a :class:`SocietyBatch` at
``d = 0`` measured with :func:`ednna.order_params.overlaps` and
:func:`ednna.order_params.trust`.

Writes ``polarisation``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.order_params import balance, overlaps, sign_balance, trust  # noqa: E402
from ednna.plotting import save, text_width  # noqa: E402
from ednna.society import SocietyBatch  # noqa: E402
from ednna.sweep import DATA_DIR  # noqa: E402


def faction_order(rho):
    """Sort agents by the leading eigenvector of the overlap matrix.

    The sign of that eigenvector is the natural faction assignment: it is the
    partition that best explains the observed pattern of agreement, and no class
    label is involved.  Sorting by its value makes a two-bloc structure visible as
    two diagonal blocks.
    """
    vals, vecs = np.linalg.eigh(rho)
    leading = vecs[:, np.argmax(vals)]
    return np.argsort(leading)


def run(preset, P, n_agents=None, use_cache=True):
    """One unbiased society, measured before and after.

    ``n_agents`` overrides the preset: this figure is a distribution over pairs, and
    a larger society gives ~N^2 of them, so it is worth paying for a bigger run here
    than the phase diagrams use.  Everything else, including the calibrated number of
    interactions per ordered pair, is the preset's.
    """
    model = preset.model.with_(n_issues=P)
    N = int(n_agents or model.n_agents)
    cache = (
        DATA_DIR / f"polarisation_P{P}_N{N}_K{model.n_dim}"
        f"_T{model.interactions_per_channel:g}.npz"
    )
    if use_cache and cache.exists():
        with np.load(cache) as z:
            print(f"[polarisation] loaded cache {cache.name}")
            return {k: z[k] for k in z.files}

    batch = SocietyBatch(
        n_agents=N, n_dim=model.n_dim, n_issues=P, d=0.0, f_d=0.0, seed=1234,
    )
    rho0, eta0 = overlaps(batch)[0].copy(), trust(batch)[0].copy()
    steps = int(round(model.interactions_per_channel * N * (N - 1)))
    print(f"[polarisation] N={N}, {steps:,} interactions ...", flush=True)
    batch.run(steps)
    rho1, eta1 = overlaps(batch)[0].copy(), trust(batch)[0].copy()
    bal = balance(batch)

    leading = np.linalg.eigh(rho1)[1][:, -1]
    out = {
        "rho0": rho0, "eta0": eta0, "rho1": rho1, "eta1": eta1,
        "B_I": np.asarray(bal["B_I"][0]), "B_A": np.asarray(bal["B_A"][0]),
        "alpha": np.asarray(model.alpha), "N": np.asarray(N),
        "sign_balance_rho": np.asarray(sign_balance(rho1)),
        "sign_balance_eta": np.asarray(sign_balance(eta1)),
        "faction": np.asarray([int((leading <= 0).sum()), int((leading > 0).sum())]),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"[polarisation] N={N}: B_I={out['B_I']:+.3f} B_A={out['B_A']:+.3f}; "
          f"sign-balanced triples: rho {out['sign_balance_rho']:.3%}, "
          f"eta {out['sign_balance_eta']:.3%}; factions {out['faction'].tolist()}")
    return out


def unordered_pairs(M):
    """The N(N-1)/2 values above the diagonal, for a symmetric matrix."""
    return M[np.triu_indices(M.shape[0], 1)]


def ordered_pairs(M):
    """All N(N-1) off-diagonal values, for an asymmetric one."""
    return M[~np.eye(M.shape[0], dtype=bool)]


def figure(data, style, name="polarisation"):
    """Two panels: what happened to the distribution of each pairwise quantity.

    No matrices.  A histogram over pairs does not depend on how the agents are
    labelled, so it cannot be an artefact of any sorting -- which is the whole
    reason to prefer it.  What the matrices showed instead, that the sides form
    exactly two blocs rather than many, is stated as a number in the caption via
    the sign-balance fraction.
    """
    N = int(data["N"])
    width = text_width()
    fig, axes = plt.subplots(1, 2, figsize=(width * 0.86, width * 0.30))
    bins = np.linspace(-1, 1, 81)

    panels = (
        (axes[0], unordered_pairs(data["rho0"]), unordered_pairs(data["rho1"]),
         "#b2182b", r"opinion overlap $\rho_{IJ}$"),
        (axes[1], ordered_pairs(data["eta0"]), ordered_pairs(data["eta1"]),
         "#5e3c99", r"trust $\eta_{J|I}$"),
    )
    for ax, before, after, colour, xlabel in panels:
        ax.hist(before, bins=bins, density=True, color="0.82", edgecolor="0.55",
                lw=0.5, label="initial", zorder=2)
        ax.hist(after, bins=bins, density=True, color=colour, alpha=0.30,
                zorder=3)
        ax.hist(after, bins=bins, density=True, histtype="step", color=colour,
                lw=1.4, label="final", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-1.02, 1.02)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.axvline(0.0, color="0.8", lw=0.5, zorder=1)
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=6.5, handlelength=1.1,
                  loc="upper center", borderaxespad=0.2)
    axes[0].set_ylabel("density over pairs")
    fig.tight_layout(pad=0.4, w_pad=1.6)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    data = run(preset, preset.p_small, n_agents=200, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
