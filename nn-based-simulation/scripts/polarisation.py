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

from ednna.order_params import balance, overlaps, trust  # noqa: E402
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


def run(preset, P, use_cache=True):
    """One unbiased society, measured before and after. Returns a dict."""
    model = preset.model.with_(n_issues=P)
    cache = (
        DATA_DIR / f"polarisation_P{P}_N{model.n_agents}_K{model.n_dim}"
        f"_T{model.interactions_per_channel:g}.npz"
    )
    if use_cache and cache.exists():
        with np.load(cache) as z:
            print(f"[polarisation] loaded cache {cache.name}")
            return {k: z[k] for k in z.files}

    batch = SocietyBatch(
        n_agents=model.n_agents, n_dim=model.n_dim, n_issues=P,
        d=0.0, f_d=0.0, seed=1234,
    )
    rho0, eta0 = overlaps(batch)[0].copy(), trust(batch)[0].copy()
    batch.run(model.n_steps())
    rho1, eta1 = overlaps(batch)[0].copy(), trust(batch)[0].copy()
    bal = balance(batch)
    out = {
        "rho0": rho0, "eta0": eta0, "rho1": rho1, "eta1": eta1,
        "B_I": np.asarray(bal["B_I"][0]), "B_A": np.asarray(bal["B_A"][0]),
        "alpha": np.asarray(model.alpha),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"[polarisation] P={P}: B_I={out['B_I']:+.3f} B_A={out['B_A']:+.3f}")
    return out


def offdiag(M):
    N = M.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return M[mask]


def figure(data, style, name="polarisation"):
    rho1, eta1 = data["rho1"], data["eta1"]
    order = faction_order(rho1)
    width = text_width()
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.29))

    for ax, M, title, cmap in (
        (axes[0], rho1[np.ix_(order, order)], r"opinion overlap $\rho_{IJ}$", "RdBu_r"),
        (axes[1], eta1[np.ix_(order, order)], r"trust $\eta_{J|I}$", "PuOr_r"),
    ):
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, origin="upper")
        ax.set_title(title, pad=3)
        ax.set_xlabel("agent (sorted)")
        ax.set_xticks([0, M.shape[0] - 1])
        ax.set_yticks([0, M.shape[0] - 1])
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                                ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)
    axes[0].set_ylabel("agent (sorted)")

    ax = axes[2]
    bins = np.linspace(-1, 1, 41)
    ax.hist(offdiag(data["rho0"]), bins=bins, density=True, histtype="stepfilled",
            color="0.75", edgecolor="0.4", lw=0.6, label=r"$\rho$, initial")
    ax.hist(offdiag(rho1), bins=bins, density=True, histtype="step",
            color="tab:red", lw=1.2, label=r"$\rho$, final")
    ax.hist(offdiag(eta1), bins=bins, density=True, histtype="step",
            color="tab:purple", lw=1.2, ls="--", label=r"$\eta$, final")
    ax.set_xlabel("pairwise value")
    ax.set_ylabel("density")
    ax.set_xlim(-1, 1)
    # trust saturates, so its two spikes at +-1 run off the top; clipping the axis
    # keeps the opinion distributions, which are the interesting ones, readable
    ax.set_ylim(0, 3.4)
    ax.legend(loc="upper center", fontsize=6)
    ax.set_title("from unimodal to bimodal", pad=3)

    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    data = run(preset, preset.p_small, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
