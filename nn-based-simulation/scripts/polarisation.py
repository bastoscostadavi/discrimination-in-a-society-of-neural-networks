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


def unordered_pairs(M):
    """The N(N-1)/2 values above the diagonal, for a symmetric matrix."""
    return M[np.triu_indices(M.shape[0], 1)]


def ordered_pairs(M):
    """All N(N-1) off-diagonal values, for an asymmetric one."""
    return M[~np.eye(M.shape[0], dtype=bool)]


def figure(data, style, name="polarisation"):
    """Two matrices and, beside them, what happened to the distributions.

    The histograms get one panel per sector rather than sharing an axis: opinion
    overlap is symmetric and lives on N(N-1)/2 unordered pairs, trust is asymmetric
    and lives on N(N-1) ordered pairs, and trust saturates so hard that on a shared
    axis its spikes either dwarf the opinion curves or have to be clipped.  Showing
    both sectors before *and* after also makes the point that trust does not start
    saturated: the two-tone matrix in the centre is produced by the dynamics.
    """
    rho0, rho1, eta0, eta1 = data["rho0"], data["rho1"], data["eta0"], data["eta1"]
    N = rho1.shape[0]
    order = faction_order(rho1)
    # the boundary is where the sorted eigenvector changes sign, not the count of
    # positive entries: faction_order sorts ascending, so the negative group is first
    leading = np.linalg.eigh(rho1)[1][:, -1][order]
    boundary = int(np.searchsorted(np.sign(leading), 0.5)) - 0.5

    width = text_width()
    fig = plt.figure(figsize=(width, width * 0.33))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.92], hspace=0.55, wspace=0.42,
                          left=0.06, right=0.98, top=0.90, bottom=0.16)

    for col, (M, title, cmap) in enumerate((
        (rho1, r"opinion overlap $\rho_{IJ}$", "RdBu_r"),
        (eta1, r"trust $\eta_{J|I}$", "PuOr_r"),
    )):
        ax = fig.add_subplot(gs[:, col])
        im = ax.imshow(M[np.ix_(order, order)], cmap=cmap, vmin=-1, vmax=1,
                       origin="upper")
        # mark where the two factions meet
        ax.axhline(boundary, color="k", lw=0.6, alpha=0.6)
        ax.axvline(boundary, color="k", lw=0.6, alpha=0.6)
        ax.set_title(title, pad=3, fontsize=8)
        ax.set_xlabel("agent, sorted by faction")
        ax.set_xticks([0, N - 1]); ax.set_yticks([0, N - 1])
        if col == 0:
            ax.set_ylabel("agent, sorted by faction")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb.outline.set_linewidth(0.4)

    bins = np.linspace(-1, 1, 41)
    panels = (
        (gs[0, 2], unordered_pairs(rho0), unordered_pairs(rho1), "tab:red",
         rf"$\rho$, {N*(N-1)//2} pairs"),
        (gs[1, 2], ordered_pairs(eta0), ordered_pairs(eta1), "tab:purple",
         rf"$\eta$, {N*(N-1)} ordered pairs"),
    )
    for spec, before, after, colour, title in panels:
        ax = fig.add_subplot(spec)
        ax.hist(before, bins=bins, density=True, histtype="stepfilled",
                color="0.8", edgecolor="0.45", lw=0.6, label="initial")
        ax.hist(after, bins=bins, density=True, histtype="step",
                color=colour, lw=1.3, label="final")
        ax.set_title(title, pad=3, fontsize=7.5)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="upper center", handlelength=1.2)
    ax.set_xlabel("pairwise value")
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    data = run(preset, preset.p_small, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
