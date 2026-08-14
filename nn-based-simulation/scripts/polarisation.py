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
from matplotlib.patches import Patch

from _cli import setup  # noqa: E402

from ednna.order_params import balance, overlaps, sign_balance, trust  # noqa: E402
from ednna.plotting import HIST_BLUE, HIST_RED, framed_axes, panel, save  # noqa: E402
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


def run(preset, P, n_agents=None, n_runs=None, use_cache=True):
    """Independent unbiased societies, measured before and after.

    Two knobs the phase diagrams do not need. ``n_agents`` because this figure is a
    distribution over pairs and a society of ``N`` supplies ``O(N^2)`` of them; and
    ``n_runs`` because :class:`SocietyBatch` evolves independent societies in
    lockstep, so pooling several costs little more than one and buys the same
    smoothness as a much larger single society without pretending the pairs within
    one society are independent.  Everything else is the preset's, including the
    calibrated interaction count per ordered pair.
    """
    model = preset.model.with_(n_issues=P)
    N = int(n_agents if n_agents is not None else preset.polarisation_agents)
    R = int(n_runs if n_runs is not None else preset.polarisation_runs)
    cache = (
        DATA_DIR / f"polarisation_P{P}_N{N}x{R}_K{model.n_dim}"
        f"_T{model.interactions_per_channel:g}.npz"
    )
    if use_cache and cache.exists():
        with np.load(cache) as z:
            print(f"[polarisation] loaded cache {cache.name}")
            return {k: z[k] for k in z.files}

    batch = SocietyBatch(
        n_agents=N, n_dim=model.n_dim, n_issues=P,
        d=np.zeros(R), f_d=np.zeros(R), seed=1234,
    )
    iu = np.triu_indices(N, 1)
    offd = ~np.eye(N, dtype=bool)
    pool = lambda M, mask: np.concatenate([m[mask] for m in M])

    rho_before = pool(overlaps(batch), iu)
    eta_before = pool(trust(batch), offd)
    steps = int(round(model.interactions_per_channel * N * (N - 1)))
    print(f"[polarisation] {R} societies of N={N}, {steps:,} interactions each ...",
          flush=True)
    batch.run(steps)
    rho_m, eta_m = overlaps(batch), trust(batch)
    rho_after, eta_after = pool(rho_m, iu), pool(eta_m, offd)
    bal = balance(batch)

    # R_wmu needs the two matrices paired, which the flattened pools cannot give
    # back, so compute it here rather than reconstruct it later
    iu_r, iu_c = iu
    r_wmu = np.array([
        ((e[iu_r, iu_c] + e[iu_c, iu_r]) * m[iu_r, iu_c]).sum() / (N * (N - 1))
        for m, e in zip(rho_m, eta_m)
    ])

    sb_rho = np.array([sign_balance(m) for m in rho_m])
    sb_eta = np.array([sign_balance(m) for m in eta_m])
    factions = np.array([[int((np.linalg.eigh(m)[1][:, -1] <= 0).sum()) for m in rho_m],
                         [N - int((np.linalg.eigh(m)[1][:, -1] <= 0).sum()) for m in rho_m]])
    out = {
        "rho_before": rho_before, "rho_after": rho_after,
        "eta_before": eta_before, "eta_after": eta_after,
        "B_I": bal["B_I"], "B_A": bal["B_A"],
        "alpha": np.asarray(model.alpha), "N": np.asarray(N), "R": np.asarray(R),
        "R_wmu": r_wmu,
        "sign_balance_rho": sb_rho, "sign_balance_eta": sb_eta,
        "factions": factions,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    n_tri = N * (N - 1) * (N - 2) // 6
    print(f"[polarisation] {R}x N={N}: B_I={bal['B_I'].mean():+.3f} "
          f"B_A={bal['B_A'].mean():+.3f}; sign balance rho {sb_rho.mean():.6f} "
          f"eta {sb_eta.mean():.6f} (0 unbalanced of {n_tri*R:,} triples if 1.0); "
          f"R_wmu={r_wmu.mean():+.3f}; faction split {factions[0].min()}-{factions[0].max()} of {N}; "
          f"pooled pairs: rho {rho_after.size:,}, eta {eta_after.size:,}")
    return out


def unordered_pairs(M):
    """The N(N-1)/2 values above the diagonal, for a symmetric matrix."""
    return M[np.triu_indices(M.shape[0], 1)]


def ordered_pairs(M):
    """All N(N-1) off-diagonal values, for an asymmetric one."""
    return M[~np.eye(M.shape[0], dtype=bool)]


def figure(data, style, name="polarisation"):
    """Two panels: what happened to the distribution of each pairwise quantity.

    Styled after the histograms of Costa (2021): solid pastel bars with a thin
    darker edge rather than step outlines, the two series overlaid with enough
    transparency that the overlap reads as a third tone, a closed frame with
    inward ticks on all four sides, and no grid.  No legend either --- the colours
    are named in the caption, which is what that layout does and which keeps the
    saturated trust spikes from having to share space with a key.
    """
    fig, axes = plt.subplots(1, 2, figsize=panel(1.0, 0.34 / 0.92))
    bins = np.linspace(-1, 1, 65)

    for ax, before, after, xlabel in (
        (axes[0], data["rho_before"], data["rho_after"], r"opinion overlap $\rho_{IJ}$"),
        (axes[1], data["eta_before"], data["eta_after"], r"trust $\eta_{J|I}$"),
    ):
        for values, (fill, edge) in ((before, HIST_BLUE), (after, HIST_RED)):
            ax.hist(values, bins=bins, density=True, color=fill, alpha=0.62,
                    edgecolor=edge, linewidth=0.35, zorder=2)
        ax.set_xlabel(xlabel)
        # a little air beyond the walls: both quantities pile up at exactly +-1, and
        # a bar flush against the frame reads as clipped rather than as a peak
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_ylim(bottom=0)
        framed_axes(ax)
    axes[0].set_ylabel("density over pairs")

    # swatches drawn like the bars themselves, above the axes: the trust panel is
    # occupied at both walls and the opinion panel in the middle, so there is no
    # in-axes corner that is free in both
    handles = [
        Patch(facecolor=fill, edgecolor=edge, alpha=0.62, linewidth=0.35, label=lab)
        for (fill, edge), lab in ((HIST_BLUE, "initial"), (HIST_RED, "final"))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.03), handlelength=1.5, handleheight=0.9,
               columnspacing=1.8, borderaxespad=0.0)
    fig.tight_layout(pad=0.4, w_pad=1.8, rect=(0, 0, 1, 0.93))
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    data = run(preset, preset.p_small, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
