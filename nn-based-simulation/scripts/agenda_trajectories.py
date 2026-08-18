#!/usr/bin/env python3
"""Does distrust come first, or disagreement?  It depends on the agenda.

With no discrimination field at all, a society still polarizes: learning anneals
away frustration, driving both the ideological balance ``B_I`` and the affective
balance ``B_A`` from zero (half the triples frustrated, as at random
initialization) towards one (no frustration).  The *path* it takes through the
``(B_I, B_A)`` plane depends on the complexity of the agenda ``alpha = P/K``:

* small ``alpha`` -- few issues, "discussing only symbols" -- the trajectory
  runs up the ``B_A`` axis first: affective alignment forms quickly and then
  slowly drags opinions into line.  Agents distrust each other first and come
  to disagree afterwards.
* large ``alpha`` -- many issues discussed in detail -- the trajectory stays
  below the diagonal: ideological alignment forms first and affective alignment
  follows.  Agents disagree first and come to distrust each other afterwards.

Writes ``agenda_trajectories``.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.order_params import balance  # noqa: E402
from ednna.plotting import framed_axes, panel, pastel, save  # noqa: E402
from ednna.society import SocietyBatch  # noqa: E402
from ednna.sweep import DATA_DIR  # noqa: E402


def trajectories(preset, use_cache=True, verbose=True):
    """B_I and B_A over time for each agenda size, averaged over repeats."""
    model = preset.model
    issues = tuple(preset.trajectory_issues)
    n_rep = preset.n_trajectory_repeats
    n_samples = preset.n_trajectory_samples
    total = model.n_steps()

    cache = (
        DATA_DIR
        / f"trajectories_{preset.name}_N{model.n_agents}_K{model.n_dim}"
        f"_T{model.interactions_per_channel:g}_r{n_rep}.npz"
    )
    if use_cache and cache.exists():
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
        if verbose:
            print(f"[trajectories] loaded cache {cache.name}")
        return out

    # linear in time, matching the draft: the markers crowd near the end
    # because the annealed dynamics slows down, not because sampling does
    times = np.unique(np.linspace(total / n_samples, total, n_samples).astype(int))

    B_I = np.zeros((len(issues), times.size))
    B_A = np.zeros((len(issues), times.size))
    for i, P in enumerate(issues):
        batch = SocietyBatch(
            n_agents=model.n_agents,
            n_dim=model.n_dim,
            n_issues=P,
            d=np.zeros(n_rep),
            f_d=np.zeros(n_rep),
            seed=4000 + i,
            dtype=model.numpy_dtype(),
        )
        samples = batch.run(
            total, measure_at=times.tolist(), measure_fn=lambda s: balance(s)
        )
        B_I[i] = [s["B_I"].mean() for s in samples]
        B_A[i] = [s["B_A"].mean() for s in samples]
        if verbose:
            print(
                f"[trajectories] P={P:>6d} (alpha={P/model.n_dim:8.4g}): "
                f"B_I={B_I[i, -1]:+.3f}  B_A={B_A[i, -1]:+.3f}"
            )

    out = {
        "issues": np.asarray(issues),
        "times": times,
        "B_I": B_I,
        "B_A": B_A,
        "n_dim": np.asarray(model.n_dim),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def figure(data, style):
    issues = data["issues"]
    K = int(data["n_dim"])
    fig, ax = plt.subplots(figsize=panel(0.60, 0.52/0.72))
    # both axes are balances on the same [0, 1] scale and the reference line is the
    # diagonal, which only means "equal" if the box is square
    ax.set_box_aspect(1)
    colours = pastel("viridis_r", 0.30)(np.linspace(0.05, 0.95, len(issues)))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=1)
    for i, P in enumerate(issues):
        ax.plot(
            data["B_I"][i],
            data["B_A"][i],
            marker="<",
            markersize=3.2,
            lw=0.8,
            color=colours[i],
            label=rf"$\alpha = {P/K:.2f}$",
            zorder=2,
        )
    ax.set_xlabel(r"$B_I$")
    ax.set_ylabel(r"$B_A$")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.05, 1.05)
    framed_axes(ax, minor=False)
    ax.legend(loc="lower right", fontsize=6, ncol=1, framealpha=0.9, frameon=True)
    fig.tight_layout(pad=0.4)
    return save(fig, "agenda_trajectories", style)


def main():
    args, preset = setup(__doc__)
    data = trajectories(preset, use_cache=not args.no_cache)
    figure(data, args.style)


if __name__ == "__main__":
    main()
