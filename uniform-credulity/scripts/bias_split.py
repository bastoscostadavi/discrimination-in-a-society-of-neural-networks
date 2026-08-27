#!/usr/bin/env python3
"""What a uniform field does to the two groups it creates, at a point.

The phase plane is drawn from aggregates.  This is the picture underneath them:
a handful of societies at full and at half prevalence, measured with the paper's
order parameters, with the class channels as controls, and with the directed
trust matrix split by the only partition a uniform field induces -- the agents
that carry it against the agents that do not.

Two things are worth separating there, and the aggregate ``T_mu`` mixes them.
The trust a biased agent *extends* is the field acting: it shifts the sign test
that decides whether a message reads as agreement, so of course a credulous agent
trusts more.  The trust a biased agent *receives* is not: ``D[r, e] = a`` says
nothing whatever about the emitter, so any separation there is something the
population worked out for itself.

Writes ``uniform_split`` and prints the table.
"""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from credulity.order_params import (  # noqa: E402
    PAPER_NAMES, bias_block_trust, measure, trust, trust_given_per_agent,
    trust_received_per_agent,
)
from credulity.plotting import (  # noqa: E402
    HIST_BLUE, HIST_RED, framed_axes, panel, save,
)
from credulity.society import SocietyBatch  # noqa: E402

#: One condition per row of the table: a label, the strength, the prevalence.
#: The two full-prevalence rows say what the field does when it is the whole
#: population's; the two half-prevalence rows are where the partition exists and
#: so where the emergent margin can be read at all.
CONDITIONS = (
    ("none",              0.0, 0.0),
    ("$a=+1$\nall",      +1.0, 1.0),
    ("$a=+1$\nhalf",     +1.0, 0.5),
    ("$a=-1$\nall",      -1.0, 1.0),
    ("$a=-1$\nhalf",     -1.0, 0.5),
)

#: The two conditions the histograms are drawn for: the mixed populations, one
#: per sign.  At full prevalence there is only one group and nothing to compare.
HIST_CONDITIONS = ("$a=+1$\nhalf", "$a=-1$\nhalf")


def run(preset, conditions=CONDITIONS, verbose=True):
    """One batch holding every condition, repeated ``preset.demo_runs`` times."""
    model = preset.model
    N = preset.demo_agents
    reps = preset.demo_runs
    a = np.repeat([c[1] for c in conditions], reps)
    f = np.repeat([c[2] for c in conditions], reps)

    batch = SocietyBatch(
        n_agents=N, n_dim=model.n_dim, n_issues=model.n_issues,
        a=a, f=f, seed=20260821,
        dtype=model.numpy_dtype(), shared_schedule=model.shared_schedule,
    )
    n_steps = int(round(model.interactions_per_channel * N * (N - 1)))
    if verbose:
        print(f"[bias_split] {batch.R} societies of N={N}, "
              f"{n_steps:,} interactions each")
    batch.run(n_steps)
    return batch, reps


def _nanmean(v, axis):
    """``np.nanmean`` with the all-nan warning silenced.

    A condition at full or zero prevalence has an empty group, so some rows are
    entirely ``nan`` by construction.  ``nan`` is the right answer there and the
    warning is noise, but silencing it has to be scoped to this call: an all-nan
    slice anywhere else would be a bug worth hearing about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=axis)


def _pool(values, reps):
    """Mean over the repeats of each condition, ignoring empty-group ``nan``."""
    v = np.asarray(values, dtype=float).reshape(-1, reps)
    return _nanmean(v, axis=1)


def _pool_se(values, reps):
    """Standard error of that mean, over the repeats that are not ``nan``.

    Reported for the block quantities because the emergent margin is a small
    difference between two of them, and a small difference between two noisy
    numbers is not a result until the noise is on the page next to it.
    """
    v = np.asarray(values, dtype=float).reshape(-1, reps)
    n = np.isfinite(v).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(v, axis=1, ddof=1)
    return np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)


def _fmt(values, width=9, prec=3):
    return "".join(f"{'nan':>{width}}" if not np.isfinite(v)
                   else f"{v:>{width}.{prec}f}" for v in values)


def table(batch, reps, conditions=CONDITIONS):
    """Print the table, and return it as a dict of pooled arrays."""
    stats = measure(batch)
    pooled = {k: _pool(v, reps) for k, v in stats.items()}
    blocks = bias_block_trust(batch).reshape(-1, reps, 2, 2)
    block_m = _nanmean(blocks, axis=1)

    names = [c[0].replace("$", "").replace("\n", " ").replace("=", " = ")
             for c in conditions]
    width = max(len(n) for n in names) + 1

    # The paper's five come from PAPER_NAMES, not from a hand-written tuple:
    # writing them out by hand is how B_rho came to be swept and never shown.
    overall = ("T_mu", "rho_mean") + tuple(
        k for k in PAPER_NAMES if k not in ("R_muc", "R_cw"))
    controls = ("R_muc", "R_cw", "R_cred", "R_stat")
    margins = ("T_give_b", "T_give_u", "T_get_b", "T_get_u")
    opinion = ("rho_bb", "rho_uu", "rho_bu")

    head = (f"{'condition':<{width}}" + _titles(overall)
            + " |" + _titles(controls)
            + " |" + _titles(margins)
            + " |" + _titles(opinion))
    print("\n" + head)
    print("-" * len(head))
    for i, name in enumerate(names):
        print(f"{name:<{width}}"
              + _fmt([pooled[k][i] for k in overall])
              + " |" + _fmt([pooled[k][i] for k in controls])
              + " |" + _fmt([pooled[k][i] for k in margins])
              + " |" + _fmt([pooled[k][i] for k in opinion]))

    print("\nT_mu^{b->} is the mean trust a biased agent extends, T_mu^{->b} the mean")
    print("trust it receives.  The first is the field; the second is not.")

    leak = -1.0 / batch.N
    print(f"\nThe four class parameters are controls: the dynamics never reads the "
          f"class\nlabel.  Three of them sit at zero.  R_muc does not, and the "
          f"amount is known:\nwith the diagonal excluded the uniform channel "
          f"leaks into the matching one at\n-1/(N-1) = {leak * batch.N / (batch.N - 1):+.4f}.")
    print(f"    {'condition':<{width}}{'R_muc':>9}{'predicted':>11}")
    for i, name in enumerate(names):
        pred = -pooled["T_mu"][i] / (batch.N - 1)
        print(f"    {name:<{width}}{pooled['R_muc'][i]:>9.3f}{pred:>11.3f}")

    print("\nMean trust by block, [receiver, emitter], biased first, "
          "+/- the standard\nerror over realizations:")
    blocks_flat = bias_block_trust(batch).reshape(len(names) * reps, 4)
    block_se = np.stack([_pool_se(blocks_flat[:, j], reps) for j in range(4)],
                        axis=1)
    print(f"    {'condition':<{width}}" + "".join(
        f"{t:>16}" for t in ("b<-b", "b<-u", "u<-b", "u<-u")))
    for i, name in enumerate(names):
        cells = "".join(
            f"{'nan':>16}" if not np.isfinite(block_m[i].ravel()[j])
            else f"{block_m[i].ravel()[j]:>10.3f} +/-{block_se[i, j]:>5.3f}"
            for j in range(4))
        print(f"    {name:<{width}}" + cells)

    # The emergent margin is the difference of the last two columns, and it is
    # the one number here that the field does not put in by hand, so it gets its
    # own line with the error of the difference rather than being left for the
    # reader to subtract two columns and guess at.
    print("\nThe emergent margin, u<-b minus u<-u: the same unbiased listeners "
          "comparing\ntwo speakers who differ only in whether *they* carry the "
          "field.")
    for i, name in enumerate(names):
        m = block_m[i, 1, 0] - block_m[i, 1, 1]
        se = np.hypot(block_se[i, 2], block_se[i, 3])
        if np.isfinite(m):
            sig = "" if not np.isfinite(se) or se == 0 else f"   ({m / se:>5.1f} se)"
            print(f"    {name:<{width}}{m:>+8.3f} +/-{se:>6.3f}{sig}")

    return pooled, block_m


def _titles(keys):
    return "".join(f"{k:>9}" for k in keys)


def figure(batch, reps, style, conditions=CONDITIONS, name="uniform_split"):
    """Per-agent trust extended and received, split by group.

    Two rows, one per sign of the field, and two columns: what each agent gives
    and what each agent gets.  The left column is the field, and it separates by
    construction.  Whether the right column separates is the question.
    """
    eta = trust(batch)
    given = trust_given_per_agent(batch, eta=eta)      # (R, N)
    received = trust_received_per_agent(batch, eta=eta)
    labels = [c[0] for c in conditions]

    rows = [labels.index(k) for k in HIST_CONDITIONS]
    bins = np.linspace(-1.0, 1.0, 41)

    fig, axes = plt.subplots(len(rows), 2, figsize=panel(1.0, 0.42), squeeze=False)
    for i, cond in enumerate(rows):
        lo, hi = cond * reps, (cond + 1) * reps
        biased = batch.biased[:, lo:hi].T          # (reps, N)
        for j, (values, what) in enumerate(((given, "extends"),
                                            (received, "receives"))):
            ax = axes[i][j]
            v = values[lo:hi]
            for member, (fill, edge), tag in (
                (biased, HIST_RED, "biased"),
                (~biased, HIST_BLUE, "unbiased"),
            ):
                ax.hist(v[member], bins=bins, color=fill, edgecolor=edge,
                        linewidth=0.5, alpha=0.72, label=tag)
            framed_axes(ax, minor=False)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel(r"mean $\eta$ an agent " + what, labelpad=1)
            if j == 0:
                ax.set_ylabel("agents", labelpad=1)
            title = labels[cond].replace("\n", ", ")
            ax.set_title(f"{title} -- {what}", pad=3)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left", frameon=False,
                          handlelength=1.0, labelspacing=0.22, borderpad=0.2)
    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    batch, reps = run(preset)
    table(batch, reps)
    figure(batch, reps, args.style)


if __name__ == "__main__":
    main()
