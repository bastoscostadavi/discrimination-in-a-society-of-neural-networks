#!/usr/bin/env python3
"""What each field component does, and which order parameters can see it.

One society per field component, at full strength with every agent prejudiced,
measured with both sets of order parameters: the five the main line of work
reports, and the four class-symmetry channels of the directed trust matrix.

The point is one row of the table.  Under a pure credulity field the population
is maximally split -- one class trusts everyone and the other trusts nobody,
itself included -- and every published order parameter reads zero.  The
cancellation is exact for equal class sizes, so the table also prints what the
algebra says each entry should be, and the two agree to within the fluctuation of
a finite society.

The status row underneath it is the same story with the indices exchanged, and
the two rows agreeing on every published parameter is not a coincidence: a pure
``b`` and a pure ``c`` give trust matrices that are transposes of each other, and
the published parameters use only the symmetric part.

Writes ``cred_asymmetry_invisibility`` and prints the table.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from credfield.order_params import (  # noqa: E402
    CHANNEL_OF, PAPER_NAMES, balance_by_composition, class_block_trust,
    correlations, credulity_per_agent, measure, trust, trust_channels,
)
from credfield.plotting import (  # noqa: E402
    HIST_BLUE, HIST_RED, LABELS, framed_axes, panel, save,
)
from credfield.society import SocietyBatch

#: One condition per row of the table: a label, the field, and the fraction.
#: Full strength and full prevalence throughout, since the question here is what
#: each component *can* do, not how much of it a given quorum buys.
#: Two-line labels: the component on top, what it means underneath.  One line
#: does not fit -- five conditions across half a text width collide -- and
#: rotating them costs more vertical space than the second line does.
CONDITIONS = (
    ("none",              dict(),      0.0),
    ("$a=1$\nuniform",    dict(a=1.0), 1.0),
    ("$b=1$\ncredulity",  dict(b=1.0), 1.0),
    ("$c=1$\nstatus",     dict(c=1.0), 1.0),
    ("$p=1$\nmatching",   dict(p=1.0), 1.0),
)

CHANNELS = ("T_mu", "R_cred", "R_stat", "R_muc")

#: Plain-text column headings for the within-class balances.
LABELS_ASCII = {"B_eta_A": "B_eta^AA", "B_eta_B": "B_eta^BB",
                "B_rho_A": "B_rho^AA", "B_rho_B": "B_rho^BB"}


def run(preset, conditions=CONDITIONS, verbose=True):
    """One batch holding every condition, repeated ``preset.demo_runs`` times."""
    model = preset.model
    N = preset.demo_agents
    reps = preset.demo_runs
    fields = {k: [] for k in ("a", "b", "c", "p")}
    frac = []
    for _, kw, f in conditions:
        for _ in range(reps):
            for k in fields:
                fields[k].append(kw.get(k, 0.0))
            frac.append(f)

    batch = SocietyBatch(
        n_agents=N, n_dim=model.n_dim, n_issues=model.n_issues,
        f=np.asarray(frac), seed=20260821,
        dtype=model.numpy_dtype(), shared_schedule=model.shared_schedule,
        **{k: np.asarray(v) for k, v in fields.items()},
    )
    n_steps = int(round(model.interactions_per_channel * N * (N - 1)))
    if verbose:
        print(f"[invisibility] {batch.R} societies of N={N}, "
              f"{n_steps:,} interactions each")
    batch.run(n_steps)
    return batch, reps


def _pool(values, reps):
    """Mean and standard error over the repeats of each condition."""
    v = np.asarray(values, dtype=float).reshape(-1, reps)
    return v.mean(axis=1), v.std(axis=1, ddof=1) / np.sqrt(reps) if reps > 1 else \
        (v.mean(axis=1), np.zeros(v.shape[0]))


def table(batch, reps, conditions=CONDITIONS):
    """Print the table, and return it as a dict of pooled arrays."""
    eta = trust(batch)
    stats = dict(correlations(batch, eta=eta))
    stats.update(measure(batch))
    stats.update(trust_channels(batch, eta=eta))
    blocks = class_block_trust(batch, eta=eta)

    pooled = {}
    for k, v in stats.items():
        m, se = _pool(v, reps)
        pooled[k] = (m, se)
    block_m = blocks.reshape(-1, reps, 2, 2).mean(axis=1)

    names = [c[0].replace("$", "").replace("\n", " ").replace("=", " = ")
             for c in conditions]
    width = max(len(n) for n in names) + 1

    # All five the paper reports, not a chosen four.  B_rho was omitted here
    # while the prose above claimed "every order parameter the paper reports",
    # which left the headline claim displaying four fifths of its own evidence.
    paper = PAPER_NAMES
    # Every within-class balance, both sectors.  B_rho_A and B_rho_B were swept
    # into every cache, given labels, ranges and colour maps, and displayed
    # nowhere -- so the one thing they say (the classes are reorganized in the
    # trust sector and left identical in the opinion sector) went unreported.
    within = ("B_eta_A", "B_eta_B", "B_rho_A", "B_rho_B")
    head = (f"{'condition':<{width}}" + "".join(f"{k:>9}" for k in paper)
            + " |" + "".join(f"{k:>9}" for k in CHANNELS)
            + " |" + "".join(f"{LABELS_ASCII[k]:>10}" for k in within)
            + " |" + "".join(f"{t:>7}" for t in ("A<-A", "A<-B", "B<-A", "B<-B")))
    print("\n" + head)
    print("-" * len(head))
    for i, name in enumerate(names):
        row = f"{name:<{width}}"
        row += "".join(f"{pooled[k][0][i]:>9.3f}" for k in paper)
        row += " |" + "".join(f"{pooled[k][0][i]:>9.3f}" for k in CHANNELS)
        row += " |" + "".join(f"{pooled[k][0][i]:>10.3f}" for k in within)
        row += " |" + "".join(f"{block_m[i].ravel()[j]:>7.2f}" for j in range(4))
        print(row)
    print("\nA<-B = mean trust a class-A receiver places in a class-B emitter.")
    print("The two sectors' within-class balances say different things: B_eta^AA "
          "and B_eta^BB\nsplit apart under b and c, while B_rho^AA and B_rho^BB "
          "stay equal to each other\nin every row -- the field reorganizes the "
          "classes in trust and leaves them\nideologically alike.")
    print("The credulity row is the result: R_cred saturated, every paper "
          "parameter at zero.")
    print("Its blocks read across rather than down -- class A trusts both "
          "classes, class B\nneither -- which is the transpose of the status "
          "row underneath it, and the reason\nthe two rows agree on every "
          "published parameter.")

    # the finite-size leakage, which is the one non-zero the algebra predicts
    g = -1.0 / (batch.N - 1)
    i_a = names.index("a = 1 uniform")
    print(f"\nThe a row's R_muc is not noise: with the diagonal excluded the uniform "
          f"channel\nleaks into the matching one at -1/(N-1), so R_muc = "
          f"{g * pooled['T_mu'][0][i_a]:+.3f} is predicted "
          f"against {pooled['R_muc'][0][i_a]:+.3f} measured.")

    # the parity signature, for the credulity condition.  Each of the three
    # agents in the cycle eta_IJ eta_JK eta_KI appears exactly once as the
    # *receiver* -- the first index -- so under a pure b the sign of the product
    # is (-1)^(number of suspicious agents in the triple).  Under a pure c the
    # same argument runs on the emitter index, which is why the two rows of the
    # table above have the same B_eta.
    i_b = names.index("b = 1 credulity")
    lo, hi = i_b * reps, (i_b + 1) * reps
    values, counts = balance_by_composition(batch, eta=eta)
    parity = values[lo:hi].mean(axis=0)
    print("\nTrust balance by how many class-B agents a triple holds "
          "(credulity condition):")
    print("   k =      0       1       2       3     counts "
          + str([int(c) for c in counts]))  # plain ints: numpy >= 2 reprs its own
    print("      " + "".join(f"{v:>8.3f}" for v in parity)
          + "   <- (-1)^k, and the two parities are equinumerous,")
    print("                                              which is why the aggregate is 0.")
    return pooled, block_m, parity, counts


#: Prevalences for the residual check.  Away from 0 and 1 the prejudiced agents
#: are drawn unevenly between the classes, which is the one thing that can break
#: the cancellation, so this is where to look for it.
RESIDUAL_FRACTIONS = (0.25, 0.5, 0.75)


def residual(preset, fractions=RESIDUAL_FRACTIONS, reps=None, verbose=True,
             component="b"):
    """Is the cancellation exact per population, or only on average?

    At ``f = 0`` and ``f = 1`` every class holds the same number of prejudiced
    agents and the AA and BB terms cancel term by term, so ``R_muc`` is
    identically zero.  In between, which agents are prejudiced is drawn
    independently of class, so one class can hold more of them than the other and
    the cancellation is only as good as that balance.

    This runs several populations at each intermediate prevalence and reports the
    spread of ``R_muc`` together with its correlation against the class imbalance
    among the prejudiced agents, ``sum kappa_I over prejudiced I``.  A high
    correlation means the residual is that imbalance and nothing else: mean zero,
    shrinking with N, and not a reading of the hierarchy.
    """
    model = preset.model
    N = preset.demo_agents
    reps = reps or 2 * preset.demo_runs
    frac = np.repeat(np.asarray(fractions, dtype=float), reps)
    batch = SocietyBatch(
        n_agents=N, n_dim=model.n_dim, n_issues=model.n_issues,
        f=frac, seed=20260822,
        dtype=model.numpy_dtype(), shared_schedule=model.shared_schedule,
        **{component: 1.0},
    )
    n_steps = int(round(model.interactions_per_channel * N * (N - 1)))
    if verbose:
        print(f"\n[residual] {batch.R} societies of N={N} at {component}=1, "
              f"{n_steps:,} interactions each")
    batch.run(n_steps)

    eta = trust(batch)
    r_muc = correlations(batch, eta=eta)["R_muc"]
    resp_name = CHANNEL_OF[component]
    r_resp = trust_channels(batch, eta=eta)[resp_name]
    imbalance = (batch.prejudiced * batch.kappa[:, None]).sum(axis=0)

    print(f"\n{'f':>6}{'R_muc':>9}{'sd':>7}{'|max|':>7} |{resp_name:>9}{'sd':>7} |"
          f"{'imbalance':>11}{'corr':>7}")
    rows = []
    for i, f in enumerate(fractions):
        sl = slice(i * reps, (i + 1) * reps)
        rm, rs, imb = r_muc[sl], r_resp[sl], imbalance[sl]
        corr = (float(np.corrcoef(rm, imb)[0, 1])
                if np.std(imb) > 0 and np.std(rm) > 0 else float("nan"))
        print(f"{f:>6.2f}{rm.mean():>9.3f}{rm.std():>7.3f}{np.abs(rm).max():>7.3f} |"
              f"{rs.mean():>9.3f}{rs.std():>7.3f} |{imb.mean():>11.2f}{corr:>7.2f}")
        rows.append((f, rm, rs, imb, corr))
    print("The residual is the class imbalance among the prejudiced agents, and "
          f"nothing\nelse: mean zero, and it shrinks with N. {resp_name} rises "
          "with the prevalence meanwhile.")
    return rows


def figure(batch, reps, style, conditions=CONDITIONS, name="cred_asymmetry_invisibility"):
    """Two panels: what the published parameters see, and what is there.

    Left, three published order parameters per condition as grouped bars,
    alongside the credulity channel.  The three published bars are flat on the
    credulity and status columns, which is the finding.  Right, the distribution
    over agents of the trust each one *gives*, split by class, for the credulity
    condition: two modes at opposite signs, which is the split the left panel
    cannot report.
    """
    eta = trust(batch)
    stats = dict(measure(batch))
    stats.update(trust_channels(batch, eta=eta))

    keys = ("R_muc", "R_cw", "R_wmu", "R_cred")
    names = [c[0] for c in conditions]
    fig, axes = plt.subplots(1, 2, figsize=panel(1.0, 0.44),
                             gridspec_kw=dict(width_ratios=(1.55, 1.0)))

    ax = axes[0]
    n_cond = len(conditions)
    x = np.arange(n_cond)
    w = 0.19
    colors = ("#c0392b", "#27ae60", "#2c6fad", "#8e44ad")
    for j, (key, col) in enumerate(zip(keys, colors)):
        m, se = _pool(stats[key], reps)
        ax.bar(x + (j - 1.5) * w, m, w, yerr=se, color=col, label=LABELS[key],
               error_kw=dict(lw=0.6, capsize=1.4, capthick=0.6))
    ax.axhline(0.0, color="0.35", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, linespacing=1.25)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel("order parameter")
    ax.legend(ncol=4, fontsize=6, loc="upper center", handlelength=0.9,
              columnspacing=0.9, handletextpad=0.4, borderpad=0.2)
    framed_axes(ax, minor=False)
    ax.set_title("what each field does, as measured", pad=3, fontsize=8)

    ax = axes[1]
    # The per-agent quantity to show is the one the swept component writes into.
    # Under b that is the trust each agent *gives*, not the trust it receives:
    # plotting the received trust here would give one unimodal blob, since under
    # a pure credulity field every agent is trusted by the credulous half and
    # distrusted by the suspicious half regardless of its own class.
    i_b = [n for n, _, _ in conditions].index("$b=1$\ncredulity")
    recv = credulity_per_agent(batch, eta=eta)[i_b * reps:(i_b + 1) * reps]
    kap = batch.kappa
    bins = np.linspace(-1, 1, 33)
    for cls, (face, edge), lab in ((0, HIST_BLUE, "class $A$"),
                                   (1, HIST_RED, "class $B$")):
        v = recv[:, kap == (1.0 if cls == 0 else -1.0)].ravel()
        ax.hist(v, bins=bins, color=face, edgecolor=edge, lw=0.6, alpha=0.85,
                label=lab)
    ax.set_xlabel("trust given")
    ax.set_ylabel("agents")
    ax.legend(fontsize=6.5, loc="upper center", handlelength=1.0, borderpad=0.2)
    framed_axes(ax, minor=False)
    ax.set_title("the same population, per agent", pad=3, fontsize=8)

    fig.tight_layout(pad=0.4)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    batch, reps = run(preset)
    table(batch, reps)
    residual(preset)
    figure(batch, reps, args.style)


if __name__ == "__main__":
    main()
