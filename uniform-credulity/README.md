# Uniform credulity

The component of the prejudice field that names nobody.

A class-dependent shift of an agent's opinion field has four independent
components. Writing `kappa = +1` for class A and `-1` for class B,

```
D[r, e] = a  +  b kappa_r  +  c kappa_e  +  p kappa_r kappa_e
```

`p` is the paper's discrimination field and lives in
[`../nn-based-simulation/`](../nn-based-simulation/). `c` (status) and `b`
(credulity asymmetry) live in
[`../directional-prejudice/`](../directional-prejudice/). This directory is the
fourth: `a`, the same shift whoever is speaking and whoever is listening.

It is the odd one out of the four, and that is the reason to run it rather than
the reason not to.

---

## What a label-free field does

`a` does not correlate anything with anything, because it refers to no label. It
moves the **trust separatrix**. A receiver decides whether a message agrees with
it by the sign of the opinion field `h_w`; adding `a` biases that decision
uniformly. `a > 0` is a population disposed to read whatever it hears as
agreement, `a < 0` one disposed to read it as disagreement.

That is not a small perturbation of the ordinary dynamics, because of the two
properties the modulation functions have:

- **agreement builds trust**, so a shift in perceived agreement is a shift in
  how readily trust forms at all;
- **a distrusted source is anti-learned from** rather than ignored, so pushing
  a population into distrust does not slow its learning down, it reverses it.

So the two halves of the plane are two different states rather than two
strengths of one, and neither of them is the polarization the society reaches
with no field. The strength axis is therefore swept over both signs, which is
also why this sweep is twice the width of the `b` and `c` sweeps next door:
negating `b` or `c` is the relabelling `A <-> B` and buys nothing, while negating
`a` is a relabelling of nothing.

## The `(a, f_a)` plane

`python scripts/uniform_phase.py --preset full` sweeps strength against
prevalence at the resolution of the paper's own phase diagrams: **200x200,
`N = 40`, `K = 30`, `P = 5`, at the calibrated `Delta t = 500`**, one realization
per pixel.

The composite reads: **red** where the population has been driven into universal
distrust, **green** where into universal trust, **blue** where the ordinary
opinion-trust alignment `R_wmu` survives. Unlike the composite next door this one
does *not* take `|T_mu|`. There, `|.|` is right because the sign of the field is
a relabelling that maps the ensemble to itself; here the sign is the one
distinction the plane exists to draw, and collapsing it would be throwing the
result away.

No regions are named on it. The four states of the `p` plane were identified from
a sweep before they were labelled, and the same is owed to this one.

## What is measured, and why the usual five are not enough here

Everything the rest of the project reports is measured. Of it, exactly the
quantities that **reference the class label** are controls here, because the
label is in the measurement and nowhere in the model — `tests/test_society.py`
pins that down directly, by permuting which agents are class A and checking the
trajectory is bit-for-bit unchanged. That is four quantities, and they come from
two different sets:

| | | |
|---|---|---|
| `R_muc`, `R_cw` | two of the paper's five | control |
| `R_cred`, `R_stat` | the sibling's two extra channels | control |
| `R_wmu`, `B_rho`, `B_eta` | the paper's other three | **not** controls |

The last row matters and is easy to get wrong. Those three reference no label at
all, and on this plane they are the *result* rather than the control: `R_wmu`
runs from `0.55` with no field to `0.00` under strong suspicion, and `B_eta` from
`+0.95` to `−0.97`. Saying "four of the five are controls" would be false, not
merely unproven.

Three of the four controls do sit at zero. **`R_muc` does not, and the amount is exactly
predictable.** The four class-symmetry weights are orthogonal over all `N^2`
pairs but not over the `N(N-1)` that exclude the diagonal — and excluding it is
right, since an agent's trust in itself is a convention rather than a
measurement. With equal class sizes the only non-zero off-diagonal Gram entry is
`<1, kappa_r kappa_e> = -N`, so the uniform channel leaks into the matching one
at `-1/(N-1)`:

> **A population with uniformly high trust and no class structure whatsoever has
> `R_muc = -T_mu / (N-1)`, not zero** — about `-2.6%` of `T_mu` at `N = 40`.

That is a property of the published parameter, not of this rewriting of it, and
the uniform field is the cleanest place in the whole basis to measure it, because
here the leak is the *only* thing `R_muc` can be reading. `uniform_cut` plots
`-(N-1) R_muc` against `T_mu` so the identity is visible at full scale rather
than as a flat line at two per cent of the axis.

### The partition that does matter

Not classes: the fraction `f_a` that carries the field against the rest. That
partition is not symmetric the way a class is, because the field acts on the
**receiver**. So the directed trust matrix splits into four blocks, and two
different questions live in them:

| | |
|---|---|
| `T_mu^{b->} - T_mu^{u->}` | the trust each group **extends**. This is the field acting, and its size at a given `a` is close to a restatement of the field. |
| `T_mu^{u<-b} - T_mu^{u<-u}` | the trust a **biased and an unbiased speaker receive from the same unbiased listeners**. Nothing in `D[r, e] = a` mentions the emitter, so anything here is emergent. |

The second is measured between agents that carry no field at all, comparing
speakers who differ only in whether *they* do. That is deliberately narrower than
pooling over all receivers: a pooled margin averages the biased agents' own rows
in, and those rows are the field rather than a response to it.

## Reading a threshold off this plane

Both sibling directories quote a threshold — the field strength at which the
responding channel is half ordered — and the natural next question is whether
this plane's agrees. It is a question with more ways to get it wrong than it
looks, and the machinery here exists because of them rather than in advance
of them.

**There are two definitions and they answer different questions.** Half of each
row's own saturation asks *where a given population's transition happens*. A
fixed absolute level of the channel asks *how much bias buys a stated degree of
order*. They disagree systematically, because at low prevalence the channel
saturates lower: a bar set at half of the row's own ceiling comes down with it,
a fixed bar does not and so takes more strength as fewer agents carry the field.
One reports a threshold that barely moves with prevalence; the other reports a
strength-prevalence trade-off. Neither is wrong about its own question.

**Only the relative definition can locate a transition.** A fixed bar
manufactures the trade-off it is being asked about. On a synthetic plane whose
transition sits at a fixed strength *by construction*, a fixed bar at 0.6
reports "the product is conserved" at every transition width from 0.06 upwards,
with a healthy prevalence span and margins up to 3.9 — clearing every guard. The
margin *grows* with the transition width, so a comfortable one there is the
artifact strengthening rather than reassurance. `threshold_summary` therefore
refuses to name a locus from a fixed level at all, while still reporting its
numbers and what it *would* have said.

**The relative definition is more robust, not immune.** On a synthetic plane
whose transition sits at fixed `f|a|`, it inverts at a transition width of 0.20,
on a margin of 1.78. `MIN_RATIO = 2.0` is set to exclude exactly that, at the
cost of also declining a correct verdict at width 0.15. With both guards — a
prevalence span of at least 0.3 and a margin of at least 2.0 — no wrong verdict
survives at any width tested on either synthetic plane. Declining a correct
answer costs nothing; naming a wrong one costs a paper.

**A margin says nothing about the accuracy of the value.** The recovered
threshold drifts as the transition broadens even where the verdict holds: on the
fixed-`f|a|` plane the recovered product falls from `0.300` to `0.247` between
widths `0.03` and `0.20`. The sibling planes sit at width `~0.12`, where the
drift is a few per cent — so `s ~ 0.4` is good to two figures and a third should
not be written.

**And the transition width is not a second opinion on the locus.** It is
tempting: a width in strength is only a meaningful single number if the
transition is located in strength, so the converse feels as though it follows.
It does not. Two planes with the *same* transition, both at `f|a| = 0.30`, give
opposite answers to that comparison — 210x one way, 14x the other — depending
only on whether the sigmoid's width was written in the product or in the
strength. The width is reported as *context*, because every failure above gets
worse as it grows. It is not evidence about location, and no margin makes it so.

`tests/test_thresholds.py` holds all of this down against planes whose answers
are known in advance, including the failures, asserted deliberately.

## Status of the numbers in this file

**The tables below are from `--preset quick` (`N = 24`, `Delta t = 125`, 32x32),
which is a development setting and not comparable with the paper's.** They are
here so the file is not empty of numbers; the `full` run at 200x200 and `N = 40`
replaces them, and until it has been run nothing here should be quoted.

At full prevalence along the strength axis:

| `a` | -1.00 | -0.55 | -0.29 | -0.03 | 0.23 | 0.48 | 1.00 |
|---|---|---|---|---|---|---|---|
| `T_mu` | -0.98 | -0.94 | -0.21 | 0.02 | 0.41 | 0.98 | 0.98 |
| `rho_mean` | 0.02 | 0.01 | -0.01 | 0.02 | 0.18 | 0.48 | 0.37 |
| `R_wmu` | -0.02 | -0.01 | 0.29 | 0.51 | 0.42 | 0.47 | 0.36 |
| `B_eta` | -0.93 | -0.82 | 0.50 | 0.87 | 0.85 | 0.93 | 0.94 |

Three states, and the middle one is the society with no field:

- **`a` strongly negative.** Universal distrust, `B_eta = -0.93`: not a
  two-camp society but a frustrated one, in which almost every triple is
  unbalanced. Opinion decouples from trust entirely, `R_wmu -> 0`, and no
  consensus forms.
- **`a ~ 0`.** Ordinary polarization: `R_wmu ~ 0.5`, `B_eta ~ 0.87`, two balanced
  camps, which is the state the rest of the project studies perturbations of.
- **`a` strongly positive.** Universal trust *and* consensus — `rho_mean` goes
  from zero to positive, which the suspicious half never does. The two halves are
  not mirror images: distrust destroys the opinion sector's order, credulity
  creates a different one.

The emergent margin, over pixels where both groups are populated
(`0.25 < f_a < 0.75`):

| region | emergent | direct | `rho_bb - rho_uu` |
|---|---|---|---|
| suspicious (`a < -0.6`) | -0.06 | -0.92 | 0.00 |
| neutral (`|a| < 0.1`) | 0.02 | 0.01 | 0.01 |
| credulous (`a > 0.6`) | **0.15** | 0.80 | 0.08 |

and at the single point `a = +1`, `f_a = 0.5`, the block table is

```
                b<-b    b<-u    u<-b    u<-u
a = +1 half     0.98    0.97    0.23   -0.03
a = -1 half    -0.98   -0.98   -0.02   -0.05
```

Read the last two columns: an agent that carries **no field at all** trusts a
credulous speaker at `0.23` and a sceptical one at `-0.03`. A credulous agent
ends up believed more than its neighbours, by neighbours who were given no
reason to believe it.

The mechanism is available and testable rather than assumed: a credulous agent
learns from everyone, so it drifts towards the population's consensus, so more of
what it says is agreed with, and agreement is what builds trust. The prediction
is that the margin should track opinion alignment rather than standing alone —
and `rho_bb - rho_uu` is `0.08` in the credulous region against `0.00` in the
suspicious one, which is the right sign. **Whether this survives at `N = 40` and
`Delta t = 500` is exactly what the `full` run is for**; at `N = 24` with one
realization per pixel it is suggestive and not more.

## Install and run

```bash
pip install -r requirements.txt

python scripts/make_all.py --preset quick     # ~3 min, coarse grid, N=24
python scripts/make_all.py --preset medium    # ~25 min, 64x64 at N=40
python scripts/make_all.py --preset full      # ~4 h, 200x200 at N=40
python scripts/uniform_phase.py --preset full # the plane alone
python scripts/bias_split.py                  # the point table and histograms
pytest                                        # 138 tests, ~11 s
```

`full` is the paper's resolution and is not the default, because a stray
invocation of it costs an afternoon. Add `--style iclr` to render at the paper's
column width. Figures are PDF only, into `figures/{paper,iclr}/`; to look at one,
rasterize it on demand with
`pdftoppm -r 150 -png figures/iclr/uniform_phase.pdf /tmp/out`. Sweeps are cached
in `data/` keyed by a hash of the configuration **and of the set of quantities
measured**, so adding an order parameter invalidates the cache rather than
half-loading it, and restyling or re-plotting never re-simulates.

## Layout

```
credulity/
  modulation.py     Phi, Z, F_w, F_C, F_mu, F_V -- copied unchanged
  society.py        batched dynamics under a uniform receiver-side shift
  order_params.py   the paper's five, the class channels as controls, and the
                    trust and opinion blocks of the bias partition
  sweep.py          (a, f_a) grids: batching, workers, caching
  config.py         model parameters and resolution presets
  plotting.py       shared style, the bias-partition vocabulary, the composite
scripts/
  uniform_phase.py  the plane: channels, composite, cut
  bias_split.py     the point table and the per-agent histograms
  make_all.py       both of the above
tests/
  test_society.py       the field, the ignored class labels, the rng stream
  test_order_params.py  the blocks, the leak, which parameters are controls
  test_sweep.py         caching, the strips, both signs
  test_thresholds.py    the threshold routine against known-answer planes,
                        including the two definitions and where each fails
  test_outputs.py       that every swept quantity reaches a reader -- and
                        mutation tests that this check can actually fail
```

## Relationship to the sibling directories

`modulation.py` is copied verbatim — the modulation functions belong to the
inference problem and know nothing about labels or about who is biased, so no
field can touch them. `society.py`, `sweep.py`, `config.py` and `plotting.py` are
adapted copies: the dynamics is identical and the parameters are the same, so a
sweep here and a sweep there differ only in the field.

Two things were **removed** rather than carried over, and both are the point of
the directory. The 2x2 field machinery is gone: `D` here is one number per agent,
not a block matrix, because the uniform component does not depend on the emitter.
And the class labels are assigned but read by nothing in the dynamics, which
`tests/test_society.py` enforces by permuting them and demanding an identical
trajectory.

The random stream is drawn in the same order and the same shapes as in
`../directional-prejudice/`, so a run here and a run there at one seed bias the
same agents and can be compared pixel for pixel. That is asserted in
`tests/test_society.py` against the sequence written out by hand, rather than
against the other package, which is not importable from here.

## What this does not do

- **No mixed fields.** Every run here sets `a` and nothing else. What `a` and `p`
  together do — whether a credulous population discriminates more readily or less
  — is not addressed, and Table I of the companion manuscript never separates
  them either.
- **The classes are exactly the same size,** and the leakage identity
  `R_muc = -T_mu/(N-1)` assumes it. Unequal classes change the Gram entry, which
  `credulity/order_params.py` computes from `kappa` rather than assuming, but
  nothing here explores it.
- **One agenda size.** `P = 5`, `alpha = 1/6`, the simple agenda. The main line of
  work finds the order of polarization reverses at `alpha ~ 1.7`, and whether the
  trust separatrix cares about that has not been looked at.
- **The emergent margin has one proposed mechanism and one consistency check,**
  not a demonstration. Blocking the drift-to-consensus route directly — freezing
  the biased agents' opinions while leaving their trust updates alone — would
  settle it, and has not been run.
