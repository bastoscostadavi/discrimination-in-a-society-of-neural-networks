# Directional prejudice fields

A class-dependent shift of an agent's opinion field has four independent
components. The main line of work in
[`../nn-based-simulation/`](../nn-based-simulation/) studies one of them. This
directory studies the two that nothing in that work can see.

---

## The four components

A prejudiced receiver adds to its opinion field an amount that may depend on its
own class and on the emitter's, `h_w -> h_w + D[class(r), class(e)]`. With
`kappa = +1` for class A and `-1` for class B, the 2x2 matrix `D` decomposes
orthogonally and uniquely:

```
D[r, e] = a  +  b kappa_r  +  c kappa_e  +  p kappa_r kappa_e
```

| | depends on | is |
|---|---|---|
| `a` | nobody | uniform credulity; refers to no label at all |
| `b` | who is **listening** | one class trusts everyone, the other trusts nobody |
| `c` | who is **speaking** | one class is believed more **by everyone, its own members included**: status, or its negative, stigma |
| `p` | whether the two **match** | in-group favouritism and out-group hostility at once |

`p` is the paper's discrimination field, and it is the natural place to start: it
is the only component that both refers to the label and survives relabelling the
two classes, so any asymmetry it produces is spontaneous rather than imposed.

Decomposing the six cases the companion manuscript tabulates
(`dirfield.fields.TABLE_I`) shows something the table itself does not:

| case | `a` | `b` | `c` | `p` | |
|---|---|---|---|---|---|
| 1 | 1/4 | 1/4 | 1/4 | 1/4 | A favours its own; B indifferent |
| 2 | -1/4 | -1/4 | 1/4 | 1/4 | A hostile to B; B indifferent |
| 3 | 0 | 0 | 1/2 | 1/2 | both of the above |
| 4 | 1/2 | 0 | 0 | 1/2 | both classes favour their own |
| 5 | -1/2 | 0 | 0 | 1/2 | both hostile to the other |
| 6 | 0 | 0 | 0 | 1 | the symmetric case, pure `p` |

Only case 6 is a pure component, and **`c` never appears alone** — it shows up in
exactly the three cases where one class discriminates and the other does not,
always in equal measure with `p`. No entry in the table isolates a status
asymmetry, which is a fair explanation of why one has not been looked at.

## What a status field does

Run a population under pure `c` at full strength with every agent prejudiced
(`python scripts/invisibility.py`), and measure it with the paper's order
parameters and with the four channels defined below:

| condition | `R_muc` | `R_cw` | `R_wmu` | `B_eta` | `T_mu` | `R_cred` | **`R_stat`** | `B_eta^AA` | `B_eta^BB` | A←A | A←B | B←A | B←B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | -0.015 | -0.008 | 0.561 | 0.956 | 0.006 | 0.007 | 0.007 | 0.956 | 0.956 | 0.00 | 0.02 | 0.02 | -0.02 |
| `a=1` uniform | -0.025 | -0.013 | 0.437 | 0.975 | 0.991 | -0.000 | 0.000 | 0.974 | 0.975 | 0.99 | 0.99 | 0.99 | 0.99 |
| `b=1` credulity | 0.000 | -0.001 | 0.001 | -0.000 | 0.000 | **0.991** | -0.025 | 0.972 | -0.972 | 0.99 | 0.99 | -0.99 | -0.99 |
| `c=1` status | -0.000 | 0.005 | 0.005 | -0.000 | 0.000 | -0.025 | **0.991** | 0.972 | -0.971 | 0.99 | -0.99 | 0.99 | -0.99 |
| `p=1` matching | **0.992** | 0.428 | 0.425 | 0.975 | -0.025 | -0.000 | -0.000 | 0.975 | 0.975 | 0.99 | -0.99 | -0.99 | 0.99 |

`N = 40` at the calibrated `Delta t = 500`, four realizations per row, full
strength with every agent prejudiced. `A<-B` is the mean trust a class-A receiver
places in a class-B emitter.

The status row is the result. The population is **maximally ordered** — everyone
trusts A at 0.99 and distrusts B at -0.99, *including B's own members*, which is
internalized stigma — and **every order parameter the paper reports is zero.**
The credulity row is the same story with the rows and columns exchanged. The
matching row is the control: it is the paper's own field, and its `R_muc` and
`R_cw` behave as published.

## Why the paper's parameters cannot see it, exactly

`eta[r, e]` is a *directed* matrix: how far `r` trusts `e` need not equal how far
`e` trusts `r`. All three of the paper's correlations use it only through the
symmetric combination `eta_{I|J} + eta_{J|I}`, which discards the antisymmetric
part — and `b` and `c` write the label into precisely that part. The two
cancellations are exact, not approximate, and both are two lines:

**`R_muc`.** Under pure `c` the trust matrix is `eta[r, e] = s(kappa_e)`, so the
symmetrized trust of a pair is `(s_I + s_J)/2`: `+s` on AA pairs, `-s` on BB
pairs, `0` on AB pairs, against a class indicator of `+1, +1, -1`. The AA and BB
terms cancel term by term whenever the two classes are the same size.

**`B_eta`.** In the cycle `eta_IJ eta_JK eta_KI` each of the three agents appears
exactly once as the *emitter* — the second index — so the sign of the product is
`(-1)^(number of class-B agents in the triple)`. Balance is decided by the
**parity** of how many stigmatized members a triple has, and with equal class
sizes the two parities are equinumerous. Measured, at `c=1`:

```
   k =      0       1       2       3     counts [1140, 3800, 3800, 1140]
         0.972  -0.972   0.972  -0.971
```

Textbook `(-1)^k`, and `1140 + 3800 = 3800 + 1140`, which is why the aggregate is
exactly zero.

The cancellation is exact per population wherever the prejudiced agents are
split evenly between the classes, which at `f = 0` and `f = 1` they are by
construction: measured `R_muc = 0.000` with a standard deviation of `0.000` over
sixteen realizations. In between, who is prejudiced is drawn independently of
class, so one class can hold more of them than the other. That, and nothing else,
is the residual: `R_muc` correlates with the class imbalance among the prejudiced
agents at **0.93 to 0.99**, with mean zero and a spread that shrinks with `N`
(sd `0.11` at `N=24` against `0.05` to `0.09` at `N=40`, the same order as the
realization-to-realization spread the paper already quotes for its own neutral
region). It never reads as signal.

## What replaces them

The class structure of a directed matrix has four channels, orthogonal for the
same reason and in the same way as the four components of the field. So the
completed set is a rotation, not an addition:

| field component | trust channel | in the paper |
|---|---|---|
| `a` | `T_mu = <eta>` | no |
| `b` | `R_cred = <kappa_r eta[r,e]>` | no |
| `c` | `R_stat = <kappa_e eta[r,e]>` | no |
| `p` | `R_muc = <kappa_r kappa_e eta[r,e]>` | **yes** |

The fourth is the paper's `R_muc` *exactly*: `kappa_r kappa_e` is symmetric under
swapping the pair, so averaging the directed product over ordered pairs is the
same as averaging the symmetrized trust over unordered ones. The other three cost
nothing extra to measure — same elicited trust matrix, just not symmetrized —
which matters for the audit argument the paper makes in its discussion: two
elicited matrices do suffice, **but they must not be symmetrized.**

Two further quantities earn their place. `B_eta` computed inside each class
separately separates a bloc from a dust: under a directional field the credited
class is internally balanced (`+1`) while every member of the other distrusts
every other member, so every triple inside it is frustrated (`-1`). And the trust
each agent *receives* (`status_per_agent`) is the per-agent picture, bimodal by
class, that the aggregate is the first moment of.

### One caveat, exactly quantified

The four weights are orthogonal over all `N^2` pairs but not over the `N(N-1)`
that exclude the diagonal, and excluding it is right because an agent's trust in
itself is a convention rather than a measurement. The only non-zero off-diagonal
Gram entries are `<1, kappa_r kappa_e> = <kappa_r, kappa_e> = -N`, so the uniform
channel leaks into the matching one and credulity into status, each at
`-1/(N-1)`: about `-2.6%` at `N=40`.

This has a consequence for the published parameter, not just for ours. **A
population with uniformly high trust and no class structure at all has
`R_muc = -T_mu/(N-1)`, not zero.** The `a=1` row above reads `-0.025` against
`-0.991/39 = -0.025` predicted — the agreement is exact, and it is worth knowing
before reading a weak negative `R_muc` as reverse discrimination.
`trust_channels(..., orthogonalize=True)` removes the leakage by inverting the
Gram matrix, which is a pair of 2x2 systems in closed form. The default is off,
so that `R_muc` stays the paper's number.

## The `(c, f_c)` plane

`python scripts/directional_phase.py` sweeps strength against prevalence. Only the
positive half is swept: negating `c` is exactly the relabelling `A <-> B`, which
maps the ensemble to itself, so the other half is the mirror image and costs half
a sweep to learn nothing (checked in `tests/test_sweep.py` rather than assumed).

`N = 40` at the calibrated `Delta t = 500`, **200x200** -- the grid the paper's own
phase diagram uses (6.1 h on ten cores, sharing the machine with another sweep;
`PSD clips: 0`). A 64x64 run agrees with everything below and is kept as a
cross-check.

**One sharp transition in strength, at `c ~ 0.4`.** Half-saturation crossings,
with the 64x64 values beside them:

| | 200x200 | 64x64 |
|---|---|---|
| `R_stat` | `c = 0.399` | `0.374` |
| atomization | `c = 0.433` | `0.411` |
| `R_wmu` falls below 0.1 | `c = 0.412` | `0.413` |

The hierarchy leads the atomization by about `0.03` in `c`, consistently on both
grids. Each pixel is one realization, so we would not defend the gap as a
two-stage transition on this evidence; what the two grids do establish is that the
transition is at `c ~ 0.4` and that the collapse of ordinary polarization
coincides with it.

**The boundary is a curve, not a vertical line.** Fitting where each prevalence
row crosses `R_stat = 0.5` over the 112 rows that cross it at all, `f c` is
constant to a relative spread of `0.12` (`f c = 0.332 +- 0.039`) against `0.24`
for `c` alone -- so the threshold is set by the product of strength and
prevalence, roughly, rather than by strength alone. That is the form the toy
Landau treatment in [`../landau-small-cv-phase/`](../landau-small-cv-phase/)
derives for the `p` field, where the boundaries go as `f_d d = const`; finding it
again here, for a field that writes into a different sector, is a point in its
favour. Note the consequence for the swept box: with `c <= 1` the curve exits
through the top, so no population with fewer than about 40% prejudiced agents
reaches half-saturation *within the range swept* (the lowest crossing row is
`f = 0.432`). That is the hyperbola leaving the box, not a hard quorum.

**The invisibility, stated over the plane.** Across all 2021 pixels where the
hierarchy is saturated (`R_stat >= 0.9`), `R_muc` has mean `-0.0008` and standard
deviation `0.031`, and **never exceeds `0.084` in magnitude anywhere in that
region** -- the same bound, to three decimals, as on the 64x64 grid. The
published parameter does not merely average to zero over the hierarchical phase;
it stays inside `+-0.09` pixel by pixel.

The largest `|R_muc|` anywhere on the plane is `0.401`, and it is worth saying
where it is not: at `c = 0.116`, `f = 0.417`, where `R_stat = 0.099` -- that is
*below* the transition, in the ordinary polarized state, where a spontaneous
two-camp split happens to align with the class labels by chance. It is the
realization noise of the paper's own neutral region and has nothing to do with the
status field. Resolved by regime:

| region | pixels | `R_muc` sd | max `\|R_muc\|` |
|---|---|---|---|
| below the transition (`R_stat <= 0.2`) | 20978 | 0.047 | 0.401 |
| transition band (`0.2 < R_stat < 0.9`) | 17001 | 0.080 | 0.387 |
| saturated hierarchy (`R_stat >= 0.9`) | 2021 | **0.031** | **0.084** |

The transition band carries the widest spread, which is the class-imbalance
residual biting hardest where the population is only partly organized; the
saturated phase is the quietest place on the plane.

No regions are named on the composite. The four states of the `p` plane were
identified from a sweep before they were labelled, and the same is owed to this
one.

## Install and run

```bash
pip install -r requirements.txt

python scripts/make_all.py --preset quick     # minutes; coarse grids
python scripts/make_all.py --preset medium    # N=40 at the calibrated Delta t
python scripts/make_all.py --preset full --batch-size 512   # the 200x200 grid, hours
python scripts/invisibility.py --preset quick # the table and its figure alone
python scripts/directional_phase.py --component b  # the credulity field instead
pytest                                        # 68 tests, ~4 s
```

`--batch-size` and `--workers` override the preset when the machine has less
headroom than it assumes: memory scales as `batch * N * K^2`, and `full`'s default
of 1024 wants ~3.3 GB across ten workers. Note that the batch size is part of the
cache key, because the key hashes the whole sweep configuration; and since each
batch is seeded from its offset, two batch sizes genuinely draw different
realizations of the same grid rather than sharing a cache.

Add `--style iclr` to render at the paper's column width. Figures are PDF only,
into `figures/{paper,iclr}/`; to look at one, rasterize it on demand with
`pdftoppm -r 150 -png figures/iclr/directional_phase.pdf /tmp/out`. Sweeps are
cached in `data/` keyed by a hash of the configuration, so restyling or
re-plotting never re-simulates.

## Layout

```
dirfield/
  fields.py         the four-component basis, and Table I decomposed in it
  modulation.py     Phi, Z, F_w, F_C, F_mu, F_V -- copied unchanged
  society.py        batched dynamics under a general class-dependent field
  order_params.py   the paper's five, the four trust channels, the within-class
                    balances, and the parity breakdown
  sweep.py          (strength, fraction) grids: batching, workers, caching
  config.py         model parameters, presets, and which signs are worth sweeping
  plotting.py       shared style, component-aware axes, the composite
scripts/
  invisibility.py   what each component does and which parameters see it
  directional_phase.py  the (strength, prevalence) plane: channels, composite, cut
  make_all.py       both of the above
tests/              68 tests: the basis, the exact cancellations, the dynamics,
                    and every component through the plotting path
```

### Relationship to `nn-based-simulation`

`modulation.py` is copied verbatim — the modulation functions belong to the
inference problem and know nothing about classes, so a general field cannot touch
them. `society.py`, `sweep.py`, `config.py` and `plotting.py` are adapted copies:
the dynamics is identical and the parameters are the same, so a sweep here and a
sweep there differ only in the field. `order_params.py` shares the closed-form
triple sums and adds everything else. A copy rather than an import, because the
field construction, the order parameters and the figures all diverge, and
`tests/test_fields.py` holds the two in agreement where they overlap (pure `p`
must reproduce case 6 exactly).

## What this does not do

- **`a` and `b` are implemented but not swept here.** `--component a` and
  `--component b` work and the plumbing is tested; only `c` has been run over a
  plane in this directory. `b` is the transpose of `c` and its aggregate
  signatures are identical (`B_eta^AA = +0.972`, `B_eta^BB = -0.972` in both rows
  above), so only `R_cred` against `R_stat` distinguishes them — which is the
  argument for measuring all four rather than one. A sibling exploration has since
  mapped `(b, f_b)` at the same resolution and agrees where it should: saturated
  channel `0.969` against `0.968`, threshold `0.395` against `0.398`, atomization
  crossing `0.428` against `0.427`.

  Two claims about `b` and `c` need keeping apart, because conflating them invites
  a hunt for a bug that is not there. The identity `eta_b = eta_c^T` is exact
  **per realization at one seed** — a statement about a single trust matrix and its
  transpose, measured at mean elementwise `0.0047` with correlation `1.0000`, and
  that is the proof of it. Two *planes* of `b` and `c` are a different matter:
  their pixels are independent draws, so they agree in distribution and not pixel
  for pixel. Ensemble statistics match; single-pixel extremes like `max |R_muc|`
  neither match nor should, being extremes of a heavy-tailed noise distribution
  over 40 000 draws.
- **No mixed fields.** Every run here sets one component and zeroes the rest.
  `ModelConfig.background` holds the others fixed if wanted, but the interesting
  question — what `c` and `p` together do, since Table I never separates them —
  is not addressed.
- **Nothing here is contagious**, and this one we went looking for. Prejudiced
  agents impose the hierarchy on their own rows and unbiased agents in the same
  population do not adopt it at all. At `N=40`, `Delta t=500`, `c=1`, prejudiced
  receivers give `R_stat = 0.991` and unbiased ones `0.003` at `f=0.3` and
  `0.005` at `f=0.6` — zero to three decimals, not merely small. The mechanism is
  visible in the opinion overlaps, which stay at `rho = 0.007` within class A,
  `-0.007` between the classes and `-0.008` within class B: the stigmatized class
  is atomized rather than pushed into a coherent bloc, so its opinions never
  drift off consensus and there is nothing for an unbiased agent to genuinely
  disagree with. We expected the stigma to become self-fulfilling and it does
  not. Under `p` the question does not even arise, since no class is wrong for
  everybody.
- **The classes are exactly the same size.** Both exact cancellations need it.
  Unequal classes leave a residue of the same order as the imbalance, which
  `tests/test_order_params.py` checks but nothing here explores.
