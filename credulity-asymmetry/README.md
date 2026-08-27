# The credulity asymmetry

A class-dependent shift of an agent's opinion field has four independent
components. The main line of work in
[`../nn-based-simulation/`](../nn-based-simulation/) studies one of them, `p`.
The sibling directory [`../directional-prejudice/`](../directional-prejudice/)
studies `c`, in which one class is *believed* more by everyone. This directory
studies `b`, in which one class *believes* everyone.

---

## Where `b` sits

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
| `c` | who is **speaking** | one class is believed more by everyone, its own members included |
| `p` | whether the two **match** | in-group favouritism and out-group hostility at once |

`b` is the component that makes credulity itself a class attribute. A prejudiced
agent of class A reads every message as more agreeable than it is; a prejudiced
agent of class B reads every message as less so; and neither cares who is
speaking. Since perceived agreement is what builds trust, class A comes to trust
everyone and class B to trust nobody — **itself included**, which is what makes
this a split in credulity rather than a preference between groups.

Of the six cases the companion manuscript tabulates
(`credfield.fields.TABLE_I`), `b` appears in exactly two, and in both of them all
four components are present at the same magnitude:

| case | `a` | `b` | `c` | `p` | |
|---|---|---|---|---|---|
| 1 | 1/4 | **1/4** | 1/4 | 1/4 | A favours its own; B indifferent |
| 2 | -1/4 | **-1/4** | 1/4 | 1/4 | A hostile to B; B indifferent |
| 3 | 0 | 0 | 1/2 | 1/2 | both of the above |
| 4 | 1/2 | 0 | 0 | 1/2 | both classes favour their own |
| 5 | -1/2 | 0 | 0 | 1/2 | both hostile to the other |
| 6 | 0 | 0 | 0 | 1 | the symmetric case, pure `p` |

So `b` is the *least* isolated of the four in that table: `c` at least appears in
three cases, and in case 3 it appears with only `p` for company. `b` appears
twice, never with fewer than three companions, and never alone. That is a fair
explanation of why a credulity asymmetry has not been looked at on its own, and
no reason it is not there.

## The exact statement of the invisibility

`eta[r, e]` is a *directed* matrix: how far `r` trusts `e` need not equal how far
`e` trusts `r`. All three of the paper's correlations use it only through the
symmetric combination `eta_{I|J} + eta_{J|I}`, which discards the antisymmetric
part, and `b` writes the class label into precisely that part.

Under a pure `b` at strength `s` the trust matrix is `eta[r, e] = s(kappa_r)`.
Three consequences, all exact for equal class sizes rather than approximate:

**`R_muc = 0`.** The symmetrized trust of a pair is `(s_I + s_J)/2`: `+s` on AA
pairs, `-s` on BB pairs, `0` on AB pairs, against a class indicator of
`+1, +1, -1`. The AA and BB terms cancel term by term whenever the two classes
are the same size, however strong the split is.

**`B_eta = 0`.** In the cycle `eta_IJ eta_JK eta_KI` each of the three agents
appears exactly once as the **receiver** — the first index — so the sign of the
product is `(-1)^(number of suspicious agents in the triple)`. Balance is decided
by the *parity* of how many suspicious members a triple has, and with equal class
sizes the two parities are equinumerous.

**And the paper cannot even tell `b` from `c`.** A pure `b` and a pure `c` of the
same strength give trust matrices that are exact **transposes** of each other —
`s kappa_r` against `s kappa_e`. A transpose has the same symmetric part and the
negated antisymmetric part, so every parameter of the main line of work takes
*identical* values on the two: not merely both zero, but the same number wherever
they are non-zero. A credulity split and a status hierarchy are, to the published
five, one and the same measurement — and it reads zero on both.

### How far the transpose result reaches

It is a statement about a field every agent carries, and it stops being true below
that. The prejudice mask sits on the **receiver** under both fields:
`D[r,e] = b kappa_r 1[r prejudiced]` against `D[r,e] = c kappa_e 1[r prejudiced]`.
Under `b` the field and the mask are both functions of `r` and align; under `c`
the field is indexed by the emitter while the mask is still on the receiver, so
transposing `eta` does not transpose the mask. Measured on one seed at `N = 16`:

| `f` | mean \|`eta_b - eta_c^T`\| | correlation |
|---|---|---|
| 1.00 | 0.012 | 0.9999 |
| 0.75 | 0.333 | 0.732 |
| 0.50 | 0.712 | 0.354 |
| 0.25 | 0.021 | 0.9995 |

So the two fields are the same experiment read two ways at full prevalence, and
genuinely different ensembles in between — worst around `f = 0.5`, and agreeing
again near `f = 0` only because almost nobody carries a field there. Everything
this section claims is at `f = 1`, which is where the table above is measured and
where the algebra is exact.

The practical consequence, for anyone comparing this plane against the status
plane: quantities that are transpose-invariant are entitled to *differ* away from
full prevalence, and do. The saturated corner and the thresholds agree because
that is where both planes are near `f = 1`; the `R_muc` dispersion over the rest
of the plane does not, and should not be read as a discrepancy between two
measurements of one thing. Checked in `tests/test_society.py` rather than assumed.

That last point is what makes the two directions worth running separately. If the
published parameters could distinguish them, one sweep would stand for both. They
cannot, so the distinction has to be made by the channel that does see it, and
the two planes are two experiments rather than one.

### What replaces them

The class structure of a directed matrix has four channels, orthogonal for the
same reason and in the same way as the four components of the field, so the
completed set is a rotation rather than an addition:

| field component | trust channel | in the paper |
|---|---|---|
| `a` | `T_mu = <eta>` | no |
| `b` | **`R_cred = <kappa_r eta[r,e]>`** | no |
| `c` | `R_stat = <kappa_e eta[r,e]>` | no |
| `p` | `R_muc = <kappa_r kappa_e eta[r,e]>` | yes |

`R_cred` is the channel this directory sweeps for. It costs nothing extra to
measure — the same elicited trust matrix, just not symmetrized — which is the
point for the audit argument the paper makes in its discussion: two elicited
matrices do suffice, **but they must not be symmetrized.**

Two further quantities earn their place. `B_eta` computed inside each class
separately separates a bloc from a dust: under a saturated `b` the credulous
class is internally balanced (`+1`) while every member of the suspicious class
distrusts every other member, so every triple inside it is frustrated (`-1`). And
the trust each agent *gives* (`credulity_per_agent`) is the per-agent picture,
bimodal by class, that `R_cred` is the first moment of. Note that this is the
readout that has to change with the component: under `b` it is the trust *given*
that is bimodal, while the trust *received* is one unimodal blob, since every
agent is trusted by the credulous half and distrusted by the suspicious half
whatever its own class.

### One caveat, exactly quantified

The four weights are orthogonal over all `N^2` pairs but not over the `N(N-1)`
that exclude the diagonal, and excluding it is right because an agent's trust in
itself is a convention rather than a measurement. The only non-zero off-diagonal
Gram entries are `<1, kappa_r kappa_e> = <kappa_r, kappa_e> = -N`, so the uniform
channel leaks into the matching one and **credulity leaks into status**, each at
`-1/(N-1)`: about `-2.6%` at `N = 40`.

This matters more here than it does for the published parameter. A pure credulity
field at full strength puts a spurious `R_stat = -R_cred/(N-1)` into the status
channel — about `-0.025`, which is small but is *not* noise and does not shrink
with more realizations. Read carelessly it says a credulity split comes with a
faint status hierarchy in the opposite direction. It does not; it is the excluded
diagonal. `trust_channels(..., orthogonalize=True)` removes the leakage by
inverting the Gram matrix, which is a pair of 2x2 systems in closed form. The
default is off, so that `R_muc` stays the paper's number.

## What a credulity field does

Run a population under each pure component at full strength with every agent
prejudiced (`python scripts/invisibility.py`), and measure it with the paper's
order parameters and with the four channels:

| condition | `R_wmu` | `R_muc` | `R_cw` | `B_rho` | `B_eta` | `T_mu` | **`R_cred`** | `R_stat` | `B_eta^AA` | `B_eta^BB` | A←A | A←B | B←A | B←B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.548 | 0.000 | 0.000 | 0.177 | 0.949 | -0.009 | -0.006 | -0.006 | 0.949 | 0.949 | -0.02 | -0.01 | -0.01 | 0.00 |
| `a=1` uniform | 0.391 | -0.025 | -0.009 | 0.064 | 0.974 | 0.991 | 0.000 | -0.000 | 0.975 | 0.974 | 0.99 | 0.99 | 0.99 | 0.99 |
| `b=1` credulity | 0.002 | 0.000 | -0.005 | 0.002 | -0.000 | 0.000 | **0.991** | -0.025 | 0.972 | -0.972 | 0.99 | 0.99 | -0.99 | -0.99 |
| `c=1` status | 0.000 | -0.000 | -0.003 | 0.002 | -0.000 | 0.000 | -0.025 | **0.991** | 0.972 | -0.972 | 0.99 | -0.99 | 0.99 | -0.99 |
| `p=1` matching | 0.378 | 0.991 | 0.381 | 0.058 | 0.975 | -0.025 | -0.000 | 0.000 | 0.975 | 0.974 | 0.99 | -0.99 | -0.99 | 0.99 |

The first five columns are the paper's five, all of them — an earlier version of
this table showed four, omitting `B_rho`, while the sentence below claimed "every
order parameter the paper reports". `B_rho` is not a formality here: it reads
`0.177` with no field and `0.002` under a saturated credulity split, so it is a
real quantity being driven to zero rather than one that was always small.

The within-class balances are shown in **both sectors** for the same reason, and
they say the more interesting thing. `B_eta^AA` and `B_eta^BB` split apart under
`b` and under `c` — `+0.97` against `-0.97` — while `B_rho^AA` and `B_rho^BB` stay
equal to each other in every row of the table, and their asymmetry over the whole
plane has mean `-0.0002` and standard deviation `0.015`. **The field reorganizes
the two classes in the trust sector and leaves them ideologically alike.** The
atomization that the composite draws as its green channel is a trust-sector
asymmetry specifically, not a general splitting of the population, and the opinion
sector is what says so. Both of these quantities were swept into every cache from
the start, with labels, ranges and colour maps, and displayed nowhere — which is
why `tests/test_outputs.py` now asserts that every swept quantity reaches a
reader, and fails if one is added without a home.

That check needed fixing once itself, in a way worth recording. Its first version
searched the report for each name as a substring, and so could not detect the very
bug it was written for: `"B_rho"` is present whenever `B_rho^AA` is printed, so
dropping the aggregate column left it green. It now matches column headings as
exact tokens, and there are mutation tests that drop each aggregate and require it
to notice — because a guard against a specific past failure that cannot see that
failure is worse than no guard, being read as evidence.

`N = 40` at the calibrated `Delta t = 500`, eight realizations per row, full
strength with every agent prejudiced. `A<-B` is the mean trust a class-A receiver
places in a class-B emitter.

The credulity row is the result. The population is **maximally split** — class A
trusts everyone at 0.99 and class B distrusts everyone at -0.99, *including its
own members*, which is the part that makes it a split in credulity rather than a
preference between groups — and **every order parameter the paper reports is
zero.** `R_wmu`, `R_muc`, `R_cw`, `B_rho` and `B_eta` all read 0.00. The matching
row is the control: it is the paper's own field, and its `R_muc` and `R_cw` behave
as published.

Read the blocks of the credulity row across and the status row down. Class A
trusts both classes and class B trusts neither (`0.99, 0.99, -0.99, -0.99`);
everyone trusts A and nobody trusts B (`0.99, -0.99, 0.99, -0.99`). One is the
transpose of the other, and the two rows agree — to three decimals — on all five
published parameters and on both within-class balances, differing only by
exchanging `R_cred` with `R_stat`. That is the transpose argument above, measured:
a population split in credulity and a population sorted into a status hierarchy
are indistinguishable to the paper's five parameters, and one channel apart.

The `a` row is the calibration of the finite-size leakage. `T_mu = 0.991` with no
class structure at all predicts `R_muc = -0.991/39 = -0.025`, and `-0.025` is
measured. The same leakage is what puts `-0.025` in the credulity row's `R_stat`,
which is a leak and not a hierarchy.

The parity signature, at `b = 1`:

```
   k =      0       1       2       3     counts [1140, 3800, 3800, 1140]
         0.972  -0.972   0.972  -0.972
```

Textbook `(-1)^k`, and `1140 + 3800 = 3800 + 1140`, which is why the aggregate
`B_eta` is exactly zero. This is the same table the status field produces, and it
is the same table for a different reason: there each agent appears once in the
cycle as the emitter, here once as the receiver.

### The residual, where the classes are unevenly prejudiced

At `f = 0` and `f = 1` the prejudiced agents are split evenly between the classes
by construction and the cancellation is exact. In between, who is prejudiced is
drawn independently of class, so one class can hold more of them than the other.
Sixteen populations at each of three prevalences, `N = 40`:

| `f` | `R_muc` | sd | \|max\| | `R_cred` | sd | corr with imbalance |
|---|---|---|---|---|---|---|
| 0.25 | 0.005 | 0.037 | 0.119 | 0.266 | 0.077 | -0.20 |
| 0.50 | 0.007 | 0.032 | 0.093 | 0.534 | 0.080 | -0.14 |
| 0.75 | -0.010 | 0.063 | 0.141 | 0.753 | 0.049 | -0.48 |

`R_muc` has mean zero to within its standard error at each of the three
prevalences while `R_cred` climbs from 0.27 to 0.75, and the largest
single-population `|R_muc|` among these 48 populations is 0.14 — the same order as
the realization-to-realization spread the paper already quotes for its own neutral
region.

That "among these 48" is doing real work and is not a hedge. Swept over the whole
plane rather than three prevalences at full strength, `|R_muc|` reaches `0.582`,
and where it does is the subject of a later section. Three prevalences at one
strength are not a bound on the plane, and this table should not be read as one.

One difference from the status field is worth recording rather than smoothing
over. In the `c` plane the residual is *explained* by the class imbalance among
the prejudiced agents, correlating with it at 0.93 to 0.99. Here it does not:
the correlations are `-0.20, -0.14, -0.48`, and at sixteen realizations
(standard error about 0.22 on a correlation) none of them is clearly non-zero.
So the `b` residual is small for the same reason but not by the same mechanism,
and this directory does not claim the imbalance accounts for it.

## The `(b, f_b)` plane

`python scripts/cred_asymmetry.py` sweeps strength against prevalence. `N = 40`
at the calibrated `Delta t = 500`, `200x200`, one population per grid point,
40 000 populations, six hours forty-four minutes on ten cores in five strips of
`92, 92, 90, 85, 46` minutes. Treat those as wall clock rather than as a
benchmark: the machine was shared for part of the run, which is most of why the
last strip took half as long as the first. Only the positive half of the strength
axis is swept, for the reason given at the bottom of this page.

Along the strength axis, pooled over the ten rows with `f_b >= 0.95`:

| `b` | 0.00 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| **`R_cred`** | 0.02 | 0.04 | 0.14 | 0.43 | 0.60 | 0.74 | 0.76 | 0.97 | 0.97 |
| atomization | 0.00 | 0.01 | 0.04 | 0.23 | 0.40 | 0.58 | 0.62 | 0.96 | 0.94 |
| `R_wmu` | 0.56 | 0.54 | 0.40 | 0.20 | 0.11 | 0.07 | 0.05 | 0.00 | 0.00 |
| `B_eta` | 0.95 | 0.95 | 0.74 | 0.33 | 0.17 | 0.10 | 0.07 | 0.00 | 0.00 |
| `R_muc` | 0.02 | 0.02 | 0.12 | 0.14 | 0.12 | 0.05 | 0.04 | 0.00 | 0.00 |

Three things happen at once, and the coincidence is the finding:

- **The transition is sharp and single.** `R_cred` and the atomization cross half
  their saturated values (`0.969` and `0.932`) at `b = 0.383` and `b = 0.428`, and
  `R_wmu` falls below `0.1` at `b = 0.408` — each to about `+-0.02`, for the reason
  given two sections down. A credulity split has a threshold in
  the strength of the individual bias, as discrimination does in the paper, and
  the split and the internal differentiation of the two classes arrive together
  rather than in stages.
- **Ordinary polarization is destroyed by it.** `R_wmu` goes from `0.56` below the
  threshold to `0.00` above. The population is strongly trust-ordered by class
  while opinion is decoupled from trust entirely.
- **`B_eta` goes to exactly zero, not to a small number.** Below the threshold it
  is `0.95`, a balanced two-camp society; above it the parity cancellation is
  exact.

Up the prevalence axis at full strength, `R_cred` rises `0.21, 0.37, 0.59, 0.78,
0.99` at `f_b = 0.2, 0.4, 0.6, 0.8, 1`, reaching half its saturated value at
`f_b = 0.48` — close to linear, so a credulity split needs a quorum but a milder
one than `p` needs at moderate strength.

### It looks very much like the status plane

The sibling's `c` plane was re-run at `200x200` while this one was running, so both
are now at the same resolution and can be compared directly. Everything below is
computed by `credfield.thresholds` on both planes under one definition and one
smoothing — `python scripts/thresholds.py --also ../directional-prejudice/data/sweep_c_*.npz`
reproduces all of it, including for the sibling's cache, since the routine infers
the responding channel from the plane rather than being told:

| | channel half-saturation | atomization half-saturation | `R_wmu` below 0.1 | saturated channel | channel up the prevalence axis |
|---|---|---|---|---|---|
| `b`, here | 0.383 | 0.428 | 0.408 | 0.969 | 0.21, 0.37, 0.59, 0.78, 0.99 |
| `c`, sibling | 0.401 | 0.427 | 0.421 | 0.968 | 0.21, 0.41, 0.60, 0.78, 0.99 |

**The precision here is about `0.02`, not `0.001`, and that matters for what can
be claimed.** Extracting a crossing from a pooled profile depends on whether the
profile is smoothed first, and dropping the smoothing moves these numbers by
`0.007` to `0.022` — as much as the two planes differ from each other. So the
honest statement is not that the planes agree to three decimals but that **they
are indistinguishable at the precision this measurement supports**, which the
script prints alongside every crossing rather than leaving to be discovered.

The row-wise threshold says the same thing with a spread attached: `0.395` for `b`
against `0.398` for `c`, with a row-to-row standard deviation of about `0.017` on
each. A difference of `0.003` against a spread of `0.017` is agreement.

There is a systematic error on top of that spread, and it is worth separating from
it because it bounds the digits rather than the comparison. The extraction is
biased **low** at finite transition width — measured on the synthetic planes, the
recovered locus is out by `0.6%` at width `0.10`, `1.3%` at `0.12`, `3.0%` at
`0.15` and `7.7%` at `0.20`, monotonically. At the width these planes actually
have, `0.12`, that is about one per cent. So `s ~ 0.4` is quotable and a third
significant figure is not; the `0.395` and `0.398` above are the routine's output,
not a claim to a part in a thousand. The comparison survives it because the bias
depends on the width and the two planes have the same width to within `0.005`, so
both are displaced alike — a difference between planes is on firmer ground here
than either absolute value.

That is still a stronger statement than the transpose identity, because it is not
forced by it: the identity is about a saturated trust matrix, whereas these are two
different dynamics. Under `c` a prejudiced receiver's shift depends on who is
speaking, so it treats its neighbours differently; under `b` it applies the same
shift to everyone and the asymmetry enters only through which agents are
prejudiced. Those two routes reach the same threshold, and nothing required them to.

### What the threshold is a threshold in

The sibling's README reports that its transition is set by the *product* of
strength and prevalence, `f c = 0.332 +- 0.039`, relative spread `0.12` against
`0.24` for `c` alone — a hyperbola rather than a vertical line. Run on its own
cached plane that reproduces exactly, and so does the opposite, and both are true
of the same data. Which holds depends entirely on how the threshold is defined:

| threshold defined as | `s_c` | rel. spread of `s_c` | `f s_c` | of `f s_c` | locus verdict |
|---|---|---|---|---|---|
| half of each row's own saturation | 0.395 | 0.044 | 0.290 | 0.205 | **vertical line**: strength, by 4.7x |
| crossing a fixed level of 0.5 | 0.476 | 0.222 | 0.331 | 0.097 | withheld — would have said the product |

The fixed-level row is the sibling's number: `f s_c = 0.331` at relative spread
`0.097` here, against its reported `0.332 +- 0.039`. And both planes behave the
same way under both definitions — under the relative definition `c` gives
`0.398` with spread `0.047` and strength winning by `4.7x`, the same as `b`.

The reason the definitions disagree says what the prevalence axis does. At low
prevalence the channel saturates at a lower value, so half of *its own* saturation
is a lower bar and is reached at much the same strength; a fixed absolute bar does
not move, so reaching it takes more strength as fewer agents carry the bias.
Neither is wrong. "Half its own saturation" asks where a given population's
transition happens, and the answer is `b ~ 0.4` almost regardless of how many
agents are biased. A fixed level asks how much bias it takes to reach a stated
degree of order, and the answer is a trade-off between strength and prevalence.

So the two descriptions are both correct, and they answer different questions.
But they are not interchangeable, and one of them cannot answer the locus question
at all. That is established on synthetic planes whose answer is known by
construction (`tests/test_thresholds.py` builds one with its transition at a fixed
strength and one at a fixed product, and requires the routine to recover the
number that was put in and to *name* the right quantity):

- **a fixed absolute level manufactures the very effect it is asked about.** On a
  plane that is vertical by construction it reports the strength threshold
  climbing as prevalence falls — not because the transition moves but because a
  fixed bar is a harder bar for a row that saturates lower — and from about
  transition width `0.06` upwards it inverts the verdict outright, with a
  prevalence span of `0.39` and a margin of `3.4x`. **Both guards pass and the
  answer is wrong**, which is asserted as a test, wrong answer and all. So
  `summarise` withholds the locus verdict from a fixed level regardless of span
  and margin, while still reporting its numbers, because they answer the
  trade-off question honestly;
- **the relative definition is much more robust, and not immune.** On the planes
  built here it gets both answers right at all nine widths tried between `0.03`
  and `0.40`, by margins of seven times and up. But a peer session's independently built plane
  inverts it near width `0.20` at a margin of `1.78`, and that inversion does not
  reproduce on this construction at any width. So the robustness is a property of
  the plane as much as of the definition. The margin gate is set at `2.0x` to
  exclude that case rather than the observed one, at the cost of declining some
  correct verdicts near that width.

Since every failure mode of both definitions is a function of how broad the
transition is, **the width belongs next to the verdict**, and is measured rather
than assumed. On these two planes the 25-75% width in strength is:

| | median | range | rows wider than 0.15 |
|---|---|---|---|
| `b`, here | 0.120 | 0.063 – 0.188 | 15% |
| `c`, sibling | 0.115 | 0.054 – 0.175 | 9% |

`0.12` is inside the regime where the relative definition is reliable on both
constructions, and it is not a large safety factor — the nearest observed
inversion of that definition is at `0.20`. So the vertical-line reading stands and
should be quoted with its width, not as a bare number.

The width is a scale, and **not** a second piece of evidence about the locus,
though it looks like one. Since a width in strength is only a meaningful single
number if the transition is located in strength, the converse is tempting: a flat
width in strength ought to argue for a strength locus. It does not. Two synthetic
planes can put the transition at exactly the same place — `f s = 0.30` — and
differ only in whether the sigmoid's width is measured in strength or in the
product, and the width comparison then gives *opposite* answers on them, both
confidently, while the threshold definition correctly says "product" for both
(`tests/test_thresholds.py`). The width diagnostic measures how the width is
parameterised, not where the transition is.

On these planes it would not have settled anything anyway. The width in strength
is flatter than `f x width` on both, which is the direction the locus verdict
points, but only by `1.09x` for `b` and `1.29x` for `c` — well inside the `2.0x`
margin this directory declines to name a verdict on. It is also not perfectly
constant: fitting `width = k/f + c0` gives `k = 0.029, c0 = 0.078` for `b`, so
there is a real but minor `1/f` component and part of the `0.063 - 0.188` range is
that trend rather than scatter. The median is a fair scale; the range should not
be read as pure noise.

The practical consequence for this repository: the hyperbola is a good description
of how much bias buys a stated degree of order, and it should not be read as
locating the transition on a hyperbola. Under the definition that can locate it,
both planes put it on a vertical line at `s ~ 0.4`, at a transition width of
`0.12`.

### Where the invisibility is exact, and where it is only small

Across the plane `R_muc` has mean `+0.009` and standard deviation `0.054`, and it
never becomes a reading of the split: the correlation it would have to show is
`R_cred`, which reaches `0.99`. But the plane is not uniform, and resolving it by
regime says more than the aggregate. Both planes, same computation:

| region | pixels | `R_muc` sd | max `\|R_muc\|` |
|---|---|---|---|
| `b`: below the transition (`R_cred <= 0.2`) | 20889 | 0.047 | 0.508 |
| `b`: transition band (`0.2 < R_cred < 0.9`) | 17105 | 0.064 | **0.582** |
| `b`: saturated split (`R_cred >= 0.9`) | 2006 | **0.023** | **0.071** |
| `c`: below the transition | 20978 | 0.047 | 0.401 |
| `c`: transition band | 17001 | 0.080 | 0.387 |
| `c`: saturated hierarchy | 2021 | 0.031 | 0.084 |

**Where the split is saturated, the invisibility is airtight.** `R_muc` has mean
`+0.0007` and never exceeds `0.071` in magnitude across all 2006 pixels — a
slightly tighter bound than the status plane's `0.084`. The published parameter
does not merely average to zero over the phase where the population is maximally
split; it stays inside `+-0.08` pixel by pixel.

**Away from it, a single population can read as discriminating.** The largest
`|R_muc|` anywhere on this plane is `0.582`, against `0.401` on the status plane,
and the two extremes sit in different places. The status plane's worst pixel is
*below* its transition, at `c = 0.116` where `R_stat = 0.099` — the realization
noise of the paper's own neutral region, where a spontaneous two-camp split
happens to align with the labels, and nothing to do with the field. This plane's
worst pixel is *inside* the transition band, at `b = 0.367`, `f_b = 0.985`, where
`R_cred = 0.355`, and `77%` of the 147 pixels above `0.3` are in that band. So
`0.58` is a population being actively reorganized by the field while the paper's
parameter reports something that is not happening to it.

Two candidate explanations are ruled out by where that pixel is. It is not the
neutral-region noise that explains the status plane's extreme, because the field
there is well past `b = 0`. And it is not the class imbalance among the prejudiced
agents, which is the residual the sibling directory identifies at intermediate
prevalence: at `f_b = 0.985` at most one agent in forty is unprejudiced, so the
imbalance is capped at one agent, and the measured correlations with imbalance are
`-0.20, -0.14, -0.48` at sixteen realizations. What is left is proximity to the
threshold — the transition band is where a finite population's fluctuations are
largest, and where a spurious correlation has the most room to appear. Consistent
with that, it is the transition band that carries the widest spread on both planes.

None of this rescues the published parameter. A false positive in a narrow band is
still not a reading of the split, and above the threshold `R_muc` is zero to two
decimals while `R_cred` sits at `0.99`. But it sharpens what the audit argument
should claim. Reading `R_muc` alone, a population in the transition band can look
discriminatory when it is not, and a saturated one looks unbiased when it is
maximally split. In the band the parameter is not merely blind but actively
misleading, and only the channel tells the two apart.

### A note on the strips

The five strips are visible in `data/` and in nothing else. Mean row-to-row change
across the four strip boundaries is within `1.5` standard deviations of the same
quantity computed over the 195 interior boundaries, for `R_cred`, `R_muc`, `R_wmu`
and `B_eta^AA` alike, so the seams are not detectable in the maps.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/make_all.py                  # 200x200 at N=40, about four hours
python scripts/make_all.py --preset quick   # a couple of minutes, coarse
pytest                                      # 125 tests, seconds
```

This directory's defaults are `--component b --preset full --style iclr`: `full`
is the 200x200 grid at `N = 40` the paper's own phase diagram uses, and producing
this plane at that resolution is what the directory is for. `--component c`
reproduces the sibling directory's status plane from this code, which is how the
transpose claim above is checked rather than asserted.

Sweeps are cached in `data/` under a hash of the configuration, so restyling a
figure never re-simulates. The cache key covers the model and the sweep but not
the directory, so two folders running the same configuration produce the same
filename in their own `data/` — which is why each of these directories keeps its
own, and why none of them writes into another's.

### Why the plane is swept in strips

`sweep` has no checkpointing: it runs every batch and writes one `.npz` at the
very end, so an interrupt at three hours fifty-five minutes costs exactly what an
interrupt at thirty seconds costs. That is not hypothetical — it cost the first
full run of this plane, thirty minutes in, with nothing recoverable.

So `--strips 5` (the default here) cuts the prevalence axis into five contiguous
bands of forty rows, sweeps each as its own cached sweep, and concatenates. The
work is identical — same 40 000 grid points, same `Delta t` — but a kill loses at
most one band, and re-running reloads the finished bands from cache and resumes.
`--strips 1` is the single-shot path.

Two details make the strips a partition rather than an approximation of one, both
checked in `tests/test_sweep.py`:

- each band's endpoints are read off the **full** axis, not computed from the
  band's own share of the unit interval. Slicing `(0, 0.2), (0.2, 0.4), ...` and
  running `linspace` inside each duplicates every boundary row and spaces rows
  differently within a band than between bands; reading the endpoints off
  `linspace(0, 1, 200)` makes the union of the bands that same axis to `1e-12`;
- each band's seed is offset by its position in the plane, `seed + row_offset *
  n_s`. `sweep` seeds a batch from its offset *within the current sweep*, which
  restarts at zero for every strip, so bands left on a common base seed would
  draw the same societies five times over and produce a plane that looks
  converged and is one band repeated. Making the seed a function of where a
  society sits also means a re-run after a kill is bit-identical rather than
  merely statistically equivalent.

The strips are a change in how the plane is computed, not in what is computed,
but they do draw different realizations than a single 40 000-society sweep would:
batch boundaries fall in different places, so a given pixel gets a different
seed. Both are valid draws of the same ensemble; the committed figures are the
striped one.

Figures are PDF only, in `figures/iclr/`, and are named `cred_asymmetry_*`
rather than `credulity_*`: `a` is *uniform* credulity and `b` is the credulity
*asymmetry*, so the bare word names neither of them unambiguously, and the
basenames of these directories have to stay distinct for the day two of them land
in `paper/figures/`. To look at one,
`pdftoppm -r 150 -png figures/iclr/cred_asymmetry_phase.pdf /tmp/out`.

## Layout

```
credfield/
  fields.py        the four-component basis and the six tabulated cases in it
  modulation.py    the evidence Z and the four modulation functions (unchanged)
  society.py       batched dynamics under a general class-dependent field
  order_params.py  the paper's five, plus the four trust channels and the
                   within-class balances
  sweep.py         (strength, fraction) sweeps with caching
  thresholds.py    where a plane's transition is, under either definition of
                   "threshold", and R_muc resolved by regime
  config.py        model parameters and resolution presets
  plotting.py      shared figure style
scripts/
  invisibility.py     what each field component does, and what is measured
  cred_asymmetry.py   the (b, f_b) plane: channels, composite, cut
  thresholds.py       every threshold number in this README, from the caches;
                      --also runs it on a sibling's plane under one definition
  make_all.py         the first two
```

## Parameters, and where they come from

`N = 40`, `K = 30`, `P = 5` (so `alpha = P/K = 1/6`, the simple agenda), and
`Delta t = 500` interactions per ordered pair. These are the main line of work's
parameters, unchanged, so that a sweep here and a sweep there differ only in the
field; their provenance is the paper's parameter appendix. `K = 30` is recovered
from the companion manuscript's agenda-complexity figure rather than chosen;
`Delta t = 500` is calibrated against its published trajectory endpoints, and is a
real parameter because the dynamics anneals rather than reaching a stationary
state.

Only the positive half of the strength axis is swept. Negating `b` is exactly the
relabelling `A <-> B`, which maps the ensemble to itself since the two classes are
the same size and nothing else about an agent refers to its class, so the negative
half is the mirror image and costs half a sweep to learn nothing. This is checked
in `tests/test_sweep.py` rather than assumed — and checked as a statement about
the *ensemble*: at a fixed seed the two signs are two different draws rather than
each other's relabelling, and a single `N = 16` society disagrees with its mirror
by up to `0.2`, so the test compares means over eight societies against their
spread.
