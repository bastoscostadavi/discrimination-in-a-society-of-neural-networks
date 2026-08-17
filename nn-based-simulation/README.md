# Societies of entropic-dynamics neural-network agents

A simulation of a society whose members learn from each other's opinions, each
carrying both an ideology (a perceptron) and an affect (a distrust it assigns to
every other agent). When a fraction of the agents let the *class label* of a
speaker shift how agreeable they find what was said, the society undergoes phase
transitions into discriminatory collective states. This package maps them.

The model as implemented is in [`../docs/model.md`](../docs/model.md). A
language-agent version of the same experiment was specified in
[`../docs/llm-study-contract.md`](../docs/llm-study-contract.md) and then shelved;
the order parameters here are substrate-independent by construction, so it remains
possible later.

## Install and run

```bash
pip install -r requirements.txt

python scripts/make_all.py --preset quick     # minutes; coarse grids
python scripts/make_all.py --preset medium    # the committed figures
python scripts/make_all.py --preset full      # publication resolution
pytest                                        # 56 tests, ~30 s
```

Add `--style iclr` to render at the paper's column width instead of the source
draft's proportions. Figures are written as PDF only, into
`figures/{paper,iclr}/`; to look at one, rasterize it on demand with
`pdftoppm -r 150 -png figures/paper/phase_diagram.pdf /tmp/out`. Sweeps are cached
in `data/` keyed by a hash of the configuration, so restyling or re-plotting never
re-simulates. Delete `data/` or pass `--no-cache` to force a fresh run.

## What each script produces

| script | figure | what it shows |
|---|---|---|
| `polarization.py` | `polarization` | direct evidence that an unbiased society splits in two: sorted overlap and trust matrices, and the overlap distribution going from unimodal to bimodal |
| `modulation_landscape.py` | `modulation_surfaces`, `modulation_contours`, `modulation_slices` | the four modulation functions; where learning happens and which sector absorbs a surprise |
| `learning_flows.py` | `learning_flows` | the flow that learning induces, and how the discrimination field bends it |
| `correlation_maps.py` | `correlation_maps` | the three pair correlations over the `(d, f_d)` plane, two agenda sizes |
| `order_parameter_maps.py` | `order_parameter_maps` | all five order parameters in one 2x5 grid; this is the version the paper uses |
| `frustration_maps.py` | `frustration_maps` | ideological and trust balance over the same plane |
| `phase_diagram.py` | `phase_diagram`, `phase_diagram_large_agenda` | the three correlations composited into one RGB map, with the four regions labelled |
| `agenda_trajectories.py` | `agenda_trajectories` | `(B_I, B_A)` trajectories across nine agenda complexities `α = P/K` |
| `sign_convention_comparison.py` | `sign_convention_comparison` | the two readings of the discrimination field, side by side |
| `calibrate.py` | — | fixes the parameters the source draft leaves unspecified |
| `draft_comparison.py` | `draft_comparison.pdf` | each figure beside the corresponding page of the source draft |

Correspondence with the figures of the source draft, for anyone reading the two
side by side: `phase_diagram` ↔ its Fig. 1, `modulation_surfaces` ↔ Fig. 2,
`modulation_contours` ↔ Fig. 3, `learning_flows` ↔ Fig. 4, `correlation_maps` ↔
Fig. 5, `frustration_maps` ↔ Fig. 6, `agenda_trajectories` ↔ Fig. 7.
`modulation_slices` and `sign_convention_comparison` have no counterpart: the
first is the figure the draft's text discusses but never includes, the second
documents a discrepancy found while building this.

## The model in one screen

Each agent `I` holds weights `w_I` with covariance `C_I`, and for every other
agent a distrust `mu` with variance `V`. At each step an issue `x̂`, an emitter
`e` and a receiver `r` are drawn; the emitter states `σ_e = sign(w_e · x̂)` and
the receiver updates all four quantities through two scaled fields,

```
γ_C = √(1 + x̂·C_r x̂)        h_w  = (w_r·x̂) σ_e / γ_C  + D[class(r), class(e)]
γ_V = √(1 + V)               h_mu = mu / γ_V
```

and the evidence `Z = Φ(h_w) + Φ(h_mu) − 2Φ(h_w)Φ(h_mu)`, whose log-derivatives
are the four modulation functions that scale the updates. `h_w` is positive when
the receiver already agrees; `h_mu` is positive when it distrusts the emitter.
Learning is concentrated where `Z` is small — agreeing with someone you distrust,
or disagreeing with someone you trust.

`D` is the discrimination field: zero for the `1 − f_d` of agents that do not
discriminate, and `±d` for the rest according to whether the emitter shares their
class. **`d > 0` means tolerance towards one's own class and intolerance towards
the other.** That sign is forced by the algorithm and is the opposite of what the
source draft's Table I states; see
[`../docs/discrimination-field-sign.md`](../docs/discrimination-field-sign.md),
which works the discrepancy through and shows both versions.

## Parameters, and where they come from

The source draft states no simulation parameters at all — no `N`, no interaction
count (its text has a literal `Δt = ????`), no agenda sizes. One number can be
recovered from it and the rest are choices, listed here so that nothing is
implicit.

| parameter | value | provenance |
|---|---|---|
| `K`, embedding dimension | 30 | **recovered.** The draft's trajectory figure uses `α = P/K ∈ {0.03, 0.17, 0.23, 0.33, 0.50, 0.67, 1.67, 3.33, 333.33}`, which is exactly `P/30` for `P ∈ {1, 5, 7, 10, 15, 20, 50, 100, 10⁴}`. Its LLM protocol's "thirty issues" agrees. |
| `N`, society size | 40 | chosen. Large enough that the order parameters are not dominated by finite-size noise, small enough that a phase diagram is affordable; total work scales as `N³`. |
| `P`, agenda size | 5 and 100 | chosen, to put `α = P/K` either side of 1, which is where the polarization order reverses. |
| `Δt`, interactions per ordered pair | 500 | **calibrated** against the draft's trajectory endpoints; see below. |
| `C₀`, `V₀` | `I`, `1` | uninformative Gaussian prior. |
| `μ₀` | `U(−1, 1)` | gives `B_I ≈ B_A ≈ 0` initially, i.e. half the triples frustrated, as the draft describes. |
| class split | `N/2` each | the draft's two classes, `A` and `B`. |
| repeats per grid point | 1 | the draft's maps are visibly single-realization. |

### How `Δt` was calibrated

The dynamics anneals — both uncertainties shrink monotonically — so a society does
not reach a stationary state; it slows down. The measurement time is therefore a
real parameter, and the draft omits it. `scripts/calibrate.py` scans `Δt` and
compares the endpoints of all nine balance trajectories against the values
digitized from the draft's figure, and separately checks four features that do not
depend on `Δt` at all: the sign flip of the trust–class correlation at `d = 0`,
the opinion–class wedge at large `d` and `f_d`, and simple agendas finishing above
the diagonal with complex ones below it.

The result (`N=40`, `K=30`, 8 repeats, `d = 0`):

| Δt (interactions/channel) | 60 | 125 | 250 | **500** | 1000 |
|---|---|---|---|---|---|
| rms distance to the draft's endpoints | 0.631 | 0.308 | 0.152 | **0.128** | 0.146 |

The minimum is flat-bottomed between 250 and 1000 and sits at **Δt = 500**, which
is the value used everywhere. Per-curve comparison at that value:

| `α = P/K` | simulated `(B_I, B_A)` | draft `(B_I, B_A)` |
|---|---|---|
| 0.03 | (0.03, 0.97) | (0.02, 0.99) |
| 0.17 | (0.21, 0.95) | (0.24, 0.98) |
| 0.23 | (0.27, 0.94) | (0.54, 0.97) |
| 0.33 | (0.38, 0.95) | (0.55, 0.94) |
| 0.50 | (0.54, 0.93) | (0.55, 0.94) |
| 0.67 | (0.70, 0.93) | (0.62, 0.92) |
| 1.67 | (0.87, 0.88) | (0.80, 0.89) |
| 3.33 | (0.91, 0.79) | (0.90, 0.65) |
| 333  | (0.97, 0.77) | (0.97, 0.68) |

Read this honestly: **no single Δt reproduces all nine endpoints.** Seven of the
nine agree to within about 0.1, and the ideological balance `B_I` matches the
three largest agendas almost exactly, but the draft's `α = 0.23` and `α = 0.33`
endpoints sit well above ours and its two largest agendas have a lower `B_A` than
we obtain at any Δt that fits the rest. Two explanations are available and we
cannot distinguish them: the draft's `N` differs from ours (its endpoints depend
on `N` through the number of channels each agent must resolve), or those middle
curves cannot be digitized reliably — three of them overlap within a few pixels in
the printed figure, so which endpoint belongs to which `α` is partly guesswork on
our side. Dropping the two suspect readings moves the selected Δt not at all
(rms 0.093 at 250, 0.076 at 500, 0.129 at 1000), which is why we treat 500 as
settled and the residual as a limit on how precisely the draft can be re-run
rather than as a defect in either.

The Δt-independent checks all pass at Δt = 500: simple agendas finish above the
diagonal and complex ones below it, with the crossover at `α ≈ 1.67` where the
draft also puts it; `R_μc` flips sign sharply at `d ≈ 0`; and the `R_cw` wedge
appears only at large `d` and large `f_d` (+0.30 in that corner against −0.003
elsewhere).

## Performance

The inner loop is memory-bandwidth bound. Three design choices matter, and the
two obvious alternatives are both slower:

- **Agent-major layout.** State is stored as `w:(N,R,K)`, `C:(N,R,K,K)`,
  `mu,V:(N,N,R)` for `R` societies simulated at once.
- **A shared interaction schedule** across the `R` societies in a batch, so
  `C[r]` and `mu[r,e]` are contiguous *views*. A run-major layout with per-society
  indices needs fancy indexing, which copies the entire covariance tensor every
  step and is ~4× slower than the scalar loop it was meant to beat.
- **Independent initial conditions per society** — weights, distrust, agenda, and
  which agents discriminate are all drawn per society, so sharing the schedule
  couples only the *order* of interactions. Neighbouring grid points act as
  common-random-number pairs, which slightly reduces the noise between them.
  `shared_schedule=False` gives fully independent schedules at ~3× the cost.

Measured: ~0.6M agent-updates per second per core in isolation, dropping to
~0.2M/s/core once ten workers are competing for memory bandwidth. At `N=40`,
`K=30`, `Δt=500`, one sweep is ~45 minutes on ten cores at the `medium`
resolution (64×64) and ~3 hours at `full` (128×128), so budget double that for
the two agenda sizes. Total work scales as `N³ · Δt · grid²`.
`ModelConfig(dtype="float32")` roughly halves it and changes no order parameter
by more than 0.1 (tested).

## Layout

```
ednna/
  modulation.py       Φ, Z, F_w, F_C, F_mu, F_V
  society.py          batched dynamics; the agent-major layout
  discrimination.py   the six field matrices and the sign convention
  order_params.py     ρ, η, the three correlations, the two balance aggregates
  config.py           model parameters and the three presets
  sweep.py            (d, f_d) grids: batching, multiprocessing, caching
  plotting.py         shared style, colour maps, the RGB composite
scripts/              one per figure, plus calibrate and draft_comparison
tests/                51 tests: identities, invariants, known limits
figures/{paper,iclr}/ committed output
data/                 sweep caches (gitignored)
```

## Conventions that differ from the source draft

Both are implemented either way, and both are documented in
[`../docs/model.md`](../docs/model.md).

1. **Sign of the discrimination field** — see above and the dedicated note.
2. **Class indicator.** The draft's Eq. 28 writes `G_IJ ∈ {0,1}`, but its
   published trust–class map spans `−1…1`, which requires the signed form
   `G_IJ = κ_I κ_J`. That is the default; `class_indicator="01"` restores the
   literal version. The draft's normalization also caps its `R_cw` at 1/2, since
   that correlation has one term per pair rather than two; we scale it to share
   the range `[−1,1]` (`literal_norm=True` restores the draft's factor).
3. **Numerical floors** on `Z`, on `V`, and a positive-definiteness clip on the
   covariance update. The draft is silent on all three; the first two are needed
   because the exact modulation functions diverge in the deep dissonance corners
   and `F_V < 0` drives `V` to zero. The clip is the only one that could alter a
   trajectory, and the runs reported here never triggered it (the counter is
   `SocietyBatch.n_psd_clips`).
