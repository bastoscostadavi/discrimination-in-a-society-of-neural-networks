# ICLR rewrite

Rebuilding *Discrimination in a society of neural networks* around one claim:

> Class-correlated representation errors can turn locally motivated trust and
> opinion updates into group-correlated discrimination and polarization in a
> population of interacting neural agents.

Everything for the submission lives here. `code/` is the standalone, anonymous
simulation package; `paper/` is the manuscript; `figures/` is generated output.
Nothing outside this directory is modified — `nn-based-simulation/`,
`paper-iclr2027/`, `small-cv-phase-diagram/` and `controlled-small-cv-reduction/`
remain as reference.

```bash
cd code
pip install -e .
pytest                                    # 87 tests
python -c "from socsim.campaign import cost_table; print(cost_table())"
```

## What was wrong with the plan, and what changed

`ICLR_oriented_rewrite_plan.md` is the governing document, but four of its
specifications do not survive contact with the model. Each is corrected in the
code and each correction is pinned by a test.

**The discrimination field's sign is inverted in §4.1.** It writes
`D_{e|r} = −z_r d c_e c_r`, i.e. `−d` for same-class pairs, and says this matches
the symmetric case. It does not. `F_μ` carries the prefactor `(1 − 2Φ(h_w))`,
which is negative for `h_w > 0`, and `μ` is *distrust* — so raising `h_w` builds
trust, and the tolerant entry must be `+d`. Implementing §4.1 literally would
mirror the entire regime map in `d`. The correct compact form is
`D_{e|r} = +z_r d κ_r κ_e` (`discrimination.py`, `test_discrimination.py`).

**§4.3's "repaired" order parameters are a rename, not a repair.** `C_CT`, `C_CO`
and `C_TO` are numerically *identical* to the existing `R_muc`, `R_cw`, `R_wmu`
given the normalisations already in use. `test_golden.py` asserts equality to
1e-12 against the reference implementation. The paper must say "renamed", or a
reader comparing against earlier figures will conclude the numbers moved.

**§5.2's frozen-trust control is vacuous as specified.** Freezing distrust at
`μ = 0` gives `h_μ = 0`, hence `1 − 2Φ(0) = 0`, hence `F_w ≡ 0` — the opinion
sector would not learn at all. The control freezes at the random initialisation
instead, and `test_controls` pins the degeneracy so it cannot return.

**§4.6's common threshold makes the headline result an artefact.** Applying one
`τ` to two order parameters with different noise floors and dynamic ranges means
the *sign* of the polarisation-time difference partly reflects that mismatch.
First passage is measured on baseline-standardised polarisations built to be
comparable, reported as `log₁₀(t_T/t_O)`, with censoring handled explicitly.

## Two measured facts that changed the plan

**float64 is faster than float32 here.** Measured with workers saturated:
2.99 societies/s at float64 against 2.39 at float32 — float32 is 25% *slower*,
not the ~2× faster the reference README claims. `scipy.special.ndtr` computes in
double regardless, so the single-precision path pays for conversions on every
call and never recovers the memory bandwidth. The campaign is float64
throughout: faster, and one fewer numerical approximation to defend.

**The whole campaign costs about 10 hours, not 40.** With the throughput
measured rather than assumed, the full table — main diagram at 49×49 with 24
replicates, both agendas, all controls, three finite sizes, five discrimination
cases and the replica-overlap diagnostic — comes to ~108,000 societies.

## What the local analysis actually gives

Stronger than the plan asks for, and all of it executable in `theory.py` and
verified in `test_theory.py`:

- **Exact rigid translation.** `X_D(h_w, h_μ) = X_0(h_w + D, h_μ)`. The field
  translates the phase portrait; no perturbation theory, no small-`d` expansion.
- **Two `Z₂` symmetries at `D = 0`**, both from `Φ(−z) = 1 − Φ(z)`: point
  reflection and sector exchange. The first is what the field breaks.
- **The separatrix is exactly the line `h_μ = h_w + D`** — globally invariant, not
  merely tangent — with a hyperbolic saddle at `(−D, 0)` and eigenvalues `∓2/π`.
- **A parameter-free prediction.** A channel is captured by the trust attractor
  exactly when its initial dissonance gap falls below `D`, giving
  `C_CT = f_d·G(d)` with `G(D) = 2Ψ(D) − 1`. At the model's own initialisation
  `Ψ` is closed-form, so `G` has no free parameters at all; `G'(0) = 0.9655`.
- **No critical field strength for `C_CT`, and there cannot be one.** `F_μ`
  vanishes identically on `h_w = 0`, so the class-trust contrast has no linear
  restoring force at the symmetric state. The sign change at `d = 0` is a forced
  zero-crossing of an odd response, not a bifurcation. This explains, structurally,
  why the earlier semi-analytic attempt in `controlled-small-cv-reduction/`
  measured a largest eigenvalue that was identically zero across its whole grid.

The prediction is *local*: it treats each channel as deciding independently.
Measured values exceed it, and the excess `Δ = C_CT − f_d G(d)` is the collective
amplification — a better result than a theory that happened to fit.

## Design decisions worth knowing

**Seeds are keyed, not counted.** The reference implementation seeded one
generator per batch, so what a grid point received depended on batch size and on
its position in the flattened array. Here every society is identified by a
`RunKey` and every stream is derived by hashing it, which buys batch invariance,
resumability by set difference, and separable quenched/thermal disorder.

**Controls are paired with their baseline.** Sharing `crn_group` means a control
society starts from the same weights, distrust and interaction order as its
baseline partner, so differences are reported paired. This is the largest
variance reduction available and it costs nothing.

**Nothing is averaged on write.** The reference `sweep.py` collapsed its repeats
with `.mean(axis=2)` before saving, so no uncertainty could ever be recovered
from a stored result — after the compute had already been spent.

**Grids have odd point counts.** `linspace(-1, 1, 48)` does not contain `d = 0`,
so the baseline column that calibrates every threshold would be missing. Odd
counts also make each coarse grid an exact subgrid of the fine one, which is what
makes pairing possible.

**Regimes are classified, not labelled by hand.** Thresholds are calibrated from
the no-discrimination null, each replicate is classified separately and the modal
label reported with an agreement fraction, and the whole map is recomputed at
several threshold multiples. "Spin glass" is not used: that name needs overlap
distributions and initialisation dependence, so the region is named for what is
measured — negative affective balance.

**The dynamics anneals and never reaches a stationary state.** `F_C` and `F_V`
are negative almost everywhere, so both uncertainties shrink monotonically. Every
reported quantity is conditional on the measurement time. The plan never says
this and it is the likeliest reviewer objection.

## Running the campaign: one job at a time

The two codebases cannot run concurrently on a 14-core / 16 GB machine. The
reference `make_all.py --preset full` holds about 620 MB per worker (its `full`
preset uses `batch_size=1024`), so ten workers take ~6 GB; adding a second
campaign drove swap to 6.2 GB of 7.2 GB and the load average past 150, at which
point both jobs crawled. Run them sequentially.

This package is built for that: `run_campaign` writes a shard per chunk and
resumes by set difference, so it can be stopped and restarted at any point with
nothing lost. Chunk size is chosen from a time target, so an interruption costs
at most a few minutes of work. The reference implementation cannot be paused —
it writes only on completion — so it is the one to let run.

## Status

- [x] Package, packaging, 87 tests
- [x] Golden regression: reproduces the reference dynamics bit-for-bit
- [x] Seeded sweeps retaining every replicate; resumable
- [x] Control fields, magnitude-matched by assertion
- [x] Observables, including class-independent polarisation
- [x] Local analysis, verified numerically
- [x] Regime classifier and threshold sensitivity
- [x] Plotting, regime map with contested-point hatching
- [ ] Main campaign — `A1_main_P5` paused at 979/57,624 societies (11 shards
      preserved; resumes with `run_campaign(get_run("A1_main_P5"))`), yielding
      the machine to the reference run
- [ ] Figures
- [ ] Manuscript
