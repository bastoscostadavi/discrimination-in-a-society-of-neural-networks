# The opinion update under in-context learning

The sibling directory `../llm-agent-modulation` shows that a language model with
frozen weights updates its **trust** in an interlocutor the way `F_mu` says it
should. This directory measures the other sector: the **opinion** update `F_w`,
which is the function the paper's discrimination mechanism actually runs
through, and which has never been measured. It measures `F_mu` again alongside
it, on the same conditions, so that the two properties the paper asserts about
the pair — the reflection symmetry `F_w(x,y) = F_mu(y,x)`, and the crossover at
`h_mu = h_w` where the less certain sector moves the more — can be tested at all.

Model `gpt-5.6-luna` at `reasoning_effort=low`.

## The thing worth knowing first

The model is never asked for a number.

The sibling experiment asks for a chance from 0 to 100 and takes its probit.
That is a fair instrument and it is also the source of three of that
experiment's four limitations: the answers clip, so `|h_mu| <= 2.05` and the
theory's turnover sits outside the observable window; the conviction has to be
elicited rather than set, so the `(h_w, h_mu)` plane is sampled in two clumped
bands wherever the model happens to land; and a reader can reasonably say that
what was measured is a model reporting a number rather than a model behaving.

Here the model is only ever asked which of two invented things the records
favour, and it answers with one of the two names. Everything reported — the
receiver's conviction, its trust in the colleague, and both updates — is read off
**where those one-word verdicts turn over**.

## The instrument

Sampling the same briefing sixteen times almost always gives sixteen identical
answers: on crisp evidence this model is close to deterministic, and a
psychometric curve swept over evidence is very nearly a step. That rules out
reading a frequency as the measurement. It is exactly what a *nulling* method
wants, though — a sharp step has a well-located position, and the position is the
reading.

So the question is never "how sure are you". It is "how much counter-evidence
does it take before you change your answer". That quantity is in pieces of
evidence, it is signed, and it has no ceiling: push the belief harder and the
null point simply moves further out.

A briefing carries three things in three separate channels, so they move
independently:

| | | sets |
|---|---|---|
| `s` | the receiver's own evidence, net of the two directions (out of six pieces) | `h_w` |
| `k` | the colleague's record: right `k` of the last 20 settled questions | `h_mu` |
| `t` | a later consignment, net of the two directions | the ladder |

The ladder is walked adaptively — one rung at the origin, geometric steps out
until the verdict flips, then bisection into the bracket — and a probit fit over
every rung visited places the crossing between integers. Individual verdicts are
binary; the crossing they bracket is continuous.

Each condition is measured twice, with the colleague present and **silent** and
with the colleague **speaking**. Both arms share a framing, so everything
constant about the briefing cancels in the difference, and the difference is
what the colleague's word was worth.

## What the screen found

Asked a question it has nothing to go on, the model names the entity mentioned
**first** — in 14 of 16 worlds, at near-certainty, reversing exactly when the two
names are swapped. It is a position bias, not a prior, so it cancels in the
framing average and cancels again in every before-and-after difference. It is
also large enough that an experiment that did not counterbalance framing would
have measured it instead of the update. Worlds are screened on the
framing-*averaged* rate, which a content prior would move and a position bias
would not; 11 of 16 pass.

Stage 1 then checks that the ladder reads evidence faithfully. It does: after
averaging the two framings the residual bias is under a quarter of one piece of
evidence, and four pieces of evidence read back as 3.5 to 4.1.

## Units, and the one fitted constant

The ladder works in pieces of evidence; the paper works in probits. One number
converts, `lam`, and it is **not** fitted to the update — that would be a second
free parameter, and the sibling fit already spends the one that is unavoidable.
`lam` is pinned on the trust axis instead, from information the design already
contains: a colleague right `k` of 20 times has a stated flip probability of
`1 - k/20`, so the distrust they ought to carry is known before any update is
measured. Requiring the measured testimony weights to reproduce those known
values fixes `lam`, and the residual of that requirement is itself a result.

What is left over is one positive scale between the measured updates and `F_w`,
standing for the unobservable covariance in `w += (F_w / gamma_C) sigma_e C x` —
exactly the scale the sibling fit spends on `V`.

## Layout

```
llmmod2/
  llm.py        one cached call; adapted from the sibling, extended with n
  worlds.py     the invented settings, generated once and frozen
  prompts.py    every briefing the model sees
  ladder.py     the nulling ladder and its probit fit
  fields.py     pieces of evidence -> the paper's fields
  plotting.py   the figures
scripts/
  stage0_worlds.py     invent and freeze the worlds
  stage1_calibrate.py  screen them, calibrate the ladder
  stage2_opinion.py    the F_w measurement
  stage3_trust.py      the F_mu measurement, on the same grid
  figure.py            the whole of the analysis, and the figures
```

The theory is **imported** from `../nn-based-simulation/ednna/modulation.py`,
never transcribed; two copies of the paper's equations in one repository is one
copy too many. `scripts/figure.py` will not run without that sibling present.

## Reproducing

```sh
pip install -r requirements.txt
python scripts/stage0_worlds.py            # cached; free on a rerun
python scripts/stage1_calibrate.py
python scripts/stage2_opinion.py           # --quick for a smoke run
python scripts/stage3_trust.py
python scripts/figure.py
pytest
```

Caching is not an optimization here, it is part of the method: every request is
keyed by a hash of exactly what was sent, so a rerun with nothing changed replays
from disk and returns the same verdicts, and a rerun with anything changed is a
different key and actually goes out. Delete `data/cache/` to force a fresh draw.
Both measurement stages take `--max-cost` and abort rather than overrun it.

## What it found

`F_w` reproduces, over 550 conditions from 11 worlds, with no censored cells.

| | |
|---|---|
| correlation with the fitted `alpha * F_w` | **r = 0.80** |
| sign agreement | **94.7%** |
| `lam` calibration against the stated track records | **r = 0.99** |
| reach | `h_w` to 3.00, `h_mu` to 2.25 in magnitude |
| cost | 10 323 live calls, 11 668 cached responses, **$18.47** |

**The trust gate.** Signed movement toward the message, in pieces of evidence,
by the colleague's record: `-3.02` at 0/20, `-2.10` at 4/20, `+0.30` at 10/20,
`+1.75` at 16/20, `+4.26` at 20/20. The crossing is at the uninformative record,
and the sign is carried by trust and not by agreement: it is the same on both
sides of `h_w` at every record, and 100% in the predicted direction on both
sides of the trust axis. A colleague that has been wrong more often than right
moves the receiver **away** from what it asserts — including away from a
conclusion the receiver had itself reached.

**Dissonance amplification.** The size of the move is not symmetric in
agreement, and the asymmetry reverses with trust, which is what `1/Z` does and
what no trust-weighted learning rate would do:

| | receiver disagreed | receiver agreed |
|---|---|---|
| trusted emitter | **+2.37** | +1.28 |
| distrusted emitter | −1.15 | **−2.17** |

Pooled over both sides these cancel exactly (1.76 vs 1.73), which is why the
effect has to be read per side.

**The trust sector does not move.** `F_mu` was measured on the same conditions
and does not reproduce: `alpha_mu` comes out negative. The cause is in the
paper's own equation. The update is `F_mu * V / gamma_V`, and `V` is how
uncertain the receiver is about the colleague; a record of 20 settled questions
leaves almost no uncertainty, so one further exchange — on an invented question
whose answer is never revealed — has nothing to move. The measured weights say
exactly this: their *level* tracks the stated reliability almost perfectly
(`-2.47, -0.80, +0.33, +1.02, +2.87` across the five records, calibration
`r = 0.99`), while their *change* is `0.26` against the opinion sector's `1.49`.
A short-record probe (2 of 4 instead of 10 of 20) recovers the predicted sign
(`+0.13` after agreement, `-0.28` after disagreement), so the limit is the
settled prior and not the readout. Testing the reflection symmetry and the
`h_mu = h_w` crossover needs both sectors re-run against a short record.

## Figures

`scripts/figure.py` writes one figure here, `figures/iclr/opinion_plane.pdf`:
the measured opinion updates as points over the analytic `F_w`, framed and
scaled on the measurement itself. `--cuts` additionally writes two
one-dimensional cuts through it.

It also writes the paper's Figure 2, `paper/figures/llm_modulation.pdf`, which
is that panel beside the sibling experiment's `F_mu`. That one goes to the paper
and is not kept here, because half of what is in it was not measured here; the
sibling's rows are read and redrawn rather than its finished figure copied,
because two panels of one figure have to be drawn by one piece of code or they
drift apart.

The two panels share their conventions and share nothing else. An earlier
version forced Figure 1's fixed range and clip on both and was clearly worse:
the opinion sector reaches `|h_w|` of 3 and moves by up to 5, the trust sector
reaches `|h_w|` of 1.6 and moves by up to 4, so one frame wide enough for both
leaves each cloud in the middle of a mostly empty panel, and one clip large
enough for both — set by a divergence neither experiment can reach — washes the
colours out. Frame, colour scale and colour bar are therefore per panel.

The fitted scale is not a plain least squares either. `F_w` carries `1/Z` and
spans `0.003` to `1.64` over this grid, so weighting by `F^2` puts two thirds of
the weight on the 9% of conditions in the divergent corner and draws the whole
bulk of the measurement five to twenty times too pale (`alpha = 2.84` against
`4.81`). The scale is fitted on the clipped comparison the figure actually
draws. No conclusion moves with it: every statistic quoted here is a sign or a
rank.

## Limitations

The domain is synthetic. That is what buys the controlled conviction — on a real
topic the model arrives with a prior nobody set — but it is a toy, and nothing
here shows the same structure holds on questions people actually argue about.

One model, one reasoning effort. The sibling's caveat applies unchanged: this is
qualitative evidence about the *structure* of an update, not evidence that
in-context learning implements the paper's equations.

`lam` assumes evidence accumulates in log-odds, which is the assumption that
makes pieces of evidence and probits commensurable at all. Stage 1's linearity
check supports it over the measured range and says nothing beyond it.
