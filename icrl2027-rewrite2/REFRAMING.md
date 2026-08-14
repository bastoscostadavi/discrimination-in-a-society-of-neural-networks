# The reframing

Written from `../Discrimination2025(1).pdf` alone. The LLM section is assumed removed.

---

## 1. What is wrong with the current framing

The source is a statistical-mechanics paper: APS format, the phase diagram as the
deliverable, a society as the object of study. An ICLR reviewer reads that as
computational sociology and asks what it teaches them about learning
representations. The paper has a good answer. It never states it.

Three specific liabilities:

- **The contribution is descriptive.** "We simulated a society and observed four
  phases." There is no ablation isolating a mechanism, so nothing is *shown* to
  cause anything.
- **The vocabulary signals the wrong field.** Ideology, affect, distrust,
  hominins, *Homo Entropicus*. Each anthropomorphism is a reason for a reviewer
  to reclassify the paper out of scope.
- **There is no related work.** Seventeen references, all statistical mechanics
  and 1990s online learning, the most recent ML-adjacent one being the authors'
  own. This alone will draw a reject.

## 2. The ICLR-legible claim

> Each agent maintains a **learned per-source reliability estimate** and uses it
> to weight incoming supervision. That estimate is exactly the mechanism that
> converts a mild, individually harmless spurious feature into a population-wide
> group split.

Two things make this land at this venue rather than a physics venue.

**Per-source reliability weighting is standard ML equipment.** Dawid–Skene and the
crowdsourcing lineage, annotator modelling in RLHF, client reweighting in
federated learning, Byzantine-robust aggregation, peer weighting in decentralised
SGD. In every one of those it is introduced as the component that makes learning
*robust to unreliable sources*. The result here says it is also the component that
turns a proxy feature into durable group structure. That is a useful, slightly
alarming inversion of a standard design assumption.

**The injected defect is a representation error, not a preference.** The
discriminating agent appends to each input a feature that carries no task
information but correlates with the source's group label. That is the
shortcut-learning / spurious-correlation setup verbatim, lifted from one
classifier to a population of them. The venue's name is not a pun here; it is the
actual mechanism.

### Claim structure

| | |
|---|---|
| **Primary** | A spurious feature of strength `d`, held by a fraction `f_d` of learners, is amplified by peer learning into class-correlated polarisation above a sharp boundary `d_c(f_d)`. The amplifier is the learned trust sector. |
| **Secondary** | Task diversity `α = P/K` controls *what crystallises first* — trust structure before opinion structure, or the reverse. |
| **Frame** | The update rule is provably optimal for a *pair*. Deployed in a population it is pathological. Objective misspecification by scale. |

The `α` result is currently a side observation on p. 18 of the source. It is the
most novel-to-ICLR thing in the paper and should be a headline contribution.
"Task diversity determines whether a population of learners aligns its
representations first or its trust relations first" is clean, quotable and
nonobvious.

The *frame* is the one piece to state and then leave alone. It is provocative and
it is what makes the paper memorable, but "optimal" is doing heavy lifting and a
reviewer will attack it. Say it twice — abstract and discussion — and let the
ablations carry the weight.

## 3. Terminology

Two renames, applied without exception.

| source draft | here | why |
|---|---|---|
| ideological sector | **opinion sector** | It holds a classifier. "Ideology" invites a reading the model does not support, and collides with `I` as an agent index. |
| affective sector | **trust sector** | Its technical name is a per-source reliability estimate. That is the name that makes the paper legible here. |
| `B_I`, `B_A` | **`B_ρ`, `B_η`** | Named after the quantity each is built from, once `I`/`A` no longer stand for anything. |

Keep "distrust" for the signed variable `μ` where its sign matters (`μ > 0` is
distrust), but call the sector trust. In the technical sections prefer
*reliability estimate*; use the social reading once, in the introduction, and once
in the discussion.

## 4. What to cut

- **Two of the three update presentations.** Eqs. (9)–(12), (13)–(16), (17)–(20)
  are one update written three ways. Main text gets one boxed form; the entropic
  dynamics derivation goes to an appendix.
- **Five of six `D` matrices.** Keep the symmetric case in the main text, appendix
  the rest with one sentence on which asymmetries matter.
- **The Darwin epigraph, *Homo Entropicus*, and the hominin argument.** Reviewers
  punish unfalsifiable evolutionary storytelling severely. The small-group /
  large-group point survives as two sentences about objective misspecification by
  scale, stated as a property of the algorithm.
- **The modulation-function tour.** Figs. 2 and 3 of the source spend two pages on
  the shape of `F_w, F_μ, F_C, F_V`. Main text needs one panel and two properties
  (sign reversal, blame attribution). The rest is appendix material.
- **The parsing-error motivation as a separate story.** `v_p = ‖w‖⁻²` is elegant
  but it is a second noise source competing for the reader's attention with the
  one that matters. State it in the setup, move the derivation to the appendix.

## 5. What to add — this decides accept/reject

### 5.1 The ablations. Non-negotiable.

The source *asserts* (p. 6) that Hebbian and Perceptron rules do not produce this
behaviour. Right now that is a sentence. It has to be a figure. Four conditions,
each run over the same `(d, f_d)` grid:

| | condition | prediction | what it establishes |
|---|---|---|---|
| **A0** | **Random-label control.** `D` keyed to a per-emitter random bit drawn independently of class, same magnitude `d`. | `R_μc ≈ 0`; degraded learning but no group structure. | Separates *correlation with a label* from *added noise*. The source raises this on p. 20 and never tests it. |
| **A1** | **Frozen trust.** `F_μ = F_V = 0`, `μ` held at initialisation. | No transition in `R_cw` either. | Trust is not merely the *readout* of the split — it is the carrier. Measuring `R_μc` here is trivial; **measure `R_cw`**, that is the nontrivial part. |
| **A2** | **No anti-learning.** Replace `F_w`'s prefactor `(1 − 2Φ(h_μ))` with `max(0, 1 − 2Φ(h_μ))`, so a distrusted emitter produces no update rather than a reversed one. | Polarisation collapses, including the class-free kind at `d = 0`. | The sharpest single test. The effective antiferromagnetic coupling is the whole engine. |
| **A3** | **Rule baselines.** Hebbian and Perceptron updates, same `D`, same grid. | No class-correlated phase. | Converts the source's p. 6 assertion into evidence, and shows the effect is a property of reliability-weighted inference, not of learning-from-peers in general. |

These four panels are what turn "we observed a phase diagram" into "we identified
a mechanism". Without them the paper is a simulation study.

### 5.2 Make the boundary quantitative

`d_c(f_d)` is currently a colour boundary in a pixel map. Extract it at a fixed
`R_μc` threshold, fit it, and check finite-size scaling across `N ∈ {20, 40, 80}`.
A collapse plot is worth more to a reviewer than several pages of interpretation,
and it converts the central claim from a picture into a prediction.

### 5.3 One experiment beyond perceptrons

A single panel with either a committee machine or with issues drawn from real
sentence embeddings instead of Gaussian `x̂` removes the largest objection — that
this is a perceptron artefact. It does not have to be the main result. It has to
exist.

### 5.4 Statistical hygiene

The source's maps are visibly single-realisation, and its measurement time is a
literal `Δt = ????` (p. 18). State `N`, `P`, `K`, `Δt`, initialisation and seed
count; put error bars on the extracted boundary; show `N`-dependence. This is
cheap and its absence is expensive.

### 5.5 State what the covariances are, at Eq. 1

The source draft introduces `C` and `V` as posterior widths and never says the
other half: they premultiply every update, so they are simultaneously the step
sizes. The limit makes it concrete — as `C → 0` an agent is completely certain of
its opinion and its weights stop moving; as `V → 0` it is completely certain of
how far to trust a given source and that estimate stops moving. A fully certain
agent is a frozen agent, because a Bayesian update against a delta-function prior
returns the prior. No learning-rate hyperparameter appears anywhere in the model
because the covariances already are one.

This is worth a paragraph at Eq. 1 rather than a remark later, for three reasons.
It is the cleanest statement of what the Gaussian state is *for*. It makes the
annealing behaviour a consequence rather than an observation — `F_C, F_V < 0`
means the population is running toward the frozen limit, which is why it slows
down instead of equilibrating and why measurement time is a parameter. And it
gives `C_0 = I`, `V_0 = 1` a meaning: the initialisation sets the total movement
available to the population over its entire history, which is not obvious and is
currently reported as an uninformative-prior default.

### 5.6 Settle the sign convention first

Eq. (25) with Table I, read literally, puts in-group tolerance at `d < 0`, while
the text and every figure put the discriminatory phases at `d > 0`. One global
sign reconciles them. This has to be fixed *before* the phase diagram is the
centrepiece of a submission — the `x` axis of the headline figure cannot be
ambiguous. The convention used here: **`d > 0` is in-group tolerance and
out-group intolerance**, i.e. `D = d · [[+1, −1], [−1, +1]]`, which is the reading
the sign of `F_μ` forces.

## 6. Related work, essentially from scratch

At minimum:

- **Shortcut learning and spurious correlations** — Geirhos et al.; group
  robustness (Sagawa et al.); proxy variables in algorithmic fairness.
- **Learning with noisy labels and annotator reliability** — the Dawid–Skene
  lineage through to modern annotator modelling and RLHF reward-model work.
- **Decentralised and federated learning** — gossip protocols, client reweighting,
  Byzantine robustness. This is where per-source reliability weighting actually
  ships.
- **Model collapse and self-consuming training loops** — the closest living
  relative of this result. Its absence would be conspicuous.
- **Multi-agent emergent conventions**, opinion dynamics as it appears in ML
  venues, and structural balance in networks (for `B_ρ`, `B_η`).

## 7. Page budget (9 pages main text)

| § | content | pages |
|---|---|---|
| 1 | Introduction and contributions | 1.0 |
| 2 | Setup: a population of learners with per-source reliability | 0.5 |
| 3 | The learning rule — one boxed update, two properties | 1.0 |
| 4 | A spurious feature as a field shift | 0.5 |
| 5 | Observables | 0.5 |
| 6 | Results: phase diagram and boundary; **ablations**; agenda complexity | 4.0 |
| 7 | Related work | 0.75 |
| 8 | Limitations and discussion | 0.75 |

Check the ICLR 2027 CFP for the actual limit; 9 + unlimited appendix is assumed.

## 8. Risk assessment

**Fit is real but not automatic.** This is a science-of-learning-systems paper.
Those land at ICLR — grokking, model collapse, emergence — but only when the
ablation isolating the mechanism is airtight. With §5.1 I think it is a solid
submission. Without it, it reads as computational social science with neural
networks as substrate, and being interesting will not save it.

**Do not oversell to human discrimination.** The source hedges once (p. 6,
"being careful that these are simple machines"). Keep exactly that hedge, put it
in Limitations, and make the social claim nowhere else. Reviewers are far more
forgiving of a mechanism paper that gestures at a social interpretation than of a
social claim propped up by perceptrons.

**The venue is not the only option.** FAccT or AIES would take the discrimination
framing more readily and demand less mechanism. That is a worse paper and a
smaller audience, and the mechanism is the interesting part — so ICLR, but with
§5.1 done properly.

## 9. Title

Recommended: **Learned Trust Amplifies Spurious Features into Group Structure**

Alternatives:

- *When Reliability Estimation Becomes Discrimination* — sharper, more hostile,
  higher variance with reviewers.
- *Discrimination in a Society of Neural Networks* — the source title. Memorable,
  and fine if the abstract is ML-first from word one, but it spends the title on
  the interpretation rather than the mechanism.
- *Optimal Pairwise Learning Is Collectively Discriminatory* — the strongest hook
  and the easiest to attack. Only with the ablations in hand.
