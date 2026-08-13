# The model as implemented

Reference for both pillars of the project: the neural-network society in
`../nn-based-simulation/`, and the language-agent society specified in
[`llm-study-contract.md`](llm-study-contract.md). It records the equations actually implemented, the
parameters and where each comes from, and the three places where a literal
reading of the source draft (`../Discrimination2025(1).pdf`) is ambiguous or
inconsistent. The sign of the discrimination field is important enough to have
its own note: [`discrimination-field-sign.md`](discrimination-field-sign.md).

## State

A society of `N` agents. Agent `I` holds

- **ideological sector**: weights `w_I ∈ R^K` with covariance `C_I ∈ R^{K×K}`,
- **affective sector**: for every other agent `J`, a distrust `μ_{J|I} ∈ R` with
  variance `V_{J|I} > 0`.

The belief is the product of Gaussians `N(ŵ_I, C_I) · Π_J N(μ_{J|I}, V_{J|I})`.
The means say what the agent thinks; the (co)variances say how firmly, and
therefore how fast it can be moved.

Each agent also carries an immutable class label `κ_I = ±1`, visible to others and
carrying no information about any issue.

An **issue** is a unit vector `x̂ ∈ R^K`. The **agenda** is a fixed set of `P`
issues. An agent's opinion on an issue is `σ_I = sign(w_I · x̂)`.

## One interaction

Draw an issue `x̂`, an emitter `e` and a receiver `r`. The emitter states `σ_e`.
The receiver forms two scaled fields,

```
γ_C  = √(1 + x̂·C_r x̂)                γ_V  = √(1 + V_{e|r})
h_w  = (ŵ_r · x̂) σ_e / γ_C  +  D      h_μ  = μ_{e|r} / γ_V
```

where `D` is the discrimination field (below), and the evidence

```
Z = Φ(h_w) + Φ(h_μ) − 2 Φ(h_w) Φ(h_μ)
```

with `Φ` the standard normal CDF. `h_w > 0` means the receiver already agrees with
the message; `h_μ > 0` means it distrusts the emitter. `Z` is small in the two
*dissonant* quadrants — agree-and-distrust, disagree-and-trust — and large in the
two consonant ones.

The four modulation functions are the log-derivatives of the evidence:

```
F_w  = ∂ln Z/∂h_w   = (1 − 2Φ(h_μ)) g(h_w) / Z
F_μ  = ∂ln Z/∂h_μ   = (1 − 2Φ(h_w)) g(h_μ) / Z
F_C  = ∂²ln Z/∂h_w² = −F_w (F_w + h_w)
F_V  = ∂²ln Z/∂h_μ² = −F_μ (F_μ + h_μ)
```

and the receiver updates

```
ŵ_r      += (F_w / γ_C) σ_e C_r x̂          C_r      += (F_C / γ_C²) (C_r x̂)(C_r x̂)ᵀ
μ_{e|r}  += (F_μ / γ_V) V_{e|r}             V_{e|r}  += (F_V / γ_V²) V_{e|r}²
```

Three consequences are worth naming, because the collective behaviour follows
from them:

- **`F_w`'s prefactor `1 − 2Φ(h_μ)` is negative for a distrusted emitter**, so the
  receiver moves its opinion *away* from what a distrusted agent says. Effective
  antiferromagnetic couplings appear without being put in.
- **`F_μ`'s prefactor `1 − 2Φ(h_w)` is negative when the receiver agrees**, so
  agreement builds trust and disagreement erodes it. This is the hinge the
  discrimination field turns.
- **`F_C, F_V < 0` almost everywhere**, so both uncertainties shrink: the dynamics
  anneals. A society therefore does not reach a stationary state — it slows down —
  and *when* you measure is a parameter of the experiment.

## The discrimination field

A fraction `f_d` of agents, drawn independently of class, extend their
representation of a message with a feature that carries no information about it
and depends only on the emitter's class. The effect is an additive shift of the
opinion field,

```
h_w  →  h_w + D[class(receiver), class(emitter)]
```

with `D` one of six matrices (`ednna/discrimination.py`). The default, and the one
all phase diagrams use, is the fully symmetric case in which both classes favour
their own and are hostile to the other:

```
D = d · [[+1, −1], [−1, +1]]
```

**`d > 0` means tolerance towards in-group and intolerance towards out-group.**
This is forced by the sign of `F_μ` above and is the opposite of what the draft's
Table I states; see [`discrimination-field-sign.md`](discrimination-field-sign.md).

## Order parameters

Per pair of agents: the **ideological overlap** `ρ_IJ = cos(w_I, w_J)`, and the
**trust** `η_{e|r} = 1 − 2Φ(h_μ)`, which is `+1` for a fully trusted emitter and
`−1` for a fully distrusted one. With `G_IJ = κ_I κ_J`,

```
R_wμ = ⟨(η_{I|J} + η_{J|I}) ρ_IJ⟩       opinion–trust
R_μc = ⟨G_IJ (η_{I|J} + η_{J|I})⟩       trust–class     (the discrimination parameter)
R_cw = ⟨G_IJ ρ_IJ⟩                      opinion–class
```

averaged over unordered pairs, each normalised to `[−1, 1]`. Social balance over
triples,

```
b_I  = ρ_IJ ρ_JK ρ_KI
b_A  = (η_IJ η_JK η_KI + η_JI η_IK η_KJ)/2
B_I  = ⟨b_I⟩,   B_A = ⟨b_A⟩
```

separates *organised* disagreement from disorder: two coherent, mutually opposed
blocs give `B = 1` whatever they are made of, a random society gives `0`, and a
society that cannot settle goes negative. Both aggregates are computed in closed
form — for any `M` with unit diagonal,

```
Σ_{distinct ordered (I,J,K)} M_IJ M_JK M_KI = tr(M³) − 3(Σ_IJ M_IJ M_JI − N) − N
```

and each unordered triple appears six times — which replaces an `O(N³)`
enumeration with two matrix products. Verified against enumeration in the tests.

The order parameters mention neither `K` nor the weights: they need only a signed
opinion alignment and a signed trust per pair. That is deliberate, and it is what
lets the language-agent society be measured on the same axes.

## Parameters

| parameter | value | provenance |
|---|---|---|
| `K` embedding dimension | 30 | **recovered** from the draft's `α = P/K` values (see below) |
| `N` agents | 40 | chosen; work scales as `N³` |
| `P` agenda size | 5 (simple), 100 (complex) | chosen, straddling `α = 1` |
| `Δt` interactions per ordered pair | 500 | **calibrated** against the draft's trajectory endpoints |
| `C₀`, `V₀` | `I`, `1` | uninformative prior |
| `μ₀` | `U(−1,1)` | starts the society with half its triples frustrated |
| class split | `N/2` / `N/2` | the draft's two classes |
| discrimination case | 6 | the symmetric case; the other five are implemented |

`K = 30` is recoverable rather than chosen: the draft's trajectory figure is
labelled with `α = P/K ∈ {0.03, 0.17, 0.23, 0.33, 0.50, 0.67, 1.67, 3.33, 333.33}`,
and these are exactly `P/30` for `P ∈ {1, 5, 7, 10, 15, 20, 50, 100, 10⁴}`. The
draft's own LLM protocol ("the opinions from −1 to 1 on thirty issues") agrees.

Everything else the draft leaves open, including a literal `Δt = ????` in its
text. `nn-based-simulation/scripts/calibrate.py` fixes the remaining freedom
against features of the published figures; see the simulation README for the
result and the residuals.

## Where a literal reading of the draft does not work

1. **Sign of the discrimination field.** Eq. 25 plus Table I put the
   discriminatory phase at `d < 0`; the text and every figure put it at `d > 0`.
   One global sign reconciles them. Full argument, both variants, and a
   side-by-side figure: [`discrimination-field-sign.md`](discrimination-field-sign.md).
   Flag: `ModelConfig(literal_draft_sign=True)`.

2. **The class indicator.** Eq. 28 defines `G_IJ` as `1` for same-class pairs and
   `0` otherwise. But the published `R_μc` map spans `−1…1`, which a `{0,1}`
   indicator cannot reach: under perfect reverse discrimination it returns
   `−(N/2−1)/(N−1)`, and its range depends on `N`. Since reverse discrimination is
   half the phase diagram, we use the signed `G_IJ = κ_I κ_J`.
   Flag: `class_indicator="01"`.

   Relatedly, the draft normalises all three correlations by `N(N−1)`. That is
   right for the two that sum `η_{I|J} + η_{J|I}` over unordered pairs, but `R_cw`
   has a single term per pair, so the same divisor caps it at `1/2`. We use
   `2/(N(N−1))` there. Flag: `literal_norm=True`.

3. **Numerical safeguards, which the draft does not mention.** The exact
   modulation functions diverge where `Z → 0` (deep in the dissonant corners), and
   `F_V < 0` drives `V` monotonically to zero, so floors on `Z` and `V` are
   required for a long run to survive at all. The covariance update
   `C → C + a (C x̂)(C x̂)ᵀ` also stays positive semi-definite only while
   `1 + a x̂·C x̂ ≥ 0`, and `F_C` is unbounded below, so `a` is clipped to that
   boundary. Only the clip could alter a trajectory; it did not trigger in any run
   reported here, and `SocietyBatch.n_psd_clips` counts it.
