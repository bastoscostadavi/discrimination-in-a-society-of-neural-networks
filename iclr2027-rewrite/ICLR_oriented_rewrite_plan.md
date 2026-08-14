# ICLR-Oriented Rewrite Plan

## Working paper

**Current title:** *Discrimination in a Society of Neural Networks*

**Target:** ICLR 2027 main track

**Scope of this plan:** Remove the unfinished LLM section and rebuild the manuscript around the neural-agent model. This is not a copyedit. It is a scientific reframing, validation, and compression effort designed for a 9-page ICLR submission, with derivations and secondary results moved to appendices.

**Status note:** Statements about the present manuscript describe what is currently in the draft. Proposed analyses and experiments below are recommendations, not results already established by the draft.

---

## 1. Submission thesis

The paper should make one central claim:

> Class-correlated representation errors can turn locally motivated trust and opinion updates into group-correlated discrimination and polarization in a population of interacting neural agents.

The paper should not be sold primarily as a new derivation of the Entropic Dynamics learning rule. That machinery is inherited from prior work. The new paper should instead focus on what happens when an irrelevant class-dependent feature perturbs the field used by the learning dynamics.

A strong ICLR framing is:

1. Each agent is a linear neural classifier with a learned opinion vector.
2. Each directed pair of agents also has a learned trust variable.
3. Opinion and trust co-adapt through an entropic approximation to Bayesian online learning.
4. A subset of agents uses class identity, which is irrelevant to the issue being classified, in its internal representation of an interaction.
5. This class-correlated representation error breaks the symmetry of the trust-opinion dynamics.
6. At population scale, the symmetry breaking produces distinct collective regimes as the fraction of biased agents and the bias strength vary.
7. The complexity of the issue agenda controls whether affective polarization or ideological polarization develops first.

The final paper should be understandable even to an ICLR reviewer who has never read the authors' earlier work on entropic neural agents.

---

## 2. The contribution statement to build toward

The introduction should promise no more than the paper can demonstrate. A defensible contribution list would be:

1. **A formal model of class-correlated representation error.** We introduce a group-identity feature that is irrelevant to issue labels but shifts the perceived opinion field of discriminating receivers.

2. **A mechanism connecting local inference to collective discrimination.** We show how this field shift changes the basins of attraction of the coupled opinion-trust dynamics, favoring different consonant states for in-group and out-group interactions.

3. **A quantitative population-level phase diagram.** Through replicated simulations, we characterize class-uncorrelated, frustrated, ideology-aligned discriminatory, and class-dominant discriminatory regimes as functions of the discriminator fraction and discrimination strength.

4. **A temporal result about agenda complexity.** We show that the ratio of the number of issues to representation dimension changes the ordering of affective and ideological polarization.

5. **Robustness and mechanism tests.** We establish that the collective effect depends on class correlation, not merely on additional noise, and evaluate sensitivity to population size, initialization, learning rule, class balance, and discrimination pattern.

A stronger version of the paper would add:

6. **An analytical transition result.** A reduced mean-field or linear-stability analysis predicts at least one transition boundary, scaling law, or sufficient condition for the onset of class-correlated polarization.

The sixth contribution is highly desirable. Without it, the paper can still be viable, but the empirical work must be substantially stronger and the claims must consistently describe the phase diagram as simulation-based.

---

## 3. Questions the rewritten paper must answer

The manuscript should be organized around four research questions.

### RQ1. What exactly is the representation error?

The paper must define what information is task-relevant, what information is class-related, and why the class-related information is irrelevant to the issue label. At present, the discrimination field is introduced operationally as a shift in the opinion field. The rewrite should derive or motivate that shift from an explicit augmented representation.

### RQ2. Why does the error generate discrimination rather than ordinary noise?

The central control is an equally strong perturbation that is not correlated with class. If class-shuffled or zero-mean random perturbations do not produce the same class-aligned phases, the paper can attribute the effect to class correlation rather than generic model misspecification.

### RQ3. Which collective regimes exist, and how are they defined?

Every phase label must correspond to explicit order parameters, uncertainty estimates, and a reproducible classification rule. The paper should not rely only on visually inspecting heatmaps.

### RQ4. What controls the causal ordering of affective and ideological polarization?

The agenda-complexity result should be stated in terms of measurable polarization times, not only trajectories in a balance-balance plane. The paper should quantify when trust polarization precedes opinion polarization and when the ordering reverses.

---

## 4. Scientific repairs required before prose polishing

### 4.1 Formalize the discrimination mechanism

Use class labels

```text
c_i in {-1, +1}
```

and a discriminator indicator

```text
z_i in {0, 1}, with P(z_i = 1) = f_d.
```

For the symmetric discrimination pattern currently called `D6`, write the class-dependent field compactly as

```text
D_{e|r} = -z_r d c_e c_r.
```

This gives `-d` for same-class interactions and `+d` for cross-class interactions, matching the current D6 table. State clearly which index is the emitter and which is the receiver.

Then define the field used by a discriminating receiver:

```text
h_w^D = h_w + D_{e|r}.
```

The rewrite should explain this as an augmented representation. One possible formulation is:

```text
phi_r(x, c_e, c_r) = [x, z_r s(c_e, c_r)],
```

where the added coordinate contains no information about the issue's task label but enters the receiver's compatibility calculation. Show how this construction reduces to the additive field shift above.

The main text should study D6 because it gives a clean symmetric notion of in-group favoritism and out-group intolerance. The other five discrimination matrices should be moved to the appendix and treated as generalization tests.

### 4.2 State precisely what is optimized

The phrase "optimal learning algorithm" is too broad. The rewrite must specify:

- the student-teacher setting in which the update was derived;
- the probabilistic objective or approximation criterion;
- the assumptions under which the update is optimal or asymptotically optimal;
- which parts of the result are inherited from prior publications;
- why applying the rule in a many-agent population is out of distribution relative to the derivation.

The paper should avoid implying that the population dynamics are globally optimal. A safer formulation is:

> Agents use an entropic approximation to a Bayesian online update derived for a noisy student-teacher interaction. We study the population-level consequences of deploying this locally motivated rule in a many-agent system.

### 4.3 Repair and simplify the order parameters

The current text describes the class matrix as taking values 1 for same class and 0 otherwise, while the corresponding plots span negative and positive values. That definition must be corrected.

Use a signed class relation:

```text
s_ij = c_i c_j in {-1, +1}.
```

Define directed trust scores from the distrust fields:

```text
t_{j|i} = 1 - 2 Phi(h_mu,j|i),
```

where positive values indicate trust and negative values indicate distrust. Define ideological overlap:

```text
q_ij = <w_i, w_j> / (||w_i|| ||w_j||).
```

A clean set of population order parameters is:

```text
C_CT = average_{i != j} s_ij t_{j|i}
C_CO = average_{i < j} s_ij q_ij
C_TO = average_{i < j} ((t_{j|i} + t_{i|j}) / 2) q_ij
```

Interpretation:

- `C_CT > 0`: same-class trust and cross-class distrust.
- `C_CO > 0`: ideological alignment follows class.
- `C_TO > 0`: trust aligns with ideological agreement.

Also report class-independent polarization measures. Otherwise, a low class correlation could be confused with an unpolarized population even when the population is strongly polarized along a class-independent axis.

Possible class-independent measures include:

- the leading two-cluster structure of the opinion-overlap matrix;
- the magnitude of the first nontrivial eigenvalue of the centered overlap matrix;
- a two-cluster silhouette score in opinion space;
- the bimodality or variance of pairwise opinion overlaps;
- the equivalent quantities for the symmetrized trust matrix.

The exact choice should be fixed before the final simulation campaign.

### 4.4 Use defensible phase names

The current "neutral" region can still contain polarization unrelated to class. Rename it to something such as:

- **class-uncorrelated polarized phase**, if polarization is present;
- **weakly structured phase**, if both class correlation and global polarization are low.

Do not use "spin glass" unless the paper supplies appropriate diagnostics, such as overlap distributions, metastability, susceptibility, nonzero Edwards-Anderson-type order, or strong initialization dependence. Without those diagnostics, use:

- **frustrated mixed phase**;
- **class-tolerant frustrated phase**;
- **disordered frustrated phase**.

The final phase names should describe measured behavior, not invoke a physical analogy that is not tested.

### 4.5 Separate balance from frustration

The current quantities `B_I` and `B_A` increase toward one as balance improves, but the text sometimes calls them frustrations. Use one convention consistently.

Option A, balance convention:

```text
B_O = mean sign(q_ij q_jk q_ki)
B_T = mean sign(t_bar_ij t_bar_jk t_bar_ki)
```

Option B, frustration convention:

```text
F_O = (1 - B_O) / 2
F_T = (1 - B_T) / 2.
```

If continuous products are used instead of signs, state that they are continuous balance scores and do not describe them as the fraction of balanced triples.

For directed trust, define whether the triangle score uses:

- symmetrized pairwise trust;
- both directed cycle orientations;
- or a genuinely directed structural-balance measure.

The current symmetrization should not remain implicit.

### 4.6 Quantify temporal ordering

The agenda-complexity result is potentially one of the paper's strongest findings. Replace the qualitative statement with explicit times:

```text
t_O(tau) = first time ideological polarization exceeds threshold tau
t_T(tau) = first time affective polarization exceeds threshold tau
Delta t = t_T(tau) - t_O(tau)
```

Then plot `Delta t` against

```text
alpha = P / K
```

with confidence intervals over seeds and sensitivity to the threshold `tau`.

Interpretation:

- `Delta t < 0`: affective polarization occurs first.
- `Delta t > 0`: ideological polarization occurs first.

Keep the current trajectory plot as a secondary visualization, but make the first-passage analysis the primary evidence.

### 4.7 Add a minimum analytical result

At minimum, analyze the reduced field dynamics shown in the current flow diagrams:

```text
d h_w / dt proportional to F_w(h_w + D, h_mu)
d h_mu / dt proportional to F_mu(h_w + D, h_mu).
```

The analysis should:

1. identify the symmetry at `D = 0`;
2. show how `D = +/-d` shifts the separatrix or basin geometry;
3. connect same-class and cross-class shifts under D6 to different probabilities of reaching the trust-agree and distrust-disagree attractors;
4. state a testable prediction for the population simulations.

A stronger analysis would derive an approximate instability condition for the class-uncorrelated state. Candidate targets include:

- a critical product or nonlinear function of `f_d` and `d`;
- a mean-field equation for `C_CT` near zero;
- a susceptibility that diverges or peaks near the transition;
- finite-size scaling of the transition location.

If no population-level analytical boundary is obtained, the abstract and introduction must not imply that the phase diagram itself was solved mathematically.

---

## 5. Experimental program

### 5.1 Fully specify the simulation protocol

The main paper or appendix must report:

- population size `N`;
- representation dimension `K`;
- number of issues `P`;
- class proportions;
- discriminator sampling procedure;
- discrimination matrix and sign convention;
- grid or sampling scheme for `f_d` and `d`;
- number of independent seeds;
- initialization of weights, covariances, distrust means, and distrust variances;
- emitter, receiver, and issue sampling rules;
- number of interactions or stopping rule;
- update schedule and any clipping or numerical stabilization;
- measurement times;
- software, hardware, and random-number-generation details.

Every main heatmap should display a mean over independent runs. Report uncertainty either as standard error, bootstrap intervals, or seed-to-seed standard deviation. Near transition boundaries, increase the number of seeds.

A reasonable minimum is 20 independent seeds per central grid point and more near estimated boundaries. If that is too expensive, use adaptive refinement near boundaries rather than presenting a dense but statistically weak grid.

### 5.2 Essential controls

The paper needs the following controls.

#### Control 1: No discrimination

Set `d = 0` or `f_d = 0`. This establishes the class-uncorrelated baseline.

#### Control 2: Uncorrelated representation noise

Give discriminating agents perturbations of the same magnitude as `d`, but draw the sign independently of class. This tests whether the effect is caused by class correlation rather than added noise.

#### Control 3: Shuffled class labels

Preserve agent dynamics and the distribution of perturbations, but shuffle the class labels used to compute the reported class correlations. This establishes a null distribution for `C_CT` and `C_CO`.

#### Control 4: Frozen trust sector

Hold distrust variables fixed while opinions learn. This tests whether co-adaptation of trust is necessary for the discriminatory phase.

#### Control 5: Frozen opinion sector

Hold opinion vectors fixed while trust learns. This isolates class-based affective sorting from ideological feedback.

#### Control 6: Alternative learning rules

Compare the entropic update with simpler Hebbian and perceptron-style rules already discussed in the manuscript. Quantify whether the discriminatory phases weaken, disappear, or change. This is important because it can establish that the phenomenon is tied to the inference structure rather than being universal to any interacting classifier.

### 5.3 Robustness analyses

At least the following should appear in the appendix, with the most important one summarized in the main text:

- population size `N`;
- representation dimension `K`;
- issue count `P` at fixed `P/K`;
- class imbalance;
- multiple initial distrust distributions;
- multiple initial opinion distributions;
- asynchronous sampling variations;
- positive and negative `d`;
- D1 through D6 discrimination structures;
- sparse or modular interaction graphs, if computationally feasible;
- different uncertainty initializations for the ideological and affective sectors.

The paper should distinguish true qualitative robustness from cases where only the transition location shifts.

### 5.4 Phase-boundary estimation

Do not define phases by hand-drawn regions alone. Use one of these procedures:

1. pre-register numerical thresholds on the order parameters and show sensitivity to those thresholds;
2. cluster points in the space `(C_CT, C_CO, C_TO, global opinion polarization, global trust polarization, frustration)` and verify cluster stability;
3. estimate ridges in susceptibilities or derivatives of the order parameters and use them as transition curves.

The strongest presentation combines order-parameter heatmaps with boundary estimates and uncertainty bands.

### 5.5 Representative trajectories and microstructure

For one parameter point in each regime, show:

- time series of the three principal order parameters;
- distributions of same-class and cross-class trust;
- distributions of same-class and cross-class opinion overlap;
- a reordered trust matrix and opinion-overlap matrix;
- final agent embeddings projected into two dimensions only as a visualization, not as primary evidence.

This will make the macroscopic phase labels interpretable.

---

## 6. Recommended main-paper structure and page budget

ICLR 2027 allows at most 9 pages of main text at initial submission. References do not count, and appendices are unlimited, but reviewers are not required to read them. The main paper therefore has to contain all evidence needed for the central claims.

### Abstract: approximately 180 to 220 words

The abstract should contain:

1. the problem: local representation bias in interacting learners;
2. the model: neural classifiers with co-adaptive directed trust;
3. the mechanism: a class-correlated irrelevant feature shifts the opinion field;
4. the main result: distinct collective regimes in `f_d` by `d` space;
5. the temporal result: agenda complexity reverses the ordering of affective and ideological polarization;
6. the scope: a mechanistic agent model, not a direct model of human behavior.

Delete all LLM references and all placeholders. Do not begin with a broad moral claim.

### 1. Introduction: 1.0 to 1.2 pages

Paragraph 1: Representation choices can include task-irrelevant group information. In an isolated classifier this can produce biased predictions; in interacting learners it can also change who learns from whom.

Paragraph 2: Existing work studies online neural learning, opinion dynamics, trust, fairness, and structural balance, but the collective effect of class-correlated representation error in co-adapting trust-opinion systems remains unclear.

Paragraph 3: Introduce the model in plain language.

Paragraph 4: State the central mechanism and results.

End with a concise contribution list.

Remove:

- the Darwin epigraph;
- the unfinished LLM framing;
- unsupported claims about hominin evolution;
- lengthy general discussion of all forms of human discrimination;
- claims that the model directly explains human societies.

### 2. Model: 1.4 to 1.6 pages

Include only the definitions needed to understand the new result:

- issues `x_mu` in `R^K`;
- perceptron opinion `sign(w_i . x_mu)`;
- directed distrust variable and trust transformation;
- pairwise interaction protocol;
- compact evidence or update equations;
- discrimination field and D6 definition.

The full Gaussian projection derivation, covariance updates, and all modulation derivatives should move to Appendix A.

Add a schematic showing:

```text
issue + emitter opinion + emitter/receiver class relation
                         |
                         v
             receiver representation
                         |
                         v
         opinion update <-> trust update
```

### 3. Mechanism: 1.0 to 1.2 pages

Explain dissonance in model terms:

- agreement with a distrusted emitter;
- disagreement with a trusted emitter.

Show the unbiased flow and the shifted same-class and cross-class flows in one clean figure. State the symmetry-breaking result or proposition. Derive a prediction for the sign of the class-trust correlation.

This section should replace the current long tour through multiple modulation-function surfaces.

### 4. Experimental setup and observables: 0.7 to 0.9 pages

Define:

- simulation protocol;
- `C_CT`, `C_CO`, and `C_TO`;
- class-independent polarization;
- balance or frustration;
- phase-classification rule;
- seed count and uncertainty reporting.

Put the full parameter table in the appendix, but include the central parameter values in the main text.

### 5. Collective regimes: 2.0 to 2.3 pages

Present:

1. the main phase diagram;
2. the underlying order-parameter heatmaps;
3. representative points or trajectories;
4. uncertainty or boundary stability;
5. the uncorrelated-noise control;
6. one finite-size result.

The narrative should distinguish:

- class-uncorrelated polarization;
- frustrated or tolerant mixed behavior;
- discrimination with ideological alignment;
- class-dominant distrust that persists even when opinions are similar.

Do not claim four sharp thermodynamic phases unless finite-size evidence supports that wording. "Collective regimes" is safer when boundaries are crossovers.

### 6. Agenda complexity: 0.8 to 1.0 pages

Plot ideological and affective polarization times against `alpha = P/K`. Report the crossover and uncertainty. Use the balance trajectories as supporting evidence.

The main conclusion should be model-specific:

> Restricted agendas favor earlier affective sorting, whereas more complex agendas favor earlier ideological sorting, under the studied dynamics and parameter range.

Avoid converting this directly into a claim about political history without empirical validation.

### 7. Robustness, limitations, and conclusion: 0.6 to 0.8 pages

Summarize the most decision-relevant robustness results. State limitations explicitly:

- binary, fixed class labels;
- linear classifiers;
- synthetic issue embeddings;
- mostly well-mixed interactions;
- no direct behavioral or societal validation;
- model-specific trust semantics;
- finite population and finite simulation horizon.

Conclude with the ML insight: a locally motivated update can have harmful collective consequences when representations contain a systematically group-correlated irrelevant feature.

### Required statements outside the main page count

Prepare:

- AI use statement;
- ethics statement, recommended because the paper concerns discrimination and social interpretation;
- reproducibility statement;
- anonymous code and configuration package.

ICLR 2027 requires an AI-use disclosure. Because generative AI is being used to suggest structure and edit the manuscript, that use should be disclosed according to the conference policy.

---

## 7. Figure plan

### Main Figure 1: Model and local mechanism

A two-part figure:

1. a schematic of the interaction and class-correlated representation feature;
2. reduced field flows for `D < 0`, `D = 0`, and `D > 0`, with identical axes and clearly marked attractors or basins.

This should be a redesigned version of the current Figure 4. Remove editorial text from the figure and caption.

### Main Figure 2: Population phase diagram

Show:

- categorical regime map with estimated boundaries;
- small adjacent panels for `C_CT`, `C_CO`, and `C_TO`;
- uncertainty or boundary variability across seeds.

This consolidates the current Figures 1 and 5.

### Main Figure 3: Mechanism controls and finite-size behavior

Possible panels:

- class-correlated field;
- same-magnitude uncorrelated noise;
- shuffled-label null;
- transition curves for several `N` values.

This figure directly answers the likely reviewer objection that the result is generic noise or a finite-size artifact.

### Main Figure 4: Agenda-complexity crossover

Primary panel:

```text
Delta t = t_T - t_O versus alpha = P/K.
```

Secondary panels:

- representative time series for low and high `alpha`;
- balance trajectories corresponding to the current Figure 7.

### Appendix figures

Move or expand the following to the appendix:

- full modulation-function surfaces;
- contour plots for all four modulation functions;
- D1 through D6 phase diagrams;
- additional frustration maps;
- initialization sensitivity;
- all hyperparameter sweeps;
- sparse-network results;
- additional representative trajectories.

---

## 8. Table plan

### Main Table 1: Definitions and regime signatures

Columns:

- quantity;
- mathematical definition;
- interpretation;
- expected sign in each regime.

This table should include `C_CT`, `C_CO`, `C_TO`, global polarization, and frustration.

### Appendix Table A1: Simulation parameters

Include every value and distribution required for reproduction.

### Appendix Table A2: Discrimination patterns

Retain D1 through D6 with an unambiguous emitter/receiver orientation and a short verbal description.

### Appendix Table A3: Robustness summary

For each perturbation, report whether the same regimes appear and how the transition location changes.

---

## 9. Rewrite map from the current manuscript

| Current material | Action | Destination |
|---|---|---|
| Title | Replace with a mechanism-focused title | Main paper |
| Abstract | Rewrite from zero; remove LLM text and Portuguese placeholder | Main paper |
| Darwin quotation | Delete | None |
| Broad catalogue of human discrimination | Compress to two or three sentences | Introduction |
| LLM approach in the introduction | Delete | None |
| Prior neural-agent and entropic-dynamics background | Condense and cite clearly | Related work and model |
| Anthropological and hominin speculation | Delete unless supported and essential | None |
| Full Bayesian/entropic derivation | Short summary only | Appendix A |
| Modulation-function surfaces | Keep only the mechanism-relevant view | Main Figure 1; remainder in appendix |
| Six discrimination matrices | Use D6 in main text; test all in supplement | Appendix B |
| Equation `h_w^D = h_w + D_e|r` | Promote to the central model equation | Main paper |
| Current phase diagram | Recompute with seeds, uncertainty, and explicit regime rule | Main Figure 2 |
| Current order parameters | Correct class encoding and simplify notation | Main paper |
| Current "spin glass" label | Diagnose properly or rename | Main paper |
| Current balance/frustration section | Redefine quantities and add first-passage times | Main paper and appendix |
| Empty phase-diagram subsections | Replace with full results narrative | Main paper |
| Unfinished LLM society section | Delete entirely | None |
| Discussion | Rewrite around ML implications, scope, and limitations | Main paper |
| Conclusions | Write a concrete one-paragraph conclusion | Main paper |
| Literature-review comments in Appendix A | Convert into a real related-work section or delete | Main paper/references |
| Incomplete citations and `??` references | Resolve all | Entire paper |
| Author contributions and acknowledgments | Keep only in non-anonymous final version | Camera-ready |

---

## 10. Related-work strategy

The present bibliography mostly supports the historical development of the learning rule. The rewrite needs a broader and more current related-work section. Organize it by conceptual relation, not by a long chronological list.

### 10.1 Online Bayesian and entropic neural learning

Explain which update equations are inherited, what was previously established, and what is new in this paper.

### 10.2 Neural-agent opinion and trust dynamics

Position the coupled opinion-trust model relative to neural-agent societies, adaptive signed networks, and coevolving beliefs and ties.

### 10.3 Group polarization and structural balance

Connect the triangle-balance measures and polarization dynamics to structural-balance and signed-network work. Clarify whether the paper studies equilibrium, transient dynamics, or both.

### 10.4 Fairness, sensitive attributes, and spurious features

Frame the discrimination field as a task-irrelevant group-correlated representation feature. Distinguish this model from conventional supervised fairness settings: the central outcome here is not only prediction disparity but endogenous population structure.

### 10.5 Collective effects of local learning rules

Position the work within research on how individually sensible learning or optimization rules can generate undesirable system-level outcomes.

The related-work search must be done before finalizing novelty claims. Do not state that the mechanism or phase diagram is the first of its kind until that search is complete.

---

## 11. Reviewer threat model

### Objection 1: "The novelty is only adding a scalar bias to an old model."

**Mitigation:** Derive the bias from an explicit representation model, provide a symmetry-breaking analysis, establish class-correlation controls, and show a robust population-level phase structure that was not present in the earlier model.

### Objection 2: "The phase diagram is based on arbitrary visual labels."

**Mitigation:** Define order parameters, phase-classification rules, uncertainty, and boundary estimation before presenting the categorical map.

### Objection 3: "The result is just extra noise."

**Mitigation:** Add same-magnitude uncorrelated-noise and shuffled-class controls.

### Objection 4: "The model has little connection to modern machine learning."

**Mitigation:** Center the paper on representation error, online probabilistic learning, co-adaptive trust, and system-level consequences. Avoid presenting it mainly as a sociological analogy.

### Objection 5: "The paper overclaims implications for humans."

**Mitigation:** State that this is a mechanistic computational model. Remove evolutionary speculation and historical claims that are not tested.

### Objection 6: "There is no theory behind the phase boundaries."

**Mitigation:** Provide at least a local symmetry-breaking analysis. Preferably add a mean-field instability or finite-size scaling result. If that is not possible, use "regimes" and "crossovers" rather than "thermodynamic phases."

### Objection 7: "The experiments are not reproducible."

**Mitigation:** Release anonymous code, configurations, seeds, parameter tables, and scripts that regenerate every figure.

### Objection 8: "The order parameters do not measure what the text claims."

**Mitigation:** Replace the 0/1 class matrix with a signed same/different-class relation and validate each statistic on synthetic sanity-check configurations.

### Objection 9: "The agenda-complexity claim is qualitative."

**Mitigation:** Define affective and ideological polarization times, show the crossover statistically, and test threshold sensitivity.

---

## 12. Sanity checks for the mathematics and implementation

Before running the full experiment grid, create unit tests for the following cases.

### Order-parameter tests

1. Same-class agents trust each other, cross-class agents distrust each other, and opinions align by class. Expected: `C_CT`, `C_CO`, and `C_TO` all near `+1`.
2. Trust and opinions form two factions unrelated to class. Expected: global polarization high, class correlations near zero.
3. Everyone trusts everyone and has aligned opinions. Expected: `C_TO` high, class correlations near zero in a balanced class population.
4. Same-class agents distrust each other and trust the other class. Expected: `C_CT` negative.
5. Random weights and random directed trust. Expected: all signed correlations near zero as `N` grows.

### Discrimination-field tests

1. Under D6, verify that same-class pairs receive `-d` and cross-class pairs receive `+d`.
2. Verify that only discriminating receivers apply the shift.
3. Verify the emitter/receiver orientation in code and notation.
4. Verify that `d = 0` exactly recovers the non-discriminating implementation.
5. Verify invariance under swapping the names of classes A and B.

### Update-rule tests

1. Check numerical derivatives of `log Z` against implemented modulation functions.
2. Check covariance and variance updates for positive semidefiniteness.
3. Check limiting behavior for strong agreement, strong disagreement, strong trust, and strong distrust.
4. Check reproducibility under fixed seeds.
5. Check that simulation conclusions are not numerical overflow or clipping artifacts.

---

## 13. Title options

Preferred:

1. **Collective Discrimination from Class-Correlated Representation Errors in Interacting Neural Agents**

Alternatives:

2. **From Local Inference to Collective Discrimination in Neural-Agent Populations**
3. **Group-Correlated Representation Errors Break Symmetry in Interacting Neural Learners**
4. **Emergent Discrimination in Co-Adapting Opinion and Trust Networks**
5. **Class-Dependent Representation Bias in Societies of Neural Agents**

The preferred title states the mechanism and the collective outcome without implying direct human validation.

---

## 14. Abstract blueprint

Do not reuse the current abstract. Write a new abstract using this six-sentence structure:

1. **Problem:** Interacting learners may adapt not only their predictions but also their trust in information sources, allowing representation errors to propagate through a population.
2. **Model:** We study neural agents whose perceptron opinions and directed trust variables co-adapt through an entropic approximation to Bayesian online learning.
3. **Intervention:** A fraction `f_d` of agents incorporates an issue-irrelevant, class-correlated feature that shifts the field used in its update by strength `d`.
4. **Mechanism/result:** The shift breaks the symmetry between consonant trust-opinion states and generates distinct class-uncorrelated, frustrated, and discriminatory collective regimes.
5. **Temporal result:** Varying `P/K` changes whether affective or ideological polarization emerges first.
6. **Scope:** These results identify a mechanism by which group-correlated representation errors can create population-level harm even when the underlying local update is motivated by probabilistic inference.

Every result sentence must be updated after the final experiments so it contains quantitative evidence rather than qualitative adjectives.

---

## 15. Writing rules for the rewrite

1. Define every symbol before use.
2. Use one notation for distrust mean, distrust field, and trust score.
3. Use "receiver" and "emitter" consistently, or replace them with "learner" and "source" throughout.
4. Distinguish opinion agreement from task accuracy. The model has no external ground-truth social opinion unless one is explicitly introduced.
5. Distinguish class correlation from polarization.
6. Distinguish balance from low frustration.
7. Distinguish a finite-population crossover from a thermodynamic phase transition.
8. Mark inherited equations and new equations explicitly.
9. Avoid cognitive or psychological terms such as "blame" unless they are introduced as metaphors and not measurable claims.
10. Avoid human or political causal claims that the simulations do not test.
11. Use captions that state the experiment, parameters, statistic, aggregation over seeds, and conclusion.
12. Remove every placeholder, editorial comment, missing citation marker, and unresolved section reference.
13. Use sentence case for the title and headings.
14. Keep the main text focused on the mechanism and evidence. Move derivations and exhaustive variants to appendices.

---

## 16. Appendix architecture

### Appendix A. Derivation of the entropic update

- probabilistic model;
- likelihood and evidence;
- Gaussian projection;
- weight, covariance, distrust, and variance updates;
- modulation functions;
- symmetry identities;
- numerical derivative checks.

### Appendix B. Discrimination representations

- augmented representation derivation;
- D1 through D6 matrices;
- emitter/receiver convention;
- verbal interpretation of each matrix.

### Appendix C. Order parameters and phase classification

- full definitions;
- synthetic sanity checks;
- thresholds or clustering procedure;
- sensitivity analysis.

### Appendix D. Simulation details

- pseudocode;
- parameter tables;
- stopping rules;
- compute resources;
- random seeds;
- code structure.

### Appendix E. Additional phase diagrams

- different `N`, `K`, `P`, class balance, initialization, and learning rule;
- all D matrices;
- negative `d`;
- uncertainty maps.

### Appendix F. Agenda-complexity analysis

- first-passage threshold sensitivity;
- full trajectories;
- alternative temporal metrics;
- finite-size results.

### Appendix G. Limitations and additional ethics discussion

- interpretation limits;
- potential misuse or overgeneralization;
- relation between synthetic class labels and protected attributes;
- why the model should not be used to infer properties of real demographic groups.

---

## 17. Go or no-go criteria

The paper is ready for ICLR submission only if all of the following are true:

- [ ] The new contribution can be stated without describing the inherited entropic-learning derivation as novel.
- [ ] The discrimination field is derived from an explicit class-correlated, task-irrelevant representation feature.
- [ ] The signed class relation and all order parameters are mathematically correct and unit-tested.
- [ ] Main phase diagrams aggregate enough independent seeds and report uncertainty.
- [ ] The phase classification rule is explicit and robust.
- [ ] At least one finite-size analysis is complete.
- [ ] Uncorrelated-noise and shuffled-class controls are complete.
- [ ] The agenda-complexity crossover is quantified with polarization times and confidence intervals.
- [ ] At least a local symmetry-breaking analysis is included.
- [ ] "Spin glass" and "phase transition" are either supported by diagnostics or replaced with safer terminology.
- [ ] All central claims are visible in the 9-page main text.
- [ ] The related-work section establishes novelty relative to current fairness, polarization, signed-network, and multi-agent-learning research.
- [ ] Code, configurations, seeds, and figure scripts are available anonymously.
- [ ] The manuscript is fully double blind.
- [ ] The AI-use, ethics, and reproducibility statements are complete.
- [ ] Every placeholder, unresolved reference, Portuguese editorial note, and incomplete section has been removed.

If the work reaches only "remove the LLM section and polish the English," it does not meet this go criterion.

---

## 18. Suggested execution schedule for ICLR 2027

Official deadlines are September 18, 2026 for the abstract and September 25, 2026 for the full paper, both Anywhere on Earth.

### August 13 to August 17: Freeze the scientific story

- agree on the one-sentence thesis;
- decide whether D6 is the main model;
- correct notation and order parameters;
- audit the current code against the equations;
- define the exact experiment matrix and compute budget.

### August 18 to August 27: Run the central experiments

- baseline phase diagrams with seeds;
- uncorrelated-noise and shuffled-label controls;
- finite-size sweep;
- representative trajectories;
- first-passage analysis for agenda complexity.

### August 28 to September 3: Complete theory and robustness

- local symmetry-breaking analysis;
- alternative learning-rule ablation;
- D1 through D6 appendix runs;
- initialization and class-imbalance checks;
- finalize regime classification.

### September 4 to September 10: Write the new paper

- write abstract and introduction from zero;
- write compact model and mechanism sections;
- write results around finalized figures;
- move derivations into appendices;
- write limitations, ethics, and reproducibility statements.

### September 11 to September 15: Internal review

Ask at least three readers to act as:

1. an ICLR theory reviewer;
2. an empirical ML reviewer;
3. a fairness or computational-social-science reviewer.

Have each reader answer:

- What is the novel contribution?
- Which claim is least supported?
- Is the model connected clearly enough to machine learning?
- Which figure is unnecessary?
- What would make you reject the paper?

### September 16 to September 18: Abstract submission

- finalize title and genuine abstract;
- freeze author list before the abstract deadline;
- verify all OpenReview profiles;
- verify reciprocal-reviewing eligibility;
- submit the abstract.

### September 19 to September 24: Finalization

- complete final checks and anonymous code package;
- verify 9-page limit;
- verify anonymity in PDF metadata, code, acknowledgments, and repository history;
- rerun all figure-generation scripts from a clean environment;
- proofread equations, captions, and cross-references.

### September 25: Full submission

Submit early enough to allow technical validation of the uploaded PDF and supplement.

---

## 19. Minimum viable and strong versions

### Minimum viable ICLR version

- corrected model and order parameters;
- class-correlated representation derivation;
- replicated phase diagrams with uncertainty;
- uncorrelated-noise and shuffled-label controls;
- finite-size analysis;
- quantified agenda-complexity crossover;
- local symmetry-breaking analysis;
- strong limitations and reproducibility package.

### Strong ICLR version

Everything above, plus:

- mean-field or stability prediction for a transition boundary;
- scaling analysis near the transition;
- robust results across several discrimination matrices;
- comparison across learning rules;
- sparse-network or topology generalization;
- a sharper result explaining when class-dominant distrust becomes independent of ideological similarity.

---

## 20. Final positioning

The manuscript should end as a machine-learning paper about representation and interacting learning systems, not as a broad theory of human prejudice.

The strongest defensible message is:

> A task-irrelevant feature can have consequences beyond individual prediction bias. When learners also infer whom to trust, a class-correlated representation error changes the information flow of the whole population, producing group-aligned collective structure and altering the order in which affective and ideological polarization develop.

That message is within ICLR's scope. Its acceptance case depends on isolating the mechanism, repairing the observables, adding rigorous controls and uncertainty, and showing that the collective findings are new rather than merely a visual consequence of inserting a class-dependent scalar into an existing model.
