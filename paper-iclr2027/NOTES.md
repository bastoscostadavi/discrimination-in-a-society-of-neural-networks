# Notes on the ICLR 2027 submission

Working notes, not part of the paper.

## Status

The language-agent study was called off, and the paper was reworked to stand on
the neural-network society alone. That was not a deletion: the abstract, the
introduction's framing and contribution list, the related-work positioning, the
order-parameter section's closing argument, the discussion's limitations, and the
ethics and reproducibility statements all made promises about a second substrate
and had to be rewritten. What remains claims only what the simulation shows.

Body is within the 9-page limit, references and appendix follow, and the PDF has
no unresolved markers. All figures are at the `full` preset (200×200, matching the
draft's grid).

## Venue fit — worth a decision before submitting

The paper is now a statistical-mechanics result about societies of learning
agents, with no language-model experiment. That is a coherent contribution, but it
is further from ICLR's centre of gravity than the two-substrate version was, and a
reviewer may reasonably ask what makes it a *learning representations* paper. The
current answer, made explicitly in the introduction and discussion, is:

- the effect is a property of the learning rule, and specifically of the sector
  that estimates other agents' reliability — so it is a hazard for any multi-agent
  system whose members model each other;
- it is invisible to per-agent evaluation, because the individual bias is `O(d)`
  while the population state is discontinuous in `d`;
- the order parameters are substrate-independent and cheap, so they are usable as
  a population-level audit.

Three options if that feels thin:

1. **Run the Hebbian/perceptron ablation** (below). It is the cheapest way to add
   an empirical result and it directly supports the "specific to good learners"
   claim, which is currently argued rather than measured.
2. **Reinstate a reduced language-agent experiment** — even one condition at one
   `f_d`, showing the sign of `R_μc` responds to instructed tolerance, would
   restore the second substrate at a fraction of the cost.
3. **Retarget** to a venue where the result sits more naturally (a complex-systems
   or computational-social-science venue), where it would be a strong fit as is.

## The one unmeasured claim

The draft asserts that simple learning rules — Hebbian, perceptron — do not produce
the discriminatory phases. We now say plainly in the discussion that we argue this
rather than measure it. The argument is sound as far as it goes: the effect acts
through `F_mu`, and a rule with no affective sector has none to bias; under a plain
Hebbian update the discrimination field has no effect whatever, since the update
does not depend on `h_w`.

Running it is cheap: add a `rule` option to `SocietyBatch` selecting between the
entropic update, Hebbian, and perceptron, then one sweep per rule at
`--preset quick`. Flat maps would turn a paragraph of reasoning into a
measurement. This is the single most valuable addition left.

## Venue mechanics

- **Template.** ICLR has not released a 2027 style file. This directory carries
  the ICLR 2026 template with "ICLR 2026" replaced by "ICLR 2027"
  (`iclr2027_conference.sty`, `.bst`). **Replace both with the official files when
  they appear** and re-check the page limit.
- **Anonymity.** `\iclrfinalcopy` is commented out, so the submission builds
  anonymised. Restore the author block, and add the FAPESP 2024/18736-8 and TELUS
  Digital Research Hub acknowledgements, only for the camera-ready. Note the
  GitHub repository is public and under a personal account: linking it from the
  submission would deanonymise.
- **The companion manuscript is cited anonymously.** `references.bib` has
  `caticha2026discrimination` as `{Anonymous}`, "companion manuscript, under
  review". Restore the full citation for the camera-ready, or cite it normally if
  it is posted to arXiv first.

## Adaptation decisions

- **Reframed from statistical mechanics to multi-agent learning.** The draft opens
  on discrimination in society and treats the neural-network society as the object
  of study. Here the opening question is whether discriminatory structure can
  emerge from the learning rule alone, and the results are presented for what they
  say about building and evaluating multi-agent systems.
- **Title.** "Discrimination as a Phase of Learning: Emergent In-Group Trust in
  Societies of Adaptive Agents".
- **Compression.** Equations 1–20 of the draft become the model section's
  essentials plus `app:derivation`; the six discrimination matrices become one
  displayed matrix plus `app:cases`; the on-line-learning genealogy is in
  `app:genealogy`. The two map figures merged into one 2×5
  `order_parameter_maps`, which is a better figure as well as a smaller one.
- **A figure the draft refers to but does not contain.** Its text discusses the
  crossover between sectors at `h_mu = h_w` ("figure ??"); that is now the
  right-hand panel of the modulation figure. Its promised histograms of the
  per-agent class-trust `u_I` ("Panel blabla") are still unused — the helper
  exists (`class_trust_per_agent`) and there is now space for it if wanted.
- **Sign convention corrected throughout**, stated explicitly the first time `d`
  appears. See `../docs/discrimination-field-sign.md` and `app:sign`.
- **Cleaned up.** Portuguese TODOs removed, `??` references resolved, empty `[]`
  citations filled. The Darwin epigraph and the Lincoln quotation are both cut;
  there is room for the epigraph again if you want it back.
- **Order-parameter conventions** stated rather than left implicit: signed class
  indicator, and the `R_cw` normalisation the draft's Eq. 29 caps at 1/2. Both
  alternatives are implemented (`app:conventions`).

## Remaining

- [ ] Decide the venue question above.
- [ ] Optionally run the Hebbian/perceptron ablation.
- [ ] Replace the placeholder ICLR style files when the official ones appear.
- [ ] Camera-ready only: author block, acknowledgements, real citation for the
      companion manuscript.
