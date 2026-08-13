# Notes on the ICLR 2027 submission

Working notes, not part of the paper. Two sections: what was decided in adapting
the source material to this venue, and what remains to be filled.

## Venue mechanics

- **Template.** ICLR has not released a 2027 style file as of writing. This
  directory carries the ICLR 2026 template with every occurrence of "ICLR 2026"
  replaced by "ICLR 2027" (`iclr2027_conference.sty`, `.bst`). **Replace both
  with the official 2027 files when they appear** and re-check the page limit,
  which has been 9 pages of main text plus unlimited references and appendix.
- **Anonymity.** `\iclrfinalcopy` is commented out, so the submission builds
  anonymised. The author block currently reads "Anonymous authors"; restore the
  real block, and add the FAPESP 2024/18736-8 and TELUS Digital Research Hub
  acknowledgements, only for the camera-ready.
- **The companion manuscript is cited anonymously.** `references.bib` has
  `caticha2026discrimination` as `{Anonymous}`, "companion manuscript, under
  review", because citing the source draft by author would deanonymise the
  submission. Restore the full citation for the camera-ready. If the draft is
  posted to arXiv before submission, cite it normally in third person instead.
- **Required sections.** Reproducibility statement and ethics statement are
  present, before the bibliography, per ICLR practice.

## Adaptation decisions

- **Reframed from statistical mechanics to multi-agent learning.** The source
  draft opens on discrimination in society and treats the neural-network society
  as the object of study. Here the opening question is whether discriminatory
  structure can emerge in a population of learning agents from the learning rule
  alone, and the phase diagram is presented as a *prediction* about societies of
  interacting learners that the language-agent study then tests. The physics is
  intact; the framing, the contribution list, and the discussion are new.
- **Title.** "Discrimination as a Phase of Learning: Emergent In-Group Trust in
  Societies of Adaptive Agents". The draft's title names the substrate; this one
  names the phenomenon and the claim.
- **Compression.** Equations 1-20 of the draft become the model section's
  essentials plus `app:derivation`. The six discrimination matrices become one
  displayed matrix in the main text plus `app:cases`. The modulation-function
  contours and the sector-crossover cuts merge into one main-text figure; the
  3D surfaces are generated but unused, and the flow-field panels moved to
  `app:flows` when the page limit bit.
- **Two figures the draft refers to but does not contain.** Its text discusses
  the crossover between sectors at `h_mu = h_w` ("figure ??") — that is now the
  right-hand panel of the modulation figure. Its text also promises histograms of
  the per-agent class-trust `u_I` ("Panel blabla"); the helper exists in the code
  (`class_trust_per_agent`) but no figure uses it yet. Add it if space allows.
- **Sign convention corrected throughout.** See
  `../docs/discrimination-field-sign.md` and `app:sign`. The paper states the
  convention explicitly the first time `d` appears, which the draft never does.
- **Cleaned up.** All Portuguese TODOs removed; all `??` cross-references
  resolved; the draft's empty `[]` citations filled from
  `references.bib`. Both the Darwin epigraph and the Lincoln
  quotation are dropped: the agenda-complexity result carries the latter's point
  directly, and the former was the cheapest thing to cut for the page limit.
- **Order-parameter conventions** stated rather than left implicit: signed class
  indicator, and the `R_cw` normalisation that the draft's Eq. 29 caps at 1/2.
  Both alternatives are implemented in the code (`app:conventions`).

## An unverified claim, deliberately weakened

The source draft asserts that simple learning rules — Hebbian, perceptron — do not
produce the discriminatory phases, but it does not show this, and neither do we.
`sections/results.tex` therefore makes only the argument that can be made without
the experiment: a rule with no affective sector has no `F_mu` for the class label
to bias, and under a plain Hebbian update the discrimination field has no effect
whatsoever because the update does not depend on `h_w`.

Running the actual control is a genuinely worthwhile addition and is not
expensive: add a `rule` option to `SocietyBatch` selecting between the entropic
update, a Hebbian update, and a perceptron update, then re-run one sweep per rule
at `--preset quick`. Flat maps would turn a paragraph of reasoning into a
measurement, and reviewers at an ML venue are likely to ask for exactly this
ablation. It is scoped out of the current pass only because it goes beyond
reproducing what the draft reports.

## To fill

- [ ] **The language-agent study** (`sections/llm.tex`). The protocol,
  measurement and failure-mode discussion are written; `\llmtodo{...}` marks
  every hole. Specifically: the ladder of instruction strengths mapping to `d`,
  the group-label scheme and its swap control, the results subsection, and
  `app:llm-prompts`.
- [ ] **Replace the two placeholder figures and uncomment the figure block.**
  `figures/placeholder_llm_*.pdf` are generated by `make_placeholders.py`. The
  figure environment in `sections/llm.tex` is **commented out**, with a marked
  RESERVED SLOT block: a figure with no data in it does not earn space against a
  9-page limit. Uncomment it when the study runs and pay for the space from the
  reserve list above.
- [ ] **Abstract**: two `\llmtodo` spans for the empirical result.
- [ ] **Intro contribution 4**: one sentence of findings.
- [ ] Set `\llmtodofalse` in `main.tex` once no notes remain, and confirm nothing
  purple is left in the PDF.
- [ ] Re-run `make sim PRESET=full` for the final figures.
- [x] **Page count.** Done: the body now ends within 9 pages (the reproducibility
  statement starts partway down p9), references on p10, appendix to p17. Getting
  there took: the two map figures merged into one 2x5 `order_parameter_maps`; the
  learning-flow figure and the on-line-learning genealogy moved to the appendix;
  the LLM protocol detail moved to `app:llm-prompts`; the derivation sketch cut to
  its essentials; captions, abstract, discussion and intro tightened; the Darwin
  epigraph dropped.
  **Reserve list for when the LLM results land** (pagination is float-limited, so
  prefer removing figure area over prose): (i) uncomment the LLM figure in
  `sections/llm.tex` and, to pay for it, either drop the `modulation_slices` panel
  or move the whole modulation figure to the appendix; (ii) the six-case table is
  already in the appendix; (iii) `phase_diagram_large_agenda` is generated but not
  included anywhere -- it is available if a reviewer asks.