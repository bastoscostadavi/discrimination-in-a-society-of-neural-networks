# Notes on the ICLR 2027 submission

Working notes, not part of the paper.

## Structure

Presentation order, per the author's revision:

1. Introduction — 2. Related work — 3. Polarization in a society of learning agents
(3.1 microscopic variables, 3.2 dynamics, 3.3 emergent polarization, 3.4 the
agenda-size effect) — 4. Discrimination in a society of learning agents (4.1 an
error of representation, 4.2 the phase diagram) — 5. Discussion.

Two sections, one per phenomenon. §3 defines the model and then immediately says
what it does unaided; §4 adds the bias and maps the consequences. The alternative —
model, then unbiased results, then bias, then biased results — split each story
across a section boundary. The agenda-complexity result belongs in §3 because it is
measured at `d = 0`: it is a property of the model, not of the bias.

Order parameters arrive where first needed rather than in a block: `rho`, `eta`,
`B_I`, `B_A` open §3.3; the three correlations (`R_wmu`, `R_muc`, `R_cw`) sit at the
end of §4.1, since all three are read off the phase diagram together and `R_wmu` is
not needed to establish that the society polarizes.

Terminology: the sector is the **opinion** sector, never "ideological". The symbols
`B_I` and `b^I` keep their subscript for continuity with the source draft, with a
parenthetical in §3.3 noting it is historical.

**The paper is a single file.** `main.tex` contains every section inline; there is
no `sections/` subfolder, and the template's `math_commands.tex` is not loaded
either (the paper uses none of its 427 macros, and `\eqref` comes from amsmath).
The only other files the build needs are `references.bib`, the two
`iclr2027_conference` template files, `fancyhdr.sty`, and `figures/`.

## Orientation of the phase maps

`f_d` increases **upwards**, which is the opposite of the source draft. The draft's
maps put `f_d = 0` at the top, which is `imshow`'s row-major default rather than a
choice, and it makes the quorum result read backwards: the threshold at `f_d ~ 0.4`
appears as a band creeping up from the bottom instead of a line one crosses going up.
`sweep()` still returns `(n_fd, n_d)` arrays with row 0 at `f_d = 0`; the flip lives
in `phase_map`, which draws them with `origin="lower"`, and `tests/test_plotting.py`
asserts that the origin and the extent agree.

**Flag this to the coauthors.** Our maps are vertical mirrors of the ones in the
companion manuscript, so a reader comparing the two side by side will misread them
unless told.

## Figure placement, and why

Pagination is float-limited, not text-limited: cutting 400 words of prose moved the
body boundary by 17 words, because the figures pin to page tops and text reflows
around them. Only figure area buys pages. The main text therefore carries the four
figures that do the most work per unit of space:

- `modulation_slices` (§3.2) — the sector crossover, and the one thing the
  discrimination field acts on.
- `polarization` (§3.3) — the two histograms that show the split.
- `agenda_trajectories` (§3.4) — the `alpha` result.
- `order_parameter_maps` (§4.2) — all five order parameters, both agendas.
- `phase_diagram` (§4.2) — the two composites side by side.

Both agendas stay in the body, because the difference between them *is* one of the
results: only `α < 1` has region (IV). Both balances stay too, because `B_A` is the
only quantity that separates region (I) from an absence of structure, and the
composite is exactly where that information is lost — see below.

**Why the composite carries three order parameters and not five.** Because an RGB
image has three channels. There is no principled reason beyond that, and the cost is
real: region (I) is black in the composite because all three correlations vanish
there, so the composite cannot tell frustration from nothing-happening. `B_A < 0` is
what does, and it appears only in `order_parameter_maps`. The composite is a
classification device with a known blind spot, not a summary of the measurements.

Every correlation map runs white-to-saturated across the whole of its range,
including the signed `R_muc`. A diverging map reads better in isolation, but these
three panels are also the three colour channels of the phase diagram, whose red
channel is exactly this white-to-red ramp over `[-1, 1]`; a blue negative arm made
the panel disagree with the composite it feeds.

Three moved to the appendix, in this order of reluctance: `modulation_contours`
(`app:modulation`, a visualization of equations already displayed);
`learning_flows` (`app:flows`, which carries the mechanism and would be the first
thing to bring back if space appeared); `sign_convention_comparison`
(`app:sign`, documentation rather than result).

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
  anonymized. Restore the author block, and add the FAPESP 2024/18736-8 and TELUS
  Digital Research Hub acknowledgements, only for the camera-ready. Note the
  GitHub repository is public and under a personal account: linking it from the
  submission would deanonymize.
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
- **Title.** The source draft's title is kept: "Discrimination in a Society of
  Neural Networks". At the template's `\LARGE` it wraps to a second line and
  hyphenates ("Neural Net-works"), so `main.tex` steps it down one size to
  `\Large`, which fits it on one line. That is the only deviation from the
  template's typography; revert it if a future title is shorter.
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
  indicator, and the `R_cw` normalization the draft's Eq. 29 caps at 1/2. Both
  alternatives are implemented (`app:conventions`).

## Remaining

- [ ] Decide the venue question above.
- [ ] Optionally run the Hebbian/perceptron ablation.
- [ ] Replace the placeholder ICLR style files when the official ones appear.
- [ ] Camera-ready only: author block, acknowledgements, real citation for the
      companion manuscript.
