# icrl2027-rewrite2

An independent reframing of `Discrimination2025(1).pdf` for ICLR, derived from the
source PDF alone. Structure and argument only — no new simulations were run. All
figure slots point at output already in `../nn-based-simulation/figures/iclr/`.

## What is here

| file | what it is |
|---|---|
| [`REFRAMING.md`](REFRAMING.md) | The argument. What the ICLR-legible claim is, what to cut, what must be added, and the honest risk assessment. **Read this first.** |
| [`FIGURES.md`](FIGURES.md) | Every figure slot in the skeleton, mapped to an existing file or marked as a new experiment. |
| `paper/main.tex` | Compiles the skeleton. Drops in the official ICLR style if present, falls back to `article` if not. |
| `paper/sections/*.tex` | The restructured paper. Real prose where the source supports it; `\todo{}` where an experiment is missing. |

## The claim, in one sentence

Each agent maintains a **learned per-source reliability estimate** and uses it to
weight incoming supervision — and that estimate is exactly the mechanism that
converts a mild, individually harmless spurious feature into a population-wide
group split with a sharp threshold.

Per-source reliability weighting is standard equipment in ML: Dawid–Skene and
crowdsourcing, annotator modelling in RLHF, client reweighting in federated
learning, Byzantine-robust aggregation, peer weighting in decentralised SGD. It is
always introduced as the thing that makes learning *robust to bad sources*. This
paper says it is also the channel that turns a proxy feature into persistent group
structure.

## Two vocabulary changes from the source draft

Applied throughout, deliberately:

- The `w` sector is the **opinion sector**, never "ideological". It holds a
  classifier; "ideology" invites a reading the model does not support and
  collides with the `I` subscripts used for agent indices.
- The `μ` sector is the **trust sector**, never "affective". Its technical name
  is a per-source reliability estimate, and that is the name that makes the
  paper legible to this venue.

This renames the balance order parameters: the source's `B_I` (ideological) and
`B_A` (affective) become **`B_ρ`** and **`B_η`**, named after the quantity each is
built from. The figures currently in `nn-based-simulation/figures/` still carry the
old axis labels; see [`FIGURES.md`](FIGURES.md).

## Build

```bash
cd paper
# optional: cp /path/to/iclr2027_conference.{sty,bst} .
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures resolve through `\graphicspath` to `../../nn-based-simulation/figures/iclr/`.
Nothing needs to be copied.

## Relationship to the other directories

This is a *parallel* proposal, not a replacement for `../iclr2027-rewrite/`. It was
written without reading that directory, on request, so that the framing is an
independent read of the source rather than a revision of an existing one. Compare
and take what is useful.

The folder name is spelled as requested (`icrl`, not `iclr`) — say the word and
it gets renamed.
