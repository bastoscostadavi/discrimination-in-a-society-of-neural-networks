# Discrimination in a society of LLMs

Can a population of learning agents sort itself into mutually distrustful groups
along a label that carries no information — with no biased data and no group-level
preference anywhere in the system?

This repository answers that for a society of agents whose learning rule can be
written down and analyzed, and maps the result as a phase diagram.

| | |
|---|---|
| [`nn-based-simulation/`](nn-based-simulation/) | A society of perceptron agents learning from each other under an optimal on-line rule that carries an explicit, dynamical trust for every other agent. Analyzable, and mapped as a phase diagram in the strength `d` of a class-correlated bias and the fraction `f_d` of agents carrying it. |
| [`docs/llm-study-contract.md`](docs/llm-study-contract.md) | A language-agent version of the same experiment, specified but **shelved**. Kept because the order parameters were built to make it possible later. |
| [`paper-iclr2027/`](paper-iclr2027/) | The ICLR 2027 submission drawing on both, with the language-agent section scoped and its figure slots reserved. |
| [`docs/`](docs/) | The model as implemented, and the discrepancies found in the source material. |
| [`paper-iclr2027/NOTES.md`](paper-iclr2027/NOTES.md) | Every venue-adaptation decision, and what is still to fill. |

The starting point is `Discrimination2025(1).pdf` (Caticha et al.), which sets up
the model and reports the phase diagram. It is the guide for the work here, not
its organizing principle: the simulation stands on its own, and where a literal
reading of the draft does not work, the code follows the model and the
discrepancy is documented rather than silently reproduced.

## Start here

```bash
cd nn-based-simulation
pip install -r requirements.txt
python scripts/make_all.py --preset quick    # all figures, a few minutes
pytest                                       # 51 tests
```

Then [`nn-based-simulation/README.md`](nn-based-simulation/README.md) for what
each figure shows and where every parameter comes from, and
[`docs/model.md`](docs/model.md) for the equations.

## Two findings worth knowing before reading anything else

**The discrimination field's sign is inconsistent in the source draft.** Its
Eq. 25 and Table I, read literally, place the discriminatory phases at `d < 0`,
while its text and all of its figures place them at `d > 0`. One global sign
reconciles them; we use the reading the algorithm forces, in which `d > 0` means
in-group tolerance and out-group intolerance.
[`docs/discrimination-field-sign.md`](docs/discrimination-field-sign.md) works it
through, shows both versions side by side, and lists what needs fixing in the
draft.

**The draft states no simulation parameters.** No society size, no agenda size, a
literal `Δt = ????` for the measurement time. One number, the embedding dimension
`K = 30`, is recoverable from the `α = P/K` labels of its trajectory figure; the
rest are choices, and they are calibrated against features of the published
figures by `nn-based-simulation/scripts/calibrate.py` and tabulated with their
provenance rather than left implicit.
