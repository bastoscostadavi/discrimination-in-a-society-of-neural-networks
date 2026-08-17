# What a language-agent study would have to produce

> **Shelved.** The language-agent study was called off; the paper now stands on
> the neural-network society alone. This note is kept because the interface it
> specifies is the hard part, and because the order parameters were deliberately
> defined to make such a study possible later. Nothing in the code or the paper
> depends on it.

`../nn-based-simulation/` predicts a phase diagram in two quantities: the strength
`d` of a bias correlated with an agent's group label, and the fraction `f_d` of
agents carrying it. The open question is whether a population of language-model
agents, instructed to varying degrees of tolerance towards another group, lands in
the same phases. This note fixes the interface such a study would need, so that
its numbers would be *comparable* with the simulation's rather than merely
analogous.

## The two matrices

For each condition — a point in `(d, f_d)`, at an agenda size `P` — a run must end
with two matrices over the `N` agents:

| quantity | shape | meaning | how |
|---|---|---|---|
| `rho[I, J]` | `(N, N)`, symmetric, unit diagonal | ideological overlap | elicit an opinion vector `v_I ∈ [-1,1]^P` over the agenda from each agent, then `rho_IJ = cos(v_I, v_J)` |
| `eta[r, e]` | `(N, N)`, **asymmetric**, unit diagonal | trust receiver `r` places in emitter `e` | ask each agent, for every other agent, how far it trusts that agent's statements; map to `[-1, 1]` |

Given those, the order parameters come straight from the existing code — nothing
needs reimplementing, and nothing should be. `ednna/order_params.py` currently
reads them off a `SocietyBatch` (`society.w`, `society.mu`, `society.V`,
`society.kappa`); the clean move is to factor its three correlation functions and
`_mean_triple_product` to take `(rho, eta, kappa)` directly, so that the identical
code path measures both societies. That refactor is small and worth doing before
the first LLM run rather than after.

## Requirements the comparison depends on

1. **Inert group labels.** Labels must name arbitrary partitions with no content
   related to any issue on the agenda, or the model retrieves associations it
   already holds about real social categories and the experiment measures those
   instead. Include a label-swap control: the order parameters are invariant under
   the swap by construction, so any dependence on it signals label-driven priors.
2. **Asymmetric trust.** Ask each agent separately; do not symmetrize. `B_A` is
   defined on the two cycle orientations, and symmetrizing destroys the
   distinction between organized hostility and frustration — which is exactly the
   distinction that identifies the spin-glass phase.
3. **Measurement outside the loop.** Elicit opinions and trust in separate passes
   that never enter any agent's interaction history, so measuring does not perturb
   the dynamics.
4. **Both agenda sizes.** The agenda-complexity reversal needs a simple agenda and
   a complex one. In the theory the crossover is at `α = P/K ≈ 1.7`, where `K` is
   the dimension of the representation; for language agents `K` is unknown, so `P`
   has to be varied as widely as the context budget allows and the crossover
   located empirically.
5. **Trajectories, not just endpoints.** Measure after every round. The `(B_I,
   B_A)` *path* is the result, and its position relative to the diagonal is what
   distinguishes distrust-first from disagreement-first.
6. **Repeats.** Every condition needs several independent runs. The theory's maps
   are single-realization and visibly noisy; one LLM run is not evidence of a
   phase.
7. **A ladder of `d` near zero.** The predicted transition is sharp at `d ≈ 0` and
   needs a quorum `f_d ≳ 1/3`. A ladder with only extremes will find a difference
   and miss the phase boundary, which is the actual claim.

## If it is ever revived

The paper's discussion names this as the obvious next question and says plainly
that it is unanswered. Reviving the study means writing a new section against the
protocol above; the figure slots and placeholder machinery that used to hold its
layout have been removed from `paper-iclr2027/`, so it would start clean.
