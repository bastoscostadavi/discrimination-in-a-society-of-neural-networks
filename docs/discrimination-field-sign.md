# The sign of the discrimination field

**Summary.** In the source draft (`Discrimination2025(1).pdf`), Eq. 25 and Table I,
read literally, contradict the draft's own text and every one of its figures.
The two readings differ by one global sign, and they are not cosmetically
different: they exchange the two halves of the phase diagram, so that the
discriminatory phases land at `d < 0` instead of `d > 0`. The simulation in
`nn-based-simulation/` and the paper in `paper-iclr2027/` both use the
consistent reading, in which **`d > 0` means intolerance toward the out-group**.
This note records the argument and the code path that reproduces both.

## What the draft says

Three statements are in play.

1. **Eq. 25** — a discriminating receiver evaluates the modulation functions at a
   shifted opinion field:

   $$h_w^D = h_w + D_{e|r}$$

2. **Table I** — for `d > 0`, the in-group entries of the six matrices carry
   `-d` and the out-group entries `+d`. For the fully symmetric case
   `D_6`, that is `D[A,A] = D[B,B] = -d` and `D[A,B] = D[B,A] = +d`.

3. **The text and the figures** — "For `d > 0`, A is more tolerant towards A in
   cases 1, 3, 4 and 6, and more intolerant towards B in cases 2, 3, 5 and 6";
   the phase diagram and the correlation maps place the discriminatory phases
   (high trust–class correlation) at `d > 0`, and the frustrated,
   reverse-discrimination region at `d < 0`.

Statements 1 and 2 together imply the *opposite* of statement 3.

## Why they conflict

The affective modulation function (Eq. 23) is

$$F_\mu(h_w, h_\mu) = \bigl(1 - 2\Phi(h_w)\bigr)\,\frac{g(h_\mu)}{Z},$$

and `mu` is *distrust*, updated as `mu <- mu + F_mu V / gamma_V`. The factor
`(1 - 2 Phi(h_w))` is negative whenever `h_w > 0`. So:

- **Perceived agreement (`h_w > 0`) drives `mu` down: it builds trust.**
- **Perceived disagreement (`h_w < 0`) drives `mu` up: it builds distrust.**

A receiver that adds `+d` to `h_w` therefore reads its counterpart as more
agreeable than it is and grows *more trusting* of it — the tolerant response. A
receiver that adds `-d` manufactures disagreement that was not there and grows
*more distrustful* — the intolerant response. Hence, under Eq. 25:

| entry in `D` | effect on the receiver | in-group entry in Table I |
|---|---|---|
| `+d` | more tolerant, builds trust | — |
| `-d` | less tolerant, builds distrust | `-d` for `d > 0` |

Table I gives the in-group the `-d` entry, which by the algorithm is the
*intolerant* one. Under a literal reading, a society with `d > 0` and case `D_6`
therefore ends up distrusting its own class and trusting the other — reverse
discrimination — and the discriminatory phases appear at `d < 0`.

The same conflict shows up in the draft's own description of its flow-field
figure. Adding `D` moves the separatrix of the learning flow to
`h_mu = h_w + D`. With `D = +d > 0` the separatrix moves *up*, enlarging the
"trust and agree" basin and shrinking "distrust and disagree". But the draft
states that "for `d > 0` the basin of attraction for consonance with *distrust
and disagree* increases, see right image of figure 4", where the right image is
labelled `D_{e|r} = d`. That is only true if the shift enters with the opposite
sign. Likewise the draft labels its `D_{e|r} = -d` panel "More tolerant", while
by the argument above `-d` is the intolerant entry.

So the draft has one sign error, expressed three times over: it is not the case
that the figures were produced with different code than the text describes, but
that the text's `+`/`-` assignment is inverted relative to the algorithm.

## The fix

Any one of the following resolves it; they are all the same single change.

- **Flip Table I**: in-group entries `+d`, out-group entries `-d`. *(What we
  do.)*
- Keep Table I and write Eq. 25 as `h_w^D = h_w - D_{e|r}`.
- Keep both and redefine `d < 0` as the discriminatory direction, relabelling
  every figure axis.

We take the first because it leaves Eq. 25 — the equation that follows from the
model — untouched, and preserves the meaning of the axis in every published
figure. In `nn-based-simulation/ednna/discrimination.py` the templates are
therefore written with `+1` on the tolerant entry:

```python
_TEMPLATES = {                    # rows: receiver's class, cols: emitter's class
    6: [[+1.0, -1.0],
        [-1.0, +1.0]],            # both classes favour their own, are hostile to the other
    ...
}
```

with `field_matrix(d, case=6)` returning that template times `d`.

## Seeing the difference

`field_matrix(..., literal_draft=True)` negates the matrix, reproducing the
draft's literal Table I under the draft's own Eq. 25. Every layer of the code
exposes the switch (`ModelConfig(literal_draft_sign=True)`), and

```
python scripts/sign_convention_comparison.py
```

runs the same sweep under both and writes
`figures/<style>/sign_convention_comparison.pdf`: the trust–class correlation
`R_mu,c` and the affective balance `B_A` over the `(d, f_d)` plane, once per
convention. The two rows are mirror images in `d`. Only the top row matches the
maps printed in the draft.

Numerically, at `N = 40`, `K = 30`, `P = 5`, `f_d = 0.9`, case `D_6`:

| convention | `R_mu,c` at `d = -0.8` | `R_mu,c` at `d = +0.8` |
|---|---|---|
| consistent (used here) | `-0.82` | `+0.89` |
| draft, literal | `+0.89` | `-0.82` |

The draft's printed maps show `R_mu,c` strongly positive on the right-hand half
of the plane, which is the first row.

## What this means for the physics

Nothing. The model, the phase diagram and every conclusion of the draft are
unaffected — the two discriminatory phases, the spin-glass region of reverse
discrimination, the neutral region, and the agenda-complexity result are all the
same objects. What changes is only which half of the `d` axis is labelled
"discriminatory", and the draft's figures already make the intended choice. The
correction needed is in the draft's Table I (or, equivalently, one sign in
Eq. 25) plus the two figure-4 panel labels and the sentence about basins of
attraction.

## Checklist for the source draft

- [ ] Table I: swap the signs, so in-group entries are `+d` and out-group `-d`
      for `d > 0` (or negate Eq. 25 instead).
- [ ] Figure 4 panel labels: `D_{e|r} = -d` is the *less* tolerant case,
      `D_{e|r} = +d` the *more* tolerant one.
- [ ] The sentence "For `d > 0` the basin of attraction for consonance with
      'distrust and disagree' increases, see right image of figure 4" — with the
      corrected table this is the `D_{e|r} = -d` panel, i.e. the response to an
      out-group emitter.
- [ ] Add one sentence stating the convention explicitly, e.g. "throughout,
      `d > 0` denotes tolerance toward in-group and intolerance toward
      out-group emitters", so the reader never has to reconstruct it.
