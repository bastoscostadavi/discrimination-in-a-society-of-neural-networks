# $F_\mu$ under LLM in-context learning

Figure 1 of the paper says something about a mechanism, not about perceptrons.
Given how firmly a receiver holds its own view on an issue (`h_w`, signed by
whether the message agrees) and how far it distrusts the sender (`h_mu`), the
modulation functions say how much of the surprise goes into revising the opinion
(`F_w`) and how much into revising the trust (`F_mu`). They are derived from the
inference problem, so they ought to describe *any* agent that has to revise a
belief about a source and a belief about an issue from the same evidence.

This directory tests that on an agent whose weights never move. A large language
model is shown an interlocutor's history in its context window and asked for a
credence before and after that interlocutor speaks; the update happens in the
forward pass, with no gradient step and nothing shared with the paper's learning
rule. **`F_mu` reproduces** — sign, dissonance gating and conviction ordering,
over the whole range the readout reaches.

Everything here produces Figure 11 of `paper/main.tex` (Appendix E).

Model `gpt-5.6-luna` at `reasoning_effort=low`; 3 916 calls, about **$0.66**.
Every call is cached on disk by request hash, so a rerun in a working tree that
has `data/cache/` costs nothing. That directory is gitignored, so a fresh clone
re-runs against the API and pays the $0.66 once.

## What is measured

Three questions, and the reason all three are about the *other* agent.

| question | asked | gives |
|---|---|---|
| **reliability** — "the chance this agent's judgement on this theme is sound" | before and after the message | `h_mu = -Phi^-1(r)`, and `Delta h_mu` as the difference |
| **agreement** — "the chance their next statement is one you agree with" | before | the evidence `Z`, and by inversion a check on the conviction |
| **conviction** — the agreement question, asked about a third party held to be sound | before | `\|h_w\|` directly, which places the point along the `h_w` axis |

Nothing asks the agent how strongly it believes its own opinion. A model told
that a view is its own reads holding it as a commitment, and answers such a
question at the ceiling or by stepping outside the persona — two different
constructs with no way to tell which came back. It is the same failure that put
all 820 baselines of an earlier five-rung design at `+2`. Conviction is obtained
from a prediction about someone else instead, and only the algebra is about the
receiver: since `q = c + e - 2ce`, a source with `e -> 0` gives `q = c` exactly.

Prior trust is likewise never asserted, only **learned in context** from a
history of two to five previous statements by the interlocutor, `k` of them
agreeing. There is no stated trust level for the model to repeat back, and the
before/after pair is a difference of two answers to one identical question.

Answers are probabilities from 0 to 100, not rungs on an ordinal scale. This is
not cosmetic: it is the paper's own variable (`eta = 1 - 2 Phi(h_mu)` is a
probability-scaled trust), and it has room to move at both ends, which five named
rungs do not.

## What comes out

Over 356 conditions (20 themes x 9 track records x 2 conviction framings) and 712
measured updates:

* the **sign** of `Delta h_mu` follows the sign of the agreement rather than the
  prior trust, in 95.2% and 84.8% of conditions for agreeing and disagreeing
  messages;
* its **size** is 4.0x and 5.0x larger in the dissonant quadrants than in the
  consonant ones — learning driven by surprise;
* told to hold its view firmly rather than tentatively, the model revises trust
  more (`1.34` against `0.90`), the ordering the prefactor `1 - 2 Phi(h_w)`
  requires;
* against `F_mu` over the plane: `r = 0.85`, 86.0% sign agreement, under one
  fitted positive scale.

The fitted conviction is a **lower bound**, `|h_w| >= 2.9`. Where `|F_mu|` peaks
along `h_mu` is set by `|h_w|` alone and moves outward as conviction rises; in
the limit `F_mu` becomes the Gaussian hazard rate and has no peak at all. At the
fitted value the peak sits outside the `+-2.05` a clipped probability can reach
through a probit, so the measurement covers the monotone flank and the turnover
is not observable in either direction.

`F_w` is not reported. It needs a *change* in conviction, which is a difference of
two conviction readings and inherits the noise of both; it did not come out, and
the paper says nothing about it.

## Reproducing Figure 11

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env      # only needed if the cache is deleted

python scripts/trust_curve.py --tag curve   # the run; replays from cache, $0
python scripts/trust_figure.py --tag curve  # writes figures/iclr/trust_llm.pdf
```

`trust_curve.py` also prints the numbers quoted above and in Appendix E.

The stimulus material in `data/themes/` is already built and is what the run
reads. To rebuild it from scratch — 20 themes, each with the agent's own opinion
and opinions generated at five degrees of agreement and then rated back to decide
which column they actually belong to — run `python scripts/stage1.py`. It
overwrites `data/themes/`, so every measurement downstream would need redoing.

## Layout

```
llmmod/
  llm.py        one call, the on-disk cache, and bounded concurrency
  themes.py     the 20 themes and why they were chosen
  generate.py   stage 1: builds and rates the stimulus material
  prompts.py    every prompt sent, and what each one measures
  scale.py      the five-point scale, used only to build the material
  fields.py     stated probabilities in, the paper's two fields out
  plotting.py   Figure 11
scripts/
  stage1.py       rebuild data/themes/ (not needed to reproduce the figure)
  trust_curve.py  the measurement -> data/trust/curve.rows.jsonl
  trust_figure.py the figure -> figures/iclr/trust_llm.pdf
data/
  themes/  the stimulus material, one JSON per theme
  cache/   3 916 responses keyed by request hash (gitignored)
  trust/   one row per measured update
```

## Caveats, in the order they matter

1. The conviction question asks what a sound third party would say, which a
   receiver may answer from how common it takes its view to be rather than how
   sure it is. **Perceived consensus and conviction are not separated** by this
   instrument.
2. Both conviction readings are compressed relative to the curve fit — median
   `|h_w|` is `0.67` direct and `1.15` by inversion against the `>= 2.9` the
   shape of the curve prefers. Signs survive; the plane panel is conservative.
3. The inversion `c = (q - e) / (1 - 2e)` is admissible only where `q` lies in
   `[r, 1-r]`, and against a distrusted interlocutor that band is narrow: the
   agents routinely predict disagreement more strongly than `c = 1` permits. It
   survives on 280 of 712 conditions and the censoring falls hardest in the
   distrusted half, which is why the direct question is the one plotted.
4. Making the histories more hostile does not help. Stated reliability **floors
   near 0.24 and rises** with a longer disagreeing history, from `0.236` at two
   contradicting statements to `0.341` at five.
5. `|h_mu|` cannot exceed `2.05`, being a stated probability clipped before the
   probit, so the deep corners of Figure 1 are out of reach in every condition.

None of this shows that in-context learning implements the paper's update rule.
It shows that an agent solving the same inference problem, by a mechanism that
shares nothing with the derivation, revises trust the way the derivation says.
