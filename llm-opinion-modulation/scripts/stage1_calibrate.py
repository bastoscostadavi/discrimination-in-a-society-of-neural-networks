"""Stage 1: screen the worlds, and calibrate the ladder on them.

Two things have to be true before a world can carry a measurement, and neither
can be assumed.

*Prior-free.*  The conviction knob only works if the model arrives with no view.
Every world is put the bare question with no evidence and no colleague, in both
framings.  What this turns up is worth stating plainly, because it decides how
the whole experiment has to be run: asked a question it has nothing to go on,
the model names the entity that was mentioned **first**, in almost every world
and almost every draw.  That is a property of the question, not of the world --
it reverses exactly when the two names are swapped -- so the screen is on the
rate *averaged over the two framings*, which is what a content prior would move
and a position bias would not.  A world whose averaged rate leaves the band can
be answered from the names themselves, and is dropped.

The position bias is why every measurement below is run in both framings and
averaged.  It is large, it is not a nuisance that can be ignored, and it is also
harmless: it is a constant of the framing, so it cancels in the framing average
and cancels again in every before-and-after difference the experiment reports.

*Faithful.*  The null point should sit where the evidence puts it.  With the
colleague present but silent, the belief the receiver holds ought to be worth
about as many pieces of evidence as it was given: measuring the null at ``s = 0``
gives the framing's own bias, and measuring it at ``s = 4`` says whether four
pieces of evidence are worth four.  The bias is reported and not corrected --
it is a constant of the framing and cancels in every difference the experiment
reports -- but a world whose null point ignores its evidence is dropped.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import _cli  # noqa: F401  - path setup
import numpy as np

from llmmod2 import prompts, worlds
from llmmod2.ladder import DRAWS, measure_null
from llmmod2.llm import MAX_N, ask, usage_total

ROOT = _cli.ROOT
OUT = ROOT / "data" / "rows" / "calibration.json"

#: A world passes if its framing-averaged no-evidence answer rate is this close
#: to a coin.  The band is on the average, not on either framing separately: a
#: world that answers 1.00 one way round and 0.00 the other is perfectly
#: prior-free, and is the common case.
SCREEN_TOL = 0.20

#: Evidence level used for the faithfulness check, and the fraction of it the
#: measured leaning must recover.
CHECK_S = 4
CHECK_MIN = 0.4


def screen(world, issue, draws):
    """No-evidence answer rate, per framing."""
    out = {}
    for flip in (0, 1):
        schema = prompts.verdict_schema(issue, flip)
        user = prompts.screen_prompt(world, issue, flip)
        answers = []
        for nonce in range((draws + MAX_N - 1) // MAX_N):
            answers += ask(prompts.SYSTEM, user, schema, nonce=nonce,
                           n=min(MAX_N, draws - len(answers)))
        out[flip] = sum(1 for a in answers if a["answer"] == issue.a) / len(answers)
    return out


def baseline(world, issue, s, flip, draws):
    """Null point with the colleague present and silent."""
    seed = f"{world.key}|cal|{s}|{flip}"
    schema = prompts.verdict_schema(issue, flip)

    def render(t):
        return prompts.opinion_prompt(world, issue, s=s, k=prompts.TRACK_TOTAL // 2,
                                      t=t, message_dir=None, flip=flip, seed=seed)

    return measure_null(render, schema, issue.a, prompts.SYSTEM, draws=draws)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--quick", action="store_true",
                    help="screen only; skip the null-point calibration")
    args = ap.parse_args()

    all_worlds = worlds.load()
    print(f"screening {len(all_worlds)} worlds")
    with ThreadPoolExecutor(args.workers) as pool:
        screens = list(pool.map(
            lambda w: screen(w, w.issue(0), args.draws), all_worlds))

    rows = {}
    kept = []
    for w, sc in zip(all_worlds, screens):
        mean = 0.5 * (sc[0] + sc[1])
        position = 0.5 * (sc[0] - sc[1])
        ok = abs(mean - 0.5) <= SCREEN_TOL
        rows[w.key] = {"screen": {str(k): v for k, v in sc.items()},
                       "prior": float(mean), "position_bias": float(position),
                       "kept": ok}
        print(f"  {w.key:10s} p(a) = {sc[0]:.2f} / {sc[1]:.2f}   "
              f"prior = {mean:.2f}  position = {position:+.2f}   "
              f"{'keep' if ok else 'DROP'}")
        if ok:
            kept.append(w)
    print(f"{len(kept)}/{len(all_worlds)} pass the prior screen; "
          f"mean position bias "
          f"{np.mean([abs(r['position_bias']) for r in rows.values()]):.2f}")

    if not args.quick:
        jobs = [(w, s, f) for w in kept for s in (0, CHECK_S) for f in (0, 1)]
        print(f"calibrating {len(jobs)} null points")
        with ThreadPoolExecutor(args.workers) as pool:
            nulls = list(pool.map(
                lambda j: baseline(j[0], j[0].issue(0), j[1], j[2], args.draws),
                jobs))
        by_world = {}
        for (w, s, f), np_ in zip(jobs, nulls):
            by_world.setdefault(w.key, {})[f"{s}|{f}"] = np_.as_dict()
        for w in kept:
            d = by_world[w.key]
            bias = np.mean([-d[f"0|{f}"]["t_star"] for f in (0, 1)])
            gain = np.mean([-d[f"{CHECK_S}|{f}"]["t_star"] for f in (0, 1)]) - bias
            ok = gain >= CHECK_MIN * CHECK_S
            rows[w.key].update(nulls=d, bias=float(bias), gain=float(gain),
                               kept=bool(rows[w.key]["kept"] and ok))
            print(f"  {w.key:10s} bias = {bias:+.2f}   "
                  f"{CHECK_S} pieces read as {gain:+.2f}   "
                  f"{'keep' if ok else 'DROP'}")
        kept = [w for w in kept if rows[w.key]["kept"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"kept": [w.key for w in kept], "worlds": rows},
                              indent=2))
    print(f"\n{len(kept)} worlds kept -> {OUT}")
    u = usage_total()
    print(f"calls={u['calls']} cached={u['cached']} ${u['dollars']:.3f}")


if __name__ == "__main__":
    main()
