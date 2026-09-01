"""Stage 2: the opinion update.

One condition is a world, a conviction ``s``, a track record ``k``, and a
direction for the colleague to assert.  It is measured twice with the same
briefing: once with the colleague present and silent, once with the colleague
saying something.  Each measurement is a walk down the ladder to the point where
the verdict turns over, and the *difference* between the two null points is what
the colleague's statement was worth, in pieces of evidence.

That difference is the observable.  It is signed by the direction the colleague
asserted, so a positive value means the receiver moved towards the message and a
negative value means it moved away -- and moving away is not a failure mode
here, it is the prediction: ``F_w = (1 - 2 Phi(h_mu)) g(h_w) / Z`` changes sign
with the trust of the emitter, so a receiver that hears agreement from a source
it has watched be wrong should end up believing its own conclusion *less*.

Both arms of a condition share a framing, so the first-named-entity bias stage 1
found is common to them and cancels in the difference.  Framings are assigned
across the grid rather than within a condition, which balances them over the
design at half the cost of running both.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import _cli  # noqa: F401  - path setup

from llmmod2 import prompts, worlds
from llmmod2.ladder import DRAWS, NullPoint, measure_null
from llmmod2.llm import usage_total, cost_estimate

ROOT = _cli.ROOT
CALIB = ROOT / "data" / "rows" / "calibration.json"
OUT = ROOT / "data" / "rows" / "opinion.jsonl"

#: Conviction levels, as the net direction of the receiver's own evidence.  With
#: ``OWN_TOTAL = 6`` these are the five splits from one-against-five to
#: five-against-one, so the *amount* of evidence is constant along the axis and
#: only its balance moves.
S_LEVELS = (-4, -2, 0, 2, 4)
S_QUICK = (-2, 0, 2)

#: Track records, spanning a source that has been wrong every time to one that
#: has been right every time, through the uninformative middle.
K_LEVELS = (0, 4, 10, 16, 20)
K_QUICK = (0, 10, 20)


def _flip(w_index, s, k):
    """Framing for a condition.  Balanced across the grid, fixed within one."""
    return (w_index + S_LEVELS.index(s) + K_LEVELS.index(k)) % 2


def _null(world, s, k, flip, message_dir, draws):
    issue = world.issue(0)
    seed = f"{world.key}|op|{s}|{k}|{flip}"
    schema = prompts.verdict_schema(issue, flip)

    def render(t):
        return prompts.opinion_prompt(world, issue, s=s, k=k, t=t,
                                      message_dir=message_dir, flip=flip,
                                      seed=seed)

    return measure_null(render, schema, issue.a, prompts.SYSTEM, draws=draws)


def _safe(fn, *args):
    """A cell that will not measure is recorded as censored, not raised.

    These sweeps run for an hour or more against a network service.  An
    exhausted retry on one rung would otherwise discard every cell measured
    before it; instead the cell is marked and the fits drop it, which is the same
    treatment a belief that ran off the end of the ladder gets.
    """
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - counted, not swallowed
        print(f"  [cell failed] {type(exc).__name__}: {str(exc)[:100]}",
              flush=True)
        return NullPoint(t_star=float("nan"), beta=float("nan"), censored=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-cost", type=float, default=25.0,
                    help="abort before exceeding this many dollars")
    args = ap.parse_args()

    if not CALIB.is_file():
        raise SystemExit("run scripts/stage1_calibrate.py first")
    kept = json.loads(CALIB.read_text())["kept"]
    pool_worlds = worlds.load(keep=kept)
    s_levels, k_levels = S_LEVELS, K_LEVELS
    if args.quick:
        pool_worlds = pool_worlds[:3]
        s_levels, k_levels = S_QUICK, K_QUICK

    cells = [(i, w, s, k) for i, w in enumerate(pool_worlds)
             for s in s_levels for k in k_levels]
    print(f"{len(pool_worlds)} worlds x {len(s_levels)} convictions x "
          f"{len(k_levels)} records = {len(cells)} conditions, "
          f"{len(cells) * 3} null points")

    def guard():
        if cost_estimate() > args.max_cost:
            raise SystemExit(f"aborting: past --max-cost {args.max_cost}")

    with ThreadPoolExecutor(args.workers) as pool:
        pre = list(pool.map(
            lambda c: _safe(_null, c[1], c[2], c[3], _flip(c[0], c[2], c[3]), None,
                                  args.draws), cells))
    guard()
    print(f"  pre done, ${cost_estimate():.2f}")

    posts = {}
    for m in (+1, -1):
        with ThreadPoolExecutor(args.workers) as pool:
            posts[m] = list(pool.map(
                lambda c: _safe(_null, c[1], c[2], c[3], _flip(c[0], c[2], c[3]), m,
                                      args.draws), cells))
        guard()
        print(f"  message {m:+d} done, ${cost_estimate():.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as fh:
        for idx, (i, w, s, k) in enumerate(cells):
            flip = _flip(i, s, k)
            p = pre[idx]
            for m in (+1, -1):
                q = posts[m][idx]
                fh.write(json.dumps({
                    "world": w.key, "s": s, "k": k, "flip": flip, "sign": m,
                    "track_total": prompts.TRACK_TOTAL,
                    "own_total": prompts.OWN_TOTAL,
                    "lean_pre": p.leaning, "lean_post": q.leaning,
                    "d_lean": q.leaning - p.leaning,
                    "censored": bool(p.censored or q.censored),
                    "beta_pre": p.beta, "beta_post": q.beta,
                    "pre": p.as_dict(), "post": q.as_dict(),
                }) + "\n")
                n += 1
    print(f"wrote {n} rows -> {OUT}")
    u = usage_total()
    print(f"calls={u['calls']} cached={u['cached']} ${u['dollars']:.2f}")


if __name__ == "__main__":
    main()
