"""Stage 3: the trust update, on the conditions stage 2 already measured.

The receiver's opinion is not the only thing that moves when the colleague
speaks.  The paper's rule updates two sectors at once, and it claims they are
the same function with its arguments exchanged: ``F_w(x, y) = F_mu(y, x)``, with
whichever sector is the less certain moving the more, and the crossover sitting
on ``h_mu = h_w``.  Neither of those has ever been measured on a language model,
and neither can be measured from one sector alone -- which is why this stage
runs on the same grid as stage 2 rather than on a grid of its own.

How trust is read.  There is no reliability question, for the same reason there
is no credence question anywhere else here.  The receiver is turned to a second,
unrelated issue in the same records on which it has examined nothing itself, and
the colleague states a view on it; the ladder then measures how much evidence has
to be laid against that view before the receiver stops following it.  That
quantity *is* the weight of the colleague's word, in the same units as
everything else, signed, with no ceiling.  Measured once with the colleague
silent on the focal issue and once with them speaking, its change is the trust
update.

The pre arm carries the whole focal briefing too, so the two arms differ by the
colleague's message and by nothing else -- not by context length, not by the
focal evidence, not by the framing.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import _cli  # noqa: F401  - path setup

from llmmod2 import prompts, worlds
from llmmod2.ladder import DRAWS, NullPoint, measure_null
from llmmod2.llm import cost_estimate, usage_total
from stage2_opinion import K_LEVELS, K_QUICK, S_LEVELS, _flip

ROOT = _cli.ROOT
CALIB = ROOT / "data" / "rows" / "calibration.json"
OUT = ROOT / "data" / "rows" / "trust.jsonl"

#: Convictions carried over from stage 2.  A subset, because the trust sector is
#: needed on enough of the plane to test the symmetry rather than on all of it.
S_LEVELS_TRUST = (-2, 0, 2)
S_QUICK = (0,)

#: The direction the colleague asserts on the fresh issue.  Fixed; the framing
#: assignment already varies whether that entity is the one named first.
FRESH_DIR = +1


def _weight_null(world, s, k, flip, focal_dir, draws):
    focal, fresh = world.issue(0), world.issue(1)
    seed = f"{world.key}|tr|{s}|{k}|{flip}"
    schema = prompts.verdict_schema(fresh, flip)

    def render(t):
        return prompts.trust_prompt(world, focal, fresh, s=s, k=k,
                                    focal_dir=focal_dir, t=t,
                                    fresh_dir=FRESH_DIR, flip=flip, seed=seed)

    return measure_null(render, schema, fresh.a, prompts.SYSTEM, draws=draws)


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
    ap.add_argument("--max-cost", type=float, default=25.0)
    args = ap.parse_args()

    if not CALIB.is_file():
        raise SystemExit("run scripts/stage1_calibrate.py first")
    kept = json.loads(CALIB.read_text())["kept"]
    pool_worlds = worlds.load(keep=kept)
    s_levels, k_levels = S_LEVELS_TRUST, K_LEVELS
    if args.quick:
        pool_worlds = pool_worlds[:3]
        s_levels, k_levels = S_QUICK, K_QUICK

    cells = [(i, w, s, k) for i, w in enumerate(pool_worlds)
             for s in s_levels for k in k_levels]
    print(f"{len(cells)} conditions, {len(cells) * 3} null points")

    def guard():
        if cost_estimate() > args.max_cost:
            raise SystemExit(f"aborting: past --max-cost {args.max_cost}")

    with ThreadPoolExecutor(args.workers) as pool:
        pre = list(pool.map(
            lambda c: _safe(_weight_null, c[1], c[2], c[3],
                            _flip(c[0], c[2], c[3]), None, args.draws), cells))
    guard()
    print(f"  pre done, ${cost_estimate():.2f}")

    posts = {}
    for m in (+1, -1):
        with ThreadPoolExecutor(args.workers) as pool:
            posts[m] = list(pool.map(
                lambda c: _safe(_weight_null, c[1], c[2], c[3],
                                _flip(c[0], c[2], c[3]), m, args.draws), cells))
        guard()
        print(f"  focal message {m:+d} done, ${cost_estimate():.2f}")

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
                    "fresh_dir": FRESH_DIR,
                    "track_total": prompts.TRACK_TOTAL,
                    "weight_pre": FRESH_DIR * p.leaning,
                    "weight_post": FRESH_DIR * q.leaning,
                    "d_weight": FRESH_DIR * (q.leaning - p.leaning),
                    "censored": bool(p.censored or q.censored),
                    "pre": p.as_dict(), "post": q.as_dict(),
                }) + "\n")
                n += 1
    print(f"wrote {n} rows -> {OUT}")
    u = usage_total()
    print(f"calls={u['calls']} cached={u['cached']} ${u['dollars']:.2f}")


if __name__ == "__main__":
    main()
