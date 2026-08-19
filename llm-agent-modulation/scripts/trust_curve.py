#!/usr/bin/env python3
"""The trust-curve run: three questions per condition, all about the emitter.

What is measured, and why it takes three calls.

``r_before``   "the chance this agent's judgement on this theme is sound", asked
               after a track record of prior statements.  Gives ``h_mu``, the
               axis.
``q``          "the chance their next statement is one you would agree with",
               asked before any message.  This is the evidence ``Z``; with
               ``r_before`` it inverts for the receiver's conviction and so for
               ``|h_w|``.  It is *not* interchangeable with the first question:
               it is symmetric under swapping who is likely wrong, so on its own
               it cannot say whether a predicted disagreement is the emitter's
               fault or the receiver's.
``r_after``    the first question again, with one message added and nothing else
               changed.  ``Delta h_mu`` is the difference of the two.

The three fields on the plane are then ``h_mu`` (from ``r_before``), ``h_w``
(sign from the message, magnitude from the inversion) and ``Delta h_mu``.

Prior trust is built, never asserted.  A track record of two or three previous
statements, each drawn from the theme's agreeing or disagreeing columns, sets
``k = agreements - disagreements`` in ``-3..+3``.  Nothing in any prompt tells
the agent how much it trusts the emitter, so there is no rung to echo back and
the before/after pair is a difference of two answers to one question.

Conviction is manipulated, not measured: the receiver is told it holds its
opinion firmly or tentatively.  The model's prediction is that the firm arm
shows the larger ``|Delta h_mu|``, which this run can contradict.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from _cli import ROOT  # noqa: E402

from llmmod.fields import (P_CLIP, conviction, h_mu_from_reliability,  # noqa: E402
                           h_w_from, to_p)
from llmmod.generate import load_all  # noqa: E402
from llmmod.llm import MODEL, ask_many, usage_total  # noqa: E402
from llmmod.prompts import (CHANCE_SCHEMA, CONVICTION, SYSTEM,  # noqa: E402
                            agreement_prompt, conviction_prompt,
                            reliability_prompt)
from llmmod.themes import by_key  # noqa: E402

OUT = ROOT / "data" / "trust"
PRICE_IN, PRICE_OUT = 0.20, 1.20

#: Track records.  ``k`` is agreements minus disagreements, over a history whose
#: length is the smallest that can express it: two statements for an even ``k``,
#: three for an odd one, and ``|k|`` once that exceeds them.  The length
#: therefore varies between columns, which is a mild confound (more statements is
#: more evidence) and is recorded in the rows so it can be checked against.
#:
#: The record runs further into disagreement than into agreement, and not for
#: symmetry's own sake.  The conviction inversion is admissible only where the
#: predicted agreement ``q`` lies in ``[r, 1-r]``, and against a mildly
#: distrusted emitter that band is narrow: at ``r = 0.28`` the agents routinely
#: predict disagreement more strongly than ``c = 1`` allows, and the condition is
#: censored rather than measured.  The band widens as ``r`` falls, so the extra
#: columns buy identification in exactly the half-plane where the opinion sector
#: is most interesting and was most thinly covered.
TRACK = (-5, -4, -3, -2, -1, 0, 1, 2, 3)


def _record(k):
    """``(agreements, disagreements)`` in a track record for column ``k``.

    The length matches ``k`` in parity, so the split is always whole, and is the
    shortest such length that can reach ``k``.
    """
    m = max(abs(k), 3 if k % 2 else 2)
    j = (m + k) // 2
    return j, m - j


def _pools(rec, per_sign):
    """Message texts and track-record texts, disjoint by construction.

    Messages are reserved first, from the extreme columns, because the message
    is what carries the sign and should be unambiguous.  The track record then
    takes what is left of the extremes before falling back on the softer
    columns, so a record of three statements is as strong as the material
    allows.  A statement is never both a message and part of the history.
    """
    used = {s: [o for o in rec["opinions"] if o.get("used") and o["slot"] == s]
            for s in (-2, -1, 1, 2)}
    msg = {+1: used[2][:per_sign], -1: used[-2][:per_sign]}
    hist = {+1: used[2][per_sign:] + used[1],
            -1: used[-2][per_sign:] + used[-1]}
    return msg, hist


def _history(hist, k, key):
    """The track record for column ``k``, in a fixed but unpatterned order.

    Ordered by a hash of the theme and column rather than left grouped, so the
    agreeing statements do not all arrive first; the order is deterministic so
    that repeated draws of the same condition are repeats of the *same* prompt,
    which is what the cache's ``nonce`` is for.
    """
    j, l = _record(k)
    if len(hist[+1]) < j or len(hist[-1]) < l:
        return None
    picked = hist[+1][:j] + hist[-1][:l]
    seed = int(hashlib.sha256(f"{key}:{k}".encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(len(picked))
    return [picked[i]["text"] for i in order]


def _mean_p(answers):
    """Draws of one question, averaged as probabilities.

    Averaged before the probit, not after: the answers are bounded and the
    probit is not, so one draw at the top of the scale would otherwise carry the
    mean with it.  Returns ``(mean probability, how many draws hit the clip)``.
    """
    vals = [a["chance"] for a in answers if "__error__" not in a]
    if not vals:
        return None, 0
    ps = [to_p(v) for v in vals]
    clipped = sum(1 for v in vals if not P_CLIP[0] < v / 100.0 < P_CLIP[1])
    return float(np.mean(ps)), clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="curve")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--per-sign", type=int, default=1,
                    help="messages reserved per sign, per theme")
    ap.add_argument("--draws-before", type=int, default=3)
    ap.add_argument("--draws-q", type=int, default=2)
    ap.add_argument("--draws-after", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = load_all()
    conditions, skipped = [], []
    for rec in records:
        key = rec["theme"]["key"]
        msg, hist = _pools(rec, args.per_sign)
        for k in TRACK:
            history = _history(hist, k, key)
            if history is None:
                skipped.append((key, k))
                continue
            for conv in CONVICTION:
                conditions.append({"theme": key, "k": k, "conviction": conv,
                                   "history": history, "messages": msg,
                                   "own": rec["own_opinion"]})
    if skipped:
        print(f"  skipped {len(skipped)} conditions for want of material: "
              f"{skipped[:6]}{' ...' if len(skipped) > 6 else ''}")

    def ctx(c):
        return (by_key(c["theme"]), c["own"], c["conviction"], c["history"])

    before = [{"system": SYSTEM, "user": reliability_prompt(*ctx(c)),
               "schema": CHANCE_SCHEMA, "model": args.model, "nonce": d}
              for c in conditions for d in range(args.draws_before)]
    qs = [{"system": SYSTEM, "user": agreement_prompt(*ctx(c)),
           "schema": CHANCE_SCHEMA, "model": args.model, "nonce": d}
          for c in conditions for d in range(args.draws_q)]
    conv_before = [{"system": SYSTEM, "user": conviction_prompt(*ctx(c)),
                    "schema": CHANCE_SCHEMA, "model": args.model, "nonce": d}
                   for c in conditions for d in range(args.draws_q)]
    after_jobs = [(ci, sign, o, d)
                  for ci, c in enumerate(conditions)
                  for sign in (+1, -1)
                  for o in c["messages"][sign]
                  for d in range(args.draws_after)]
    after = [{"system": SYSTEM,
              "user": reliability_prompt(*ctx(conditions[ci]), message=o["text"]),
              "schema": CHANCE_SCHEMA, "model": args.model, "nonce": d}
             for ci, sign, o, d in after_jobs]

    total = len(before) + len(qs) + len(conv_before) + len(after)
    print(f"{len(conditions)} conditions -> {len(before)} before + {len(qs)} q "
          f"+ {len(conv_before)} conv + {len(after)} after = {total} calls")
    if args.dry_run:
        return

    r_out = ask_many(before, workers=args.workers, label="before")
    q_out = ask_many(qs, workers=args.workers, label="q")
    a_out = ask_many(after, workers=args.workers, label="after")
    cb_out = ask_many(conv_before, workers=args.workers, label="conv")

    nb, nq = args.draws_before, args.draws_q
    per_cond = []
    for i, c in enumerate(conditions):
        r, r_clip = _mean_p(r_out[i * nb:(i + 1) * nb])
        q, q_clip = _mean_p(q_out[i * nq:(i + 1) * nq])
        cd, c_clip = _mean_p(cb_out[i * nq:(i + 1) * nq])
        per_cond.append((r, q, cd, r_clip + q_clip + c_clip))

    grouped = {}
    for (ci, sign, o, _d), ans in zip(after_jobs, a_out):
        grouped.setdefault((ci, sign, o["id"]), []).append(ans)

    rows = []
    for (ci, sign, oid), answers in grouped.items():
        c = conditions[ci]
        r, q, cd, clip = per_cond[ci]
        ra, ra_clip = _mean_p(answers)
        if r is None or q is None or ra is None:
            continue
        h_mu = h_mu_from_reliability(r)
        # two readings of the same conviction: the inversion, kept as a check and
        # undefined on more than half the conditions, and the direct question,
        # which is what places the point along h_w
        cw = conviction(q, r)
        hd = h_w_from(cd, sign) if cd is not None else float("nan")
        rows.append({
            "theme": c["theme"], "k": c["k"], "conviction": c["conviction"],
            "sign": sign, "message_id": oid, "history_len": len(c["history"]),
            "r_before": r, "q": q, "r_after": ra,
            "h_mu": h_mu, "c": cw, "h_w": h_w_from(cw, sign),
            "c_direct": cd, "h_w_direct": hd,
            "h_mu_after": h_mu_from_reliability(ra),
            "delta_mu": h_mu_from_reliability(ra) - h_mu,
            "clipped": clip + ra_clip,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.tag}.rows.jsonl"
    with path.open("w") as fh:
        for r_ in rows:
            fh.write(json.dumps(r_) + "\n")
    print(f"\n[rows] {path}  ({len(rows)} rows)")

    _report(rows)
    u = usage_total()
    print(f"\n[usage] {u['calls']} calls ({u['cached']} cached) -> "
          f"${(u['prompt'] * PRICE_IN + u['completion'] * PRICE_OUT) / 1e6:.3f}")


def _report(rows):
    """The three things worth seeing before any figure is drawn."""
    arr = lambda key, f=lambda r: True: np.array(  # noqa: E731
        [r[key] for r in rows if f(r) and r[key] is not None
         and np.isfinite(r[key])])

    print("\nh_mu by track record (does the history move trust at all?)")
    print("     k   n     r_before      h_mu")
    for k in TRACK:
        sub = [r for r in rows if r["k"] == k]
        if not sub:
            continue
        print(f"  {k:+3d}  {len(sub):3d}   {np.mean([r['r_before'] for r in sub]):.3f}"
              f"      {np.mean([r['h_mu'] for r in sub]):+.3f}")

    print("\nDelta h_mu by message sign and conviction "
          "(sign should follow the message, magnitude the conviction)")
    for conv in ("firm", "tentative"):
        for sign, name in ((+1, "agrees   "), (-1, "disagrees")):
            v = arr("delta_mu", lambda r, s=sign, c=conv:
                    r["sign"] == s and r["conviction"] == c)
            sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            print(f"  {conv:<10s} message {name}  n={v.size:4d}   "
                  f"{v.mean():+.3f} +- {sem:.3f}")

    print("\nthe two conviction estimates, on the same conditions")
    inv = arr("c")
    dirc = arr("c_direct")
    print(f"  inversion  c = (q-e)/(1-2e):  {inv.size:4d} identified of "
          f"{len(rows)},  mean {inv.mean():.3f},  "
          f"outside (0,1): {int(np.sum((inv <= 0) | (inv >= 1)))}")
    print(f"  direct     c = q(sound source): {dirc.size:4d} identified of "
          f"{len(rows)},  mean {dirc.mean():.3f},  "
          f"outside (0,1): {int(np.sum((dirc <= 0) | (dirc >= 1)))}")
    both = np.array([(r["c"], r["c_direct"]) for r in rows
                     if r["c"] is not None and r["c_direct"] is not None
                     and np.isfinite(r["c"]) and 0 < r["c"] < 1])
    if both.size:
        print(f"  where both are defined (n={len(both)}): "
              f"pearson r = {np.corrcoef(both[:, 0], both[:, 1])[0, 1]:+.3f}")

    print(f"clipped answers: {sum(r['clipped'] for r in rows)}")


if __name__ == "__main__":
    main()
