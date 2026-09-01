"""Every briefing the model is shown, and the shape of the answer it gives back.

The design rule this module exists to enforce: **the model is never asked for a
number.**  It is asked which of two invented things the records favour, and it
answers with one of the two names.  Everything the analysis reports -- the
receiver's conviction, its trust in the colleague, and both updates -- is
recovered from where those one-word verdicts turn over.  There is no stated
probability anywhere in the experiment, and so no scale for the model to
saturate and no ceiling on the readout.

Three quantities are laid into a briefing, and they are kept in separate
channels so that they can be varied independently:

``s``  the receiver's *own* evidence, net of the two directions.  This sets the
       conviction the receiver holds when the colleague speaks: the ``h_w``
       coordinate.  Held fixed within a measurement.
``k``  the colleague's track record out of ``TRACK_TOTAL``.  This sets ``h_mu``.
       Held fixed within a measurement.
``t``  a later consignment of evidence, net of the two directions.  This is the
       ladder: it is swept to find the point at which the verdict turns over.

The colleague's paragraph is present in both arms of a measurement and their
*message* is present in only one, so the difference between the two null points
isolates the message and carries no other change in the context -- including the
first-named-entity bias, which is a constant of the framing and cancels.
"""

from __future__ import annotations

import hashlib

#: Questions in the colleague's track record.  Twenty is the sibling's history
#: length rounded to a number a reader can hold, and it makes the uninformative
#: record -- ten right, ten wrong, ``h_mu = 0`` -- exactly representable.
TRACK_TOTAL = 20

#: The receiver's own evidence is always this many pieces, whatever its net
#: direction, so that conviction varies without the *amount* of evidence
#: varying with it.
OWN_TOTAL = 6

SYSTEM = (
    "You are an analyst working on a body of historical records. You judge each "
    "question only from the briefing you are given, and you always commit to the "
    "answer the briefing makes more likely, however slightly."
)


def verdict_schema(issue, flip):
    """Strict schema whose only two admissible answers are the entity names."""
    names = [issue.b, issue.a] if flip else [issue.a, issue.b]
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": names}},
            "required": ["answer"],
            "additionalProperties": False}


def _order(issue, flip):
    return (issue.b, issue.a) if flip else (issue.a, issue.b)


def question(world, issue, flip):
    first, second = _order(issue, flip)
    return f"which {world.predicate}, {first} or {second}?"


def _evidence_lines(world, issue, net, total, label, seed):
    """``total`` pieces of evidence with the given ``net`` direction, shuffled.

    Order is fixed by a hash of the cell rather than by an RNG, so the same cell
    always renders the same briefing and the cache is meaningful.
    """
    if total <= 0:
        return []
    n_plus = (total + net) // 2
    tags = [+1] * n_plus + [-1] * (total - n_plus)
    h = hashlib.sha256(seed.encode()).digest()
    order = sorted(range(total), key=lambda i: h[i % len(h)] * total + i)
    tags = [tags[i] for i in order]
    return [f"  {label} {i}: indicates that {issue.name(d)} {world.predicate}."
            for i, d in enumerate(tags, 1)]


def _own_block(world, issue, s, seed):
    unit = world.unit_singular
    lines = _evidence_lines(world, issue, s, OWN_TOTAL, unit.capitalize(),
                            seed + "|own")
    if not lines:
        return (f"You have not yet examined any {world.unit_plural} bearing on "
                f"this question.")
    return (f"You have examined {OWN_TOTAL} {world.unit_plural} bearing on it:\n"
            + "\n".join(lines))


def _probe_block(world, issue, t, seed):
    """The rung of the ladder, worded so that it reads the same in both prompts.

    The standing of this evidence has to be stated -- a consignment the receiver
    silently discounts would make the ladder's units drift -- but it cannot be
    stated by pointing at what came before, because on the fresh issue of the
    trust prompt nothing did.  So it is pinned to the collection rather than to
    the briefing.
    """
    if t == 0:
        return None
    n = abs(t)
    d = 1 if t > 0 else -1
    unit = world.unit_plural if n > 1 else world.unit_singular
    verb = {1: "It indicates", 2: "Both indicate"}.get(n, f"All {n} indicate")
    return (f"A later consignment has since reached you: {n} further {unit} "
            f"bearing on this question, of the same standing as any other "
            f"record in the collection. {verb} that "
            f"{issue.name(d)} {world.predicate}.")


def _colleague_block(world, issue, k, message):
    """The track record, and the message when there is one.

    The record is stated as a count of settled questions rather than as a
    reliability, so that the receiver has to weigh a history rather than read a
    number off a scale.  ``k`` above half is a source worth following, ``k``
    below half a source worth contradicting, and ``k`` at half carries nothing.
    """
    lines = [f"{world.colleague} works on the same records. Of the last "
             f"{TRACK_TOTAL} questions of this kind whose answers were later "
             f"established beyond doubt, {world.colleague} answered correctly "
             f"{k} times and incorrectly {TRACK_TOTAL - k} times."]
    if message is not None:
        lines.append(f'{world.colleague} has now stated: "{message}"')
    return "\n".join(lines)


def claim(world, issue, d):
    return f"{issue.name(d)} {world.predicate}."


def opinion_prompt(world, issue, *, s, k, t, message_dir, flip, seed):
    """The focal briefing.

    ``message_dir`` is ``None`` for the arm in which the colleague is present
    but silent, and ``+1``/``-1`` for the arm in which they assert one of the two
    entities.  Nothing else differs between the two arms.
    """
    message = None if message_dir is None else claim(world, issue, message_dir)
    parts = [f"{world.setting} You are one of its analysts.",
             f"Question: {question(world, issue, flip)}",
             _own_block(world, issue, s, seed),
             _colleague_block(world, issue, k, message)]
    probe = _probe_block(world, issue, t, seed)
    if probe:
        parts.append(probe)
    parts.append(f"On the whole of the above, {question(world, issue, flip)}")
    return "\n\n".join(parts)


def trust_prompt(world, focal, fresh, *, s, k, focal_dir, t, fresh_dir, flip, seed):
    """The fresh-issue briefing, used to weigh the colleague's testimony.

    The focal exchange is carried in the context, so this measures the weight the
    receiver puts on the colleague *after* whatever happened on the focal issue.
    Run with ``focal_dir=None`` it measures the weight before, with the focal
    briefing still present so that the two arms differ only by the message.
    """
    message = None if focal_dir is None else claim(world, focal, focal_dir)
    parts = [f"{world.setting} You are one of its analysts.",
             f"Earlier today you worked on this question: "
             f"{question(world, focal, flip)}",
             _own_block(world, focal, s, seed),
             _colleague_block(world, focal, k, message),
             "You have now turned to a second, unrelated question in the same "
             "records, on which you have examined nothing yourself.",
             f"Question: {question(world, fresh, flip)}",
             f'{world.colleague} has stated: "{claim(world, fresh, fresh_dir)}"']
    probe = _probe_block(world, fresh, t, seed)
    if probe:
        parts.append(probe)
    parts.append(f"On the whole of the above, {question(world, fresh, flip)}")
    return "\n\n".join(parts)


def screen_prompt(world, issue, flip):
    """No evidence, no colleague: does the model already have a view?

    A world only earns its place if the answer here is near chance.  A world in
    which the model can guess from the names is one in which the conviction knob
    is not the only thing setting the conviction.
    """
    return "\n\n".join([
        f"{world.setting} You are one of its analysts.",
        f"Question: {question(world, issue, flip)}",
        f"No {world.unit_plural} bearing on this question survive, and nothing "
        f"else is known about it.",
        f"You must still answer. On your best guess, "
        f"{question(world, issue, flip)}"])
