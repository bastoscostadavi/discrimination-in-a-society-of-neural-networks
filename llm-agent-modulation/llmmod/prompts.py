"""Every prompt the experiment sends.

Kept minimal on purpose.  An earlier version of this file carried instructions
nobody asked for -- a system prompt telling the agent not to "hedge or qualify
into vagueness", and a request that its opinion "take a definite position rather
than surveying the possibilities".  Those pushed the agent to the end of its own
scale: it rated its agreement with its own opinion at ``+2`` in all 820 calls of
the run that followed, which left the ideological update no room to move and
made half the grid unmeasurable.  Instructions about *how* to hold an opinion
are not neutral scaffolding; they set the very quantity being measured.  So the
prompts here say what is needed to get a well-formed answer and nothing more.

Stage 1 builds the material: the agent's own opinion ``OA`` on a theme, opinions
``Oi`` at each degree of agreement, and a rating of each ``Oi`` that fixes which
column it belongs to.

Stage 2 measures.  It differs from the stage-1 rating in exactly one way, and
that difference is the experiment: the same opinion is now attributed to another
agent, and the agent is told how far it trusts them.  What that conditioning
does to the agent's hold on its own opinion, and to its trust, is the update.
"""

from __future__ import annotations

from .scale import AGREE_LABEL, LEVELS

__all__ = ["SYSTEM", "opinion_prompt", "generate_prompt", "generate_schema",
           "rate_prompt", "OPINION_SCHEMA", "RATE_SCHEMA", "CHANCE_SCHEMA",
           "CONVICTION", "reliability_prompt", "agreement_prompt",
           "conviction_prompt"]

#: Minimal.  Structured output already forces the answer shape, so nothing here
#: needs to ask for brevity or numbers.
SYSTEM = "You are an agent taking part in a discussion with other agents."

OPINION_SCHEMA = {
    "type": "object",
    "properties": {"opinion": {"type": "string"}},
    "required": ["opinion"], "additionalProperties": False,
}

RATE_SCHEMA = {
    "type": "object",
    "properties": {"rating": {"type": "integer", "minimum": min(LEVELS),
                              "maximum": max(LEVELS)}},
    "required": ["rating"], "additionalProperties": False,
}



def generate_schema(n):
    """Exactly ``n`` opinions.

    Built per call: a strict schema pins ``minItems`` and ``maxItems``, so a
    constant would silently cap the request however many the prompt asks for --
    a failure that leaves no trace, since the response still validates.
    """
    return {
        "type": "object",
        "properties": {
            "opinions": {"type": "array", "items": {"type": "string"},
                         "minItems": n, "maxItems": n},
        },
        "required": ["opinions"], "additionalProperties": False,
    }


def _scale(labels):
    return ", ".join(f"{d:+d}".replace("+0", "0") + f" = {labels[d]}"
                     for d in LEVELS)


def opinion_prompt(theme):
    return (f"Consider {theme.statement}.\n\n"
            f"State your opinion on it, in one or two sentences.")


def generate_prompt(theme, own_opinion, degree, n=3):
    """``n`` opinions aimed at one degree of agreement.

    The only instruction beyond the aim is that the ``n`` differ from each
    other, which is needed to get ``n`` opinions rather than one restated.
    """
    return (f"Consider {theme.statement}.\n\n"
            f"You currently have this opinion on that theme:\n\n"
            f"  \"{own_opinion}\"\n\n"
            f"Write {n} different opinions on the same theme that you would "
            f"{AGREE_LABEL[degree]} with.")


def rate_prompt(theme, own_opinion, other_opinion):
    """Where an opinion sits, with no speaker and no trust attached.

    Asked one opinion at a time: shown a list, a model ranks within the list and
    returns a tidy spread, which would manufacture the agreement between
    requested and measured degree that this rating exists to test.
    """
    return (f"Consider {theme.statement}.\n\n"
            f"You currently have this opinion on that theme:\n\n"
            f"  \"{own_opinion}\"\n\n"
            f"Another opinion on that theme is:\n\n  \"{other_opinion}\"\n\n"
            f"Rate your degree of agreement with it using the scale "
            f"{_scale(AGREE_LABEL)}.")






# ---------------------------------------------------------------------------
# The trust-curve experiment
#
# Three questions, and the reason all three are about the *other* agent.  An
# earlier design asked the receiver "what is the chance your own view is the
# correct one" to get |h_w|.  A model told "this is your opinion" reads holding
# an opinion as a commitment, so that question is either answered at the ceiling
# or answered by stepping outside the persona and pricing the proposition
# neutrally -- two different constructs with no way to tell which came back.  It
# is the same failure that put all 820 stage-1 baselines at +2, in probability
# form.  Nothing below asks the agent to price its own belief; conviction is
# recovered by inversion in ``llmmod.fields`` instead.
#
# Prior trust is *built* from a track record rather than asserted, so there is no
# rung for the model to echo back, and the before/after pair is a difference of
# two answers to the identical question.
# ---------------------------------------------------------------------------

CHANCE_SCHEMA = {
    "type": "object",
    "properties": {"chance": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["chance"], "additionalProperties": False,
}

#: How firmly the receiver is told it holds its own view.  This is a
#: manipulation of ``|h_w|``, not a measurement of it: the model's prediction is
#: that the firm arm shows the larger ``|Delta h_mu|``, and that ordering is
#: something the run can contradict.
CONVICTION = {
    "firm": "You hold that opinion firmly.",
    "tentative": "You hold that opinion only tentatively.",
}


def _context(theme, own_opinion, conviction, history):
    """Everything the three questions share, to the character.

    The three calls must differ *only* in the question asked and, for the third,
    in the added message.  Building the shared part once is what guarantees
    that: a difference introduced by editing one prompt and not another would be
    indistinguishable from an effect.
    """
    lines = [f"Consider {theme.statement}.\n",
             "You currently have this opinion on that theme:\n",
             f"  \"{own_opinion}\"\n",
             CONVICTION[conviction] + "\n",
             f"You have discussed this theme with another agent "
             f"{len(history)} times before. On each occasion they said:\n"]
    lines += [f"  {i}. \"{t}\"" for i, t in enumerate(history, 1)]
    return "\n".join(lines)


def reliability_prompt(theme, own_opinion, conviction, history, message=None):
    """The trust question, before the message and again after it.

    Asked about the emitter alone -- not about whether the receiver would agree
    with them -- because only then is the answer a function of ``h_mu`` and
    usable as an axis.  ``message=None`` is the before call; passing a message
    adds it and changes nothing else.
    """
    body = _context(theme, own_opinion, conviction, history)
    if message is not None:
        body += f"\n\nThey have now said:\n\n  \"{message}\""
    return (body + "\n\nWhat is the chance, from 0 to 100, that this agent's "
                   "judgement on this theme is sound?")


def agreement_prompt(theme, own_opinion, conviction, history, message=None):
    """The evidence ``Z``, asked as a prediction about the emitter's next move.

    Same context as :func:`reliability_prompt`, different question.  Asked
    before any message is shown, so it is a genuine prediction rather than a
    reconstruction of one.

    Asked *again* with a message added, it is the second half of the opinion
    sector: the reliability answer pins how wrong the emitter is thought to be,
    and this one then inverts for how firmly the receiver now holds its own
    view.  The difference of the two inversions is ``Delta h_w``, and it is
    obtained without ever asking the agent to price its own belief -- both
    questions are about the emitter, and only the arithmetic is about the
    receiver.
    """
    body = _context(theme, own_opinion, conviction, history)
    if message is not None:
        body += f"\n\nThey have now said:\n\n  \"{message}\""
    return (body + "\n\nWhat is the chance, from 0 to 100, that their next "
                   "statement on this theme is one you would agree with?")


def conviction_prompt(theme, own_opinion, conviction, history, message=None):
    """Conviction read directly, with no inversion and no singularity.

    The agreement prediction is ``q = c + e - 2ce``.  Asked about an emitter the
    receiver holds to be sound, ``e -> 0`` and the whole expression collapses to
    ``q = c``: the chance of agreeing with a reliable source *is* the chance of
    being right.  So the same question, pointed at a stipulated-reliable third
    party, measures the conviction that :func:`agreement_prompt` can only reach
    through a quotient which is singular at neutral trust and censored wherever
    the receiver expects to disagree more strongly than ``c = 1`` permits.

    Still nothing is asked about the agent's own belief -- this is a prediction
    about a third party, and only the algebra is about the receiver.  The third
    party is explicitly not the interlocutor, so the trust learned in context
    does not carry over.
    """
    body = _context(theme, own_opinion, conviction, history)
    if message is not None:
        body += f"\n\nThey have now said:\n\n  \"{message}\""
    return (body + "\n\nSeparately, a third agent -- one whose judgement on "
                   "this theme you consider sound -- will state their view on "
                   "it. What is the chance, from 0 to 100, that it is one you "
                   "would agree with?")
