"""The five-point scale the stimulus material is built on.

Used only in stage 1, where the opinions are generated and rated.  An opinion is
requested at a named degree of agreement and then rated back on the same named
degrees, so the check that an opinion landed where it was aimed is a comparison
of like with like.

The measurement itself does not use this scale at all.  Everything in
:mod:`llmmod.fields` is a probability from $0$ to $100$, which is what the two
fields of the paper are actually written in and what has room to move at both
ends; an earlier version of this experiment read the update off these five rungs
instead and could not measure anything at the ends of either axis, because a
receiver told it strongly trusts someone has no rung left to move up.
"""

from __future__ import annotations

__all__ = ["LEVELS", "AGREE_LABEL"]

#: The five rungs an opinion can be generated at and rated on.
LEVELS = (-2, -1, 0, 1, 2)

#: How a degree of agreement is named, both when an opinion is *requested* at it
#: and when the agent is asked to *rate* one.  The same words in both places, so
#: the cross-check measures the construct it was asked for.
AGREE_LABEL = {
    -2: "disagree",
    -1: "somewhat disagree",
    0: "neutral",
    +1: "somewhat agree",
    +2: "agree",
}
