"""The six themes.

Chosen against three constraints, each for a reason:

1. **Genuinely open.**  No fact settles them, so the agent has to reason from
   the opinion it was given rather than retrieve a correct answer.
2. **Low tribal valence.**  A theme that maps onto an existing political
   alignment measures what the model was trained to say about that alignment.
   These are all arguable in good faith from either side by the same person.
3. **Statable in a sentence or two.**  Otherwise message length starts varying
   with the degree of disagreement, which puts a gradient across the grid that
   has nothing to do with the mechanism being measured.

Twenty, because themes are the replication unit.  A cell mean is an average over
themes, and its honest error bar is the spread *between* them -- the 36 calls in a
cell are six or twenty clusters, not thirty-six independent draws.  With six
clusters that error bar is itself too noisy to trust; twenty makes it mean
something.  Themes also make the generation step checkable: if the model writes
every "disagree" opinion as a strawman, that shows up as something common to all
of them rather than as an effect.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Theme", "THEMES", "by_key"]


@dataclass(frozen=True)
class Theme:
    key: str
    statement: str


THEMES = (
    Theme("cities",
          "whether a city should be shaped mainly for the people who already "
          "live in it, or for the people who might come to live in it"),
    Theme("career",
          "whether a working life spent deepening a single craft is worth more "
          "than one spent moving between several"),
    Theme("difficulty",
          "whether difficulty is a necessary part of a good education, or "
          "mostly an obstacle that good teaching should remove"),
    Theme("translation",
          "whether a translation should stay faithful to the words of the "
          "original, or to the effect the original had on its first readers"),
    Theme("wilderness",
          "whether wild places are better left entirely alone, or actively "
          "managed to keep them as they are"),
    Theme("food",
          "whether a food tradition is better preserved as it was handed down, "
          "or allowed to change with the people who cook it"),
    Theme("library",
          "whether a public library should stock what its readers ask for, or "
          "what will still be worth reading in fifty years"),
    Theme("craft",
          "whether a craft is better learned by imitating a master closely, or "
          "by working out one's own method from the start"),
    Theme("river",
          "whether a town on a flooding river should build higher defences, or "
          "move away from the water"),
    Theme("rules",
          "whether the rules of a game should stay fixed, or be revised "
          "whenever players find a loophole in them"),
    Theme("score",
          "whether an orchestra should play a score as it is written, or as "
          "its conductor hears it"),
    Theme("usage",
          "whether a body that governs a language should record how people "
          "actually speak, or set out how they ought to"),
    Theme("museum",
          "whether a museum should return its objects to the places they were "
          "made, or keep a collection together where it can be compared"),
    Theme("reading",
          "whether a long book is better read slowly over months, or quickly "
          "in a few sittings"),
    Theme("maps",
          "whether a map should show the world as it is measured, or as it is "
          "experienced by the people travelling through it"),
    Theme("garden",
          "whether a garden should be planted for the season ahead, or for "
          "what it will have become in twenty years"),
    Theme("recipe",
          "whether a recipe is better followed exactly, or treated as a "
          "starting point to depart from"),
    Theme("studies",
          "whether a field of research advances more by many small careful "
          "studies, or by a few bold and risky ones"),
    Theme("house",
          "whether a house is better built to last centuries, or built cheaply "
          "so it can be replaced when needs change"),
    Theme("league",
          "whether a sports league should even out the strength of its teams, "
          "or let the strongest ones dominate"),
)


def by_key(key):
    for t in THEMES:
        if t.key == key:
            return t
    raise KeyError(key)
