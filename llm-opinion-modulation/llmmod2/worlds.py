"""The synthetic settings the agents reason about, and how they are built.

Why invent worlds at all.  The quantity this experiment needs is a *controlled*
conviction: the receiver must hold a belief of a known strength before anyone
speaks to it.  On a real topic that is impossible -- the model arrives with a
prior nobody set -- which is why the sibling experiment can only elicit ``h_w``
and ends up sampling the plane wherever the model happens to land.  On an
invented question the model has no prior at all, so its entire belief comes from
evidence we hand it, and the coordinate becomes a knob rather than a reading.

A world is a setting, a unit of evidence, a colleague, a predicate, and a few
issues.  Everything the briefing says is composed in :mod:`llmmod2.prompts` from
these pieces by fixed templates, so no generated text ever reaches the model in
a structural position.  The generator supplies names and a verb phrase; it does
not supply prompt scaffolding.

The set is generated once and frozen to ``data/worlds/index.json``, for the same
reason the sibling freezes ``data/themes/``: stage 2 is built on this material
and it must not shift under a half-finished analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .llm import ask

ROOT = Path(__file__).resolve().parent.parent
WORLD_DIR = ROOT / "data" / "worlds"
INDEX = WORLD_DIR / "index.json"

#: Issues per world.  The first is the focal one, on which the opinion update is
#: measured; the second is the fresh one, on which the colleague's testimony is
#: weighed before and after they speak on the first.  The third is a spare, used
#: when a world's first issue fails the prior-free screen in stage 1.
ISSUES_PER_WORLD = 3


@dataclass(frozen=True)
class Issue:
    a: str
    b: str

    def name(self, d):
        """The entity favoured by direction ``d`` (``+1`` is ``a``)."""
        return self.a if d > 0 else self.b


@dataclass(frozen=True)
class World:
    key: str
    setting: str
    place: str
    unit_singular: str
    unit_plural: str
    colleague: str
    predicate: str
    issues: tuple

    def issue(self, i):
        return self.issues[i]


WORLD_SCHEMA = {
    "type": "object",
    "properties": {
        "worlds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "place": {"type": "string"},
                    "setting": {"type": "string"},
                    "unit_singular": {"type": "string"},
                    "unit_plural": {"type": "string"},
                    "colleague": {"type": "string"},
                    "predicate": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"a": {"type": "string"},
                                           "b": {"type": "string"}},
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                        "minItems": ISSUES_PER_WORLD,
                        "maxItems": ISSUES_PER_WORLD,
                    },
                },
                "required": ["key", "place", "setting", "unit_singular",
                             "unit_plural", "colleague", "predicate", "issues"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["worlds"],
    "additionalProperties": False,
}

GENERATOR_SYSTEM = (
    "You invent neutral fictional reference material for a psychology "
    "experiment. Nothing you invent may correspond to anything real."
)


def generator_prompt(n):
    return (
        f"Invent {n} unrelated fictional scholarly settings.\n\n"
        "Each one is an archive, survey, register or catalogue belonging to a "
        "civilisation, institution or expedition that never existed. For each, give:\n\n"
        "  key             a short lowercase one-word identifier, unique across the set\n"
        "  place           the invented civilisation, institution or expedition\n"
        "  setting         one sentence naming the body of records and what it holds\n"
        "  unit_singular   the name of one piece of evidence in it, e.g. 'archival index'\n"
        "  unit_plural     the plural of that\n"
        "  colleague       a title and invented surname, e.g. 'Archivist Deth'\n"
        "  predicate       a third-person verb phrase stating a comparison that is\n"
        "                  settled by the records, written so that it can follow the\n"
        "                  name of either candidate and remain grammatical, e.g.\n"
        "                  'was completed first', 'stood at the greater altitude',\n"
        "                  'carried the heavier traffic'\n"
        "  issues          three pairs of invented entities from that setting, each\n"
        "                  pair being two things the predicate can be asked about\n\n"
        "Requirements. The two entities in a pair must be of the same kind and of "
        "comparable standing, so that neither is the obvious answer to someone who "
        "knows nothing. The predicate must be decidable in principle but not "
        "guessable from the names. Avoid any name that resembles a real place, "
        "person, work or organisation, and avoid loaded or evaluative subject "
        "matter entirely: nothing about merit, morality, groups of people, or "
        "anything a reader could hold an opinion about. These are dry questions of "
        "record."
    )


def generate(n, max_tokens=16000):
    """Ask the model for ``n`` worlds.  Cached, so a rerun costs nothing."""
    out = ask(GENERATOR_SYSTEM, generator_prompt(n), WORLD_SCHEMA,
              max_tokens=max_tokens, n=1)
    return out[0]["worlds"]


def freeze(raw):
    """Write the generated worlds to ``data/worlds/``."""
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    return INDEX


def load(keep=None):
    """The frozen worlds, optionally restricted to ``keep`` keys, in file order."""
    if not INDEX.is_file():
        raise SystemExit(f"no worlds at {INDEX}; run scripts/stage0_worlds.py first")
    raw = json.loads(INDEX.read_text())
    worlds = [World(key=w["key"], setting=w["setting"], place=w["place"],
                    unit_singular=w["unit_singular"], unit_plural=w["unit_plural"],
                    colleague=w["colleague"], predicate=w["predicate"],
                    issues=tuple(Issue(i["a"], i["b"]) for i in w["issues"]))
              for w in raw]
    if keep is not None:
        keep = set(keep)
        worlds = [w for w in worlds if w.key in keep]
    return worlds
