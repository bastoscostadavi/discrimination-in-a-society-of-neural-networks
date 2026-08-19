"""Stage 1: build the stimulus material, and measure where it actually landed.

For each theme the agent states its own opinion ``OA``, then is asked for three
opinions at each of the five degrees of agreement, then rates each of those
fifteen back on the same five-point scale.

The rating is not a formality.  **An opinion belongs in the column its rating
puts it in, not the one it was requested at.**  Asking for "somewhat disagree"
and getting something the agent then rates ``-2`` is not a failure of the
generation, it is the measurement doing its job; silently trusting the request
would put the opinion in the wrong column and smear the map.  Three opinions per
degree exist precisely so that after reassignment there is still something left
in every column.

Both numbers are kept.  ``requested`` and ``measured`` are written to the theme
file side by side, so the drift is visible in the data rather than absorbed by
it, and so the generation can be audited for the obvious artefact -- a model that
writes every "strongly disagree" opinion as a strawman will show it here, as
requested ``-2`` opinions rating ``-2`` far more cleanly than ``-1`` opinions
rate ``-1``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .llm import MODEL, ask, ask_many
from .prompts import (OPINION_SCHEMA, RATE_SCHEMA, SYSTEM, generate_prompt,
                      generate_schema, opinion_prompt, rate_prompt)
from .scale import LEVELS
from .themes import THEMES

__all__ = ["THEME_DIR", "generate_candidates", "borrow", "build_theme",
           "build_all", "load_theme", "load_all", "write_markdown"]

THEME_DIR = Path(__file__).resolve().parent.parent / "data" / "themes"

#: Opinions generated per requested degree.  Over-generated on purpose: the
#: requested degree turns out to be a weak predictor of the measured one --
#: asking for "somewhat agree" returns something rated "agree" every time -- so
#: candidates are drawn broadly at every rung and sorted afterwards by where the
#: cross-check actually puts them.
PER_DEGREE = 8

#: Candidates borrowed from *other* themes, per theme, to fill the neutral
#: column.
#:
#: Nothing written about a theme is ever rated 0 on that theme: across 90
#: within-theme opinions the neutral column came back empty, and four different
#: ways of asking for a neutral opinion (purely factual statements, deliberately
#: balanced ones, narrow sub-questions, rejections of the framing) all came back
#: rated as agreement.  The reason is that neutral does not mean lukewarm.  In
#: the model ``h_w = 0`` is *orthogonality*: the message bears on a direction the
#: receiver's opinion does not span.  An opinion about a different theme
#: entirely is exactly that, and it is the one construction that reaches the
#: middle of the axis.  These are still rated in the receiving theme's context
#: like every other candidate, so they earn their column rather than being
#: assigned it.
CROSS_THEME = 6

#: How many candidates to keep per column.  Extras are recorded but unused, so
#: the columns are balanced and a column that happened to attract twenty
#: candidates does not outweigh one that attracted four.
KEEP_PER_COLUMN = 3


def generate_candidates(theme, model=MODEL, per_degree=PER_DEGREE, workers=8):
    """Pass one: agent A's own opinion, and the within-theme candidates.

    Rating is deliberately not done here.  The neutral column is filled from
    other themes, so every theme's candidates have to exist before any theme's
    ratings can be asked for.
    """
    own = ask(SYSTEM, opinion_prompt(theme), OPINION_SCHEMA,
              model=model)["opinion"]
    gen = ask_many(
        [{"system": SYSTEM,
          "user": generate_prompt(theme, own, d, per_degree),
          "schema": generate_schema(per_degree), "model": model} for d in LEVELS],
        workers=workers, label=f"{theme.key}:generate", progress=False,
    )
    opinions = []
    for d, result in zip(LEVELS, gen):
        if "__error__" in result:
            print(f"  [{theme.key}] generation failed at {d:+d}: "
                  f"{result['__error__'][:100]}")
            continue
        for k, text in enumerate(result["opinions"]):
            opinions.append({"id": f"{theme.key}-{d:+d}-{k}".replace("+", "p"),
                             "requested": d, "source": theme.key,
                             "text": text.strip()})
    return own, opinions


def borrow(pools, key, n=CROSS_THEME):
    """Pick ``n`` opinions from the other themes, round robin and deterministic.

    Round robin so the borrowed opinions come from several themes rather than
    one, and deterministic so a rerun uses the same material -- the stimulus set
    has to be stable for the ratings and the grid to refer to the same thing.
    """
    others = [k for k in pools if k != key]
    picked = []
    i = 0
    while len(picked) < n and others:
        src = others[i % len(others)]
        pool = pools[src]
        j = i // len(others)
        if j < len(pool):
            o = pool[j]
            picked.append({"id": f"{key}-cross-{src}-{j}", "requested": None,
                           "source": src, "text": o["text"]})
        i += 1
        if i > 4 * n + len(others) * 8:
            break
    return picked


def build_theme(theme, own, opinions, model=MODEL, keep=KEEP_PER_COLUMN, workers=8):
    """Pass two: rate every candidate in this theme's context, then select."""
    rated = ask_many(
        [{"system": SYSTEM, "user": rate_prompt(theme, own, o["text"]),
          "schema": RATE_SCHEMA, "model": model} for o in opinions],
        workers=workers, label=f"{theme.key}:rate", progress=False,
    )
    for o, r in zip(opinions, rated):
        o["measured"] = None if "__error__" in r else int(r["rating"])
        o["slot"] = o["measured"]

    # the same number per column, so a column that attracted many candidates
    # does not outweigh a thin one; prefer opinions whose requested degree
    # already agreed with the measurement, as the least borderline available
    for d in LEVELS:
        pool = sorted((o for o in opinions if o["slot"] == d),
                      key=lambda o: (o["requested"] != d, o["id"]))
        for rank, o in enumerate(pool):
            o["used"] = rank < keep
    for o in opinions:
        o.setdefault("used", False)

    used = [o for o in opinions if o["used"]]
    ok = [o for o in opinions if o["slot"] is not None]
    within = [o for o in ok if o["requested"] is not None]
    return {
        "theme": {"key": theme.key, "statement": theme.statement},
        "model": model,
        "own_opinion": own,
        "keep_per_column": keep,
        "opinions": opinions,
        "coverage": {str(d): sum(1 for o in used if o["slot"] == d) for d in LEVELS},
        "candidates": {str(d): sum(1 for o in ok if o["slot"] == d) for d in LEVELS},
        "moved": sum(1 for o in within if o["slot"] != o["requested"]),
        "n_within": len(within),
    }


def build_all(themes=THEMES, model=MODEL, per_degree=PER_DEGREE,
              keep=KEEP_PER_COLUMN, cross=CROSS_THEME, workers=8):
    THEME_DIR.mkdir(parents=True, exist_ok=True)

    print("[stage1] pass 1: own opinions and within-theme candidates")
    owns, pools = {}, {}
    for theme in themes:
        own, opinions = generate_candidates(theme, model=model,
                                            per_degree=per_degree, workers=workers)
        owns[theme.key], pools[theme.key] = own, opinions
        print(f"           {theme.key:<12s} {len(opinions)} candidates")

    print("[stage1] pass 2: borrow across themes, then rate everything in context")
    records = []
    for theme in themes:
        opinions = pools[theme.key] + borrow(pools, theme.key, cross)
        rec = build_theme(theme, owns[theme.key], opinions, model=model,
                          keep=keep, workers=workers)
        (THEME_DIR / f"{theme.key}.json").write_text(json.dumps(rec, indent=2))
        cov = " ".join(f"{d:+d}:{rec['coverage'][str(d)]}/"
                       f"{rec['candidates'][str(d)]}" for d in LEVELS)
        print(f"           {theme.key:<12s} used/available {cov}   "
              f"reassigned {rec['moved']}/{rec['n_within']}")
        records.append(rec)

    (THEME_DIR / "index.json").write_text(json.dumps({
        "model": model,
        "themes": [{"key": r["theme"]["key"], "own_opinion": r["own_opinion"],
                    "coverage": r["coverage"], "candidates": r["candidates"],
                    "moved": r["moved"],
                    "n_opinions": len(r["opinions"])} for r in records],
    }, indent=2))
    write_markdown(records)
    return records


def load_theme(key):
    return json.loads((THEME_DIR / f"{key}.json").read_text())


def load_all(themes=THEMES):
    return [load_theme(t.key) for t in themes
            if (THEME_DIR / f"{t.key}.json").exists()]


def write_markdown(records, path=None):
    """A readable dump of the whole stimulus set.

    The JSON is what the code reads; this is what a person reads when deciding
    whether the generated opinions are any good, which is a judgement no
    statistic makes for you.
    """
    path = path or THEME_DIR / "OPINIONS.md"
    out = ["# Stimulus material", "",
           "Generated by stage 1, one section per theme.", "",
           "`asked` is the degree the opinion was requested at, or `cross` for "
           "one borrowed from another theme (the `from` column names which) to "
           "reach the neutral rung. `rated` is the "
           "degree agent A afterwards rated it at, on the same five-point scale — "
           "**that is the column the experiment uses**, and where the two differ "
           "the rating wins. `used` marks the opinions stage 2 actually shows; "
           "the rest are surplus from over-generation and are kept so the "
           "selection is auditable.", ""]
    for r in records:
        out += [f"## {r['theme']['key']}", "",
                f"**Theme.** {r['theme']['statement'].capitalize()}.", "",
                f"**Agent A's own opinion (OA).** {r['own_opinion']}", "",
                "| asked | rated | used | from | opinion |",
                "|--:|--:|:-:|:--|---|"]
        key = r["theme"]["key"]
        def _order(o):
            """Generated rungs first, in order, then borrowed, then hand-written.

            ``requested`` is an int for generated opinions, ``None`` for one
            borrowed from another theme and ``"hand"`` for one written by a
            person, so it cannot be sorted on directly.
            """
            req = o["requested"]
            if isinstance(req, int):
                return (0, req, o["id"])
            return (1 if req is None else 2, 0, o["id"])

        for o in sorted(r["opinions"], key=_order):
            req = o["requested"]
            asked = ("cross" if req is None else
                     req if isinstance(req, str) else f"{req:+d}")
            m = "--" if o["measured"] is None else f"{o['measured']:+d}"
            flag = (" **←**" if isinstance(req, int)
                    and o["measured"] not in (None, req) else "")
            src = "" if o["source"] == key else o["source"]
            out.append(f"| {asked} | {m}{flag} | {'x' if o['used'] else ''} | "
                       f"{src} | {o['text']} |")
        cov = ", ".join(f"`{d:+d}`: {r['coverage'][str(d)]}/{r['candidates'][str(d)]}"
                        for d in LEVELS)
        out += ["", f"Used / available per column — {cov}.", ""]
    Path(path).write_text("\n".join(out))
    print(f"[stage1] {path}")
    return path
