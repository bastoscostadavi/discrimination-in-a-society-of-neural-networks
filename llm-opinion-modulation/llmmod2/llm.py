"""One call to the model, and many of them at once.

Adapted from ``llm-agent-modulation/llmmod/llm.py``.  Three things live here and
nowhere else: where credentials come from, how a request is turned into a cache
key, and how concurrency is bounded.

The one substantive change is ``n``.  This experiment reads a *frequency* of
one-word verdicts rather than a stated number, so the unit of measurement is a
batch of independent draws from one prompt, not a single answer.  ``n`` is part
of the request, part of the cache key, and the cached value is the whole list of
parsed answers; a rung of the ladder is one cached entry.

**Caching is not an optimization here, it is part of the method.**  Stage 0
invents the worlds that everything afterwards is built on, and stage 1 decides
which of them earn a place; if a rerun redrew either, the stimulus would shift
under a half-finished analysis.  It also makes a rung of the ladder replayable:
the same briefing must return the same verdicts, or a sweep that revisits a rung
would be measuring the sampler rather than the model.  Every request is keyed by
a hash of exactly what was sent -- model, system, user, schema, effort, and the
batch size -- so a rerun with all of those unchanged replays from disk, and a
rerun with any of them changed is a different key and actually goes out.  Delete
``data/cache/`` to force a fresh draw.

Two routes
----------

The measurement reported in the README is ``gpt-5.6-luna``, called directly.  A
second route exists so the same instrument can be pointed at a model from
another family, which is the only way to tell a property of the update from a
property of one model: a name containing a ``/`` is a model *slug* and goes to
OpenRouter, which speaks the same chat-completions dialect, so nothing else in
the experiment knows which route it is on.  :data:`PRICES` lists what is wired
up and ``--model`` on every stage script chooses.  The model is part of the cache
key, so a switch neither reads another model's verdicts out of the cache nor
disturbs them -- and it does pay for the whole sweep again.

Two things differ on the routed path, and both were found by trying it rather
than by reading about it:

``n`` **is silently ignored.**  A request for four choices from an
Anthropic-served model comes back with one, HTTP 200 and no warning.  Left
alone, that is the worst kind of failure here: a rung whose cache key says eight
draws would hold one, the psychometric fit would be reading a single verdict as
a frequency, and nothing in the output would look wrong.  So the routed path
never asks for a batch.  It issues ``n`` separate requests, each paying for the
briefing again, and :func:`_choices` asserts on both routes that as many
verdicts came back as were asked for.  The ladder and the cache entries are
unchanged, so rungs stay comparable across routes; the sweep costs about eight
times the input tokens, which is the dominant term.  Budget with ``--max-cost``.

``reasoning_effort`` **has nothing to act on.**  :data:`EFFORT` asks for the
shallowest reasoning available, because the quantity being measured is a
first-order response.  Haiku 4.5 has no such dial -- it takes a thinking budget
or no thinking -- so the parameter is not sent on the routed path, leaving
thinking off, which is the shallowest that family goes; completions come back at
around seventeen tokens, which is the verdict and no deliberation.  It stays in
the cache key on both routes as the reasoning depth that was *asked* for.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

__all__ = ["MODEL", "MAX_N", "PRICES", "CACHE_DIR", "load_env", "ask",
           "ask_many", "usage_total", "cost_estimate"]

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

MODEL = "gpt-5.6-luna"

#: Where a routed model is called.  OpenRouter speaks chat completions, so the
#: only difference from a direct call is this address and the credential.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

#: $/M tokens for input and output, per model.  A model that is not listed is
#: refused rather than run: these sweeps are tens of thousands of calls, and a
#: missing price would silently disarm ``--max-cost``, which is the only thing
#: standing between a typo in ``--model`` and an unbounded bill.
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}

#: Retries per request.  Empty completions come back intermittently under
#: concurrency -- the identical call succeeds on the next attempt -- and a
#: failure that reaches the caller costs a whole generated rung, so it is
#: retried here rather than handled as data.  Nothing is cached until it parses.
RETRIES = 4

#: Reasoning depth.  These are snap judgements from a short briefing; the
#: quantity being measured is a first-order response, not the product of
#: deliberation, and a long chain of thought would be measuring something else.
EFFORT = "low"

#: Draws per request.  The OpenAI API rejects more than this with a 400, so a
#: wider rung is several calls at different ``nonce``.  The routed path has no
#: working batch parameter at all and issues the draws one at a time, but keeps
#: the same cap so that a rung splits into the same cache entries on both routes
#: and the two runs stay comparable entry for entry.
MAX_N = 8

_usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0, "cached": 0,
          "dollars": 0.0}
_lock = threading.Lock()


def load_env():
    """Read ``.env`` from this directory or the repository root into the process.

    Only fills keys that are not already set, so an exported variable still
    wins.  Values are not logged anywhere.
    """
    for candidate in (ROOT / ".env", ROOT.parent / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _routed(model):
    """Whether ``model`` is a slug to be called through OpenRouter."""
    return "/" in model


@lru_cache(maxsize=2)
def _client(routed):
    """One client per route for the process.

    The sibling builds a fresh one per call, which is fine at eight workers and
    is not at forty: each client owns a connection pool, and a sweep that opens
    one per request runs the file descriptors out before it runs the ladder out.
    The client is documented as thread-safe, so a single cached instance per
    route is the right shape here.
    """
    from openai import OpenAI
    load_env()
    if routed:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set; a routed model "
                             "cannot be called without it")
        return OpenAI(base_url=OPENROUTER_BASE, api_key=key)
    return OpenAI()


def _record(model, prompt_tokens, completion_tokens, reasoning_tokens):
    """Add one live call to the running total, priced for its own model."""
    price_in, price_out = PRICES[model]
    with _lock:
        _usage["calls"] += 1
        _usage["prompt"] += prompt_tokens
        _usage["completion"] += completion_tokens
        _usage["reasoning"] += reasoning_tokens
        _usage["dollars"] += (prompt_tokens * price_in
                              + completion_tokens * price_out) / 1e6


def _key(model, system, user, schema, effort, nonce, n):
    blob = json.dumps([model, system, user, schema, effort, nonce, n], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _choices(resp, want):
    """The parsed verdicts of one response, checked for arity.

    Choices that come back empty because the completion hit the token cap are
    dropped rather than retried: they carry no verdict, and the caller records
    the surviving count so a thin rung is visible in the data.  A response that
    is short of choices for any *other* reason is a silently ignored ``n``, and
    is raised rather than accepted -- a rung that quietly holds one draw where
    its key says eight would corrupt the frequency the ladder is built on, and
    would leave no trace in the output.
    """
    parsed = [json.loads(c.message.content)
              for c in resp.choices if c.message.content]
    if not parsed:
        raise ValueError(f"no usable choice (finish_reason="
                         f"{resp.choices[0].finish_reason!r})")
    if len(resp.choices) < want:
        raise ValueError(f"asked for {want} choices, got {len(resp.choices)}; "
                         f"this endpoint does not honour n")
    return parsed


def _call(client, system, user, schema, model, max_tokens, n, effort=None):
    """One request, retried.  Returns the parsed verdicts it carried."""
    extra = {"reasoning_effort": effort} if effort is not None else {}
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "answer", "strict": True,
                                                 "schema": schema}},
                max_completion_tokens=max_tokens,
                n=n,
                **extra,
            )
            parsed = _choices(resp, n)
            u = resp.usage
            _record(model, u.prompt_tokens, u.completion_tokens,
                    getattr(getattr(u, "completion_tokens_details", None),
                            "reasoning_tokens", 0) or 0)
            return parsed
        except Exception:  # noqa: BLE001 - retried, then re-raised
            if attempt == RETRIES - 1:
                raise
            time.sleep(0.5 * 2 ** attempt + random.random() * 0.3)


def _draw_direct(system, user, schema, model, effort, max_tokens, n):
    """A rung as one request for ``n`` choices."""
    return _call(_client(False), system, user, schema, model, max_tokens, n,
                 effort=effort)


def _draw_routed(system, user, schema, model, max_tokens, n):
    """A rung as ``n`` independent requests, because ``n`` is not honoured.

    Issued in sequence rather than concurrently.  The callers already run
    twenty-four cells at a time and each rung already splits into two batches, so
    a third layer of fan-out here would put hundreds of requests in flight and
    spend the sweep in rate-limit backoff.

    A draw that will not come back after its retries ends the rung early instead
    of losing it: the caller records the surviving count, which is the same
    treatment the direct path gives a choice that came back empty.  A rung with
    nothing in it does raise -- there is no reading in it to keep.
    """
    client = _client(True)
    out = []
    for _ in range(n):
        try:
            out += _call(client, system, user, schema, model, max_tokens, 1)
        except Exception:  # noqa: BLE001 - a thin rung, visible in the data
            if not out:
                raise
            break
    return out


def ask(system, user, schema, model=MODEL, effort=EFFORT, max_tokens=2000,
        nonce=0, n=1):
    """One structured call.  Returns a list of ``n`` parsed objects.

    ``schema`` is the JSON Schema of the answer; strict structured output means
    the response is guaranteed to parse, so there is no free-text fallback to
    maintain and no partial answers to guess at.

    ``nonce`` enters the cache key but not the request.  It is how a genuinely
    repeated draw of the *same* prompt is taken: without it the cache would
    return one batch forever, and a rung wider than :data:`MAX_N` would be the
    same eight answers repeated instead of an independent extension of them.
    """
    if n > MAX_N:
        raise ValueError(f"n={n} exceeds the batch maximum of {MAX_N}")
    if model not in PRICES:
        raise ValueError(f"no price on record for {model!r}; add it to PRICES "
                         f"so that --max-cost can hold")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(model, system, user, schema, effort, nonce, n)}.json"
    if path.exists():
        with _lock:
            _usage["cached"] += 1
        return json.loads(path.read_text())["parsed"]

    if _routed(model):
        parsed = _draw_routed(system, user, schema, model, max_tokens, n)
    else:
        parsed = _draw_direct(system, user, schema, model, effort, max_tokens, n)
    path.write_text(json.dumps({"model": model, "system": system, "user": user,
                                "parsed": parsed}, indent=2))
    return parsed


def ask_many(requests, workers=8, label="", progress=True):
    """Run ``requests`` -- dicts of :func:`ask` keyword arguments -- concurrently.

    Order is preserved.  An exception on one request is returned in its slot
    rather than raised, so one bad cell cannot lose an otherwise complete sweep;
    callers filter for it.
    """
    def run(req):
        try:
            return ask(**req)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return {"__error__": f"{type(exc).__name__}: {exc}"}

    out = [None] * len(requests)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, r): i for i, r in enumerate(requests)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
            done += 1
            if progress and done % max(1, len(requests) // 20) == 0:
                print(f"  [{label}] {done}/{len(requests)}", flush=True)
    errors = [r for r in out if isinstance(r, dict) and "__error__" in r]
    if errors:
        print(f"  [{label}] {len(errors)} failed, e.g. {errors[0]['__error__'][:120]}")
    return out


def cost_estimate():
    """Dollars spent on live calls this process.

    Accumulated per call at that call's own price, rather than applied to a token
    total at the end, so a process that touches two models is still costed
    correctly.
    """
    with _lock:
        return _usage["dollars"]


def usage_total():
    """Tokens spent this process, the call counts, and an estimate in dollars."""
    with _lock:
        return dict(_usage)
