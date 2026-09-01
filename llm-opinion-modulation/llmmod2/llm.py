"""One call to the model, and many of them at once.

Adapted from ``llm-agent-modulation/llmmod/llm.py``.  Three things live here and
nowhere else: where credentials come from, how a request is turned into a cache
key, and how concurrency is bounded.

The one substantive change is ``n``.  This experiment reads a *frequency* of
one-word verdicts rather than a stated number, so the unit of measurement is a
batch of independent draws from one prompt, not a single answer.  ``n`` is part
of the request, part of the cache key, and the cached value is the whole list of
parsed answers; a rung of the ladder is one cached entry.  The model caps ``n``
at 8, so a rung wider than that is assembled from several calls distinguished by
``nonce``.

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

__all__ = ["MODEL", "MAX_N", "CACHE_DIR", "load_env", "ask", "ask_many",
           "usage_total", "cost_estimate"]

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

MODEL = "gpt-5.6-luna"

#: Retries per request.  Empty completions come back intermittently under
#: concurrency -- the identical call succeeds on the next attempt -- and a
#: failure that reaches the caller costs a whole generated rung, so it is
#: retried here rather than handled as data.  Nothing is cached until it parses.
RETRIES = 4

#: Reasoning depth.  These are snap judgements from a short briefing; the
#: quantity being measured is a first-order response, not the product of
#: deliberation, and a long chain of thought would be measuring something else.
EFFORT = "low"

#: Draws per request.  The API rejects more than this with a 400, so a wider
#: rung is several calls at different ``nonce``.
MAX_N = 8

_usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0, "cached": 0}
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


@lru_cache(maxsize=1)
def _client():
    """One client for the process.

    The sibling builds a fresh one per call, which is fine at eight workers and
    is not at forty: each client owns a connection pool, and a sweep that opens
    one per request runs the file descriptors out before it runs the ladder out.
    The client is documented as thread-safe, so a single cached instance is the
    right shape here.
    """
    from openai import OpenAI
    load_env()
    return OpenAI()


def _key(model, system, user, schema, effort, nonce, n):
    blob = json.dumps([model, system, user, schema, effort, nonce, n], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


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

    Choices that come back empty because the completion hit the token cap are
    dropped rather than retried: they carry no verdict, and the caller records
    the surviving count so a thin rung is visible in the data.
    """
    if n > MAX_N:
        raise ValueError(f"n={n} exceeds the API maximum of {MAX_N}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(model, system, user, schema, effort, nonce, n)}.json"
    if path.exists():
        with _lock:
            _usage["cached"] += 1
        return json.loads(path.read_text())["parsed"]

    client = _client()
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "answer", "strict": True,
                                                 "schema": schema}},
                reasoning_effort=effort,
                max_completion_tokens=max_tokens,
                n=n,
            )
            parsed = [json.loads(c.message.content)
                      for c in resp.choices if c.message.content]
            if not parsed:
                raise ValueError(
                    f"no usable choice (finish_reason="
                    f"{resp.choices[0].finish_reason!r})")
            break
        except Exception:  # noqa: BLE001 - retried, then re-raised
            if attempt == RETRIES - 1:
                raise
            time.sleep(0.5 * 2 ** attempt + random.random() * 0.3)
    u = resp.usage
    with _lock:
        _usage["calls"] += 1
        _usage["prompt"] += u.prompt_tokens
        _usage["completion"] += u.completion_tokens
        _usage["reasoning"] += getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
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


#: $/M tokens for :data:`MODEL`, as used by the sibling experiment.
PRICE_IN, PRICE_OUT = 0.20, 1.20


def cost_estimate():
    """Dollars spent on live calls this process."""
    return (_usage["prompt"] * PRICE_IN + _usage["completion"] * PRICE_OUT) / 1e6


def usage_total():
    """Tokens spent this process, the call counts, and an estimate in dollars."""
    return dict(_usage, dollars=cost_estimate())
