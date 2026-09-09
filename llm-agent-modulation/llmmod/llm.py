"""One call to the model, and many of them at once.

Three things live here and nowhere else: where credentials come from, how a
request is turned into a cache key, and how concurrency is bounded.

**Caching is not an optimization here, it is part of the method.**  Stage 1
generates the opinions that stage 2 is built on; if a rerun regenerated them the
stimulus material would silently change under a half-finished analysis.  Every
request is keyed by a hash of exactly what was sent -- model, system, user,
schema, effort -- so a rerun with any of those unchanged replays from disk and
returns the same text, and a rerun with any of them changed is a different key
and actually goes out.  Delete ``data/cache/`` to force a fresh draw.

Two routes
----------

The measurement reported in the README is ``gpt-5.6-luna``, called directly.  A
name containing a ``/`` is a model *slug* and is called through OpenRouter
instead, which speaks the same chat-completions dialect; :data:`PRICES` lists
what is wired up and ``--model`` already threads through every script.  This is
how the figure's two panels can be put to a second model without either
experiment learning what it is talking to.  The model is part of the cache key,
so a switch neither reads another model's answers out of the cache nor disturbs
them.

One parameter does not survive the trip.  :data:`EFFORT` asks for the shallowest
reasoning available, because the quantity Figure 1 is about is a first-order
response; Haiku 4.5 has no such dial -- it takes a thinking budget or no
thinking -- so ``reasoning_effort`` is not sent on the routed path, which leaves
thinking off, the shallowest that family goes.  It stays in the cache key on both
routes as the reasoning depth that was *asked* for.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["MODEL", "PRICES", "CACHE_DIR", "load_env", "ask", "ask_many",
           "usage_total", "cost_estimate"]

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

MODEL = "gpt-5.6-luna"

#: Where a routed model is called.  OpenRouter speaks chat completions, so the
#: only difference from a direct call is this address and the credential.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

#: $/M tokens for input and output, per model.  A model that is not listed is
#: refused rather than run, so that a typo in ``--model`` cannot spend against a
#: price nobody recorded.
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}

#: Retries per request.  Empty completions come back intermittently under
#: concurrency -- the identical call succeeds on the next attempt -- and a
#: failure that reaches the caller costs a whole generated rung, so it is
#: retried here rather than handled as data.  Nothing is cached until it parses.
RETRIES = 4

#: Reasoning depth.  These are snap judgements about one's own view; the
#: quantity Figure 1 is about is a first-order response, not the product of
#: deliberation, and a long chain of thought would be measuring something else.
EFFORT = "low"

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


def _client(model):
    from openai import OpenAI
    load_env()
    if _routed(model):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set; a routed model "
                             "cannot be called without it")
        return OpenAI(base_url=OPENROUTER_BASE, api_key=key)
    return OpenAI()


def _key(model, system, user, schema, effort, nonce):
    blob = json.dumps([model, system, user, schema, effort, nonce], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def ask(system, user, schema, model=MODEL, effort=EFFORT, max_tokens=4000, nonce=0):
    """One structured call.  Returns the parsed object.

    ``schema`` is the JSON Schema of the answer; strict structured output means
    the response is guaranteed to parse, so there is no free-text fallback to
    maintain and no partial answers to guess at.

    ``nonce`` enters the cache key but not the request.  It is how a genuinely
    repeated draw of the *same* prompt is taken: without it the cache would
    return one answer forever, and the baseline arm -- which is the same prompt
    for every opinion in a cell -- would contribute a single fixed number
    instead of an average over draws.
    """
    if model not in PRICES:
        raise ValueError(f"no price on record for {model!r}; add it to PRICES")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(model, system, user, schema, effort, nonce)}.json"
    if path.exists():
        with _lock:
            _usage["cached"] += 1
        return json.loads(path.read_text())["parsed"]

    client = _client(model)
    extra = {} if _routed(model) else {"reasoning_effort": effort}
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
                **extra,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError(
                    f"empty completion (finish_reason="
                    f"{resp.choices[0].finish_reason!r})")
            parsed = json.loads(content)
            break
        except Exception:  # noqa: BLE001 - retried, then re-raised
            if attempt == RETRIES - 1:
                raise
            time.sleep(0.5 * 2 ** attempt + random.random() * 0.3)
    u = resp.usage
    price_in, price_out = PRICES[model]
    reasoning = getattr(getattr(u, "completion_tokens_details", None),
                        "reasoning_tokens", 0) or 0
    with _lock:
        _usage["calls"] += 1
        _usage["prompt"] += u.prompt_tokens
        _usage["completion"] += u.completion_tokens
        _usage["reasoning"] += reasoning
        _usage["dollars"] += (u.prompt_tokens * price_in
                              + u.completion_tokens * price_out) / 1e6
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
        for fut, i in futures.items():
            out[i] = fut.result()
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
