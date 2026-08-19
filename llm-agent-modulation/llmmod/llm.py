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

__all__ = ["MODEL", "CACHE_DIR", "load_env", "ask", "ask_many", "usage_total"]

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

MODEL = "gpt-5.6-luna"

#: Retries per request.  Empty completions come back intermittently under
#: concurrency -- the identical call succeeds on the next attempt -- and a
#: failure that reaches the caller costs a whole generated rung, so it is
#: retried here rather than handled as data.  Nothing is cached until it parses.
RETRIES = 4

#: Reasoning depth.  These are snap judgements about one's own view; the
#: quantity Figure 1 is about is a first-order response, not the product of
#: deliberation, and a long chain of thought would be measuring something else.
EFFORT = "low"

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


def _client():
    from openai import OpenAI
    load_env()
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_key(model, system, user, schema, effort, nonce)}.json"
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
        for fut in futures:
            pass
        for fut, i in futures.items():
            out[i] = fut.result()
            done += 1
            if progress and done % max(1, len(requests) // 20) == 0:
                print(f"  [{label}] {done}/{len(requests)}", flush=True)
    errors = [r for r in out if isinstance(r, dict) and "__error__" in r]
    if errors:
        print(f"  [{label}] {len(errors)} failed, e.g. {errors[0]['__error__'][:120]}")
    return out


def usage_total():
    """Tokens spent this process, and an estimate in dollars."""
    return dict(_usage)
