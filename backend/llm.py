"""Single gateway for all LLM calls: cached in SQLite, retried with backoff."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from typing import Any

from backend.config import CACHE_DB, GRAPHRAG_API_KEY, LLM_BASE_URL

logger = logging.getLogger(__name__)

COUNTERS: dict[str, int] = {"calls": 0, "cache_hits": 0, "tokens": 0}
_LOCK = threading.Lock()
_CLIENT: Any = None
MAX_RETRIES = 5


def _db() -> sqlite3.Connection:
    """Open the cache database, creating the table if needed."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
    return conn


def _client() -> Any:
    """Create the OpenAI client lazily so imports work without a key."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(api_key=GRAPHRAG_API_KEY, base_url=LLM_BASE_URL)
    return _CLIENT


def _key(model: str, system: str, user: str) -> str:
    """Cache key: sha256 of model + system + user."""
    return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    """Return a cached response or None."""
    with _db() as conn:
        row = conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, value: str) -> None:
    """Store a response in the cache."""
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", (key, value))


def _is_retryable(err: Exception) -> bool:
    """True for rate limits, timeouts and transient server errors."""
    if "insufficient_quota" in str(err):
        return False
    name = type(err).__name__
    return name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


def chat_text(model: str, system: str, user: str) -> str:
    """Return the model's text reply, using the cache when possible."""
    key = _key(model, system, user)
    cached = _cache_get(key)
    if cached is not None:
        with _LOCK:
            COUNTERS["cache_hits"] += 1
        return cached

    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            with _LOCK:
                COUNTERS["calls"] += 1
            resp = _client().chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            if resp.usage:
                with _LOCK:
                    COUNTERS["tokens"] += resp.usage.total_tokens
            text = resp.choices[0].message.content or ""
            _cache_put(key, text)
            return text
        except Exception as err:
            if not _is_retryable(err) or attempt == MAX_RETRIES - 1:
                raise
            logger.warning("LLM retry %d after %s: %s", attempt + 1, type(err).__name__, err)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _parse_json(raw: str) -> Any:
    """Parse JSON, tolerating markdown fences and surrounding prose."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def chat_json(model: str, system: str, user: str, schema_hint: str = "") -> dict:
    """Return the model's reply parsed as a JSON object, retrying once with a repair prompt."""
    sys_prompt = f"{system}\n\nRespond with JSON only, matching: {schema_hint}" if schema_hint else system
    raw = chat_text(model, sys_prompt, user)
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("JSON parse failed, retrying with repair instruction")
        repair = (
            f"Your previous reply was not valid JSON:\n{raw}\n\n"
            f"Return only the corrected JSON object matching: {schema_hint}"
        )
        return _parse_json(chat_text(model, sys_prompt, repair))


def reset_counters() -> None:
    """Zero the call, cache-hit and token counters."""
    with _LOCK:
        for k in COUNTERS:
            COUNTERS[k] = 0


if __name__ == "__main__":
    from backend.config import LLM_MODEL_CHEAP

    for _ in range(2):
        chat_text(LLM_MODEL_CHEAP, "Reply with one word.", "Say hello.")
    logger.info("Counters: %s", COUNTERS)
