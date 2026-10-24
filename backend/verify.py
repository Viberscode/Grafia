"""Verify claims: strict quote-substring check, then a batched support check."""
from __future__ import annotations

import json
import logging
import re

from backend.config import LLM_MODEL_CHEAP
from backend.contracts import Claim, Evidence
from backend.llm import chat_json

logger = logging.getLogger(__name__)

SUPPORT_SYSTEM = """You check whether each claim is fully supported by its quotes.
A claim is supported only if the quotes alone justify it. Return JSON:
{"results": [{"index": 0, "supported": true}]} with one entry per claim."""


def normalise(text: str) -> str:
    """Lowercase, unify quotes and dashes, and collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                         "\u2013": "-", "\u2014": "-"}))
    return re.sub(r"\s+", " ", text).strip().strip('."\'')


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """True if the normalised quote is a substring of the normalised chunk."""
    q = normalise(quote)
    return bool(q) and q in normalise(chunk_text)


def check_quotes(claims: list[dict], chunk_lookup: dict[str, str]) -> list[Claim]:
    """Keep only evidence whose quote appears verbatim in the cited chunk."""
    kept: list[Claim] = []
    for c in claims:
        evidence = [
            Evidence(chunk_id=str(e.get("chunk_id", "")), quote=str(e.get("quote", "")))
            for e in c.get("evidence", []) if isinstance(e, dict)
        ]
        valid = [e for e in evidence if quote_in_chunk(e.quote, chunk_lookup.get(e.chunk_id, ""))]
        if valid:
            kept.append(Claim(text=str(c.get("text", "")), evidence=valid))
        else:
            logger.info("Dropped claim with no valid quotes: %s", c.get("text"))
    return kept


def check_support(claims: list[Claim]) -> list[Claim]:
    """Drop claims the cheap model says are not supported by their quotes."""
    if not claims:
        return []
    payload = [{"index": i, "claim": c.text, "quotes": [e.quote for e in c.evidence]}
               for i, c in enumerate(claims)]
    try:
        res = chat_json(LLM_MODEL_CHEAP, SUPPORT_SYSTEM, json.dumps(payload, ensure_ascii=False),
                        '{"results": [{"index": 0, "supported": true}]}')
        unsupported = {int(r["index"]) for r in res.get("results", []) if r.get("supported") is False}
    except Exception as err:
        logger.warning("Support check failed, keeping quote-verified claims: %s", err)
        return claims
    return [c for i, c in enumerate(claims) if i not in unsupported]


def verify_claims(claims: list[dict], chunk_lookup: dict[str, str], support_check: bool = True) -> list[Claim]:
    """Run the quote check and optionally the support check."""
    kept = check_quotes(claims, chunk_lookup)
    return check_support(kept) if support_check else kept
