"""Route questions to GraphRAG global search, local search, or abstention."""
from __future__ import annotations

import logging

from backend.config import LLM_MODEL_CHEAP, ROUTER_MIN_CONF
from backend.llm import chat_json

logger = logging.getLogger(__name__)

ROUTES = ("global", "local", "abstain")
GLOBAL_KEYWORDS = (
    "theme", "overall", "across", "main topics", "summarise", "summarize", "summary",
    "big picture", "trends", "in general", "key issues", "what are the main",
)

SYSTEM_PROMPT = """You route questions for a knowledge-graph Q&A system over a document corpus
about a city's clean energy transition (renewables, grid, storage, financing, people and organisations).
Pick exactly one route:
- "global": broad, thematic or corpus-wide questions (themes, summaries, trends, comparisons across documents).
- "local": specific facts, named entities, relationships, or multi-hop questions linking entities.
- "abstain": off-topic, nonsensical, or clearly unanswerable from such a corpus (personal, weather, sports, coding, future predictions).
Return JSON: {"route": "global|local|abstain", "confidence": 0.0-1.0, "reason": "short reason"}."""

FEW_SHOT = """Examples:
Q: What are the main themes across these documents? -> {"route": "global", "confidence": 0.95, "reason": "Corpus-wide themes."}
Q: Summarise the biggest risks facing the energy transition. -> {"route": "global", "confidence": 0.9, "reason": "Cross-document summary."}
Q: How do different stakeholders view the transition overall? -> {"route": "global", "confidence": 0.85, "reason": "Thematic comparison."}
Q: Who supplied the turbines for the offshore wind farm? -> {"route": "local", "confidence": 0.95, "reason": "Specific fact."}
Q: How is the researcher who wrote the blackout report connected to the battery project's software vendor? -> {"route": "local", "confidence": 0.9, "reason": "Multi-hop entity link."}
Q: When did the battery storage project enter service? -> {"route": "local", "confidence": 0.9, "reason": "Specific date lookup."}
Q: What is the best pizza topping? -> {"route": "abstain", "confidence": 0.98, "reason": "Off-topic."}
Q: What will the stock price of the utility be in 2040? -> {"route": "abstain", "confidence": 0.9, "reason": "Unanswerable prediction."}
"""


def keyword_prior(question: str) -> bool:
    """True when the question contains words that suggest a thematic (global) query."""
    q = question.lower()
    return any(kw in q for kw in GLOBAL_KEYWORDS)


def route_question(question: str) -> dict:
    """Return {route, confidence, reason}; never a label outside global/local/abstain."""
    prior = keyword_prior(question)
    try:
        res = chat_json(
            LLM_MODEL_CHEAP, SYSTEM_PROMPT, f"{FEW_SHOT}\nQ: {question} ->",
            schema_hint='{"route": "global|local|abstain", "confidence": 0.0, "reason": "str"}',
        )
        route = str(res.get("route", "local")).strip().lower()
        conf = float(res.get("confidence", 0.5))
        reason = str(res.get("reason", ""))
    except Exception as err:
        logger.warning("Router LLM failed: %s", err)
        route, conf, reason = ("global", 0.6, "Keyword prior (router fallback).") if prior else (
            "local", 0.5, f"Router fallback: {err}")

    if route not in ROUTES:
        route, reason = "local", f"Invalid label replaced with local. {reason}"
    if prior and route == "local" and conf < 0.85:
        route, conf, reason = "global", max(conf, 0.7), f"Keyword prior pushed to global. {reason}"
    elif prior and route == "global":
        conf = min(1.0, conf + 0.1)
    if conf < ROUTER_MIN_CONF and route != "local":
        reason = f"Low confidence ({conf:.2f}) fell back to local. {reason}"
        route = "local"
    return {"route": route, "confidence": round(conf, 3), "reason": reason.strip()}
