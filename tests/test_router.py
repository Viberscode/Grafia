"""Router tests: offline rule checks plus an LLM accuracy run when an API key is set."""
import logging
from typing import Any

import pytest

from backend import router
from backend.config import GRAPHRAG_API_KEY, OFFLINE

logger = logging.getLogger(__name__)

LABELLED = [
    ("What are the main themes across the documents?", "global"),
    ("Summarise the overall progress of the clean energy transition.", "global"),
    ("What tensions between stakeholders appear across the corpus?", "global"),
    ("Who supplied the floating turbines for Northwind?", "local"),
    ("When did the Kestrel Battery Storage Project enter service?", "local"),
    ("How much money did the National Clean Grid Fund give to Kestrel?", "local"),
    ("How is Priya Raman connected to Gridwise Analytics?", "local"),
    ("Who mediated the dispute between Tidewater Energy and the fishermen?", "local"),
    ("What two milestones must be met before the Saltmarsh Gas Plant retires?", "local"),
    ("What is the capital of Australia?", "abstain"),
    ("Write me a Python function to reverse a list.", "abstain"),
    ("Who will win the 2032 football world cup?", "abstain"),
]


def test_low_confidence_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confidence below ROUTER_MIN_CONF falls back to local."""
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "abstain", "confidence": 0.1, "reason": "x"})
    assert router.route_question("Tell me about Kestrel")["route"] == "local"


def test_invalid_label_never_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Labels outside global/local/abstain are replaced."""
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "drift", "confidence": 0.9, "reason": "x"})
    assert router.route_question("Who is Sandra Lee?")["route"] in router.ROUTES


def test_keyword_prior_pushes_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thematic keywords push a borderline local answer to global."""
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "local", "confidence": 0.6, "reason": "x"})
    assert router.route_question("What are the main themes?")["route"] == "global"


def test_router_falls_back_when_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the LLM, the keyword prior decides between global and local."""
    def boom(*a: Any, **k: Any) -> dict:
        """Simulate an LLM outage."""
        raise RuntimeError("offline")
    monkeypatch.setattr(router, "chat_json", boom)
    assert router.route_question("Summarise the overall plan")["route"] == "global"
    assert router.route_question("Who is Sandra Lee?")["route"] == "local"


@pytest.mark.skipif(not GRAPHRAG_API_KEY or OFFLINE, reason="needs an API key")
def test_router_accuracy() -> None:
    """Router accuracy on 12 labelled corpus questions (run with -o log_cli=true to see the log)."""
    correct = 0
    for q, gold in LABELLED:
        got = router.route_question(q)["route"]
        correct += got == gold
        logger.info("%s %7s -> %-7s %s", "OK" if got == gold else "XX", gold, got, q)
    acc = correct / len(LABELLED)
    logger.info("Router accuracy: %.0f%% (%d/%d)", acc * 100, correct, len(LABELLED))
    assert acc >= 0.75
