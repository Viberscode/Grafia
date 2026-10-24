"""Router tests: offline rule checks plus an LLM accuracy run when an API key is set."""
import os

import pytest

from backend import router

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


def test_low_confidence_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "abstain", "confidence": 0.1, "reason": "x"})
    assert router.route_question("Tell me about Kestrel")["route"] == "local"


def test_invalid_label_never_returned(monkeypatch):
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "drift", "confidence": 0.9, "reason": "x"})
    assert router.route_question("Who is Sandra Lee?")["route"] in router.ROUTES


def test_keyword_prior_pushes_global(monkeypatch):
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "local", "confidence": 0.6, "reason": "x"})
    assert router.route_question("What are the main themes?")["route"] == "global"


def test_router_falls_back_when_llm_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(router, "chat_json", boom)
    assert router.route_question("Summarise the overall plan")["route"] == "global"
    assert router.route_question("Who is Sandra Lee?")["route"] == "local"


@pytest.mark.skipif(not os.getenv("GRAPHRAG_API_KEY") or os.getenv("OFFLINE") == "1",
                    reason="needs an API key")
def test_router_accuracy():
    correct = 0
    for q, gold in LABELLED:
        got = router.route_question(q)["route"]
        correct += got == gold
        print(f"{'OK ' if got == gold else 'XX '} {gold:>7} -> {got:<7} {q}")
    acc = correct / len(LABELLED)
    print(f"\nRouter accuracy: {acc:.0%} ({correct}/{len(LABELLED)})")
    assert acc >= 0.75
