"""Offline retrieval tests: global themes, multi-hop local trace, hybrid fusion, abstention, grounding."""
import networkx as nx
import pytest
from fakes import FakeVectors, graphrag_tables

from backend import pipeline, router
from backend.contracts import Claim, Evidence
from backend.retrieval import mentioned, retrieve, title_aliases
from backend.retrieval.hybrid import rrf
from backend.stores import loader
from backend.stores.graph_store import MemoryStore


@pytest.fixture
def stores() -> tuple[MemoryStore, FakeVectors]:
    """Memory graph store and fake vectors loaded from the sample tables."""
    g, v = MemoryStore(), FakeVectors()
    loader.load(graph=g, vectors=v, tables=graphrag_tables())
    return g, v


@pytest.fixture
def ui_graph(monkeypatch: pytest.MonkeyPatch) -> nx.Graph:
    """The UI graph pipeline.get_graph() returns, with edge ids like the payload."""
    g = nx.Graph()
    for n, c in [("ALPHA", "0"), ("BETA", "0"), ("GAMMA", "1"), ("DELTA", "1")]:
        g.add_node(n, community=c)
    for i, (u, v) in enumerate([("ALPHA", "BETA"), ("BETA", "GAMMA"), ("GAMMA", "DELTA")]):
        g.add_edge(u, v, id=str(i))
    monkeypatch.setattr(pipeline, "get_graph", lambda: g)
    return g


def test_global_highlights_only_entities_named_in_reports(stores: tuple) -> None:
    """One community step per report, at most 15 highlighted nodes, all named in the report text."""
    r = retrieve("global", "What are the main themes of the Alpha and Beta partnership?", *stores)
    assert [s.kind for s in r.trace] == ["community"] * len(r.trace) and 1 <= len(r.trace) <= 3
    assert r.trace[0].ids == ["0"] and r.contexts[0]["label"] == "theme"
    assert set(r.nodes) <= {"ALPHA", "BETA"} and len(r.nodes) <= 15


def test_local_multi_hop_trace(stores: tuple) -> None:
    """Two named entities give seed -> hop (shortest path) -> chunk -> document."""
    r = retrieve("local", "How is Alpha connected to Delta?", *stores)
    assert [s.kind for s in r.trace] == ["seed", "hop", "chunk", "document"]
    assert set(r.trace[0].ids) == {"ALPHA", "DELTA"}
    assert r.trace[1].ids in (["ALPHA", "BETA", "GAMMA", "DELTA"], ["DELTA", "GAMMA", "BETA", "ALPHA"])
    assert {"t1", "t2"} <= set(r.trace[2].ids) and len(r.contexts) <= 8
    assert set(r.trace[3].ids) <= {"one.txt", "two.txt"} and r.trace[3].ids


def test_unrelated_question_retrieves_nothing_and_abstains(stores: tuple, ui_graph: nx.Graph,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """No seeds and no close reports means no context, so the pipeline abstains without an LLM call."""
    assert not retrieve("local", "What is the capital of France?", *stores).contexts
    assert not retrieve("global", "What is the capital of France?", *stores).contexts
    monkeypatch.setattr(pipeline, "retrieve", lambda route, q: retrieve(route, q, *stores))
    monkeypatch.setattr(pipeline, "closest_community", lambda q: None)
    resp = pipeline.run("What is the capital of France?", mode="local")
    assert resp.abstained and resp.route == "local"


def test_hybrid_fuses_themes_and_facts(stores: tuple) -> None:
    """Hybrid contexts carry both labels and the trace has community steps then local steps."""
    r = retrieve("hybrid", "What are the main themes around Gamma, and who hires Delta?", *stores)
    labels = {c["label"] for c in r.contexts}
    assert labels == {"theme", "fact"} and r.info["split"]["theme"] >= 1
    kinds = [s.kind for s in r.trace]
    assert kinds[0] == "community" and "seed" in kinds and kinds.index("seed") > kinds.index("community")


def test_rrf_rewards_items_in_both_lists() -> None:
    """An item ranked in both lists beats items ranked once."""
    a = [{"chunk_id": "x"}, {"chunk_id": "both"}]
    b = [{"chunk_id": "both"}, {"chunk_id": "y"}]
    assert rrf([a, b])[0]["chunk_id"] == "both"


def test_aliases_match_honorifics_and_unique_first_words() -> None:
    """'Priya Raman' matches DR. PRIYA RAMAN and 'Northwind' matches the wind farm."""
    titles = ["DR. PRIYA RAMAN", "NORTHWIND OFFSHORE WIND FARM", "AURORA BAY", "AURORA BAY GREEN BOND", "RAMAN REPORT"]
    assert title_aliases(titles)["DR. PRIYA RAMAN"][1] == " priya raman "
    assert mentioned(titles, "Priya Raman studied the Northwind project") == titles[:2]
    assert mentioned(titles, "Aurora issued bonds") == []


def test_highlights_are_intersected_with_verified_quotes(stores: tuple, ui_graph: nx.Graph) -> None:
    """Only entities named in verified claims stay highlighted, joined by the trace path between them."""
    r = retrieve("local", "How is Alpha connected to Delta?", *stores)
    claims = [Claim(text="Alpha works with Beta, and Gamma hires Delta.",
                    evidence=[Evidence(chunk_id="t1", quote="Alpha works with Beta.")])]
    h = pipeline.ground_highlights("local", r, claims)
    assert set(h.nodes) == {"ALPHA", "BETA", "GAMMA", "DELTA"}
    assert set(h.edges) == {"0", "1", "2"}
    only_alpha = [Claim(text="Alpha is an organisation.", evidence=[])]
    assert pipeline.ground_highlights("local", r, only_alpha).nodes == ["ALPHA"]


def test_router_mixed_question_goes_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Theme plus fact cues give hybrid, with or without the LLM."""
    q = "What are the main themes around financing, and who issued the green bond?"
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "global", "confidence": 0.9, "reason": "x"})
    assert router.route_question(q)["route"] == "hybrid"
    monkeypatch.setattr(router, "chat_json", lambda *a, **k: {"route": "hybrid", "confidence": 0.8, "reason": "x"})
    assert router.route_question("Give the big picture on storage and the Kestrel start date")["route"] == "hybrid"

    def boom(*a: object, **k: object) -> dict:
        """Simulate an LLM outage."""
        raise RuntimeError("offline")
    monkeypatch.setattr(router, "chat_json", boom)
    assert router.route_question(q)["route"] == "hybrid"
    assert router.route_question("What are the main themes?")["route"] == "global"
