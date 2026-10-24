"""Offline tests for graph.to_payload and the demo data used by MOCK=1."""
from backend.graph import to_payload
from backend.verify import quote_in_chunk
from ui.mock import REPORT_SUMMARIES, MockData, run

QUESTIONS = ["What are the main themes?", "What is the Harbor Wind Farm?", "What is the capital of France?",
             "How is the Clean Energy Plan connected to the Electric Bus Fleet?"]


def test_payload_schema_and_cap():
    data = MockData()
    resp = run(QUESTIONS[3], data=data, delay_s=0)
    p = to_payload(data.graph, resp.highlights, False, max_nodes=20, text_units=data.text_units, titles=data.titles)
    assert set(p) == {"nodes", "edges", "highlight", "communities", "options"}
    assert len(p["nodes"]) == 20
    ids = {n["id"] for n in p["nodes"]}
    assert set(resp.highlights.nodes) <= ids
    assert all(e["source"] in ids and e["target"] in ids for e in p["edges"])
    assert set(p["highlight"]["order"]) <= set(p["highlight"]["nodes"])
    assert {"id", "label", "type", "community", "degree", "description", "sources"} <= set(p["nodes"][0])


def test_payload_expand_pins_neighbours():
    data = MockData()
    p = to_payload(data.graph, None, True, max_nodes=5, expanded=["HARBOR WIND FARM"])
    ids = {n["id"] for n in p["nodes"]}
    assert {"HARBOR WIND FARM", *data.graph.neighbors("HARBOR WIND FARM")} <= ids


def test_mock_responses_cover_routes_and_quotes_verify():
    data = MockData()
    assert data.graph.number_of_nodes() == 40 and len(data.titles) == 3
    lookup = {f"report_{k}": v for k, v in REPORT_SUMMARIES.items()}
    lookup.update({k: text for k, (_, text) in data.text_units.items()})
    routes = []
    for q in QUESTIONS:
        resp = run(q, data=data, delay_s=0)
        routes.append(resp.route)
        assert all(quote_in_chunk(e.quote, lookup[e.chunk_id]) for c in resp.claims for e in c.evidence)
    assert routes == ["global", "local", "abstain", "local"]
    assert len(run(QUESTIONS[3], data=data, delay_s=0).highlights.nodes) == 12
    assert run(QUESTIONS[0], "basic", data=data, delay_s=0).route == "basic"
