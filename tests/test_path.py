"""Tests for backend.path highlight and source mapping."""
from backend.path import build_highlights, matched_entities


def test_local_highlights_include_shortest_path(fake_graph, local_context, fake_tables):
    hl, sources = build_highlights("local", local_context, fake_graph, fake_tables)
    assert {"ALPHA", "BETA", "GAMMA"} <= set(hl.nodes)
    assert "NOPE" not in hl.nodes
    assert {"10", "11"} <= set(hl.edges)
    assert hl.communities == ["0"]
    assert [s.doc for s in sources] == ["doc_one.txt", "doc_two.txt"]


def test_global_highlights_use_report_communities(fake_graph):
    ctx = {"reports": [{"id": "1", "title": "Delta cluster"}], "text_units": []}
    hl, sources = build_highlights("global", ctx, fake_graph)
    assert set(hl.nodes) == {"DELTA", "EPSILON"}
    assert hl.communities == ["1"]
    assert sources == []


def test_missing_columns_do_not_raise(fake_graph):
    ctx = {"entities": [{"foo": "bar"}], "relationships": [{}], "text_units": [{"id": "x"}]}
    hl, sources = build_highlights("local", ctx, fake_graph, None)
    assert hl.nodes == []
    assert sources[0].doc == "unknown document"


def test_matched_entities_case_insensitive(fake_graph):
    assert matched_entities({"entities": [{"entity": "alpha"}]}, fake_graph) == ["ALPHA"]
