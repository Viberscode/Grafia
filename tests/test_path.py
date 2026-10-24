"""Tests for backend.path highlight and source mapping."""
import networkx as nx
import pandas as pd

from backend.path import build_highlights, matched_entities


def test_local_highlights_include_shortest_path(
    fake_graph: nx.Graph, local_context: dict, fake_tables: dict[str, pd.DataFrame]
) -> None:
    """Local search highlights matched entities, the path between them, and maps sources to docs."""
    hl, sources = build_highlights("local", local_context, fake_graph, fake_tables)
    assert {"ALPHA", "BETA", "GAMMA"} <= set(hl.nodes)
    assert "NOPE" not in hl.nodes
    assert {"10", "11"} <= set(hl.edges)
    assert hl.communities == ["0"]
    assert [s.doc for s in sources] == ["doc_one.txt", "doc_two.txt"]


def test_global_highlights_use_report_communities(fake_graph: nx.Graph) -> None:
    """Global search highlights members of the communities whose reports were used."""
    ctx = {"reports": [{"id": "1", "title": "Delta cluster"}], "text_units": []}
    hl, sources = build_highlights("global", ctx, fake_graph)
    assert set(hl.nodes) == {"DELTA", "EPSILON"}
    assert hl.communities == ["1"]
    assert sources == []


def test_missing_columns_do_not_raise(fake_graph: nx.Graph) -> None:
    """Records without expected columns are skipped instead of failing."""
    ctx = {"entities": [{"foo": "bar"}], "relationships": [{}], "text_units": [{"id": "x"}]}
    hl, sources = build_highlights("local", ctx, fake_graph, None)
    assert hl.nodes == []
    assert sources[0].doc == "unknown document"


def test_matched_entities_case_insensitive(fake_graph: nx.Graph) -> None:
    """Lower-case entity names match upper-case graph nodes."""
    assert matched_entities({"entities": [{"entity": "alpha"}]}, fake_graph) == ["ALPHA"]
