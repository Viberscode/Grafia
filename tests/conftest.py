"""Shared offline fixtures: a tiny fake knowledge graph and GraphRAG-like context."""
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def fake_graph() -> nx.Graph:
    """Five-node graph: A-B-C chain in community 0, D-E in community 1."""
    g = nx.Graph()
    for name, comm in [("ALPHA", "0"), ("BETA", "0"), ("GAMMA", "0"), ("DELTA", "1"), ("EPSILON", "1")]:
        g.add_node(name, type="org", description=f"{name} desc", community=comm,
                   communities=[comm, f"{comm}{comm}"], text_unit_ids=[])
    g.add_edge("ALPHA", "BETA", id="10", description="a-b", weight=1.0, text_unit_ids=["t1"])
    g.add_edge("BETA", "GAMMA", id="11", description="b-c", weight=1.0, text_unit_ids=["t2"])
    g.add_edge("DELTA", "EPSILON", id="12", description="d-e", weight=1.0, text_unit_ids=["t3"])
    g.add_edge("GAMMA", "DELTA", id="13", description="c-d", weight=1.0, text_unit_ids=["t3"])
    for n in g.nodes:
        g.nodes[n]["degree"] = g.degree(n)
    return g


@pytest.fixture
def fake_tables() -> dict[str, pd.DataFrame]:
    """Minimal documents and text_units tables."""
    docs = pd.DataFrame({"id": ["d1", "d2"], "title": ["doc_one.txt", "doc_two.txt"]})
    units = pd.DataFrame({
        "id": ["t1", "t2", "t3"],
        "text": ["Alpha works with Beta.", "Beta funds Gamma.", "Delta partners with Epsilon."],
        "document_id": ["d1", "d1", "d2"],
    })
    return {"documents": docs, "text_units": units}


@pytest.fixture
def local_context() -> dict:
    """Context shaped like graphrag local search output after engine normalisation."""
    return {
        "entities": [{"id": "1", "entity": "ALPHA"}, {"id": "3", "entity": "GAMMA"}, {"id": "9", "entity": "NOPE"}],
        "relationships": [{"id": "10", "source": "ALPHA", "target": "BETA"}],
        "reports": [],
        "text_units": [{"id": "0", "text": "Alpha works with Beta."}, {"id": "2", "text": "Delta partners with Epsilon."}],
    }
