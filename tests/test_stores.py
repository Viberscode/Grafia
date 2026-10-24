"""Offline store tests: loader rows, MemoryStore queries, Neo4j Cypher via a fake driver, embedded Kùzu."""
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeVectors, graphrag_tables

from backend.stores import loader
from backend.stores.graph_store import MemoryStore, Neo4jStore


@pytest.fixture
def tables() -> dict[str, pd.DataFrame]:
    """GraphRAG-shaped sample tables."""
    return graphrag_tables()


def test_loader_counts_match_parquet(tables: dict[str, pd.DataFrame]) -> None:
    """Every parquet row lands in the stores and missing text_unit_ids are reported."""
    g, v = MemoryStore(), FakeVectors()
    rep = loader.load(reset=True, graph=g, vectors=v, tables=tables)
    assert rep["graph"]["Document"] == 2 and rep["graph"]["Chunk"] == 3
    assert rep["graph"]["Entity"] == 4 and rep["graph"]["REL"] == 3 and rep["graph"]["Community"] == 2
    assert rep["vectors"] == {"chunks": 3, "reports": 2, "entities": 4}
    assert rep["relationships_missing_text_unit_ids"] == 1


def test_loader_is_idempotent_and_ids_deterministic(tables: dict[str, pd.DataFrame]) -> None:
    """Loading twice changes nothing and ids do not depend on GraphRAG's random uuids."""
    g, v = MemoryStore(), FakeVectors()
    first = loader.load(graph=g, vectors=v, tables=tables)
    second = loader.load(graph=g, vectors=v, tables=tables)
    assert first["graph"] == second["graph"] and first["vectors"] == second["vectors"]
    renamed = {**tables, "entities": tables["entities"].assign(id=["x1", "x2", "x3", "x4"])}
    assert [e["id"] for e in loader.build_data(renamed).entities] == [e["id"] for e in loader.build_data(tables).entities]


def test_memory_store_queries(tables: dict[str, pd.DataFrame]) -> None:
    """Neighbours, paths, subgraphs, mentions, documents and communities."""
    g = MemoryStore()
    g.upsert(loader.build_data(tables))
    assert g.neighbors(["ALPHA"], hops=2) == ["ALPHA", "BETA", "GAMMA"]
    assert g.shortest_path("ALPHA", "DELTA") == ["ALPHA", "BETA", "GAMMA", "DELTA"]
    assert {(e["source"], e["target"]) for e in g.subgraph(["ALPHA", "BETA", "DELTA"])} == {("ALPHA", "BETA")}
    assert [c["id"] for c in g.chunks_for(["DELTA"])] == ["t3"]
    assert g.documents_for(["t3"]) == {"t3": "two.txt"}
    assert {m["title"] for m in g.community_members(["1"])} == {"GAMMA", "DELTA"}


class FakeSession:
    """Records Cypher queries."""

    def __init__(self, log: list) -> None:
        """Share the driver's log."""
        self.log = log

    def __enter__(self) -> "FakeSession":
        """Context manager entry."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Context manager exit."""

    def run(self, query: str, **params: object) -> list:
        """Record and return no rows."""
        self.log.append((query, params))
        return []


class FakeDriver:
    """Minimal neo4j driver stand-in."""

    def __init__(self) -> None:
        """Start with an empty log."""
        self.log: list = []

    def session(self) -> FakeSession:
        """Open a recording session."""
        return FakeSession(self.log)


def test_neo4j_upsert_uses_constraints_and_merge(tables: dict[str, pd.DataFrame]) -> None:
    """Constraints and the title index are created, and every write is a batched MERGE."""
    drv = FakeDriver()
    Neo4jStore("bolt://fake", "neo4j", "x", driver=drv).upsert(loader.build_data(tables))
    queries = [q for q, _ in drv.log]
    assert sum("CREATE CONSTRAINT" in q for q in queries) == 4
    assert any("CREATE INDEX entity_title" in q for q in queries)
    writes = [q for q in queries if q.startswith("UNWIND")]
    assert len(writes) == 7 and all("MERGE" in q for q in writes)
    assert not any("CREATE (" in q for q in queries)


def test_kuzu_store_roundtrip(tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    """The embedded Kùzu fallback loads and answers the same queries as MemoryStore."""
    pytest.importorskip("kuzu")
    from backend.stores.kuzu_store import KuzuStore

    store = KuzuStore(tmp_path / "kuzu")
    rep = loader.load(reset=True, graph=store, vectors=FakeVectors(), tables=tables)
    assert rep["graph"]["Entity"] == 4 and rep["graph"]["REL"] == 3 and rep["graph"]["MENTIONS"] == 6
    store.upsert(loader.build_data(tables))
    assert store.stats() == rep["graph"]
    assert store.neighbors(["ALPHA"], hops=2) == ["ALPHA", "BETA", "GAMMA"]
    assert store.shortest_path("ALPHA", "DELTA") == ["ALPHA", "BETA", "GAMMA", "DELTA"]
    assert store.documents_for(["t3"]) == {"t3": "two.txt"}
