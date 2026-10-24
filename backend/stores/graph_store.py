"""Graph store protocol, the Neo4j implementation and an in-memory NetworkX store.

Schema (all ids deterministic, see backend/stores/loader.py):
  (:Document {id, title})-[:HAS_CHUNK]->(:Chunk {id, text, doc_id})-[:MENTIONS]->(:Entity {id, title, type,
  description, degree})-[:REL {id, description, weight, text_unit_ids}]-(:Entity),
  (:Entity)-[:IN_COMMUNITY]->(:Community {id, level, title, summary}).
Methods take and return entity titles, which are also the node ids of the UI graph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import networkx as nx

logger = logging.getLogger(__name__)

BATCH = 500


@dataclass
class GraphData:
    """Store-neutral rows produced by the loader."""

    documents: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    mentions: list[dict] = field(default_factory=list)
    memberships: list[dict] = field(default_factory=list)


class GraphStore(Protocol):
    """Operations the retrievers and the loader need from a graph database."""

    name: str

    def upsert(self, data: GraphData) -> None:
        """Merge all rows (idempotent)."""

    def neighbors(self, titles: list[str], hops: int = 2) -> list[str]:
        """Entity titles within `hops` REL steps of the given entities, including them."""

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> list[str]:
        """Entity titles along a shortest REL path, or [] when unconnected."""

    def subgraph(self, titles: list[str]) -> list[dict]:
        """REL edges {source, target, description, weight, text_unit_ids} among the given entities."""

    def chunks_for(self, titles: list[str], limit: int = 20) -> list[dict]:
        """Chunks {id, text, doc_id, entity} that MENTION the given entities."""

    def documents_for(self, chunk_ids: list[str]) -> dict[str, str]:
        """Map chunk id -> document title."""

    def community_members(self, community_ids: list[str]) -> list[dict]:
        """Entities {title, degree, community} in the given communities."""

    def entity_titles(self) -> list[str]:
        """All entity titles."""

    def stats(self) -> dict[str, int]:
        """Node and relationship counts by label."""

    def clear(self) -> None:
        """Delete everything."""


def batches(rows: list[dict], size: int = BATCH) -> Iterator[list[dict]]:
    """Yield successive slices of rows."""
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


class MemoryStore:
    """NetworkX-backed store: the offline fallback and the test fake."""

    name = "memory"

    def __init__(self) -> None:
        """Start empty."""
        self.clear()

    def clear(self) -> None:
        """Delete everything."""
        self.g = nx.Graph()
        self.data = GraphData()
        self.chunks: dict[str, dict] = {}
        self.docs: dict[str, str] = {}
        self.mentions: dict[str, set[str]] = {}
        self.members: dict[str, set[str]] = {}

    def upsert(self, data: GraphData) -> None:
        """Merge all rows (idempotent)."""
        by_id = {e["id"]: e for e in data.entities}
        for e in data.entities:
            self.g.add_node(e["title"], **e)
        for r in data.relations:
            s, t = by_id.get(r["source"]), by_id.get(r["target"])
            if s and t:
                self.g.add_edge(s["title"], t["title"], **r)
        self.docs.update({d["id"]: d["title"] for d in data.documents})
        self.chunks.update({c["id"]: c for c in data.chunks})
        for m in data.mentions:
            if m["entity_id"] in by_id:
                self.mentions.setdefault(by_id[m["entity_id"]]["title"], set()).add(m["chunk_id"])
        for m in data.memberships:
            if m["entity_id"] in by_id:
                self.members.setdefault(m["community_id"], set()).add(by_id[m["entity_id"]]["title"])
        self.data = data

    def neighbors(self, titles: list[str], hops: int = 2) -> list[str]:
        """Entity titles within `hops` REL steps of the given entities, including them."""
        out: set[str] = set()
        for t in titles:
            if t in self.g:
                out |= set(nx.single_source_shortest_path_length(self.g, t, cutoff=hops))
        return sorted(out)

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> list[str]:
        """Entity titles along a shortest REL path, or [] when unconnected."""
        if source not in self.g or target not in self.g or not nx.has_path(self.g, source, target):
            return []
        path = nx.shortest_path(self.g, source, target)
        return path if len(path) - 1 <= max_hops else []

    def subgraph(self, titles: list[str]) -> list[dict]:
        """REL edges among the given entities."""
        return [{"source": u, "target": v, "description": d.get("description", ""),
                 "weight": float(d.get("weight", 1.0)), "text_unit_ids": list(d.get("text_unit_ids", []))}
                for u, v, d in self.g.subgraph(titles).edges(data=True)]

    def chunks_for(self, titles: list[str], limit: int = 20) -> list[dict]:
        """Chunks that MENTION the given entities, in entity order."""
        out, seen = [], set()
        for t in titles:
            for cid in sorted(self.mentions.get(t, ())):
                if cid not in seen and cid in self.chunks:
                    seen.add(cid)
                    out.append({**self.chunks[cid], "entity": t})
        return out[:limit]

    def documents_for(self, chunk_ids: list[str]) -> dict[str, str]:
        """Map chunk id -> document title."""
        return {c: self.docs.get(self.chunks[c]["doc_id"], "") for c in chunk_ids if c in self.chunks}

    def community_members(self, community_ids: list[str]) -> list[dict]:
        """Entities in the given communities."""
        return [{"title": t, "degree": int(self.g.nodes[t].get("degree", 0)), "community": c}
                for c in community_ids for t in self.members.get(c, ()) if t in self.g]

    def entity_titles(self) -> list[str]:
        """All entity titles."""
        return list(self.g.nodes)

    def stats(self) -> dict[str, int]:
        """Node and relationship counts by label."""
        return {"Document": len(self.docs), "Chunk": len(self.chunks), "Entity": self.g.number_of_nodes(),
                "Community": len({m["community_id"] for m in self.data.memberships} |
                                 {c["id"] for c in self.data.communities}),
                "REL": self.g.number_of_edges(), "MENTIONS": sum(map(len, self.mentions.values())),
                "IN_COMMUNITY": sum(map(len, self.members.values())), "HAS_CHUNK": len(self.chunks)}


NEO4J_SCHEMA = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX entity_title IF NOT EXISTS FOR (n:Entity) ON (n.title)",
]

NEO4J_UPSERTS = [
    ("documents", "UNWIND $rows AS r MERGE (d:Document {id: r.id}) SET d.title = r.title"),
    ("chunks", "UNWIND $rows AS r MERGE (c:Chunk {id: r.id}) SET c.text = r.text, c.doc_id = r.doc_id "
               "WITH c, r MATCH (d:Document {id: r.doc_id}) MERGE (d)-[:HAS_CHUNK]->(c)"),
    ("entities", "UNWIND $rows AS r MERGE (e:Entity {id: r.id}) SET e.title = r.title, e.type = r.type, "
                 "e.description = r.description, e.degree = r.degree"),
    ("communities", "UNWIND $rows AS r MERGE (c:Community {id: r.id}) SET c.level = r.level, "
                    "c.title = r.title, c.summary = r.summary"),
    ("relations", "UNWIND $rows AS r MATCH (s:Entity {id: r.source}), (t:Entity {id: r.target}) "
                  "MERGE (s)-[x:REL {id: r.id}]->(t) SET x.description = r.description, x.weight = r.weight, "
                  "x.text_unit_ids = r.text_unit_ids"),
    ("mentions", "UNWIND $rows AS r MATCH (c:Chunk {id: r.chunk_id}), (e:Entity {id: r.entity_id}) "
                 "MERGE (c)-[:MENTIONS]->(e)"),
    ("memberships", "UNWIND $rows AS r MATCH (e:Entity {id: r.entity_id}), (c:Community {id: r.community_id}) "
                    "MERGE (e)-[:IN_COMMUNITY]->(c)"),
]


class Neo4jStore:
    """Neo4j 5 store using MERGE on deterministic ids."""

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str | None, driver: Any = None) -> None:
        """Connect (or reuse an injected driver, e.g. a test fake)."""
        if driver is None:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(user, password or ""))
            driver.verify_connectivity()
        self.driver = driver

    def _run(self, query: str, **params: Any) -> list[dict]:
        """Run one query and return its records as dicts."""
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, **params)]

    def upsert(self, data: GraphData) -> None:
        """Create constraints, then MERGE every row in batches."""
        for stmt in NEO4J_SCHEMA:
            self._run(stmt)
        for attr, query in NEO4J_UPSERTS:
            for rows in batches(getattr(data, attr)):
                self._run(query, rows=rows)

    def neighbors(self, titles: list[str], hops: int = 2) -> list[str]:
        """Entity titles within `hops` REL steps, including the seeds."""
        q = (f"MATCH (s:Entity) WHERE s.title IN $titles OPTIONAL MATCH (s)-[:REL*1..{int(hops)}]-(n:Entity) "
             "WITH collect(DISTINCT s.title) + collect(DISTINCT n.title) AS ts UNWIND ts AS t "
             "RETURN DISTINCT t AS title")
        return sorted(r["title"] for r in self._run(q, titles=titles) if r["title"])

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> list[str]:
        """Entity titles along a shortest REL path, or []."""
        q = (f"MATCH (a:Entity {{title: $a}}), (b:Entity {{title: $b}}), "
             f"p = shortestPath((a)-[:REL*..{int(max_hops)}]-(b)) RETURN [n IN nodes(p) | n.title] AS path")
        rows = self._run(q, a=source, b=target) if source != target else [{"path": [source]}]
        return list(rows[0]["path"]) if rows else []

    def subgraph(self, titles: list[str]) -> list[dict]:
        """REL edges among the given entities."""
        q = ("MATCH (s:Entity)-[r:REL]->(t:Entity) WHERE s.title IN $titles AND t.title IN $titles "
             "RETURN s.title AS source, t.title AS target, r.description AS description, "
             "r.weight AS weight, r.text_unit_ids AS text_unit_ids")
        return [{**r, "text_unit_ids": list(r["text_unit_ids"] or [])} for r in self._run(q, titles=titles)]

    def chunks_for(self, titles: list[str], limit: int = 20) -> list[dict]:
        """Chunks that MENTION the given entities."""
        q = ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.title IN $titles "
             "RETURN c.id AS id, c.text AS text, c.doc_id AS doc_id, e.title AS entity")
        rows, seen, order = self._run(q, titles=titles), set(), {t: i for i, t in enumerate(titles)}
        out = []
        for r in sorted(rows, key=lambda r: (order.get(r["entity"], 99), r["id"])):
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out[:limit]

    def documents_for(self, chunk_ids: list[str]) -> dict[str, str]:
        """Map chunk id -> document title."""
        q = "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, d.title AS title"
        return {r["id"]: r["title"] for r in self._run(q, ids=chunk_ids)}

    def community_members(self, community_ids: list[str]) -> list[dict]:
        """Entities in the given communities."""
        q = ("MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community) WHERE c.id IN $ids "
             "RETURN e.title AS title, e.degree AS degree, c.id AS community")
        return self._run(q, ids=community_ids)

    def entity_titles(self) -> list[str]:
        """All entity titles."""
        return [r["title"] for r in self._run("MATCH (e:Entity) RETURN e.title AS title")]

    def stats(self) -> dict[str, int]:
        """Node and relationship counts by label."""
        out = {}
        for label in ("Document", "Chunk", "Entity", "Community"):
            out[label] = self._run(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
        for rel in ("REL", "MENTIONS", "IN_COMMUNITY", "HAS_CHUNK"):
            out[rel] = self._run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")[0]["n"]
        return out

    def clear(self) -> None:
        """Delete everything."""
        self._run("MATCH (n) DETACH DELETE n")


def memory_from_parquet() -> MemoryStore:
    """A MemoryStore filled from the GraphRAG parquet output."""
    from backend.engine import load_tables
    from backend.stores.loader import build_data

    store = MemoryStore()
    store.upsert(build_data(load_tables()))
    return store


def open_store(backend: str | None = None) -> GraphStore:
    """Open the configured graph store (neo4j, kuzu or memory); raises if it is unreachable."""
    from backend import config

    backend = (backend or config.GRAPH_BACKEND).lower()
    if backend == "neo4j":
        return Neo4jStore(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
    if backend == "kuzu":
        from backend.stores.kuzu_store import KuzuStore

        return KuzuStore(config.KUZU_PATH)
    if backend == "memory":
        return memory_from_parquet()
    raise ValueError(f"Unknown GRAPH_BACKEND {backend!r} (use neo4j, kuzu or memory)")


_STORE: GraphStore | None = None
_FALLBACK_REASON: str | None = None


def get_store() -> GraphStore:
    """Process-wide graph store; falls back to the in-memory store when the database is unreachable."""
    global _STORE, _FALLBACK_REASON
    if _STORE is None:
        try:
            _STORE, _FALLBACK_REASON = open_store(), None
        except Exception as err:
            logger.warning("Graph store unavailable (%s); using the in-memory fallback", err)
            _STORE, _FALLBACK_REASON = memory_from_parquet(), f"{type(err).__name__}: {err}"
    return _STORE


def fallback_reason() -> str | None:
    """Why get_store() fell back to memory, or None."""
    return _FALLBACK_REASON


def reset_store() -> None:
    """Forget the cached store so the next call reconnects and reloads."""
    global _STORE, _FALLBACK_REASON
    _STORE, _FALLBACK_REASON = None, None
