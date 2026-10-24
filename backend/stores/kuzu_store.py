"""Embedded Kùzu graph store (GRAPH_BACKEND=kuzu): the no-Docker fallback for Neo4j.

Kùzu was archived upstream in October 2025; ThemeGraph pins kuzu==0.11.3. The schema mirrors Neo4jStore.
A database directory admits one writer process, so every call opens and closes its own connection
(read-only for queries) and the indexing subprocess can load while the app is running.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.stores.graph_store import GraphData, batches

logger = logging.getLogger(__name__)

SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Document(id STRING PRIMARY KEY, title STRING)",
    "CREATE NODE TABLE IF NOT EXISTS Chunk(id STRING PRIMARY KEY, text STRING, doc_id STRING)",
    "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING PRIMARY KEY, title STRING, type STRING, "
    "description STRING, degree INT64)",
    "CREATE NODE TABLE IF NOT EXISTS Community(id STRING PRIMARY KEY, level INT64, title STRING, summary STRING)",
    "CREATE REL TABLE IF NOT EXISTS HAS_CHUNK(FROM Document TO Chunk)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS `REL`(FROM Entity TO Entity, id STRING, description STRING, "
    "weight DOUBLE, text_unit_ids STRING[])",
    "CREATE REL TABLE IF NOT EXISTS IN_COMMUNITY(FROM Entity TO Community)",
]

UPSERTS = [
    ("documents", "UNWIND $rows AS r MERGE (d:Document {id: r.id}) SET d.title = r.title"),
    ("chunks", "UNWIND $rows AS r MERGE (c:Chunk {id: r.id}) SET c.text = r.text, c.doc_id = r.doc_id"),
    ("chunks", "UNWIND $rows AS r MATCH (d:Document {id: r.doc_id}), (c:Chunk {id: r.id}) "
               "MERGE (d)-[:HAS_CHUNK]->(c)"),
    ("entities", "UNWIND $rows AS r MERGE (e:Entity {id: r.id}) SET e.title = r.title, e.type = r.type, "
                 "e.description = r.description, e.degree = r.degree"),
    ("communities", "UNWIND $rows AS r MERGE (c:Community {id: r.id}) SET c.level = r.level, "
                    "c.title = r.title, c.summary = r.summary"),
    ("relations", "UNWIND $rows AS r MATCH (s:Entity {id: r.source}), (t:Entity {id: r.target}) "
                  "MERGE (s)-[x:`REL` {id: r.id}]->(t) SET x.description = r.description, x.weight = r.weight, "
                  "x.text_unit_ids = r.text_unit_ids"),
    ("mentions", "UNWIND $rows AS r MATCH (c:Chunk {id: r.chunk_id}), (e:Entity {id: r.entity_id}) "
                 "MERGE (c)-[:MENTIONS]->(e)"),
    ("memberships", "UNWIND $rows AS r MATCH (e:Entity {id: r.entity_id}), (c:Community {id: r.community_id}) "
                    "MERGE (e)-[:IN_COMMUNITY]->(c)"),
]


class KuzuStore:
    """Kùzu implementation of the GraphStore protocol."""

    name = "kuzu"

    def __init__(self, path: Path) -> None:
        """Remember the database directory, creating the schema if it does not exist yet."""
        import kuzu  # noqa: F401  (fail fast when the package is missing)

        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._exec_many(SCHEMA)

    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[Any]:
        """Open a short-lived connection (read-only unless writing)."""
        import kuzu

        db = kuzu.Database(str(self.path), read_only=not write)
        conn = kuzu.Connection(db)
        try:
            yield conn
        finally:
            conn.close()
            db.close()

    def _exec_many(self, statements: list[str]) -> None:
        """Run DDL or write statements in one writable connection."""
        with self._conn(write=True) as conn:
            for stmt in statements:
                conn.execute(stmt)

    def _rows(self, query: str, **params: Any) -> list[dict]:
        """Run a read query and return dict rows."""
        with self._conn() as conn:
            res = conn.execute(query, params)
            cols = res.get_column_names()
            out = []
            while res.has_next():
                out.append(dict(zip(cols, res.get_next())))
            return out

    def upsert(self, data: GraphData) -> None:
        """MERGE every row in batches inside one writable connection."""
        with self._conn(write=True) as conn:
            for stmt in SCHEMA:
                conn.execute(stmt)
            for attr, query in UPSERTS:
                for rows in batches(getattr(data, attr)):
                    conn.execute(query, {"rows": rows})

    def neighbors(self, titles: list[str], hops: int = 2) -> list[str]:
        """Entity titles within `hops` REL steps, including the seeds."""
        q = (f"MATCH (s:Entity)-[:`REL`*1..{int(hops)}]-(n:Entity) WHERE s.title IN $titles "
             "RETURN DISTINCT n.title AS title")
        found = {r["title"] for r in self._rows(q, titles=titles)}
        seeds = {r["title"] for r in self._rows("MATCH (s:Entity) WHERE s.title IN $titles RETURN s.title AS title",
                                                titles=titles)}
        return sorted(found | seeds)

    def shortest_path(self, source: str, target: str, max_hops: int = 4) -> list[str]:
        """Entity titles along a shortest REL path, or []."""
        if source == target:
            return [source] if self._rows("MATCH (e:Entity {title: $a}) RETURN e.id AS id", a=source) else []
        q = (f"MATCH p = (a:Entity)-[:`REL`* SHORTEST 1..{int(max_hops)}]-(b:Entity) "
             "WHERE a.title = $a AND b.title = $b RETURN properties(nodes(p), 'title') AS path LIMIT 1")
        rows = self._rows(q, a=source, b=target)
        return list(rows[0]["path"]) if rows else []

    def subgraph(self, titles: list[str]) -> list[dict]:
        """REL edges among the given entities."""
        q = ("MATCH (s:Entity)-[r:`REL`]->(t:Entity) WHERE s.title IN $titles AND t.title IN $titles "
             "RETURN s.title AS source, t.title AS target, r.description AS description, "
             "r.weight AS weight, r.text_unit_ids AS text_unit_ids")
        return [{**r, "text_unit_ids": list(r["text_unit_ids"] or [])} for r in self._rows(q, titles=titles)]

    def chunks_for(self, titles: list[str], limit: int = 20) -> list[dict]:
        """Chunks that MENTION the given entities, in entity order."""
        q = ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.title IN $titles "
             "RETURN c.id AS id, c.text AS text, c.doc_id AS doc_id, e.title AS entity")
        order = {t: i for i, t in enumerate(titles)}
        out, seen = [], set()
        for r in sorted(self._rows(q, titles=titles), key=lambda r: (order.get(r["entity"], 99), r["id"])):
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out[:limit]

    def documents_for(self, chunk_ids: list[str]) -> dict[str, str]:
        """Map chunk id -> document title."""
        q = "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, d.title AS title"
        return {r["id"]: r["title"] for r in self._rows(q, ids=chunk_ids)}

    def community_members(self, community_ids: list[str]) -> list[dict]:
        """Entities in the given communities."""
        q = ("MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community) WHERE c.id IN $ids "
             "RETURN e.title AS title, e.degree AS degree, c.id AS community")
        return self._rows(q, ids=community_ids)

    def entity_titles(self) -> list[str]:
        """All entity titles."""
        return [r["title"] for r in self._rows("MATCH (e:Entity) RETURN e.title AS title")]

    def stats(self) -> dict[str, int]:
        """Node and relationship counts by label."""
        out = {}
        for label in ("Document", "Chunk", "Entity", "Community"):
            out[label] = self._rows(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
        for rel in ("`REL`", "MENTIONS", "IN_COMMUNITY", "HAS_CHUNK"):
            out[rel.strip("`")] = self._rows(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")[0]["n"]
        return out

    def clear(self) -> None:
        """Delete everything (relationships first, then nodes)."""
        self._exec_many(["MATCH ()-[r]->() DELETE r", "MATCH (n) DELETE n"])
