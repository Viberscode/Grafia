"""Load the GraphRAG parquet output into the graph store and ChromaDB (batched, idempotent).

Usage: python -m backend.stores.loader [--reset] [--backend neo4j|kuzu|memory]
Columns follow docs/schema_notes.md. Ids: Document/Entity/REL ids are hashes of titles, Chunk ids are
GraphRAG text unit ids (content hashes, also used in relationships.text_unit_ids), Community ids are
GraphRAG community numbers (the ids the UI graph uses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from typing import Any

import pandas as pd

from backend.stores.graph_store import GraphData, GraphStore, open_store
from backend.stores.vector_store import ChromaStore, VectorStore

logger = logging.getLogger(__name__)


def hid(prefix: str, *parts: str) -> str:
    """Deterministic id from a prefix and text parts."""
    return f"{prefix}_{hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()[:16]}"


def as_list(value: Any) -> list[str]:
    """Parquet array cell (numpy array, list or None) as a list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(v) for v in list(value)]


def build_data(tables: dict[str, pd.DataFrame]) -> GraphData:
    """Convert GraphRAG tables into store-neutral rows."""
    docs, units = tables.get("documents"), tables.get("text_units")
    ents, rels = tables.get("entities"), tables.get("relationships")
    comms, reports = tables.get("communities"), tables.get("community_reports")
    data = GraphData()
    doc_ids: dict[str, str] = {}
    if docs is not None:
        for _, r in docs.iterrows():
            doc_ids[str(r["id"])] = hid("doc", str(r["title"]))
            data.documents.append({"id": doc_ids[str(r["id"])], "title": str(r["title"])})
    if units is not None:
        for _, r in units.iterrows():
            data.chunks.append({"id": str(r["id"]), "text": str(r["text"]),
                                "doc_id": doc_ids.get(str(r.get("document_id")), "")})
    ent_ids: dict[str, str] = {}
    if ents is not None:
        for _, r in ents.iterrows():
            eid = hid("ent", str(r["title"]))
            ent_ids[str(r["id"])] = eid
            data.entities.append({"id": eid, "title": str(r["title"]), "type": str(r.get("type") or ""),
                                  "description": str(r.get("description") or ""), "degree": int(r.get("degree") or 0)})
            data.mentions += [{"chunk_id": u, "entity_id": eid} for u in as_list(r.get("text_unit_ids"))]
    title_to_id = {e["title"]: e["id"] for e in data.entities}
    if rels is not None:
        for _, r in rels.iterrows():
            s, t = title_to_id.get(str(r["source"])), title_to_id.get(str(r["target"]))
            if s and t:
                data.relations.append({"id": hid("rel", str(r["source"]), str(r["target"])), "source": s,
                                       "target": t, "description": str(r.get("description") or ""),
                                       "weight": float(r.get("weight") or 1.0),
                                       "text_unit_ids": as_list(r.get("text_unit_ids"))})
    summaries = {}
    if reports is not None:
        summaries = {str(r["community"]): (str(r.get("title") or ""), str(r.get("summary") or ""))
                     for _, r in reports.iterrows()}
    if comms is not None:
        for _, r in comms.iterrows():
            cid = str(r["community"])
            title, summary = summaries.get(cid, (str(r.get("title") or f"Community {cid}"), ""))
            data.communities.append({"id": cid, "level": int(r.get("level") or 0), "title": title, "summary": summary})
            data.memberships += [{"entity_id": ent_ids[e], "community_id": cid}
                                 for e in as_list(r.get("entity_ids")) if e in ent_ids]
    return data


def vector_docs(tables: dict[str, pd.DataFrame], data: GraphData) -> dict[str, tuple[list, list, list]]:
    """(ids, texts, metadatas) for the chunks, reports and entities collections."""
    doc_titles = {d["id"]: d["title"] for d in data.documents}
    chunks = ([c["id"] for c in data.chunks], [c["text"] for c in data.chunks],
              [{"doc_id": c["doc_id"], "doc": doc_titles.get(c["doc_id"], "")} for c in data.chunks])
    reports: tuple[list, list, list] = ([], [], [])
    rep = tables.get("community_reports")
    if rep is not None:
        for _, r in rep.iterrows():
            reports[0].append(str(r["community"]))
            reports[1].append(str(r.get("full_content") or f"{r.get('title')}\n{r.get('summary')}"))
            reports[2].append({"title": str(r.get("title") or ""), "level": int(r.get("level") or 0),
                               "rank": float(r.get("rank") or 0.0)})
    entities = ([e["id"] for e in data.entities], [f"{e['title']}: {e['description']}" for e in data.entities],
                [{"title": e["title"], "type": e["type"], "degree": e["degree"]} for e in data.entities])
    return {"chunks": chunks, "reports": reports, "entities": entities}


def load(reset: bool = False, graph: GraphStore | None = None, vectors: VectorStore | None = None,
         tables: dict[str, pd.DataFrame] | None = None) -> dict:
    """Load parquet into the stores and return a count report."""
    if tables is None:
        from backend.engine import load_tables

        load_tables.cache_clear()
        tables = load_tables()
    if graph is None:
        graph = open_store()
    if vectors is None:
        from backend.config import CHROMA_PATH

        vectors = ChromaStore(CHROMA_PATH)
    data = build_data(tables)
    if reset:
        graph.clear()
        vectors.clear()
    graph.upsert(data)
    for name, (ids, texts, metas) in vector_docs(tables, data).items():
        if ids:
            vectors.upsert(name, ids, texts, metas)
    rels = tables.get("relationships")
    missing = int(sum(1 for v in rels["text_unit_ids"] if not as_list(v))) if rels is not None else 0
    report = {
        "backend": getattr(graph, "name", type(graph).__name__),
        "parquet": {k: len(v) for k, v in tables.items()},
        "graph": graph.stats(),
        "vectors": vectors.counts(),
        "relationships_missing_text_unit_ids": missing,
    }
    logger.info("Store load: %s", json.dumps(report))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints the count report as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true", help="clear both stores before loading")
    ap.add_argument("--backend", choices=["neo4j", "kuzu", "memory"], help="override GRAPH_BACKEND")
    args = ap.parse_args(argv)
    report = load(reset=args.reset, graph=open_store(args.backend) if args.backend else None)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
