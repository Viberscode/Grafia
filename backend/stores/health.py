"""Store health for the header chips: which graph and vector stores are live, with counts."""
from __future__ import annotations

import logging

from backend.stores import graph_store, vector_store

logger = logging.getLogger(__name__)

LABELS = {"neo4j": "Neo4j", "kuzu": "Kùzu", "memory": "In-memory graph", "chroma": "ChromaDB"}


def graph_chip() -> dict:
    """{name, ok, detail, counts} for the graph store."""
    from backend.config import GRAPH_BACKEND

    want = LABELS.get(GRAPH_BACKEND, GRAPH_BACKEND)
    try:
        store = graph_store.get_store()
        stats = store.stats()
        reason = graph_store.fallback_reason()
        if reason:
            return {"name": want, "ok": False, "detail": f"unreachable, using in-memory graph ({reason[:80]})",
                    "counts": stats}
        return {"name": LABELS.get(store.name, store.name), "ok": True,
                "detail": f"{stats.get('Entity', 0)} entities, {stats.get('REL', 0)} relationships", "counts": stats}
    except Exception as err:
        logger.warning("Graph health check failed: %s", err)
        return {"name": want, "ok": False, "detail": f"{type(err).__name__}: {err}"[:120], "counts": {}}


def vector_chip() -> dict:
    """{name, ok, detail, counts} for ChromaDB."""
    try:
        counts = vector_store.get_vectors().counts()
        ok = counts.get("chunks", 0) > 0
        detail = ", ".join(f"{v} {k}" for k, v in counts.items()) if ok else "empty: run .\\tasks.ps1 load"
        return {"name": "ChromaDB", "ok": ok, "detail": detail, "counts": counts}
    except Exception as err:
        logger.warning("Vector health check failed: %s", err)
        return {"name": "ChromaDB", "ok": False, "detail": f"{type(err).__name__}: {err}"[:120], "counts": {}}


def status() -> list[dict]:
    """Health chips for every store."""
    return [graph_chip(), vector_chip()]


def main() -> int:
    """Log each store's status; exit code 1 if any store is down or empty."""
    chips = status()
    for c in chips:
        logger.info("%s %s: %s", "OK  " if c["ok"] else "DOWN", c["name"], c["detail"])
    return 0 if all(c["ok"] for c in chips) else 1


if __name__ == "__main__":
    raise SystemExit(main())
