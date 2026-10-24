"""Synchronous wrapper around the graphrag.api Python package.

Installed graphrag 3.3.0 exposes (graphrag.api):
  global_search(config, entities, communities, community_reports, community_level,
                dynamic_community_selection, response_type, query, callbacks=None, verbose=False)
  local_search(config, entities, communities, community_reports, text_units, relationships,
               covariates, community_level, response_type, query, callbacks=None, verbose=False)
  drift_search(config, entities, communities, community_reports, text_units, relationships,
               community_level, response_type, query, callbacks=None, verbose=False)
  basic_search(config, text_units, response_type, query, callbacks=None, verbose=False)
  build_index(config, method='standard', is_update_run=False, callbacks=None, ...)
All return (response, context_data) where context_data is a dict of record lists keyed by
reports / entities / relationships / claims / sources (sources = text units).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import COMMUNITY_LEVEL, GRAPH_ROOT, INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

TABLES = ["documents", "text_units", "entities", "relationships", "communities", "community_reports"]
_CWD_LOCK = threading.RLock()


@dataclass
class QueryResult:
    """Answer text plus the context records GraphRAG used."""

    answer: str
    context: dict[str, list[dict]] = field(default_factory=dict)


@contextmanager
def _in_graph_root() -> Iterator[None]:
    """Run GraphRAG with cwd = GRAPH_ROOT (it resolves prompt paths relative to cwd)."""
    with _CWD_LOCK:
        cwd = os.getcwd()
        os.chdir(GRAPH_ROOT)
        try:
            yield
        finally:
            os.chdir(cwd)


def _load_graphrag_config() -> Any:
    """Load settings.yaml with absolute storage paths."""
    from graphrag.config.load_config import load_config

    overrides = {
        "cache": {"storage": {"base_dir": str(GRAPH_ROOT / "cache")}},
        "vector_store": {"db_uri": str(OUTPUT_DIR / "lancedb")},
    }
    return load_config(root_dir=GRAPH_ROOT, cli_overrides=overrides)


def ingest_files(files: list[str | Path]) -> list[Path]:
    """Copy .txt files and convert .pdf files to .txt inside the GraphRAG input folder."""
    from pypdf import PdfReader

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in map(Path, files):
        target = INPUT_DIR / f"{f.stem}.txt"
        if f.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(f)).pages)
            target.write_text(text, encoding="utf-8")
        elif f.suffix.lower() == ".txt":
            shutil.copyfile(f, target)
        else:
            logger.warning("Skipping unsupported file %s", f)
            continue
        written.append(target)
    return written


def run_index() -> bool:
    """Run the full GraphRAG indexing pipeline and return True when all workflows succeed."""
    from graphrag.api import build_index

    with _in_graph_root():
        results = asyncio.run(build_index(config=_load_graphrag_config()))
    ok = bool(results)
    for r in results:
        if r.error:
            ok = False
            logger.error("Workflow %s failed: %s", r.workflow, r.error)
        else:
            logger.info("Workflow %s done", r.workflow)
    load_tables.cache_clear()
    return ok


@lru_cache(maxsize=1)
def load_tables() -> dict[str, pd.DataFrame]:
    """Read the GraphRAG output parquet files once and cache them."""
    tables: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        path = OUTPUT_DIR / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
        else:
            logger.warning("Missing output table %s", path)
    return tables


def _to_records(value: Any) -> list[dict]:
    """Convert a DataFrame or list into plain dict records, keeping only in-context rows."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if isinstance(value, dict):
        value = [value]
    records = [dict(r) for r in value if isinstance(r, dict)]
    if records and "in_context" in records[0]:
        records = [r for r in records if r.get("in_context", True)]
    return records


def _normalise_context(raw: Any) -> dict[str, list[dict]]:
    """Map GraphRAG context_data onto entities / relationships / reports / text_units."""
    if not isinstance(raw, dict):
        return {}
    ctx = {k.lower(): _to_records(v) for k, v in raw.items()}
    return {
        "entities": ctx.get("entities", []),
        "relationships": ctx.get("relationships", []),
        "reports": ctx.get("reports", []),
        "text_units": ctx.get("sources", []) or ctx.get("text_units", []),
    }


def run_query(question: str, method: str = "global") -> QueryResult:
    """Run a GraphRAG global, local or basic search synchronously."""
    t = load_tables()
    with _in_graph_root():
        response, context = asyncio.run(_search(question, method, t))
    answer = response if isinstance(response, str) else str(response)
    return QueryResult(answer=answer, context=_normalise_context(context))


async def _search(question: str, method: str, t: dict[str, pd.DataFrame]) -> tuple[Any, Any]:
    """Dispatch to the matching graphrag.api search coroutine."""
    import graphrag.api as api

    config = _load_graphrag_config()
    response_type = "Multiple Paragraphs"
    if method == "global":
        coro = api.global_search(
            config=config, entities=t["entities"], communities=t["communities"],
            community_reports=t["community_reports"], community_level=COMMUNITY_LEVEL,
            dynamic_community_selection=False, response_type=response_type, query=question,
        )
    elif method == "local":
        coro = api.local_search(
            config=config, entities=t["entities"], communities=t["communities"],
            community_reports=t["community_reports"], text_units=t["text_units"],
            relationships=t["relationships"], covariates=None, community_level=COMMUNITY_LEVEL,
            response_type=response_type, query=question,
        )
    elif method == "basic":
        coro = api.basic_search(
            config=config, text_units=t["text_units"], response_type=response_type, query=question,
        )
    else:
        raise ValueError(f"Unknown search method: {method}")
    return await coro


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--index":
        sys.exit(0 if run_index() else 1)
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main themes?"
    m = sys.argv[2] if len(sys.argv) > 2 else "global"
    res = run_query(q, m)
    logger.info("Answer:\n%s", res.answer)
    logger.info("Context keys: %s", {k: len(v) for k, v in res.context.items()})
    for k, v in res.context.items():
        if v:
            logger.info("%s columns: %s", k, list(v[0].keys()))
