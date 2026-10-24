"""One data source for the UI: the real GraphRAG index, or demo data (MOCK=1 / no index)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import networkx as nx
import streamlit as st

from backend.config import OUTPUT_DIR
from backend.contracts import Response

logger = logging.getLogger(__name__)

MAX_NODES = 300


@dataclass
class DataSource:
    """Everything the UI reads: graph, community titles/reports, documents and text units."""

    graph: nx.Graph
    titles: dict[str, str]
    reports: list[dict]
    documents: list[dict]
    text_units: dict[str, tuple[str, str]]
    is_mock: bool
    status: str
    runner: Callable[..., Response] = field(repr=False, default=None)

    def run(self, question: str, mode: str, progress: Callable[[str], None] | None = None) -> Response:
        """Answer a question with the real pipeline or the mock."""
        return self.runner(question, mode, progress=progress)

    def stats(self) -> dict[str, int]:
        """Header counts."""
        return {"docs": len(self.documents), "entities": self.graph.number_of_nodes(),
                "relationships": self.graph.number_of_edges(), "communities": len(self.reports) or len(self.titles)}

    def report_text(self, community: str) -> str:
        """Full text of a community report (what global answers quote from)."""
        rep = next((r for r in self.reports if r["community"] == community), None) or {}
        return f"{rep.get('title', '')}\n{rep.get('content') or rep.get('summary', '')}".strip()

    def community_nodes(self, community: str) -> list[str]:
        """Graph nodes belonging to a community at any level."""
        return [n for n, d in self.graph.nodes(data=True)
                if community == d.get("community") or community in d.get("communities", [])]

    def full_chunk(self, excerpt: str) -> tuple[str, str] | None:
        """(doc, full text) of the text unit that starts with a Source excerpt."""
        head = excerpt.rstrip(".").strip()[:120]
        for doc, text in self.text_units.values():
            if text.strip().startswith(head):
                return doc, text
        return None


def mock_requested() -> bool:
    """MOCK=1 in the environment."""
    return os.environ.get("MOCK", "").strip().lower() in ("1", "true", "yes")


def index_ready() -> bool:
    """GraphRAG output parquet files exist."""
    return (OUTPUT_DIR / "entities.parquet").exists() and (OUTPUT_DIR / "relationships.parquet").exists()


def _records(df, cols: list[str]) -> list[dict]:
    """DataFrame rows as dicts with only the columns present."""
    if df is None:
        return []
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_dict("records")


@st.cache_resource(show_spinner="Loading the knowledge graph...")
def _real() -> DataSource:
    """Load graph and tables from myproj/output once per server process."""
    from backend import pipeline
    from backend.engine import load_tables
    from backend.graph import community_titles

    tables = load_tables()
    docs = {str(r["id"]): r for r in _records(tables.get("documents"), ["id", "title", "text"])}
    units = {}
    for r in _records(tables.get("text_units"), ["id", "text", "document_id"]):
        units[str(r["id"])] = (str(docs.get(str(r.get("document_id")), {}).get("title", "unknown document")), str(r["text"]))
    cols = ["community", "title", "summary", "full_content", "level", "rank", "size"]
    reports = [{"community": str(r["community"]), "title": str(r.get("title", "")), "summary": str(r.get("summary", "")),
                "content": str(r.get("full_content") or r.get("summary", "")), "level": int(r.get("level", 0)),
                "rank": float(r.get("rank") or 0), "size": int(r.get("size") or 0)}
               for r in _records(tables.get("community_reports"), cols)]
    built = datetime.fromtimestamp((OUTPUT_DIR / "entities.parquet").stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    def runner(question: str, mode: str, progress=None) -> Response:
        return pipeline.run(question, mode=mode, progress=progress)

    return DataSource(graph=pipeline.get_graph(), titles=community_titles(tables), reports=reports,
                      documents=list(docs.values()), text_units=units, is_mock=False,
                      status=f"Index ready, built {built}", runner=runner)


@st.cache_resource(show_spinner=False)
def _mock() -> DataSource:
    """Demo data source."""
    from ui import mock

    data = mock.MockData()

    def runner(question: str, mode: str, progress=None) -> Response:
        return mock.run(question, mode, progress=progress, data=data)

    return DataSource(graph=data.graph, titles=data.titles, reports=data.reports, documents=data.documents,
                      text_units=data.text_units, is_mock=True, status="Demo data", runner=runner)


def load() -> DataSource:
    """Real data when an index exists and MOCK is not set; demo data otherwise (or if loading fails)."""
    if mock_requested() or not index_ready():
        return _mock()
    try:
        return _real()
    except Exception:
        logger.exception("Failed to load the index; falling back to demo data")
        return _mock()


@st.cache_data(show_spinner=False, max_entries=64)
def payload(_data: DataSource, is_mock: bool, h_nodes: tuple[str, ...], h_edges: tuple[str, ...],
            communities: tuple[str, ...], show_full: bool, expanded: tuple[str, ...], trace_json: str = "[]",
            cited_documents: tuple[str, ...] | None = None) -> dict:
    """Cached component payload for the current answer (is_mock keys the cache per source)."""
    import json

    from backend.graph import to_payload

    highlights = {"nodes": list(h_nodes), "edges": list(h_edges), "communities": list(communities)}
    return to_payload(_data.graph, highlights, show_full, max_nodes=MAX_NODES, text_units=_data.text_units,
                      titles=_data.titles, expanded=list(expanded), trace=json.loads(trace_json),
                      cited_documents=None if cited_documents is None else list(cited_documents))
