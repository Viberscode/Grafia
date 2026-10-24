"""Turn GraphRAG query context into graph highlights and document sources."""
from __future__ import annotations

import logging
from itertools import combinations

import networkx as nx
import pandas as pd

from backend.contracts import Highlights, Source

logger = logging.getLogger(__name__)

MAX_GLOBAL_NODES = 60
MAX_LOCAL_FALLBACK = 8
EXCERPT_CHARS = 300


def _entity_title(record: dict) -> str | None:
    """Entity title from a local-search context record."""
    for key in ("entity", "title", "name"):
        if record.get(key):
            return str(record[key])
    return None


def _report_community(record: dict) -> str | None:
    """Community id from a report record (global context uses 'id' = community number)."""
    for key in ("community", "community_id", "id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    return None


def matched_entities(context: dict, graph: nx.Graph) -> list[str]:
    """Entity titles from context that exist as nodes in the graph."""
    out: list[str] = []
    for rec in context.get("entities", []):
        title = _entity_title(rec)
        if title is None:
            logger.warning("Entity record without a title column: %s", list(rec))
            continue
        for cand in (title, title.upper()):
            if cand in graph and cand not in out:
                out.append(cand)
                break
    return out


def _edges_from_relationships(context: dict, graph: nx.Graph) -> tuple[set[str], set[str]]:
    """Highlight edges named in relationship context records."""
    nodes, edges = set(), set()
    for rec in context.get("relationships", []):
        s, t = str(rec.get("source", "")), str(rec.get("target", ""))
        if graph.has_edge(s, t):
            nodes.update((s, t))
            edges.add(graph.edges[s, t].get("id", f"{s}->{t}"))
    return nodes, edges


def focus_entities(entities: list[str], focus_text: str | None) -> list[str]:
    """Entities named in the focus text, else the top-ranked context entities."""
    if focus_text:
        text = focus_text.lower()
        named = [e for e in entities if e.lower() in text]
        if named:
            return named
    return entities[:MAX_LOCAL_FALLBACK]


def _edge_within(graph: nx.Graph, edge_id: str, nodes: set[str]) -> bool:
    """True if the edge with this id connects two nodes in the set."""
    return any(d.get("id") == edge_id and u in nodes and v in nodes for u, v, d in graph.edges(data=True))


def _shortest_path(graph: nx.Graph, entities: list[str]) -> tuple[list[str], set[str]]:
    """Shortest path between the two highest-degree matched entities that are connected."""
    ranked = sorted(entities, key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)
    for u, v in combinations(ranked[:6], 2):
        if nx.has_path(graph, u, v):
            path = nx.shortest_path(graph, u, v)
            return path, {graph.edges[a, b].get("id", f"{a}->{b}") for a, b in zip(path, path[1:])}
    return [], set()


def build_sources(context: dict, tables: dict[str, pd.DataFrame] | None) -> list[Source]:
    """Map context text units to their documents, matching by chunk text."""
    units = context.get("text_units", [])
    if not units:
        return []
    text_to_doc: dict[str, str] = {}
    if tables and "text_units" in tables and "documents" in tables:
        doc_titles = dict(zip(tables["documents"]["id"].astype(str), tables["documents"]["title"].astype(str)))
        tu = tables["text_units"]
        if "document_id" in tu.columns:
            for text, doc_id in zip(tu["text"].astype(str), tu["document_id"].astype(str)):
                text_to_doc[text.strip()] = doc_titles.get(doc_id, doc_id)
        else:
            logger.warning("text_units has no document_id column; sources will lack document names")
    sources = []
    for rec in units:
        text = str(rec.get("text", ""))
        sources.append(Source(
            doc=text_to_doc.get(text.strip(), "unknown document"),
            chunk_id=str(rec.get("id", "")),
            text=text[:EXCERPT_CHARS] + ("..." if len(text) > EXCERPT_CHARS else ""),
        ))
    return sources


def build_highlights(
    method: str,
    context: dict,
    graph: nx.Graph,
    tables: dict[str, pd.DataFrame] | None = None,
    focus_text: str | None = None,
) -> tuple[Highlights, list[Source]]:
    """Return highlights and sources; focus_text (claims + quotes) narrows local highlights to cited entities."""
    nodes: list[str] = []
    edges: set[str] = set()
    communities: list[str] = []
    try:
        if method == "global":
            communities = [c for c in (_report_community(r) for r in context.get("reports", [])) if c]
            wanted = set(communities)
            members = [n for n, d in graph.nodes(data=True) if wanted & set(d.get("communities", []))]
            members.sort(key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)
            nodes = members[:MAX_GLOBAL_NODES]
        else:
            ents = focus_entities(matched_entities(context, graph), focus_text)
            rel_nodes, rel_edges = _edges_from_relationships(context, graph)
            path, path_edges = _shortest_path(graph, ents)
            nodes = list(dict.fromkeys(ents + path))
            keep = set(nodes)
            edges = path_edges | {e for e in rel_edges if _edge_within(graph, e, keep)}
            communities = sorted({graph.nodes[n].get("community", "") for n in nodes} - {""})
    except Exception as err:
        logger.warning("Could not build highlights: %s", err)
    try:
        sources = build_sources(context, tables)
    except Exception as err:
        logger.warning("Could not build sources: %s", err)
        sources = []
    return Highlights(nodes=nodes, edges=sorted(edges), communities=communities), sources
