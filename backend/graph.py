"""Build a NetworkX knowledge graph from GraphRAG output and render it with pyvis.

Column names follow docs/schema_notes.md (graphrag 3.3): entities(id, human_readable_id, title, type,
description, text_unit_ids, degree), relationships(id, human_readable_id, source, target, description,
weight, text_unit_ids), communities(community, level, title, entity_ids), community_reports(community, title).
Node id = entity title. Edge id = relationship human_readable_id (the id GraphRAG query context uses).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from backend.engine import load_tables

logger = logging.getLogger(__name__)

PALETTE = ["#4e79a7", "#59a14f", "#edc948", "#b07aa1", "#76b7b2", "#f28e2b",
           "#9c755f", "#bab0ac", "#86bcb6", "#8cd17d", "#d4a6c8", "#a0cbe8"]
HIGHLIGHT = "#ff2d2d"
DIM = "#3a3f4b"


def _as_list(value: Any) -> list:
    """Turn parquet array cells (numpy arrays, lists, None) into a Python list."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [str(v) for v in list(value)]


def _entity_communities(tables: dict[str, pd.DataFrame]) -> dict[str, dict[int, str]]:
    """Map entity uuid -> {level: community id}."""
    out: dict[str, dict[int, str]] = {}
    comms = tables.get("communities")
    if comms is None:
        return out
    for _, row in comms.iterrows():
        for eid in _as_list(row.get("entity_ids")):
            out.setdefault(eid, {})[int(row["level"])] = str(row["community"])
    return out


def load_graph(tables: dict[str, pd.DataFrame] | None = None) -> nx.Graph:
    """Build the knowledge graph; node community = top-level (level 0) community."""
    tables = tables if tables is not None else load_tables()
    g = nx.Graph()
    ents = tables.get("entities")
    rels = tables.get("relationships")
    if ents is None:
        logger.warning("No entities table; returning empty graph")
        return g
    memberships = _entity_communities(tables)
    for _, row in ents.iterrows():
        levels = memberships.get(str(row["id"]), {})
        top = levels[min(levels)] if levels else "-1"
        g.add_node(
            str(row["title"]),
            entity_id=str(row["id"]),
            type=str(row.get("type") or "unknown"),
            description=str(row.get("description") or ""),
            community=top,
            communities=sorted(set(levels.values())),
            text_unit_ids=_as_list(row.get("text_unit_ids")),
        )
    if rels is not None:
        for _, row in rels.iterrows():
            s, t = str(row["source"]), str(row["target"])
            if s in g and t in g:
                g.add_edge(
                    s, t,
                    id=str(row.get("human_readable_id", row["id"])),
                    uuid=str(row["id"]),
                    description=str(row.get("description") or ""),
                    weight=float(row.get("weight") or 1.0),
                    text_unit_ids=_as_list(row.get("text_unit_ids")),
                )
    for n in g.nodes:
        g.nodes[n]["degree"] = g.degree(n)
    return g


def community_titles(tables: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    """Return {community id: report title}, falling back to community table titles."""
    tables = tables if tables is not None else load_tables()
    titles: dict[str, str] = {}
    for name in ("communities", "community_reports"):
        df = tables.get(name)
        if df is not None and "community" in df.columns:
            for _, row in df.iterrows():
                titles[str(row["community"])] = str(row.get("title") or f"Community {row['community']}")
    return dict(sorted(titles.items(), key=lambda kv: int(kv[0]) if kv[0].lstrip("-").isdigit() else 0))


def _community_color(comm: str) -> str:
    """Stable colour for a community id."""
    try:
        return PALETTE[int(comm) % len(PALETTE)]
    except ValueError:
        return PALETTE[sum(map(ord, comm)) % len(PALETTE)]


def _select_nodes(graph: nx.Graph, h_nodes: set[str], subgraph_only: bool, max_nodes: int) -> list[str]:
    """Pick which nodes to draw."""
    by_degree = sorted(graph.nodes, key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)
    if h_nodes and subgraph_only:
        keep = set(h_nodes)
        for n in h_nodes:
            keep.update(graph.neighbors(n))
        if len(keep) > max_nodes:
            extra = [n for n in by_degree if n in keep and n not in h_nodes]
            keep = set(h_nodes) | set(extra[: max(0, max_nodes - len(h_nodes))])
        return list(keep)
    return list(set(by_degree[:max_nodes]) | h_nodes)


def build_pyvis(
    graph: nx.Graph,
    highlights: dict | Any | None = None,
    subgraph_only: bool = True,
    max_nodes: int = 250,
    height: str = "720px",
) -> str:
    """Render the graph as standalone pyvis HTML with highlighted evidence in red."""
    from pyvis.network import Network

    if highlights is not None and hasattr(highlights, "model_dump"):
        highlights = highlights.model_dump()
    highlights = highlights or {}
    h_nodes = {n for n in highlights.get("nodes", []) if n in graph}
    h_edges = set(highlights.get("edges", []))
    has_h = bool(h_nodes or h_edges)

    nodes = _select_nodes(graph, h_nodes, subgraph_only, max_nodes)
    sub = graph.subgraph(nodes)
    net = Network(height=height, width="100%", bgcolor="#0e1117", font_color="#e6e6e6", cdn_resources="in_line")

    for n, d in sub.nodes(data=True):
        hl = n in h_nodes
        color = HIGHLIGHT if hl else (_community_color(d.get("community", "-1")) if not has_h else DIM)
        title = f"{n}\nType: {d.get('type')}\nCommunity: {d.get('community')}\n\n{d.get('description', '')[:400]}"
        net.add_node(n, label=n, title=title, color=color,
                     size=10 + min(d.get("degree", 1), 15) * (2 if hl else 1),
                     font={"color": "#ffffff" if hl or not has_h else "#8a8f98"})
    for u, v, d in sub.edges(data=True):
        hl = d.get("id") in h_edges or (u in h_nodes and v in h_nodes)
        net.add_edge(u, v, title=d.get("description", "")[:300],
                     color=HIGHLIGHT if hl else ("#5b6270" if not has_h else "#262a33"),
                     width=4 if hl else 1)

    net.set_options(json.dumps({
        "physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 150},
                    "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 120}},
        "interaction": {"hover": True, "tooltipDelay": 120},
    }))
    html = net.generate_html()
    disable_physics = (
        "network.once('stabilizationIterationsDone', function () {"
        " network.setOptions({ physics: false }); });"
    )
    return html.replace("return network;", f"{disable_physics}\n        return network;", 1)


def community_color(comm: str) -> str:
    """Public stable colour for a community id."""
    return _community_color(comm)


def _excerpt(text: str, needle: str, limit: int = 240) -> str:
    """Sentence of text that mentions needle, else the opening characters."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if needle and needle.lower() in sentence.lower():
            return sentence.strip()[:limit]
    return text.strip()[:limit]


def _sources_for(unit_ids: list[str], needle: str, text_units: dict[str, tuple[str, str]], k: int = 2) -> list[dict]:
    """Up to k {doc, excerpt} items for the given text unit ids."""
    out = []
    for uid in unit_ids:
        if uid in text_units:
            doc, text = text_units[uid]
            out.append({"doc": doc, "excerpt": _excerpt(text, needle)})
        if len(out) >= k:
            break
    return out


def evidence_order(graph: nx.Graph, nodes: list[str]) -> list[str]:
    """Highlighted nodes ordered along the shortest path between the two best-connected ones."""
    present = [n for n in nodes if n in graph]
    if len(present) < 2:
        return present
    ranked = sorted(present, key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)
    sub = graph.subgraph(present)
    u, v = ranked[0], ranked[1]
    path = nx.shortest_path(sub, u, v) if nx.has_path(sub, u, v) else (
        nx.shortest_path(graph, u, v) if nx.has_path(graph, u, v) else [u])
    path = [n for n in path if n in present] or [u]
    return path + [n for n in present if n not in path]


def to_payload(
    graph: nx.Graph,
    highlights: dict | Any | None,
    show_full: bool,
    max_nodes: int = 300,
    text_units: dict[str, tuple[str, str]] | None = None,
    titles: dict[str, str] | None = None,
    expanded: list[str] | None = None,
) -> dict:
    """Serialise the graph for the interactive component: nodes, edges, highlight, communities, options."""
    if highlights is not None and hasattr(highlights, "model_dump"):
        highlights = highlights.model_dump()
    highlights = highlights or {}
    text_units = text_units or {}
    titles = titles or {}
    h_nodes = [n for n in highlights.get("nodes", []) if n in graph]
    pinned = set(h_nodes)
    for n in expanded or []:
        if n in graph:
            pinned |= {n, *graph.neighbors(n)}
    by_degree = sorted(graph.nodes, key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)
    keep = list(pinned) + [n for n in by_degree if n not in pinned][: max(0, max_nodes - len(pinned))]
    sub = graph.subgraph(keep)

    nodes = [{
        "id": n, "label": n, "type": d.get("type", ""), "community": str(d.get("community", "-1")),
        "degree": int(d.get("degree", 0)), "description": d.get("description", ""),
        "sources": _sources_for(d.get("text_unit_ids", []), n, text_units),
    } for n, d in sub.nodes(data=True)]
    edges = [{
        "id": d.get("id", f"{u}->{v}"), "source": u, "target": v,
        "label": d.get("description", "")[:60], "description": d.get("description", ""),
        "sources": _sources_for(d.get("text_unit_ids", []), u, text_units, k=1),
    } for u, v, d in sub.edges(data=True)]
    sizes: dict[str, int] = {}
    for n in nodes:
        sizes[n["community"]] = sizes.get(n["community"], 0) + 1
    communities = [{"id": c, "title": titles.get(c, f"Community {c}"), "color": community_color(c), "size": s}
                   for c, s in sorted(sizes.items(), key=lambda kv: -kv[1])]
    return {
        "nodes": nodes,
        "edges": edges,
        "highlight": {"nodes": h_nodes, "edges": list(highlights.get("edges", [])),
                      "order": evidence_order(graph, h_nodes)},
        "communities": communities,
        "options": {"show_full": bool(show_full), "depth": 1, "pinned": sorted(pinned - set(h_nodes))},
    }


if __name__ == "__main__":
    out = Path("out")
    out.mkdir(exist_ok=True)
    g = load_graph()
    logger.info("Graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    (out / "graph.html").write_text(build_pyvis(g, subgraph_only=False), encoding="utf-8")
    top = sorted(g.nodes, key=lambda n: g.nodes[n]["degree"], reverse=True)[:3]
    sample = {"nodes": top, "edges": [], "communities": []}
    (out / "graph_highlight.html").write_text(build_pyvis(g, sample), encoding="utf-8")
    logger.info("Wrote out/graph.html and out/graph_highlight.html (highlighted: %s)", top)
    sys.exit(0)
