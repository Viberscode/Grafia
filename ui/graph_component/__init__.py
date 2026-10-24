"""Bidirectional vis-network graph component (plain HTML/JS, no build step)."""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

_FRONTEND = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component("themegraph_graph", path=str(_FRONTEND))

EVENT_TYPES = {"node_click", "edge_click", "ask", "expand", "clear"}


def graph_view(payload: dict, focus: dict | None = None, key: str = "graph", height: int = 680) -> dict | None:
    """Render the graph and return the latest event dict from the browser, or None."""
    event = _component(payload=payload, focus=focus or {"nodes": [], "edges": [], "nonce": 0},
                       height=height, key=key, default=None)
    if isinstance(event, dict) and event.get("type") in EVENT_TYPES:
        return event
    return None
