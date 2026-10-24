"""Typed accessors over st.session_state."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st

from backend.contracts import Response

TAB_KEY = "tg_tab"
GRAPH_KEY = "tg_graph"
MODE_KEY = "tg_mode"
TABS = ["Graph", "Themes", "Sources", "Compare", "Results"]
MODE_LABELS = {"auto": "Auto", "global": "Global", "local": "Local", "hybrid": "Hybrid", "basic": "GraphRAG basic",
               "graphrag_local": "GraphRAG local", "graphrag_global": "GraphRAG global"}


@dataclass
class Message:
    """One chat turn; assistant turns carry a Response."""

    role: Literal["user", "assistant"]
    content: str = ""
    response: Response | None = None
    question: str = ""
    mode: str = "auto"
    streamed: bool = False
    msg_id: int = 0


@dataclass
class Focus:
    """Graph elements the component should zoom to when nonce changes."""

    nodes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    nonce: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Payload form for the component."""
        return {"nodes": list(self.nodes), "edges": list(self.edges), "nonce": self.nonce}


_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "last_response": None,
    "focus": Focus(),
    "show_full": True,
    "selected_node": None,
    "expanded": [],
    "last_event_id": None,
    "pending": None,
    "compare": {},
    "next_msg_id": 1,
    "error": None,
}


def init() -> None:
    """Create missing keys with defaults."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)
    st.session_state.setdefault(MODE_KEY, "auto")


def messages() -> list[Message]:
    """Chat history."""
    return st.session_state["messages"]


def add_message(msg: Message) -> Message:
    """Append a message with a fresh id."""
    msg.msg_id = st.session_state["next_msg_id"]
    st.session_state["next_msg_id"] += 1
    st.session_state["messages"].append(msg)
    return msg


def last_response() -> Response | None:
    """Most recent assistant Response."""
    return st.session_state["last_response"]


def set_last_response(resp: Response | None) -> None:
    """Store the latest Response and reset per-answer graph state."""
    st.session_state["last_response"] = resp
    st.session_state["expanded"] = []
    st.session_state["show_full"] = resp is None or not resp.highlights.nodes


def focus() -> Focus:
    """Current focus request."""
    return st.session_state["focus"]


def set_focus(nodes: list[str], edges: list[str] | None = None, switch_tab: bool = True) -> None:
    """Ask the graph to zoom to nodes/edges (bumps the nonce) and show the Graph tab."""
    old: Focus = st.session_state["focus"]
    st.session_state["focus"] = Focus(nodes=list(nodes), edges=list(edges or []), nonce=old.nonce + 1)
    if switch_tab:
        st.session_state[TAB_KEY] = "Graph"


def nonce() -> int:
    """Focus nonce."""
    return focus().nonce


def show_full() -> bool:
    """Whether the graph opens in full view."""
    return bool(st.session_state["show_full"])


def selected_node() -> str | None:
    """Node last clicked in the graph."""
    return st.session_state["selected_node"]


def set_selected_node(node: str | None) -> None:
    """Remember the node clicked in the graph."""
    st.session_state["selected_node"] = node


def expanded() -> list[str]:
    """Nodes whose neighbours were added to the payload."""
    return st.session_state["expanded"]


def expand(node: str) -> None:
    """Add a node's neighbours to the payload."""
    if node not in st.session_state["expanded"]:
        st.session_state["expanded"].append(node)


def mode() -> str:
    """Selected search mode."""
    return st.session_state.get(MODE_KEY) or "auto"


def ask(question: str, mode_override: str | None = None) -> None:
    """Queue a question to be answered on this run."""
    question = (question or "").strip()
    if question:
        st.session_state["pending"] = {"question": question, "mode": mode_override or mode()}


def pop_pending() -> dict | None:
    """Take the queued question, if any."""
    pending = st.session_state["pending"]
    st.session_state["pending"] = None
    return pending


def seen_event(event_id: str | None) -> bool:
    """True if this component event was already handled; records it otherwise."""
    if not event_id or event_id == st.session_state["last_event_id"]:
        return True
    st.session_state["last_event_id"] = event_id
    return False


def compare_results() -> dict[str, Response]:
    """Basic-search responses keyed by question, for the Compare tab."""
    return st.session_state["compare"]
