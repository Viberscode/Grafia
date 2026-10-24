"""ThemeGraph Streamlit app: chat on the left, interactive evidence graph and tabs on the right."""
from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="ThemeGraph", page_icon=":material/hub:", layout="wide")

from ui import chat, compare, ingest_panel, sources, state, styles, themes  # noqa: E402
from ui.data import DataSource, load, payload  # noqa: E402

logger = logging.getLogger("themegraph.app")


def handle_graph_event(event: dict | None) -> None:
    """Apply the latest component event once (deduplicated by its event id)."""
    if not isinstance(event, dict) or state.seen_event(event.get("eid")):
        return
    kind = event.get("type")
    if kind == "ask":
        state.ask(str(event.get("question", "")))
    elif kind == "expand":
        state.expand(str(event.get("id")))
    elif kind == "node_click":
        state.set_selected_node(str(event.get("id")))
    elif kind == "clear":
        state.set_selected_node(None)
    logger.info("Graph event: %s", kind)


def safe(section: str, fn, *args) -> None:
    """Render a section; show a friendly error card instead of crashing."""
    try:
        fn(*args)
    except Exception as err:
        logger.exception("Section %s failed", section)
        st.error(f"The {section} panel hit a problem and was skipped. The rest of the app still works.",
                 icon=":material/error:")
        with st.expander("Details"):
            st.code(f"{type(err).__name__}: {err}", language=None)


def graph_tab(data: DataSource) -> None:
    """Interactive graph for the latest answer."""
    from ui.graph_component import graph_view

    resp = state.last_response()
    h = resp.highlights if resp else None
    p = payload(data, data.is_mock, tuple(h.nodes) if h else (), tuple(h.edges) if h else (),
                tuple(h.communities) if h else (), state.show_full(), tuple(state.expanded()))
    n_hl = len(p["highlight"]["nodes"])
    st.caption(f"{len(p['nodes'])} entities · {len(p['edges'])} relationships shown"
               + (f" · :red[{n_hl} on the evidence path]" if n_hl else " · ask a question to highlight evidence"))
    graph_view(p, state.focus().as_dict(), key=state.GRAPH_KEY, height=640)


def right_column(data: DataSource) -> None:
    """Tabs: Graph | Themes | Sources | Compare | Results."""
    tabs = st.tabs(state.TABS, key=state.TAB_KEY, on_change="rerun", default="Graph")
    with tabs[0]:
        safe("graph", graph_tab, data)
    with tabs[1]:
        safe("themes", themes.render, data)
    with tabs[2]:
        safe("sources", sources.render, data, state.last_response())
    with tabs[3]:
        safe("compare", compare.render, data)
    with tabs[4]:
        safe("results", compare.results)


def main() -> None:
    styles.inject()
    state.init()
    data = load()
    handle_graph_event(st.session_state.get(state.GRAPH_KEY))
    if data.is_mock:
        from ui.mock import BANNER

        st.info(BANNER, icon=":material/science:")
    left, right = st.columns([4, 6], gap="large")
    with left:
        safe("header", chat.header, data)
        safe("documents", ingest_panel.render, data)
        safe("chat", chat.render, data)
    with right:
        with st.container(key="tg_right"):
            right_column(data)


main()
