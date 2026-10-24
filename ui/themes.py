"""Themes tab: community report cards with Focus and Ask."""
from __future__ import annotations

import html

import streamlit as st

from backend.graph import community_color
from ui import state
from ui.chat import edges_among
from ui.data import DataSource

MAX_CARDS = 30


def _focus(data: DataSource, community: str) -> None:
    nodes = data.community_nodes(community)
    state.set_focus(nodes, edges_among(data.graph, nodes))


def _matches(report: dict, query: str, level: str) -> bool:
    if level != "All" and str(report["level"]) != level:
        return False
    q = query.strip().lower()
    return not q or q in report["title"].lower() or q in report["summary"].lower()


def card(data: DataSource, report: dict) -> None:
    """One theme card."""
    cid = report["community"]
    with st.container(border=True, key=f"tg_theme_card_{cid}"):
        dot = f"<span class='tg-dot' style='background:{community_color(cid)}' aria-hidden='true'></span>"
        st.markdown(f"{dot}**{html.escape(report['title'])}**", unsafe_allow_html=True)
        n_nodes = len(data.community_nodes(cid))
        st.caption(f"Level {report['level']} · {n_nodes} entities · importance {report['rank']:.1f}")
        summary = report["summary"]
        st.markdown(f"<div class='tg-muted'>{html.escape(summary[:260])}{'...' if len(summary) > 260 else ''}</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("Focus", key=f"tf_{cid}", icon=":material/center_focus_strong:", width="stretch",
                  on_click=_focus, args=(data, cid), disabled=n_nodes == 0)
        c2.button("Ask", key=f"ta_{cid}", icon=":material/chat:", width="stretch",
                  on_click=state.ask, args=(f"What are the key themes around {report['title']}?",))


def render(data: DataSource) -> None:
    """Filterable grid of theme cards, most important first."""
    if not data.reports:
        st.info("No community reports found in the index.")
        return
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    query = c1.text_input("Filter themes", placeholder="Filter by title or summary", key="tg_theme_filter")
    levels = ["All"] + sorted({str(r["level"]) for r in data.reports})
    level = c2.segmented_control("Level", levels, default="All", key="tg_theme_level", required=True) or "All"
    reports = [r for r in sorted(data.reports, key=lambda r: -r["rank"]) if _matches(r, query, level)]
    st.caption(f"{len(reports)} of {len(data.reports)} themes")
    cols = st.columns(2)
    for i, report in enumerate(reports[:MAX_CARDS]):
        with cols[i % 2]:
            card(data, report)
