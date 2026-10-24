"""Chat column: header, empty state, message cards, follow-ups and the input row."""
from __future__ import annotations

import html
import logging
import re
import time
from collections import Counter
from typing import Iterator

import networkx as nx
import streamlit as st

from backend.contracts import Claim, Response
from ui import state
from ui.data import DataSource
from ui.styles import ROUTES, guide

logger = logging.getLogger(__name__)

STAGES = {"routing": "Routing the question...", "searching": "Searching the knowledge graph...",
          "verifying": "Writing and verifying claims..."}
OUT_OF_SCOPE_EXAMPLES = ["What is the capital of France?", "Who won the 2022 football World Cup?",
                         "What is a good pizza recipe?"]


# ---------- graph helpers (no LLM) ----------

def nodes_in_text(graph: nx.Graph, text: str, prefer: list[str]) -> list[str]:
    """Graph nodes named in text, preferring the highlighted ones."""
    low = text.lower()
    hits = [n for n in graph.nodes if len(n) > 2 and n.lower() in low]
    preferred = [n for n in hits if n in prefer]
    return preferred or hits


def quote_nodes(data: DataSource, resp: Response, quote: str, chunk_id: str) -> list[str]:
    """Nodes to focus for a quote: named entities, else the cited community, else all highlights."""
    nodes = nodes_in_text(data.graph, quote, resp.highlights.nodes)
    if not nodes and chunk_id.startswith("report_"):
        members = set(data.community_nodes(chunk_id.removeprefix("report_")))
        nodes = [n for n in resp.highlights.nodes if n in members] or sorted(members)
    return nodes or list(resp.highlights.nodes)


def edges_among(graph: nx.Graph, nodes: list[str]) -> list[str]:
    """Relationship ids between the given nodes."""
    present = [n for n in nodes if n in graph]
    return [str(d.get("id")) for _, _, d in graph.subgraph(present).edges(data=True)]


def followups(graph: nx.Graph, resp: Response, k: int = 3) -> list[str]:
    """Follow-up questions from neighbours of the highlighted entities."""
    hl = sorted((n for n in resp.highlights.nodes if n in graph), key=lambda n: -graph.nodes[n].get("degree", 0))
    if not hl:
        return []
    counts = Counter(m for n in hl for m in graph.neighbors(n) if m not in hl)
    neigh = [m for m, _ in counts.most_common(k)] or hl[1:k + 1]
    name = lambda n: n.title()  # noqa: E731
    out = []
    if neigh:
        out.append(f"What is {name(neigh[0])}?")
    if len(neigh) > 1:
        out.append(f"How is {name(hl[0])} connected to {name(neigh[1])}?")
    if len(neigh) > 2:
        out.append(f"What role does {name(neigh[2])} play?")
    elif len(hl) > 1:
        out.append(f"How is {name(hl[0])} connected to {name(hl[1])}?")
    return out[:k]


def closest_theme(reason: str | None) -> str | None:
    """Theme title quoted in an abstain reason."""
    m = re.search(r"\u201c(.+?)\u201d", reason or "")
    return m.group(1) if m else None


# ---------- callbacks ----------

def _show_quote(data: DataSource, resp: Response, quote: str, chunk_id: str) -> None:
    if resp is not state.last_response():
        state.set_last_response(resp)
    nodes = quote_nodes(data, resp, quote, chunk_id)
    state.set_focus(nodes, edges_among(data.graph, nodes))


def _show_theme(data: DataSource, title: str) -> None:
    ids = [c for c, t in data.titles.items() if t == title]
    nodes = [n for c in ids for n in data.community_nodes(c)]
    state.set_focus(nodes, edges_among(data.graph, nodes))


def _submit() -> None:
    state.ask(st.session_state.get("tg_input", ""))


# ---------- header & empty state ----------

def header(data: DataSource) -> None:
    """Title, corpus chips, index status and the Guide."""
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.markdown("### :material/hub: ThemeGraph", width="content")
        guide()
    s = data.stats()
    status_cls = "tg-status-demo" if data.is_mock else "tg-status-ok"
    chips = "".join(f"<span class='tg-chip' aria-label='{v} {k}'><b>{v}</b> {k}</span>" for k, v in s.items())
    st.markdown(f"<div class='tg-chips'>{chips}<span class='tg-chip {status_cls}'>&#9679; {html.escape(data.status)}</span></div>",
                unsafe_allow_html=True)


def examples(data: DataSource) -> dict[str, list[str]]:
    """Three example questions per type, built from community titles and top entities."""
    titles = [r["title"] for r in sorted(data.reports, key=lambda r: -r["rank"])] or list(data.titles.values())
    titles = [t for t in titles if t][:3] or ["the corpus"]
    top = sorted(data.graph.nodes, key=lambda n: -data.graph.nodes[n].get("degree", 0))[:4]
    thematic = ["What are the main themes in these documents?"] + [f"What are the key themes around {t}?" for t in titles[:2]]
    factual = [f"What is {top[0].title()}?", f"What role does {top[1].title()} play?",
               f"How is {top[2].title()} connected to {top[3].title()}?"] if len(top) >= 4 else []
    return {"Thematic (global)": thematic, "Specific (local)": factual, "Out of scope (abstain)": OUT_OF_SCOPE_EXAMPLES}


def empty_state(data: DataSource) -> None:
    """Example questions and a short explainer."""
    st.markdown("#### Ask about the corpus")
    st.caption("Try one of these. Each answer comes with verified quotes and a highlighted evidence path.")
    for label, qs in examples(data).items():
        st.markdown(f"<div class='tg-muted'>{label}</div>", unsafe_allow_html=True)
        for i, q in enumerate(qs):
            st.button(q, key=f"ex_{label}_{i}", on_click=state.ask, args=(q,), icon=":material/arrow_forward:")
    with st.expander("How it works", icon=":material/info:"):
        st.markdown(
            "1. **Route**: an LLM router picks *global* (themes), *local* (entities) or *abstain*.\n"
            "2. **Search**: GraphRAG queries community reports or the entity graph.\n"
            "3. **Verify**: every claim must cite a quote found verbatim in the retrieved text.\n"
            "4. **Highlight**: cited entities and the path between them light up in red on the graph.")


# ---------- message card ----------

def _stream_words(text: str, delay: float = 0.012) -> Iterator[str]:
    for token in re.split(r"(\s+)", text):
        yield token
        time.sleep(delay if token.strip() else 0)


def route_badge(resp: Response) -> None:
    """Route badge with icon, text and a reason/confidence tooltip."""
    meta = ROUTES.get(resp.route, ROUTES["basic"])
    conf = resp.router.get("confidence")
    conf_txt = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
    reason = str(resp.router.get("reason") or "No router details.")
    st.badge(meta["label"], icon=meta["icon"], color=meta["color"], help=f"{reason}{conf_txt}")


def _doc_for(data: DataSource, resp: Response, chunk_id: str) -> str:
    if chunk_id.startswith("report_"):
        cid = chunk_id.removeprefix("report_")
        return f"Community report: {data.titles.get(cid, cid)}"
    src = next((s for s in resp.sources if s.chunk_id == chunk_id), None)
    return src.doc if src else f"chunk {chunk_id}"


def claim_card(data: DataSource, resp: Response, claim: Claim, key: str) -> None:
    """Verified claim with expandable quotes and Show-in-graph buttons."""
    with st.container(border=True, key=f"tg_claim_{key}"):
        st.markdown(f":green[:material/check_circle:] {claim.text}")
        with st.expander(f"{len(claim.evidence)} verified quote{'s' if len(claim.evidence) != 1 else ''}"):
            for j, ev in enumerate(claim.evidence):
                st.markdown(f"<div class='tg-quote'>\u201c{html.escape(ev.quote)}\u201d</div>", unsafe_allow_html=True)
                c1, c2 = st.columns([3, 2], vertical_alignment="center")
                c1.caption(_doc_for(data, resp, ev.chunk_id))
                c2.button("Show in graph", key=f"sig_{key}_{j}", icon=":material/center_focus_strong:",
                          on_click=_show_quote, args=(data, resp, ev.quote, ev.chunk_id))


def abstain_card(data: DataSource, resp: Response, key: str) -> None:
    """Why the question was not answered, with the closest theme."""
    with st.container(border=True, key=f"tg_abstain_{key}"):
        st.markdown("**:orange[:material/block:] Not answered**")
        st.write(resp.abstain_reason or "The documents don't support an answer.")
        theme = closest_theme(resp.abstain_reason)
        if theme:
            st.button(f"Show closest theme: {theme}", key=f"theme_{key}", icon=":material/travel_explore:",
                      on_click=_show_theme, args=(data, theme))


def _timings(resp: Response) -> str:
    parts = [f"{k} {v} ms" for k, v in resp.timings_ms.items() if k != "total"]
    total = resp.timings_ms.get("total")
    if total is not None:
        parts.append(f"total {total / 1000:.1f} s")
    s = resp.stats
    if s.get("quotes_total"):
        parts.append(f"{s.get('quotes_valid', 0)}/{s['quotes_total']} quotes verified")
    return " · ".join(parts)


def message_card(data: DataSource, msg: state.Message, stream: bool = False) -> None:
    """Render one assistant answer."""
    resp = msg.response
    if resp is None:
        return
    route_badge(resp)
    if resp.abstained:
        abstain_card(data, resp, str(msg.msg_id))
    elif stream:
        st.write_stream(_stream_words(resp.answer))
    else:
        st.markdown(resp.answer)
    for i, claim in enumerate(resp.claims):
        claim_card(data, resp, claim, f"{msg.msg_id}_{i}")
    if resp.sources:
        with st.expander(f"Sources ({len(resp.sources)})", icon=":material/description:"):
            for s in resp.sources:
                st.markdown(f"**{html.escape(s.doc)}** <span class='tg-muted'>chunk {html.escape(s.chunk_id)}</span>",
                            unsafe_allow_html=True)
                st.caption(s.text[:300] + ("..." if len(s.text) > 300 else ""))
    st.caption(_timings(resp))
    with st.container(horizontal=True, vertical_alignment="center"):
        with st.popover("Copy", icon=":material/content_copy:", help="Copy the answer"):
            st.code(resp.answer, language=None, wrap_lines=True)
        st.feedback("thumbs", key=f"fb_{msg.msg_id}")


def followup_chips(data: DataSource, resp: Response) -> None:
    """Three follow-up questions for the latest answer."""
    qs = followups(data.graph, resp)
    if not qs:
        return
    with st.container(key="tg_followups"):
        st.caption("Follow-ups")
        for i, q in enumerate(qs):
            st.button(q, key=f"fu_{i}_{hash(q)}", on_click=state.ask, args=(q,), icon=":material/subdirectory_arrow_right:")


# ---------- turn processing ----------

def _run_with_status(data: DataSource, question: str, mode: str) -> Response:
    with st.status(STAGES["routing"], expanded=False) as status:
        def progress(stage: str) -> None:
            status.update(label=STAGES.get(stage, stage))
            status.write(STAGES.get(stage, stage))

        try:
            resp = data.run(question, mode, progress=progress)
        except Exception as err:
            logger.exception("Question failed")
            resp = Response(route="abstain", answer="Something went wrong while answering.", abstained=True,
                            abstain_reason=f"{type(err).__name__}: {err}")
        total = resp.timings_ms.get("total")
        label = f"Done in {total / 1000:.1f} s" if total else "Done"
        status.update(label=label, state="complete")
    return resp


def answer_pending(data: DataSource) -> None:
    """Answer the queued question (from input, chips, themes or graph events)."""
    pending = state.pop_pending()
    if not pending:
        return
    question, mode = pending["question"], pending["mode"]
    state.add_message(state.Message(role="user", content=question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        resp = _run_with_status(data, question, mode)
        msg = state.add_message(state.Message(role="assistant", response=resp, question=question, mode=mode, streamed=True))
        state.set_last_response(resp)
        message_card(data, msg, stream=True)
        followup_chips(data, resp)


def history(data: DataSource) -> None:
    """Previous turns; follow-ups only under the latest answer."""
    msgs = state.messages()
    last_assistant = max((m.msg_id for m in msgs if m.role == "assistant"), default=None)
    pending = st.session_state.get("pending") is not None
    for m in msgs:
        with st.chat_message(m.role):
            if m.role == "user":
                st.markdown(m.content)
                continue
            message_card(data, m)
            if m.msg_id == last_assistant and not pending and m.response:
                followup_chips(data, m.response)


def input_row() -> None:
    """Mode popover and the chat input."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption("Search mode", width="content")
        with st.popover(state.MODE_LABELS.get(state.mode(), "Auto"), icon=":material/tune:", help="Search mode"):
            st.segmented_control("Search mode", list(state.MODE_LABELS), key=state.MODE_KEY, required=True,
                                 format_func=state.MODE_LABELS.get)
            st.caption("Auto lets the router choose. Basic is the plain vector-search baseline.")
    st.chat_input("Ask about the corpus...", key="tg_input", on_submit=_submit)


def render(data: DataSource) -> None:
    """Whole chat column."""
    header(data)
    chat = st.container(height=640, key="tg_chat", border=False, autoscroll=True)
    with chat:
        if not state.messages() and not st.session_state.get("pending"):
            empty_state(data)
        history(data)
        answer_pending(data)
    input_row()
