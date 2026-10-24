"""ThemeGraph Streamlit app: chat on the left, evidence-highlighted knowledge graph on the right."""
from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from backend.config import OUTPUT_DIR
from backend.contracts import Response

st.set_page_config(page_title="ThemeGraph", page_icon="🕸️", layout="wide")

ROUTE_COLORS = {"global": "#7c3aed", "local": "#0ea5e9", "basic": "#64748b", "abstain": "#f59e0b"}
MODES = {"Auto (router)": "auto", "Global": "global", "Local": "local", "Basic (baseline)": "basic"}
INDEX_READY = (OUTPUT_DIR / "entities.parquet").exists()


@st.cache_resource(show_spinner="Loading knowledge graph...")
def get_graph() -> nx.Graph:
    """Load the NetworkX graph once per server process."""
    from backend.graph import load_graph

    return load_graph()


@st.cache_data(show_spinner=False)
def get_communities() -> dict[str, str]:
    """Community id to report title."""
    from backend.graph import community_titles

    return community_titles()


@st.cache_data(show_spinner=False)
def get_text_units() -> dict[str, str]:
    """Text unit uuid to text, for node source excerpts."""
    from backend.engine import load_tables

    tu = load_tables().get("text_units")
    return {} if tu is None else dict(zip(tu["id"].astype(str), tu["text"].astype(str)))


def badge(route: str) -> str:
    """HTML pill showing the route taken."""
    return (f"<span style='background:{ROUTE_COLORS.get(route, '#64748b')};color:white;padding:2px 10px;"
            f"border-radius:999px;font-size:0.75rem;font-weight:600'>{route.upper()}</span>")


def render_response(resp: Response) -> None:
    """Render one assistant answer with badge, evidence, sources and timings."""
    conf = resp.router.get("confidence")
    conf_txt = f" &nbsp;<small>router confidence {conf:.2f}</small>" if isinstance(conf, (int, float)) else ""
    st.markdown(badge(resp.route) + conf_txt, unsafe_allow_html=True)
    if resp.abstained:
        st.info(f"**Not answered.** {resp.abstain_reason}")
    else:
        st.markdown(resp.answer)
        with st.expander(f"Evidence ({len(resp.claims)} verified claims)"):
            for i, c in enumerate(resp.claims, 1):
                st.markdown(f"**{i}. {c.text}**")
                for e in c.evidence:
                    st.markdown(f"> \u201c{e.quote}\u201d  \n<small>chunk `{e.chunk_id}`</small>", unsafe_allow_html=True)
    if resp.sources:
        with st.expander(f"Sources ({len(resp.sources)})"):
            for s in resp.sources:
                st.markdown(f"**{s.doc}** <small>chunk `{s.chunk_id}`</small>", unsafe_allow_html=True)
                st.caption(s.text)
    if resp.highlights.communities and resp.route == "global":
        titles = get_communities()
        st.caption("Communities used: " + "; ".join(titles.get(c, c) for c in resp.highlights.communities[:8]))
    st.caption(" · ".join(f"{k} {v} ms" for k, v in resp.timings_ms.items()))


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

with st.sidebar:
    st.header("Communities")
    if INDEX_READY:
        comms = get_communities()
        st.caption(f"{len(comms)} communities detected by GraphRAG")
        for cid, title in comms.items():
            st.markdown(f"<small><b>{cid}</b> · {title}</small>", unsafe_allow_html=True)
    else:
        st.info("No index yet. Upload documents and build the knowledge graph.")

left, right = st.columns([4, 6], gap="large")

with left:
    st.title("ThemeGraph")
    st.caption("Ask thematic or factual questions. Every answer is backed by verified quotes.")

    with st.expander("Corpus upload", expanded=not INDEX_READY):
        files = st.file_uploader("Add .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)
        if INDEX_READY:
            st.caption("An index already exists in `myproj/output`. Delete it to rebuild from scratch.")
        if st.button("Build knowledge graph", disabled=INDEX_READY or not files):
            from backend.engine import ingest_files, run_index

            with st.spinner("Indexing with GraphRAG... this can take several minutes"):
                with tempfile.TemporaryDirectory() as tmp:
                    paths = []
                    for f in files:
                        p = Path(tmp) / f.name
                        p.write_bytes(f.getbuffer())
                        paths.append(p)
                    ingest_files(paths)
                ok = run_index()
            st.cache_resource.clear()
            st.cache_data.clear()
            (st.success if ok else st.error)("Index built." if ok else "Indexing failed, see logs.")
            st.rerun()

    with st.expander("Search mode"):
        mode_label = st.radio("Mode", list(MODES), horizontal=True, label_visibility="collapsed")

    chat = st.container(height=560)
    with chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_response(msg["response"])

    question = st.chat_input("Ask about the corpus...", disabled=not INDEX_READY)
    if question:
        from backend.pipeline import run

        st.session_state.messages.append({"role": "user", "content": question})
        with chat:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Routing, searching the graph and verifying quotes..."):
                    resp = run(question, mode=MODES[mode_label])
                render_response(resp)
        st.session_state.messages.append({"role": "assistant", "response": resp})
        st.session_state.last_response = resp

with right:
    st.subheader("Knowledge graph")
    if not INDEX_READY:
        st.info("The graph appears here once `myproj/output` contains the GraphRAG parquet files.")
    else:
        from backend.graph import PALETTE, HIGHLIGHT, build_pyvis

        graph = get_graph()
        resp: Response | None = st.session_state.last_response
        highlights = resp.highlights if resp else None
        c1, c2 = st.columns([1, 2])
        show_full = c1.checkbox("Show full graph", value=False)
        n_hl = len(highlights.nodes) if highlights else 0
        c2.caption(f"{graph.number_of_nodes()} entities · {graph.number_of_edges()} relationships · "
                   f"{n_hl} highlighted")
        html = build_pyvis(graph, highlights, subgraph_only=not show_full, height="600px")
        components.html(html, height=620, scrolling=False)

        legend = (f"<span style='color:{HIGHLIGHT}'>●</span> evidence path &nbsp; "
                  + " ".join(f"<span style='color:{c}'>●</span>" for c in PALETTE[:6])
                  + " community colours &nbsp; <span style='color:#3a3f4b'>●</span> context (dimmed)")
        st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)

        options = highlights.nodes if highlights and highlights.nodes else sorted(
            graph.nodes, key=lambda n: graph.nodes[n].get("degree", 0), reverse=True)[:50]
        node = st.selectbox("Inspect a node", options, index=None,
                            placeholder="Pick a highlighted entity to see its description and sources")
        if node:
            d = graph.nodes[node]
            st.markdown(f"**{node}** · `{d.get('type')}` · community {d.get('community')} · degree {d.get('degree')}")
            st.write(d.get("description", ""))
            units = get_text_units()
            for tid in d.get("text_unit_ids", [])[:3]:
                if tid in units:
                    st.caption(units[tid][:400] + "...")
