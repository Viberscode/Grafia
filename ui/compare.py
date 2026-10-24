"""Compare tab (basic vs auto side by side) and Results tab (benchmark summary)."""
from __future__ import annotations

import difflib
import html
import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.config import PROJECT_ROOT
from backend.contracts import Response
from ui import state
from ui.chat import route_badge
from ui.data import DataSource

logger = logging.getLogger(__name__)

SUMMARY_CSV = Path(PROJECT_ROOT) / "eval" / "results" / "summary.csv"


def _plain(text: str) -> str:
    return re.sub(r"[*_#`>]+", "", text)


def diff_html(mine: str, other: str) -> str:
    """mine as HTML with words that don't appear in other marked."""
    a, b = _plain(mine).split(), _plain(other).split()
    sm = difflib.SequenceMatcher(a=[w.lower().strip(".,;:") for w in a], b=[w.lower().strip(".,;:") for w in b],
                                 autojunk=False)
    out = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        chunk = html.escape(" ".join(a[i1:i2]))
        out.append(chunk if tag == "equal" else f"<mark class='only'>{chunk}</mark>")
    return " ".join(x for x in out if x)


def _docs(resp: Response, data: DataSource) -> set[str]:
    docs = {s.doc for s in resp.sources}
    docs |= {f"Report: {data.titles.get(e.chunk_id[7:], e.chunk_id)}" for c in resp.claims for e in c.evidence
             if e.chunk_id.startswith("report_")}
    return docs


def _side(data: DataSource, title: str, resp: Response, other: Response) -> None:
    st.markdown(f"**{title}**")
    route_badge(resp)
    s = resp.stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Verified claims", len(resp.claims), delta=len(resp.claims) - len(other.claims) or None)
    m2.metric("Quotes valid", f"{s.get('quotes_valid', 0)}/{s.get('quotes_total', 0)}")
    m3.metric("Time", f"{resp.timings_ms.get('total', 0) / 1000:.1f} s")
    if resp.abstained:
        st.warning(resp.abstain_reason or "Abstained", icon=":material/block:")
    st.markdown(f"<div class='tg-card tg-diff'>{diff_html(resp.answer, other.answer)}</div>", unsafe_allow_html=True)
    mine, theirs = _docs(resp, data), _docs(other, data)
    if mine:
        st.caption("Evidence from")
        for d in sorted(mine):
            st.markdown(f"- {'**' + html.escape(d) + '** (only here)' if d not in theirs else html.escape(d)}")
    st.caption(f"{len(resp.highlights.nodes)} graph entities highlighted")


def _run_basic(data: DataSource, question: str) -> None:
    with st.spinner("Running GraphRAG basic search..."):
        state.compare_results()[question] = data.run(question, "basic")


def _last_turn() -> tuple[str, Response] | None:
    for m in reversed(state.messages()):
        if m.role == "assistant" and m.response is not None:
            return m.question, m.response
    return None


def render(data: DataSource) -> None:
    """Basic search vs the routed answer for the latest question."""
    turn = _last_turn()
    if turn is None:
        st.info("Ask a question first, then compare the ThemeGraph answer with the GraphRAG basic-search baseline.")
        return
    question, auto = turn
    st.markdown(f"Question: *{html.escape(question)}*")
    basic = state.compare_results().get(question)
    if basic is None:
        st.button("Run GraphRAG basic for this question", icon=":material/compare_arrows:", type="primary",
                  on_click=_run_basic, args=(data, question))
        st.caption("Highlighted words appear in only one of the two answers.")
        return
    left, right = st.columns(2, gap="medium")
    with left:
        _side(data, f"ThemeGraph ({auto.route})", auto, basic)
        if auto.trace:
            st.caption("Trace: " + " → ".join(f"{s.kind} ({len(s.ids)})" for s in auto.trace))
    with right:
        _side(data, "GraphRAG basic (baseline)", basic, auto)
    st.caption("Highlighted words appear in only one of the two answers; bold documents are cited by one side only.")


def results() -> None:
    """Benchmark table and chart from eval/results/summary.csv."""
    if not SUMMARY_CSV.exists():
        st.info("No benchmark results yet. Run `python -m eval.run` then `python -m eval.report` "
                "to create `eval/results/summary.csv`.", icon=":material/science:")
        return
    try:
        df = pd.read_csv(SUMMARY_CSV)
    except Exception:
        logger.exception("Could not read %s", SUMMARY_CSV)
        st.error("Could not read the benchmark summary file.")
        return
    st.dataframe(df, hide_index=True, width="stretch")
    overall = df[df["type"] == "ALL"] if "type" in df.columns else df
    cols = [c for c in ("evidence_recall", "abstain_precision") if c in overall.columns]
    if cols and "system" in overall.columns:
        st.markdown("##### Evidence recall and abstain precision by system")
        st.bar_chart(overall.set_index("system")[cols], stack=False, height=300)
