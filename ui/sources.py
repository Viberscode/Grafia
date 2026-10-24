"""Sources tab: corpus documents and the chunks cited by the latest answer, quotes highlighted."""
from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from backend.contracts import Response
from ui.data import DataSource


def highlight(text: str, quotes: list[str]) -> str:
    """Escaped HTML of text with each quote wrapped in <mark> (whitespace/case tolerant)."""
    spans: list[tuple[int, int]] = []
    for q in quotes:
        words = q.strip().strip("\u201c\u201d\"'").split()
        if not words:
            continue
        m = re.search(r"\s+".join(re.escape(w) for w in words), text, flags=re.IGNORECASE)
        if m:
            spans.append(m.span())
    out, pos = [], 0
    for start, end in sorted(spans):
        if start < pos:
            continue
        out.append(html.escape(text[pos:start]))
        out.append(f"<mark>{html.escape(text[start:end])}</mark>")
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def quotes_by_chunk(resp: Response) -> dict[str, list[str]]:
    """Verified quotes grouped by chunk id."""
    out: dict[str, list[str]] = {}
    for claim in resp.claims:
        for ev in claim.evidence:
            out.setdefault(ev.chunk_id, []).append(ev.quote)
    return out


def cited_chunks(data: DataSource, resp: Response) -> list[dict]:
    """[{title, chunk_id, text, quotes}] for report chunks and text-unit sources."""
    quotes = quotes_by_chunk(resp)
    rows = []
    for cid in [c for c in quotes if c.startswith("report_")]:
        comm = cid.removeprefix("report_")
        rows.append({"title": f"Community report: {data.titles.get(comm, comm)}", "chunk_id": cid,
                     "text": data.report_text(comm), "quotes": quotes[cid]})
    for s in resp.sources:
        full = data.full_chunk(s.text)
        rows.append({"title": s.doc, "chunk_id": s.chunk_id, "text": full[1] if full else s.text,
                     "quotes": quotes.get(s.chunk_id, [])})
    return sorted(rows, key=lambda r: -len(r["quotes"]))


def documents_table(data: DataSource, resp: Response | None) -> None:
    """Corpus documents with chunk counts and whether the latest answer cites them."""
    cited = {s.doc for s in resp.sources} if resp else set()
    chunks: dict[str, int] = {}
    for doc, _ in data.text_units.values():
        chunks[doc] = chunks.get(doc, 0) + 1
    df = pd.DataFrame([{"Document": d["title"], "Chunks": chunks.get(d["title"], 0),
                        "Cited": "✓ cited" if d["title"] in cited else ""} for d in data.documents])
    st.dataframe(df, hide_index=True, width="stretch", height=min(38 + 35 * len(df), 300))


def render(data: DataSource, resp: Response | None) -> None:
    """Documents list plus cited chunks."""
    st.markdown("##### :material/folder_open: Documents")
    documents_table(data, resp)
    st.markdown("##### :material/format_quote: Cited chunks")
    if resp is None or resp.abstained:
        st.info("Ask a question to see the chunks its answer cites, with each verified quote highlighted.")
        return
    rows = cited_chunks(data, resp)
    if not rows:
        st.info("This answer has no cited chunks.")
        return
    for i, row in enumerate(rows):
        n = len(row["quotes"])
        label = f"{row['title']} · {n} quote{'s' if n != 1 else ''}"
        with st.expander(label, expanded=i < 2 and n > 0, icon=":material/description:"):
            st.markdown(f"<div class='tg-chunk'>{highlight(row['text'], row['quotes'])}</div>", unsafe_allow_html=True)
            st.caption(f"chunk {row['chunk_id']}")
