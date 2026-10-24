"""Global (theme) retrieval: top community reports from ChromaDB and the entities they name."""
from __future__ import annotations

import logging
import time

from backend.contracts import TraceStep
from backend.retrieval import Retrieved, mentioned

logger = logging.getLogger(__name__)

TOP_COMMUNITIES = 3
REPORT_MIN = 0.3
MAX_NODES = 15


def _pick_reports(question: str, vectors: object, broad: bool) -> tuple[list, str]:
    """Top reports above the cutoff; for broad questions with no close match, the highest-ranked level-0 themes."""
    hits = vectors.search("reports", question, k=50)
    close = [h for h in hits if h.score >= REPORT_MIN][:TOP_COMMUNITIES]
    if close or not broad:
        return close, "similarity"
    top = sorted((h for h in hits if int(h.metadata.get("level", 0)) == 0),
                 key=lambda h: -float(h.metadata.get("rank", 0)))
    return top[:TOP_COMMUNITIES], "rank"


def retrieve_global(question: str, store: object, vectors: object, broad: bool | None = None) -> Retrieved:
    """Community reports (theme contexts), one community trace step each, entities named in the reports."""
    from backend.router import keyword_prior

    t0 = time.perf_counter()
    broad = keyword_prior(question) if broad is None else broad
    reports, how = _pick_reports(question, vectors, broad)
    out = Retrieved(signal=len(reports))
    if not reports:
        out.info = {"communities": [], "timings_ms": {"reports": int((time.perf_counter() - t0) * 1000)}}
        return out
    cids = [h.id for h in reports]
    members = store.community_members(cids)
    for h in reports:
        title = h.metadata.get("title", f"Community {h.id}")
        note = f"{title} (similarity {h.score:.2f})" if how == "similarity" else (
            f"{title} (top-ranked theme, rank {float(h.metadata.get('rank', 0)):.1f})")
        out.trace.append(TraceStep(kind="community", ids=[h.id], note=note))
        out.contexts.append({"chunk_id": f"report_{h.id}", "text": h.text, "label": "theme",
                             "doc": f"Theme report: {title}", "why": how})
    text = " ".join(h.text for h in reports)
    by_title: dict[str, int] = {}
    for m in members:
        by_title[m["title"]] = max(by_title.get(m["title"], 0), int(m.get("degree") or 0))
    named = mentioned(list(by_title), text)
    out.nodes = sorted(named, key=lambda t: -by_title[t])[:MAX_NODES]
    out.edges = [(e["source"], e["target"]) for e in store.subgraph(out.nodes)]
    out.communities = cids
    out.info = {"communities": [{"id": h.id, "title": h.metadata.get("title", ""), "score": h.score} for h in reports],
                "selection": how, "members": len(by_title),
                "timings_ms": {"reports": int((time.perf_counter() - t0) * 1000)}}
    return out
