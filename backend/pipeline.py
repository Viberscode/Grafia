"""End-to-end question answering: route, search, answer, verify, abstain, highlight."""
from __future__ import annotations

import logging
import re
import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import networkx as nx

from backend import abstain, answer, engine, path, router, verify
from backend.contracts import Response

logger = logging.getLogger(__name__)

MODES = ("auto", "global", "local", "basic")
STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "was", "what",
             "who", "how", "why", "when", "which", "did", "does", "do", "with", "about", "by", "this"}

_GRAPH: nx.Graph | None = None


@contextmanager
def _timer(timings: dict[str, int], name: str) -> Iterator[None]:
    """Record elapsed milliseconds for a stage."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = int((time.perf_counter() - t0) * 1000)


def get_graph() -> nx.Graph:
    """Load the knowledge graph once per process."""
    global _GRAPH
    if _GRAPH is None:
        from backend.graph import load_graph

        _GRAPH = load_graph()
    return _GRAPH


def closest_community(question: str) -> str | None:
    """Community title with the largest word overlap with the question."""
    try:
        from backend.graph import community_titles

        words = set(re.findall(r"[a-z]+", question.lower())) - STOPWORDS
        best, score = None, 0
        for title in community_titles().values():
            overlap = len(words & set(re.findall(r"[a-z]+", title.lower())))
            if overlap > score:
                best, score = title, overlap
        return best
    except Exception:
        return None


def build_chunks(method: str, context: dict) -> list[dict]:
    """Numbered chunks for answer generation: reports for global, text units otherwise."""
    chunks: list[dict] = []
    if method == "global":
        for r in context.get("reports", []):
            cid = r.get("id", r.get("community", ""))
            text = r.get("content") or r.get("full_content") or r.get("summary") or ""
            if text:
                chunks.append({"chunk_id": f"report_{cid}", "text": f"{r.get('title', '')}\n{text}"})
    else:
        for tu in context.get("text_units", []):
            if tu.get("text"):
                chunks.append({"chunk_id": str(tu.get("id", len(chunks))), "text": str(tu["text"])})
    return chunks


def _signal(method: str, context: dict, chunks: list[dict]) -> int:
    """Retrieval signal: reports used (global), matched entities (local), chunks (basic)."""
    if method == "global":
        return len(context.get("reports", []))
    if method == "local":
        return len(path.matched_entities(context, get_graph()))
    return len(chunks)


def _notify(progress: Callable[[str], None] | None, stage: str) -> None:
    """Report a stage name to an optional UI callback, ignoring callback errors."""
    if progress is None:
        return
    try:
        progress(stage)
    except Exception:
        logger.debug("progress callback failed", exc_info=True)


def run(question: str, mode: str = "auto", support_check: bool = True,
        progress: Callable[[str], None] | None = None) -> Response:
    """Answer a question; never raises (errors become an abstained Response).

    progress, if given, receives "routing", "searching" and "verifying" as stages start.
    """
    timings: dict[str, int] = {}
    route_info: dict = {}
    t_start = time.perf_counter()
    try:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        _notify(progress, "routing")
        with _timer(timings, "route"):
            if mode == "auto":
                route_info = router.route_question(question)
                route = route_info["route"]
            else:
                route = mode
                route_info = {"route": mode, "confidence": 1.0, "reason": f"Forced mode: {mode}"}

        if route == "abstain":
            ok, reason = abstain.should_abstain("abstain", {}, [], closest_community(question))
            return Response(route="abstain", answer="I can't answer that from the indexed documents.",
                            abstained=True, abstain_reason=reason, timings_ms=timings, router=route_info)

        _notify(progress, "searching")
        with _timer(timings, "search"):
            result = engine.run_query(question, route)
        chunks = build_chunks(route, result.context)
        lookup = {c["chunk_id"]: c["text"] for c in chunks}

        _notify(progress, "verifying")
        with _timer(timings, "answer"):
            gen = answer.generate_claims(question, chunks)
        with _timer(timings, "verify"):
            claims = verify.verify_claims(gen["claims"], lookup, support_check=support_check)

        stats = {
            "chunks": len(chunks),
            "quotes_total": sum(len(c.get("evidence", [])) for c in gen["claims"]),
            "quotes_valid": sum(len(c.evidence) for c in verify.check_quotes(gen["claims"], lookup)),
            "claims_raw": len(gen["claims"]),
            "claims_verified": len(claims),
        }
        scores = {"context": len(chunks), "signal": _signal(route, result.context, chunks)}
        is_abstain, reason = abstain.should_abstain(route, scores, claims, closest_community(question))

        with _timer(timings, "path"):
            focus = " ".join([c.text for c in claims] + [e.quote for c in claims for e in c.evidence])
            highlights, sources = path.build_highlights(
                route, result.context, get_graph(), engine.load_tables(), focus_text=focus)
        timings["total"] = int((time.perf_counter() - t_start) * 1000)

        if is_abstain:
            return Response(route=route, answer="I couldn't find a verifiable answer in the documents.",
                            abstained=True, abstain_reason=reason, highlights=highlights, sources=sources,
                            timings_ms=timings, router=route_info, search_answer=result.answer, stats=stats)
        return Response(route=route, answer=gen["answer"], claims=claims, highlights=highlights, sources=sources,
                        timings_ms=timings, router=route_info, search_answer=result.answer, stats=stats)
    except Exception as err:
        logger.exception("Pipeline failed")
        timings["total"] = int((time.perf_counter() - t_start) * 1000)
        return Response(route="abstain", answer="Something went wrong while answering.", abstained=True,
                        abstain_reason=f"{type(err).__name__}: {err}", timings_ms=timings, router=route_info)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main themes?"
    m = sys.argv[2] if len(sys.argv) > 2 else "auto"
    sys.stdout.write(run(q, m).model_dump_json(indent=2) + "\n")
