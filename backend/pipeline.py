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
from backend.contracts import Claim, Highlights, Response, Source
from backend.retrieval import Retrieved, mentioned, retrieve

logger = logging.getLogger(__name__)

MODES = ("auto", "global", "local", "hybrid", "basic", "graphrag_local", "graphrag_global")
BASELINES = {"basic": "basic", "graphrag_local": "local", "graphrag_global": "global"}
MIN_THEME_NODES = 3
EXCERPT_CHARS = 300
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


def reset_graph() -> None:
    """Forget the cached graph, tables and store clients so the next call reloads a rebuilt index."""
    global _GRAPH
    from backend.stores import graph_store, vector_store

    _GRAPH = None
    engine.load_tables.cache_clear()
    graph_store.reset_store()
    vector_store.reset_vectors()


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


def edge_ids(pairs: list[tuple[str, str]]) -> list[str]:
    """UI edge ids (relationship human_readable_id) for entity title pairs."""
    g = get_graph()
    return sorted({str(g.edges[u, v].get("id", f"{u}->{v}")) for u, v in pairs if g.has_edge(u, v)})


def _path_between_cited(path_nodes: list[str], cited: list[str]) -> list[str]:
    """The stretch of the retrieval path between the first and last cited entity."""
    idx = [i for i, n in enumerate(path_nodes) if n in cited]
    return path_nodes[idx[0]:idx[-1] + 1] if len(idx) >= 2 else []


def ground_highlights(route: str, r: Retrieved, claims: list[Claim]) -> Highlights:
    """Retrieved nodes that verified claims or quotes name, plus the trace path joining them."""
    text = " ".join([c.text for c in claims] + [e.quote for c in claims for e in c.evidence])
    named = mentioned(r.nodes, text)
    if route == "global":
        nodes = named if len(named) >= MIN_THEME_NODES else r.nodes
    else:
        local_info = r.info.get("local", r.info)
        nodes = list(dict.fromkeys(named + _path_between_cited(local_info.get("path", []), named)))
        if not nodes:
            nodes = next((s.ids for s in r.trace if s.kind == "seed"), [])[:3]
    keep = set(nodes)
    g = get_graph()
    communities = r.communities or sorted({str(g.nodes[n].get("community", "")) for n in nodes if n in g} - {""})
    return Highlights(nodes=nodes, edges=edge_ids([(u, v) for u, v in r.edges if u in keep and v in keep]),
                      communities=communities)


def contexts_to_sources(contexts: list[dict], claims: list[Claim]) -> list[Source]:
    """Sources for the contexts, cited chunks first."""
    cited = {e.chunk_id for c in claims for e in c.evidence}
    ordered = sorted(contexts, key=lambda c: c["chunk_id"] not in cited)
    return [Source(doc=c.get("doc") or "unknown document", chunk_id=c["chunk_id"],
                   text=c["text"][:EXCERPT_CHARS] + ("..." if len(c["text"]) > EXCERPT_CHARS else ""))
            for c in ordered]


def _answer_stats(gen: dict, claims: list[Claim], lookup: dict[str, str]) -> dict[str, int]:
    """Quote and claim counts for the evaluation harness."""
    return {
        "chunks": len(lookup),
        "quotes_total": sum(len(c.get("evidence", [])) for c in gen["claims"]),
        "quotes_valid": sum(len(c.evidence) for c in verify.check_quotes(gen["claims"], lookup)),
        "claims_raw": len(gen["claims"]),
        "claims_verified": len(claims),
    }


def _run_graph(question: str, route: str, route_info: dict, timings: dict[str, int], t_start: float,
               support_check: bool, progress: Callable[[str], None] | None) -> Response:
    """ThemeGraph retrieval (global, local or hybrid) over the graph store and ChromaDB."""
    _notify(progress, "searching")
    with _timer(timings, "retrieve"):
        r = retrieve(route, question)
    timings.update({f"retrieve_{k}": v for k, v in r.info.get("timings_ms", {}).items()})
    chunks = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in r.contexts]
    lookup = {c["chunk_id"]: c["text"] for c in chunks}

    _notify(progress, "verifying")
    with _timer(timings, "answer"):
        gen = answer.generate_claims(question, chunks)
    with _timer(timings, "verify"):
        claims = verify.verify_claims(gen["claims"], lookup, support_check=support_check)
    stats = _answer_stats(gen, claims, lookup)
    is_abstain, reason = abstain.should_abstain(route, {"context": len(chunks), "signal": r.signal}, claims,
                                                closest_community(question))
    with _timer(timings, "path"):
        highlights = ground_highlights(route, r, claims)
        sources = contexts_to_sources(r.contexts, claims)
    timings["total"] = int((time.perf_counter() - t_start) * 1000)
    retrieval = {k: v for k, v in r.info.items() if k != "timings_ms"}
    retrieval["contexts"] = [{k: c.get(k) for k in ("chunk_id", "label", "doc", "why")} for c in r.contexts]
    common = dict(route=route, highlights=highlights, sources=sources, timings_ms=timings, router=route_info,
                  stats=stats, trace=r.trace, retrieval=retrieval)
    if is_abstain:
        return Response(answer="I couldn't find a verifiable answer in the documents.", abstained=True,
                        abstain_reason=reason, **common)
    return Response(answer=gen["answer"], claims=claims, **common)


def run(question: str, mode: str = "auto", support_check: bool = True,
        progress: Callable[[str], None] | None = None) -> Response:
    """Answer a question; never raises (errors become an abstained Response).

    Modes global/local/hybrid (and auto) use ThemeGraph retrieval; basic/graphrag_local/graphrag_global
    run GraphRAG's own searches as baselines. progress receives "routing", "searching" and "verifying".
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
        if route not in BASELINES:
            return _run_graph(question, route, route_info, timings, t_start, support_check, progress)

        _notify(progress, "searching")
        with _timer(timings, "search"):
            result = engine.run_query(question, BASELINES[route])
        method = BASELINES[route]
        chunks = build_chunks(method, result.context)
        lookup = {c["chunk_id"]: c["text"] for c in chunks}

        _notify(progress, "verifying")
        with _timer(timings, "answer"):
            gen = answer.generate_claims(question, chunks)
        with _timer(timings, "verify"):
            claims = verify.verify_claims(gen["claims"], lookup, support_check=support_check)

        stats = _answer_stats(gen, claims, lookup)
        scores = {"context": len(chunks), "signal": _signal(method, result.context, chunks)}
        is_abstain, reason = abstain.should_abstain(method, scores, claims, closest_community(question))

        with _timer(timings, "path"):
            focus = " ".join([c.text for c in claims] + [e.quote for c in claims for e in c.evidence])
            highlights, sources = path.build_highlights(
                method, result.context, get_graph(), engine.load_tables(), focus_text=focus)
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
