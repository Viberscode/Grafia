"""Hybrid retrieval: run global and local, fuse contexts with reciprocal rank fusion, merge traces."""
from __future__ import annotations

from backend.retrieval import Retrieved
from backend.retrieval.global_ import retrieve_global
from backend.retrieval.local import retrieve_local

RRF_K = 60
MAX_CONTEXTS = 10


def rrf(rankings: list[list[dict]], k: int = RRF_K, limit: int = MAX_CONTEXTS) -> list[dict]:
    """Reciprocal rank fusion of context lists keyed by chunk_id."""
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranking in rankings:
        for rank, ctx in enumerate(ranking):
            cid = ctx["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(cid, ctx)
    order = sorted(scores, key=lambda c: -scores[c])[:limit]
    return [{**first[c], "rrf": round(scores[c], 5)} for c in order]


def retrieve_hybrid(question: str, store: object, vectors: object) -> Retrieved:
    """Themes and facts together: contexts labelled theme or fact, trace = global steps then local steps."""
    g = retrieve_global(question, store, vectors, broad=True)
    loc = retrieve_local(question, store, vectors)
    contexts = rrf([g.contexts, loc.contexts])
    split = {"theme": sum(c["label"] == "theme" for c in contexts), "fact": sum(c["label"] == "fact" for c in contexts)}
    return Retrieved(
        contexts=contexts,
        nodes=list(dict.fromkeys(loc.nodes + g.nodes)),
        edges=list(dict.fromkeys(loc.edges + g.edges)),
        communities=list(dict.fromkeys(g.communities + loc.communities)),
        trace=g.trace + loc.trace,
        signal=g.signal + loc.signal,
        info={"global": g.info, "local": loc.info, "split": split,
              "timings_ms": {f"global_{k}": v for k, v in g.info.get("timings_ms", {}).items()}
              | {f"local_{k}": v for k, v in loc.info.get("timings_ms", {}).items()}},
    )
