"""Local (fact) retrieval: seed entities, expand 2 hops, rank with personalised PageRank, collect chunks."""
from __future__ import annotations

import logging
import time

import networkx as nx

from backend.contracts import TraceStep
from backend.retrieval import Retrieved, norm, title_aliases

logger = logging.getLogger(__name__)

MAX_SEEDS = 5
VECTOR_SEED_MIN = 0.45
HOPS = 2
PPR_ALPHA = 0.85
TOP_ENTITIES = 12
MAX_CHUNKS = 8
CHUNK_MIN = 0.3
MAX_PATH_HOPS = 4


def seed_entities(question: str, titles: list[str], vectors: object) -> list[tuple[str, str, float]]:
    """(title, how, weight) seeds: exact title, then alias; Chroma entity matches above the cutoff fill in
    only when fewer than two entities are named."""
    aliases = title_aliases(titles)
    q = norm(question)
    seeds: list[tuple[str, str, float]] = []
    taken = q
    for t in sorted(titles, key=len, reverse=True):
        full = aliases[t][0]
        if full in taken:
            seeds.append((t, "exact", 1.0))
            taken = taken.replace(full, " | ")
    for t in titles:
        if all(s[0] != t for s in seeds) and any(a in taken for a in aliases[t][1:]):
            seeds.append((t, "alias", 0.9))
    if len(seeds) >= 2:
        return seeds[:MAX_SEEDS]
    for hit in vectors.search("entities", question, k=MAX_SEEDS * 2):
        title = str(hit.metadata.get("title") or hit.text.split(":")[0])
        if hit.score >= VECTOR_SEED_MIN and title in aliases and all(s[0] != title for s in seeds):
            seeds.append((title, "vector", round(hit.score, 3)))
    return seeds[:MAX_SEEDS]


def personalised_pagerank(edges: list[dict], nodes: list[str], seeds: dict[str, float],
                          iterations: int = 100, tol: float = 1e-8) -> dict[str, float]:
    """Weighted PageRank over the expanded subgraph, restarting at the seeds (power iteration)."""
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges:
        g.add_edge(e["source"], e["target"], weight=float(e.get("weight") or 1.0))
    total = sum(seeds.get(n, 0.0) for n in g.nodes)
    if not g.number_of_nodes() or total <= 0:
        return {}
    restart = {n: seeds.get(n, 0.0) / total for n in g.nodes}
    strength = {n: sum(d.get("weight", 1.0) for _, _, d in g.edges(n, data=True)) for n in g.nodes}
    rank = dict(restart)
    for _ in range(iterations):
        dangling = sum(rank[n] for n in g.nodes if strength[n] == 0)
        nxt = {n: (1 - PPR_ALPHA) * restart[n] + PPR_ALPHA * dangling * restart[n] for n in g.nodes}
        for u, v, d in g.edges(data=True):
            w = d.get("weight", 1.0)
            nxt[v] += PPR_ALPHA * rank[u] * w / strength[u]
            nxt[u] += PPR_ALPHA * rank[v] * w / strength[v]
        done = sum(abs(nxt[n] - rank[n]) for n in g.nodes) < tol
        rank = nxt
        if done:
            break
    return rank


def _chunk_order(edges: list[dict], path_pairs: set[frozenset], score: dict[str, float]) -> list[str]:
    """Text unit ids from REL edges: path edges first, then by endpoint PageRank."""
    ranked = sorted(edges, key=lambda e: (frozenset((e["source"], e["target"])) not in path_pairs,
                                          -(score.get(e["source"], 0) + score.get(e["target"], 0))))
    return list(dict.fromkeys(u for e in ranked for u in e.get("text_unit_ids", [])))


def retrieve_local(question: str, store: object, vectors: object) -> Retrieved:
    """Seed -> 2-hop expansion -> personalised PageRank top 12 -> chunks (REL, MENTIONS, Chroma) -> documents."""
    timings: dict[str, int] = {}
    t0 = time.perf_counter()
    titles = store.entity_titles()
    seeds = seed_entities(question, titles, vectors)
    timings["seed"] = int((time.perf_counter() - t0) * 1000)
    out = Retrieved(signal=len(seeds), info={"seeds": [{"title": t, "how": h, "weight": w} for t, h, w in seeds]})
    if not seeds:
        out.info["timings_ms"] = timings
        return out
    seed_titles = [s[0] for s in seeds]
    out.trace.append(TraceStep(kind="seed", ids=seed_titles,
                               note="; ".join(f"{t} ({h}{'' if h != 'vector' else f' {w:.2f}'})" for t, h, w in seeds)))

    t1 = time.perf_counter()
    expanded = store.neighbors(seed_titles, hops=HOPS)
    edges = store.subgraph(expanded)
    timings["expand"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    score = personalised_pagerank(edges, expanded, {t: w for t, _, w in seeds})
    top = sorted(score, key=lambda n: -score[n])[:TOP_ENTITIES]
    top = list(dict.fromkeys(seed_titles + top))[:max(TOP_ENTITIES, len(seed_titles))]
    path: list[str] = []
    if len(seed_titles) >= 2:
        path = store.shortest_path(seed_titles[0], seed_titles[1], max_hops=MAX_PATH_HOPS)
    keep = list(dict.fromkeys(path + top))
    kept_edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    path_pairs = {frozenset(p) for p in zip(path, path[1:])}
    timings["rank"] = int((time.perf_counter() - t2) * 1000)
    hop_ids = path if len(path) > 1 else [n for n in top if n not in seed_titles]
    hop_note = (" -> ".join(path) + f" ({len(path) - 1} hops)") if len(path) > 1 else (
        f"{len(expanded)} entities within {HOPS} hops; personalised PageRank kept {len(top)}")
    out.trace.append(TraceStep(kind="hop", ids=hop_ids, note=hop_note))

    t3 = time.perf_counter()
    pool = {c["id"]: c for c in store.chunks_for(keep, limit=60)}
    picked: list[tuple[str, str]] = []
    for cid in _chunk_order(kept_edges, path_pairs, score):
        if cid in pool:
            picked.append((cid, "relationship"))
    ranked_keep = sorted(keep, key=lambda n: -score.get(n, 0))
    for c in store.chunks_for(ranked_keep[:6], limit=20):
        picked.append((c["id"], f"mentions {c['entity']}"))
        pool.setdefault(c["id"], c)
    for hit in vectors.search("chunks", question, k=4):
        if hit.score >= CHUNK_MIN:
            pool.setdefault(hit.id, {"id": hit.id, "text": hit.text, "doc_id": hit.metadata.get("doc_id", "")})
            picked.append((hit.id, f"vector {hit.score:.2f}"))
    chosen: dict[str, str] = {}
    for cid, why in picked:
        if cid not in chosen and len(chosen) < MAX_CHUNKS:
            chosen[cid] = why
    docs = store.documents_for(list(chosen))
    out.contexts = [{"chunk_id": cid, "text": pool[cid]["text"], "label": "fact", "doc": docs.get(cid, ""),
                     "why": why} for cid, why in chosen.items()]
    timings["chunks"] = int((time.perf_counter() - t3) * 1000)

    out.nodes = keep
    out.edges = [(e["source"], e["target"]) for e in kept_edges]
    out.trace.append(TraceStep(kind="chunk", ids=list(chosen),
                               note=", ".join(sorted({w.split(' ')[0] for w in chosen.values()})) + " evidence"))
    out.trace.append(TraceStep(kind="document", ids=list(dict.fromkeys(d for d in docs.values() if d)),
                               note=f"{len(set(docs.values()))} source documents"))
    out.info.update({"expanded": len(expanded), "path": path, "pagerank": {n: round(score.get(n, 0), 4) for n in top},
                     "timings_ms": timings})
    return out
