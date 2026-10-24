"""ThemeGraph retrievers over the graph store and ChromaDB: local (facts), global (themes), hybrid."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.contracts import TraceStep

HONORIFICS = ("dr. ", "dr ", "mr. ", "ms. ", "mrs. ", "prof. ", "the ")
GENERIC = {"aurora", "city", "state", "national", "new", "north", "south", "east", "west", "green", "clean",
           "energy", "power", "grid", "project", "program", "plan", "fund", "bay", "coastal", "public"}


@dataclass
class Retrieved:
    """Contexts plus the graph evidence and ordered trace that produced them."""

    contexts: list[dict] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    communities: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    signal: int = 0
    info: dict = field(default_factory=dict)


def norm(text: str) -> str:
    """Lowercase, punctuation to spaces, single-spaced, padded for whole-word matching."""
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip() + " "


def title_aliases(titles: list[str]) -> dict[str, list[str]]:
    """Map title -> normalised match strings: full title, title without honorifics, and the first word of
    3+ word titles when no other title starts with it (so 'Raman' does not match RAMAN REPORT)."""
    firsts: dict[str, int] = {}
    for t in titles:
        words = norm(t).split()
        if words:
            firsts[words[0]] = firsts.get(words[0], 0) + 1
    out: dict[str, list[str]] = {}
    for t in titles:
        low = t.lower()
        variants = [norm(t)]
        for h in HONORIFICS:
            if low.startswith(h):
                variants.append(norm(t[len(h):]))
        words = norm(t).split()
        if len(words) > 2 and len(words[0]) >= 5 and words[0] not in GENERIC and firsts.get(words[0]) == 1:
            variants.append(f" {words[0]} ")
        out[t] = list(dict.fromkeys(v for v in variants if v.strip()))
    return out


def mentioned(titles: list[str], text: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    """Titles named in text (full title, honorific-free form or unique first word)."""
    aliases = aliases or title_aliases(titles)
    hay = norm(text)
    return [t for t in titles if any(a in hay for a in aliases.get(t, [norm(t)]))]


def retrieve(route: str, question: str, store: object = None, vectors: object = None) -> Retrieved:
    """Run the retriever for a route (global, local or hybrid) with the process-wide stores by default."""
    from backend.retrieval import global_, hybrid, local
    from backend.stores.graph_store import get_store
    from backend.stores.vector_store import get_vectors

    store = store or get_store()
    vectors = vectors or get_vectors()
    fn = {"global": global_.retrieve_global, "local": local.retrieve_local, "hybrid": hybrid.retrieve_hybrid}[route]
    return fn(question, store, vectors)
