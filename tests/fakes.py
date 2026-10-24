"""Offline fakes shared by the store and retrieval tests."""
import re

import pandas as pd

from backend.stores.vector_store import Hit


def words(text: str) -> set[str]:
    """Lowercase word set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class FakeVectors:
    """In-memory vector store scoring by the share of question words found in the document."""

    def __init__(self) -> None:
        """Start empty."""
        self.cols: dict[str, dict[str, tuple[str, dict]]] = {}

    def upsert(self, collection: str, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """Insert or replace."""
        self.cols.setdefault(collection, {}).update({i: (t, m) for i, t, m in zip(ids, texts, metadatas)})

    def search(self, collection: str, text: str, k: int = 5) -> list[Hit]:
        """Rank by shared words."""
        q = words(text)
        hits = [Hit(i, t, len(q & words(t)) / max(len(q), 1), m) for i, (t, m) in self.cols.get(collection, {}).items()]
        return sorted(hits, key=lambda h: -h.score)[:k]

    def counts(self) -> dict[str, int]:
        """Size per collection."""
        return {c: len(v) for c, v in self.cols.items()}

    def clear(self) -> None:
        """Drop everything."""
        self.cols = {}


def graphrag_tables() -> dict[str, pd.DataFrame]:
    """GraphRAG-shaped tables: 2 docs, 3 chunks, chain ALPHA-BETA-GAMMA-DELTA, 2 communities with reports."""
    return {
        "documents": pd.DataFrame({"id": ["d1", "d2"], "title": ["one.txt", "two.txt"]}),
        "text_units": pd.DataFrame({"id": ["t1", "t2", "t3"], "document_id": ["d1", "d1", "d2"],
                                    "text": ["Alpha works with Beta.", "Beta funds Gamma.", "Gamma hires Delta."]}),
        "entities": pd.DataFrame({"id": ["u1", "u2", "u3", "u4"], "title": ["ALPHA", "BETA", "GAMMA", "DELTA"],
                                  "type": ["org"] * 4, "description": ["a", "b", "c", "d"], "degree": [1, 2, 2, 1],
                                  "text_unit_ids": [["t1"], ["t1", "t2"], ["t2", "t3"], ["t3"]]}),
        "relationships": pd.DataFrame({"id": ["r1", "r2", "r3"], "source": ["ALPHA", "BETA", "GAMMA"],
                                       "target": ["BETA", "GAMMA", "DELTA"], "description": ["ab", "bc", "cd"],
                                       "weight": [1.0, 2.0, 1.0], "text_unit_ids": [["t1"], ["t2"], []]}),
        "communities": pd.DataFrame({"community": [0, 1], "level": [0, 0], "title": ["C0", "C1"],
                                     "entity_ids": [["u1", "u2"], ["u3", "u4"]]}),
        "community_reports": pd.DataFrame({"community": [0, 1], "level": [0, 0],
                                           "title": ["Alpha and Beta partnership", "Gamma and Delta hiring"],
                                           "summary": ["s0", "s1"],
                                           "full_content": ["Alpha and Beta partnership. Alpha works with Beta.",
                                                            "Gamma and Delta hiring. Gamma hires Delta."],
                                           "rank": [5.0, 4.0]}),
    }
