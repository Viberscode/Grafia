"""ChromaDB vector store with collections for chunks, community reports and entities.

Embeddings come from Chroma's default local embedding function (all-MiniLM-L6-v2 via ONNX), so
building and querying the vector store makes no Gemini calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

COLLECTIONS = ("chunks", "reports", "entities")
BATCH = 256


@dataclass
class Hit:
    """One search result; score is cosine similarity (1 = identical)."""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    """Operations the retrievers and the loader need from a vector database."""

    def upsert(self, collection: str, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """Insert or replace documents."""

    def search(self, collection: str, text: str, k: int = 5) -> list[Hit]:
        """Top-k most similar documents."""

    def counts(self) -> dict[str, int]:
        """Document count per collection."""

    def clear(self) -> None:
        """Drop all collections."""


def _clean(meta: dict) -> dict:
    """Chroma metadata accepts only str, int, float and bool values."""
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is not None:
            out[k] = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    return out


class ChromaStore:
    """Persistent ChromaDB client."""

    name = "chroma"

    def __init__(self, path: Path) -> None:
        """Open (or create) the database directory."""
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))

    def _collection(self, name: str) -> Any:
        """Get or create a cosine-space collection with the default embedding function."""
        if name not in COLLECTIONS:
            raise ValueError(f"Unknown collection {name!r}; expected one of {COLLECTIONS}")
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def upsert(self, collection: str, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """Insert or replace documents in batches."""
        col = self._collection(collection)
        for i in range(0, len(ids), BATCH):
            col.upsert(ids=ids[i:i + BATCH], documents=texts[i:i + BATCH],
                       metadatas=[_clean(m) for m in metadatas[i:i + BATCH]])

    def search(self, collection: str, text: str, k: int = 5) -> list[Hit]:
        """Top-k most similar documents with cosine similarity scores."""
        col = self._collection(collection)
        n = col.count()
        if not n or not text.strip():
            return []
        res = col.query(query_texts=[text], n_results=min(k, n))
        return [Hit(id=i, text=d or "", score=round(1.0 - float(dist), 4), metadata=m or {})
                for i, d, dist, m in zip(res["ids"][0], res["documents"][0], res["distances"][0],
                                         res["metadatas"][0])]

    def counts(self) -> dict[str, int]:
        """Document count per collection."""
        return {name: self._collection(name).count() for name in COLLECTIONS}

    def clear(self) -> None:
        """Drop all collections."""
        existing = {getattr(c, "name", c) for c in self.client.list_collections()}
        for name in COLLECTIONS:
            if name in existing:
                self.client.delete_collection(name)


_VECTORS: VectorStore | None = None


def get_vectors() -> VectorStore:
    """Process-wide Chroma store at CHROMA_PATH."""
    global _VECTORS
    if _VECTORS is None:
        from backend.config import CHROMA_PATH

        _VECTORS = ChromaStore(CHROMA_PATH)
    return _VECTORS


def search(collection: str, text: str, k: int = 5) -> list[Hit]:
    """Search one collection of the process-wide store."""
    return get_vectors().search(collection, text, k)


def reset_vectors() -> None:
    """Forget the cached client (after a reload in another process)."""
    global _VECTORS
    _VECTORS = None
