"""Generate an answer with claims that cite verbatim quotes from numbered context chunks."""
from __future__ import annotations

import logging

from backend.config import LLM_MODEL_STRONG
from backend.llm import chat_json

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 4000

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context chunks provided.
Rules:
1. Use only facts stated in the context. Do not use outside knowledge.
2. If the context does not contain the answer, return {"answer": "not found", "claims": []}.
3. Break the answer into short atomic claims. Every claim must have one or more evidence items.
4. Each evidence item has the chunk_id exactly as shown in square brackets and a quote that is
   copied VERBATIM (character for character) from that chunk, at most 25 words. Do not paraphrase quotes.
5. "answer" is a concise 2-5 sentence answer that only restates the claims."""

SCHEMA_HINT = '{"answer": "str", "claims": [{"text": "str", "evidence": [{"chunk_id": "str", "quote": "str"}]}]}'


def format_chunks(context_chunks: list[dict]) -> str:
    """Render chunks as a numbered block the model can cite."""
    return "\n\n".join(
        f"[{c['chunk_id']}]\n{str(c.get('text', ''))[:MAX_CHUNK_CHARS]}" for c in context_chunks
    )


def generate_claims(question: str, context_chunks: list[dict]) -> dict:
    """Return {answer, claims} grounded in context_chunks using the strong model."""
    if not context_chunks:
        return {"answer": "not found", "claims": []}
    user = f"Context chunks:\n\n{format_chunks(context_chunks)}\n\nQuestion: {question}"
    res = chat_json(LLM_MODEL_STRONG, SYSTEM_PROMPT, user, SCHEMA_HINT)
    claims = [c for c in res.get("claims", []) if isinstance(c, dict) and c.get("text")]
    return {"answer": str(res.get("answer", "not found")), "claims": claims}
