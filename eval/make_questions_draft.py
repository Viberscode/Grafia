"""Draft benchmark questions from the indexed corpus into eval/questions.draft.jsonl for human review.

One cached LLM call sees every chunk with its id and returns 7 thematic, 6 factual, 6 multihop and
5 unanswerable questions with reference answers and gold_chunk_ids; ids are validated against the index.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backend.config import LLM_MODEL_STRONG
from backend.engine import load_tables
from backend.llm import chat_json

logger = logging.getLogger("eval.make_questions_draft")

DRAFT = Path(__file__).resolve().parent / "questions.draft.jsonl"
MIX = {"thematic": 7, "factual": 6, "multihop": 6, "unanswerable": 5}
PREFIX = {"thematic": "T", "factual": "F", "multihop": "M", "unanswerable": "U"}

SYSTEM = """You write a benchmark for a question-answering system over the document chunks below.
Write exactly 7 thematic, 6 factual, 6 multihop and 5 unanswerable questions.
- thematic: asks about themes, trends or the overall picture across several documents.
- factual: one specific fact stated in one chunk.
- multihop: needs facts from two or more chunks joined through a shared entity.
- unanswerable: sounds plausible for this corpus (same places, people or projects) but the chunks do not contain
  the answer; reference must be "ABSTAIN" and gold_chunk_ids empty.
Mark "mixed": true on a thematic question only if it also asks for a specific fact.
reference is a short correct answer using only the chunks. gold_chunk_ids lists every chunk id needed."""

SCHEMA = ('{"questions": [{"type": "factual", "question": "...", "reference": "...", '
          '"gold_chunk_ids": ["<chunk id>"], "mixed": false}]}')


def corpus_prompt(tables: dict) -> tuple[str, dict[str, str]]:
    """User prompt listing every chunk with its id and document, plus chunk id -> document stem."""
    docs = dict(zip(tables["documents"]["id"].astype(str), tables["documents"]["title"].astype(str)))
    chunk_doc: dict[str, str] = {}
    parts = []
    for _, row in tables["text_units"].iterrows():
        cid, doc = str(row["id"]), docs.get(str(row["document_id"]), "")
        chunk_doc[cid] = doc.rsplit(".", 1)[0]
        parts.append(f"[chunk {cid}] (document {doc})\n{str(row['text']).strip()}")
    return "\n\n".join(parts), chunk_doc


def clean(drafts: list[dict], chunk_doc: dict[str, str]) -> list[dict]:
    """Validate types and chunk ids, add gold_docs and stable ids like T1, F1, M1, U1."""
    counters = {t: 0 for t in MIX}
    out = []
    for d in drafts:
        qtype = str(d.get("type", "")).lower()
        question = str(d.get("question", "")).strip()
        if qtype not in MIX or not question:
            logger.warning("Dropping malformed draft: %s", d)
            continue
        ids = [str(c) for c in d.get("gold_chunk_ids") or []]
        unknown = [c for c in ids if c not in chunk_doc]
        if unknown:
            logger.warning("%s: dropping unknown chunk ids %s", question[:60], unknown)
        gold = [] if qtype == "unanswerable" else [c for c in ids if c in chunk_doc]
        if qtype != "unanswerable" and not gold:
            logger.warning("Dropping %s question without valid gold chunks: %s", qtype, question[:80])
            continue
        counters[qtype] += 1
        row = {"id": f"{PREFIX[qtype]}{counters[qtype]}", "type": qtype, "question": question,
               "reference": "ABSTAIN" if qtype == "unanswerable" else str(d.get("reference", "")).strip(),
               "gold_chunk_ids": gold, "gold_docs": sorted({chunk_doc[c] for c in gold})}
        if d.get("mixed") and qtype == "thematic":
            row["mixed"] = True
        out.append(row)
    for qtype, want in MIX.items():
        if counters[qtype] != want:
            logger.warning("Got %d %s questions, wanted %d; edit the draft by hand", counters[qtype], qtype, want)
    return out


def main() -> None:
    """Draft the questions and write the JSONL file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DRAFT, help="output JSONL path")
    args = ap.parse_args()
    tables = load_tables()
    if not {"text_units", "documents"} <= set(tables):
        raise SystemExit("No index found: build the knowledge graph first.")
    prompt, chunk_doc = corpus_prompt(tables)
    res = chat_json(LLM_MODEL_STRONG, SYSTEM, prompt, SCHEMA)
    rows = clean(res.get("questions", []), chunk_doc)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    logger.info("Wrote %d questions to %s; review it, then rename to questions.jsonl", len(rows), args.out)


if __name__ == "__main__":
    main()
