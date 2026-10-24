"""Benchmark metrics: evidence recall vs gold chunks, quote validity, verified claims, routing, abstention,
latency, LLM calls, and a secondary LLM-judge correctness score."""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from backend.config import LLM_MODEL_STRONG
from backend.engine import load_tables
from backend.llm import chat_json

logger = logging.getLogger(__name__)

EXPECTED_ROUTE = {"thematic": "global", "factual": "local", "multihop": "local", "unanswerable": "abstain"}

JUDGE_SYSTEM = """You grade answers against a reference answer. For each labelled answer decide whether it is
correct: it states the key facts of the reference without contradicting it (extra correct detail is fine).
An answer that declines to answer is incorrect unless the reference is ABSTAIN.
Return JSON: {"grades": {"<label>": 1 or 0}}."""

CorpusIndex = dict[str, dict]


def _stem(title: str) -> str:
    """Document title without extension."""
    return title.rsplit(".", 1)[0]


@lru_cache(maxsize=1)
def corpus_index() -> CorpusIndex:
    """Chunk ids per document stem and per community, plus each chunk's text, from the parquet tables."""
    t = load_tables()
    if not {"communities", "text_units", "documents"} <= set(t):
        return {"doc_chunks": {}, "community_chunks": {}, "chunk_text": {}}
    doc_title = dict(zip(t["documents"]["id"].astype(str), t["documents"]["title"].astype(str)))
    doc_chunks: dict[str, set[str]] = defaultdict(set)
    chunk_text: dict[str, str] = {}
    for _, row in t["text_units"].iterrows():
        doc_chunks[_stem(doc_title.get(str(row["document_id"]), ""))].add(str(row["id"]))
        chunk_text[str(row["id"])] = str(row["text"])
    community_chunks = {str(r["community"]): {str(x) for x in r["text_unit_ids"]}
                        for _, r in t["communities"].iterrows()}
    return {"doc_chunks": dict(doc_chunks), "community_chunks": community_chunks, "chunk_text": chunk_text}


def gold_chunks(question: dict, index: CorpusIndex) -> set[str]:
    """Gold chunk ids: explicit gold_chunk_ids, else every chunk of the gold documents."""
    explicit = {str(c) for c in question.get("gold_chunk_ids") or []}
    if explicit:
        return explicit
    return {c for d in question.get("gold_docs", []) for c in index["doc_chunks"].get(_stem(d), set())}


def _match_chunk(text: str, index: CorpusIndex) -> set[str]:
    """Text-unit ids whose text contains (or is contained in) a baseline source excerpt."""
    probe = text.strip()[:200]
    if not probe:
        return set()
    return {cid for cid, t in index["chunk_text"].items() if probe in t or t.strip() in text}


def cited_chunks(response: dict, index: CorpusIndex) -> set[str]:
    """Text-unit ids behind the verified evidence (community reports expand to their text units)."""
    known = index["chunk_text"]
    sources = {s["chunk_id"]: s for s in response.get("sources", [])}
    out: set[str] = set()
    for claim in response.get("claims", []):
        for ev in claim["evidence"]:
            cid = str(ev["chunk_id"])
            if cid in known:
                out.add(cid)
            elif cid.startswith("report_"):
                out |= index["community_chunks"].get(cid.removeprefix("report_"), set())
            elif cid in sources:
                src = sources[cid]
                out |= _match_chunk(src.get("text", ""), index) or index["doc_chunks"].get(_stem(src["doc"]), set())
    return out


def evidence_recall(row: dict, question: dict, index: CorpusIndex) -> float | None:
    """Fraction of gold chunks covered by verified evidence (None for unanswerable)."""
    gold = gold_chunks(question, index)
    if not gold:
        return None
    return len(cited_chunks(row["response"], index) & gold) / len(gold)


def quote_validity(row: dict) -> float | None:
    """Share of generated quotes that passed the verbatim check."""
    s = row["response"].get("stats", {})
    return s["quotes_valid"] / s["quotes_total"] if s.get("quotes_total") else None


def route_correct(row: dict, question: dict) -> float:
    """1.0 when the routed system picked the expected route; hybrid also counts for mixed questions."""
    route = row["response"]["route"]
    expected = question.get("expected_route") or EXPECTED_ROUTE[question["type"]]
    return float(route == expected or (route == "hybrid" and bool(question.get("mixed"))))


def latest_rows(rows: list[dict]) -> list[dict]:
    """Last successful row per (qid, system); rows with an error are ignored."""
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not r.get("error"):
            latest[(r["qid"], r["system"])] = r
    return list(latest.values())


def judge(question: dict, rows: list[dict], seed: int = 0) -> dict[str, int]:
    """LLM-judge correctness for all systems on one question, presented in random order."""
    if question["reference"] == "ABSTAIN":
        return {r["system"]: int(r["response"]["abstained"]) for r in rows}
    shuffled = rows[:]
    random.Random(f"{seed}-{question['id']}").shuffle(shuffled)
    labels = {f"A{i}": r["system"] for i, r in enumerate(shuffled)}
    body = "\n\n".join(
        f"[{lab}] " + ("(declined to answer)" if r["response"]["abstained"] else r["response"]["answer"])
        for lab, r in zip(labels, shuffled))
    user = f"Question: {question['question']}\nReference: {question['reference']}\n\nAnswers:\n{body}"
    try:
        grades = chat_json(LLM_MODEL_STRONG, JUDGE_SYSTEM, user, '{"grades": {"A0": 1}}').get("grades", {})
    except Exception as err:
        logger.warning("Judge failed for %s: %s", question["id"], err)
        return {}
    return {sys_name: int(grades.get(lab, 0)) for lab, sys_name in labels.items()}


def _mean(values: list[float | None]) -> float | None:
    """Mean ignoring None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute(rows: list[dict], questions: dict[str, dict], use_judge: bool = True,
            index: CorpusIndex | None = None) -> dict:
    """Aggregate metrics per (system, question type) and overall per system."""
    index = index if index is not None else corpus_index()
    rows = [r for r in latest_rows(rows) if r["qid"] in questions]
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    correctness: dict[tuple[str, str], int] = {}
    if use_judge:
        for qid, qrows in by_q.items():
            for sys_name, g in judge(questions[qid], qrows).items():
                correctness[(qid, sys_name)] = g

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["type"])].append(r)
        groups[(r["system"], "ALL")].append(r)

    table = []
    for (sys_name, qtype), grp in sorted(groups.items()):
        unans = [r for r in grp if r["type"] == "unanswerable"]
        abstained = [r for r in grp if r["response"]["abstained"]]
        routed = [r for r in grp if r["system"] == "themegraph_routed"]
        table.append({
            "system": sys_name,
            "type": qtype,
            "n": len(grp),
            "correct": _mean([correctness.get((r["qid"], sys_name)) for r in grp]) if use_judge else None,
            "evidence_recall": _mean([evidence_recall(r, questions[r["qid"]], index) for r in grp]),
            "quote_validity": _mean([quote_validity(r) for r in grp]),
            "verified_claims": _mean([float(len(r["response"]["claims"])) for r in grp]),
            "router_acc": _mean([route_correct(r, questions[r["qid"]]) for r in routed]) if routed else None,
            "abstain_precision": (sum(r["type"] == "unanswerable" for r in abstained) / len(abstained))
            if abstained else None,
            "abstain_recall": (sum(r["response"]["abstained"] for r in unans) / len(unans)) if unans else None,
            "latency_ms": _mean([r["latency_ms"] for r in grp]),
            "llm_calls": _mean([r["llm_calls"] for r in grp]),
        })
    return {"rows": table, "judged": use_judge}


def load_raw(path: Path) -> list[dict]:
    """Read raw.jsonl, skipping unreadable lines."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable line in %s", path)
    return out
