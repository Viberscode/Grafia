"""Metrics for the benchmark: evidence recall, quote validity, routing, abstention, latency, judge."""
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


def _stem(title: str) -> str:
    """Document title without extension."""
    return title.rsplit(".", 1)[0]


@lru_cache(maxsize=1)
def _report_docs() -> dict[str, set[str]]:
    """Community id -> set of document stems covered by its text units."""
    t = load_tables()
    if not {"communities", "text_units", "documents"} <= set(t):
        return {}
    doc_title = dict(zip(t["documents"]["id"].astype(str), t["documents"]["title"].astype(str)))
    tu_doc = dict(zip(t["text_units"]["id"].astype(str), t["text_units"]["document_id"].astype(str)))
    out: dict[str, set[str]] = {}
    for _, row in t["communities"].iterrows():
        out[str(row["community"])] = {_stem(doc_title.get(tu_doc.get(str(x), ""), "")) for x in row["text_unit_ids"]}
    return out


def cited_docs(response: dict) -> set[str]:
    """Documents behind the verified evidence of a response."""
    chunk_doc = {s["chunk_id"]: _stem(s["doc"]) for s in response.get("sources", [])}
    docs: set[str] = set()
    for claim in response.get("claims", []):
        for ev in claim["evidence"]:
            cid = ev["chunk_id"]
            if cid.startswith("report_"):
                docs |= _report_docs().get(cid.removeprefix("report_"), set())
            elif cid in chunk_doc:
                docs.add(chunk_doc[cid])
    return docs - {""}


def evidence_recall(row: dict, question: dict) -> float | None:
    """Fraction of gold documents covered by verified evidence (None for unanswerable)."""
    gold = set(question.get("gold_docs", []))
    if not gold:
        return None
    return len(cited_docs(row["response"]) & gold) / len(gold)


def quote_validity(row: dict) -> float | None:
    """Share of generated quotes that passed the verbatim check."""
    s = row["response"].get("stats", {})
    return s["quotes_valid"] / s["quotes_total"] if s.get("quotes_total") else None


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
        grades = {}
    return {sys_name: int(grades.get(lab, 0)) for lab, sys_name in labels.items()}


def _mean(values: list[float]) -> float | None:
    """Mean ignoring None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute(rows: list[dict], questions: dict[str, dict], use_judge: bool = True) -> dict:
    """Aggregate metrics per (system, question type) and overall per system."""
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
        routed = [r for r in grp if r["system"] == "routed"]
        table.append({
            "system": sys_name,
            "type": qtype,
            "n": len(grp),
            "correct": _mean([correctness.get((r["qid"], sys_name)) for r in grp]) if use_judge else None,
            "evidence_recall": _mean([evidence_recall(r, questions[r["qid"]]) for r in grp]),
            "quote_validity": _mean([quote_validity(r) for r in grp]),
            "router_acc": _mean([float(r["response"]["route"] == EXPECTED_ROUTE[r["type"]]) for r in routed])
            if routed else None,
            "abstain_precision": (sum(r["type"] == "unanswerable" for r in abstained) / len(abstained))
            if abstained else None,
            "abstain_recall": (sum(r["response"]["abstained"] for r in unans) / len(unans)) if unans else None,
            "latency_ms": _mean([r["latency_ms"] for r in grp]),
            "llm_calls": _mean([r["llm_calls"] for r in grp]),
        })
    return {"rows": table, "judged": use_judge}


def load_raw(path: Path) -> list[dict]:
    """Read raw.jsonl."""
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
