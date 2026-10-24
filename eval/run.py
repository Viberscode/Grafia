"""Run benchmark questions through the GraphRAG baselines and routed ThemeGraph, one pair at a time.

Every result is appended to eval/results/raw.jsonl as soon as it finishes, so a quota error loses nothing;
`--resume` skips (question, system) pairs that already have a successful row.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from backend import llm
from backend.pipeline import run

logger = logging.getLogger("eval.run")

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
RAW = RESULTS / "raw.jsonl"
QUESTIONS = EVAL_DIR / "questions.jsonl"
SYSTEMS = {"graphrag_basic": "basic", "graphrag_local": "graphrag_local", "graphrag_global": "graphrag_global",
           "themegraph_routed": "auto"}
ERROR_ANSWER = "Something went wrong while answering."
QUOTA_MARKERS = ("RateLimit", "429", "quota", "RESOURCE_EXHAUSTED")


def load_questions(limit: int | None = None, path: Path = QUESTIONS) -> list[dict]:
    """Read a questions JSONL file, optionally truncated to the first N."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def read_rows(path: Path = RAW) -> list[dict]:
    """All rows of raw.jsonl, skipping a torn last line."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable line in %s", path)
    return rows


def finished_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    """(qid, system) pairs that have at least one row without an error."""
    return {(r["qid"], r["system"]) for r in rows if not r.get("error")}


def run_pair(question: dict, system: str) -> dict:
    """Answer one question with one system and return the raw result row."""
    before = dict(llm.COUNTERS)
    t0 = time.perf_counter()
    resp = run(question["question"], mode=SYSTEMS[system])
    row = {"qid": question["id"], "type": question["type"], "system": system,
           "latency_ms": int((time.perf_counter() - t0) * 1000),
           "llm_calls": llm.COUNTERS["calls"] - before["calls"],
           "cache_hits": llm.COUNTERS["cache_hits"] - before["cache_hits"],
           "response": resp.model_dump()}
    if resp.answer == ERROR_ANSWER:
        row["error"] = resp.abstain_reason or "pipeline error"
    return row


def append_row(row: dict, path: Path = RAW) -> None:
    """Append one row and flush it to disk."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def is_quota_error(row: dict) -> bool:
    """True when the row failed because of rate limits or quota."""
    return any(m.lower() in str(row.get("error", "")).lower() for m in QUOTA_MARKERS)


def main() -> None:
    """Run the selected systems over the questions with concurrency 1."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated subset of " + ",".join(SYSTEMS))
    ap.add_argument("--resume", action="store_true", help="skip (question, system) pairs already in raw.jsonl")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between pairs (free-tier quota)")
    ap.add_argument("--questions", type=Path, default=QUESTIONS, help="questions JSONL file")
    args = ap.parse_args()

    unknown = [s for s in args.systems.split(",") if s and s not in SYSTEMS]
    if unknown:
        ap.error(f"unknown systems {unknown}; choose from {list(SYSTEMS)}")
    systems = [s for s in args.systems.split(",") if s in SYSTEMS]
    questions = load_questions(args.limit, args.questions)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.resume and RAW.exists():
        RAW.replace(RAW.with_suffix(".jsonl.bak"))
        logger.info("Moved previous results to %s (use --resume to keep adding to them)", RAW.with_suffix(".jsonl.bak"))
    done = finished_pairs(read_rows()) if args.resume else set()

    ran = skipped = failed = 0
    for q in questions:
        for system in systems:
            if (q["id"], system) in done:
                skipped += 1
                logger.info("%s %-17s skipped (already in raw.jsonl)", q["id"], system)
                continue
            if ran and args.sleep:
                time.sleep(args.sleep)
            row = run_pair(q, system)
            append_row(row)
            ran += 1
            resp = row["response"]
            logger.info("%s %-17s route=%-15s abstained=%s claims=%d calls=%d %dms%s", q["id"], system, resp["route"],
                        resp["abstained"], len(resp["claims"]), row["llm_calls"], row["latency_ms"],
                        f" ERROR {row['error']}" if row.get("error") else "")
            if row.get("error"):
                failed += 1
                if is_quota_error(row):
                    logger.error("Quota or rate limit hit; stopping. Rerun later with --resume.")
                    logger.info("Ran %d, skipped %d, failed %d", ran, skipped, failed)
                    return
    logger.info("Ran %d, skipped %d, failed %d -> %s", ran, skipped, failed, RAW)


if __name__ == "__main__":
    main()
