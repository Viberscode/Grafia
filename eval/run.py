"""Run every benchmark question through basic, local-only, global-only and routed systems."""
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
SYSTEMS = {"basic": "basic", "local_only": "local", "global_only": "global", "routed": "auto"}


def load_questions(limit: int | None = None) -> list[dict]:
    """Read questions.jsonl, optionally truncated."""
    rows = [json.loads(line) for line in (EVAL_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    """Execute all systems and save raw results to eval/results/raw.jsonl."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated subset")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "raw.jsonl"
    systems = [s for s in args.systems.split(",") if s in SYSTEMS]
    questions = load_questions(args.limit)
    with out.open("w", encoding="utf-8") as fh:
        for q in questions:
            for sys_name in systems:
                before = dict(llm.COUNTERS)
                t0 = time.perf_counter()
                resp = run(q["question"], mode=SYSTEMS[sys_name])
                latency = int((time.perf_counter() - t0) * 1000)
                calls = llm.COUNTERS["calls"] - before["calls"]
                row = {"qid": q["id"], "type": q["type"], "system": sys_name, "latency_ms": latency,
                       "llm_calls": calls, "cache_hits": llm.COUNTERS["cache_hits"] - before["cache_hits"],
                       "response": resp.model_dump()}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                logger.info("%s %-11s route=%-7s abstained=%s claims=%d %dms", q["id"], sys_name, resp.route,
                            resp.abstained, len(resp.claims), latency)
    logger.info("Saved %s (%d questions x %d systems)", out, len(questions), len(systems))


if __name__ == "__main__":
    main()
