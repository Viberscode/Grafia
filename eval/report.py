"""Print a Markdown results table and save eval/results/summary.csv."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from eval.metrics import compute, load_raw
from eval.run import RESULTS, load_questions

logger = logging.getLogger("eval.report")

COLUMNS = ["system", "type", "n", "correct", "evidence_recall", "quote_validity", "router_acc",
           "abstain_precision", "abstain_recall", "latency_ms", "llm_calls"]


def fmt(v) -> str:
    """Format a metric cell."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.0f}" if v > 10 else f"{v:.2f}"
    return str(v)


def main() -> None:
    """Compute metrics from raw.jsonl and print them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge correctness metric")
    args = ap.parse_args()
    raw = RESULTS / "raw.jsonl"
    if not raw.exists():
        logger.error("Run `python -m eval.run` first.")
        sys.exit(1)
    questions = {q["id"]: q for q in load_questions()}
    result = compute(load_raw(raw), questions, use_judge=not args.no_judge)
    df = pd.DataFrame(result["rows"])[COLUMNS]
    df.to_csv(RESULTS / "summary.csv", index=False)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    order = {"ALL": 0, "thematic": 1, "factual": 2, "multihop": 3, "unanswerable": 4}
    df = df.assign(_o=df["type"].map(order)).sort_values(["_o", "system"]).drop(columns="_o")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |")
    table = "\n".join(lines)
    (RESULTS / "summary.md").write_text(table + "\n", encoding="utf-8")
    sys.stdout.write(table + "\n")
    logger.info("Saved %s and summary.md", RESULTS / "summary.csv")


if __name__ == "__main__":
    main()
