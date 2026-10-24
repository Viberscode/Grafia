"""Compute metrics from eval/results/raw.jsonl, save summary.csv and summary.md, and print the tables."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from eval.metrics import compute, load_raw
from eval.run import QUESTIONS, RAW, RESULTS, SYSTEMS, load_questions

logger = logging.getLogger("eval.report")

COLUMNS = ["system", "type", "n", "correct", "evidence_recall", "quote_validity", "verified_claims", "router_acc",
           "abstain_precision", "abstain_recall", "latency_ms", "llm_calls"]
TYPE_ORDER = ["ALL", "thematic", "factual", "multihop", "unanswerable"]


def fmt(v: object) -> str:
    """Format a metric cell."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.0f}" if v > 10 else f"{v:.2f}"
    return str(v)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render rows as a Markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def grouped_markdown(df: pd.DataFrame) -> str:
    """One Markdown table per question type, overall first."""
    cols = [c for c in COLUMNS if c != "type"]
    parts = []
    for qtype in TYPE_ORDER + sorted(set(df["type"]) - set(TYPE_ORDER)):
        sub = df[df["type"] == qtype].sort_values("system")
        if not sub.empty:
            title = "All questions" if qtype == "ALL" else f"{qtype.capitalize()} questions"
            parts.append(f"#### {title}\n\n{markdown_table(sub, cols)}")
    return "\n\n".join(parts)


def main() -> None:
    """Compute metrics from raw.jsonl and print them."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge correctness metric")
    args = ap.parse_args()
    if not RAW.exists():
        logger.error("Run `python -m eval.run` first.")
        sys.exit(1)
    questions = {q["id"]: q for q in load_questions(path=QUESTIONS)}
    rows = [r for r in load_raw(RAW) if r.get("system") in SYSTEMS]
    result = compute(rows, questions, use_judge=not args.no_judge)
    if not result["rows"]:
        logger.error("raw.jsonl has no successful rows for the current questions.")
        sys.exit(1)
    df = pd.DataFrame(result["rows"])[COLUMNS]
    df.to_csv(RESULTS / "summary.csv", index=False)
    text = grouped_markdown(df)
    (RESULTS / "summary.md").write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    logger.info("Saved %s and summary.md", RESULTS / "summary.csv")


if __name__ == "__main__":
    main()
