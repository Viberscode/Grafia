"""End-to-end smoke test: thematic, multi-hop and unanswerable questions through the pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.contracts import Response  # noqa: E402
from backend.llm import COUNTERS  # noqa: E402
from backend.pipeline import run  # noqa: E402

logger = logging.getLogger("smoke_test")

QUESTIONS = [
    ("thematic", "What are the main themes of Aurora Bay's clean energy transition?"),
    ("multihop", "How is the author of the report on the 2019 Harbor Blackout connected to the company "
                 "that built the Kestrel control software?"),
    ("unanswerable", "What is the recipe for a classic margherita pizza?"),
]


def main() -> int:
    """Run the three questions and assert the expected behaviour."""
    failures = 0
    for kind, q in QUESTIONS:
        resp = Response.model_validate(run(q).model_dump())
        logger.info("[%s] route=%s abstained=%s claims=%d nodes=%d timings=%s\n  %s%s", kind, resp.route,
                    resp.abstained, len(resp.claims), len(resp.highlights.nodes), resp.timings_ms,
                    resp.answer[:300], f"\n  reason: {resp.abstain_reason}" if resp.abstained else "")
        if kind == "unanswerable" and not resp.abstained:
            logger.error("Unanswerable question was answered")
            failures += 1
        if kind != "unanswerable" and resp.abstained:
            logger.error("Answerable question abstained: %s", resp.abstain_reason)
            failures += 1
    logger.info("LLM counters: %s", COUNTERS)
    logger.info("SMOKE %s", "PASSED" if not failures else f"FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
