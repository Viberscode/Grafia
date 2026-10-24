"""Repeatable demo: preload the chat with three recorded answers so a demo needs no LLM quota.

Record once (answers come from cache.db if asked before): `python -m ui.demo_seed --record`.
Use: `streamlit run app.py -- --demo-seed` (or set DEMO_SEED=1).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import streamlit as st

from backend.config import PROJECT_ROOT
from backend.contracts import Response
from ui import state

logger = logging.getLogger("ui.demo_seed")

SEED_FILE = Path(PROJECT_ROOT) / "docs" / "demo_seed.json"
SEEDED_KEY = "tg_demo_seeded"
QUESTIONS = [
    ("What are the main themes of Aurora Bay's clean energy transition?", "auto"),
    ("How is Dr. Priya Raman connected to the company that built the Kestrel control software?", "auto"),
    ("What are the main themes around financing, and who issued the green bond?", "auto"),
]


def requested(argv: list[str] | None = None) -> bool:
    """True when the app was started with --demo-seed or DEMO_SEED=1."""
    return "--demo-seed" in (sys.argv if argv is None else argv) or os.getenv("DEMO_SEED") == "1"


def load_seed(path: Path = SEED_FILE) -> list[tuple[str, str, Response]]:
    """Recorded (question, mode, Response) triples."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(r["question"], r["mode"], Response.model_validate(r["response"])) for r in rows]


def apply(path: Path = SEED_FILE) -> None:
    """Preload chat history once per session when requested."""
    if not requested() or st.session_state.get(SEEDED_KEY):
        return
    st.session_state[SEEDED_KEY] = True
    if not path.exists():
        logger.warning("Demo seed requested but %s is missing; run `python -m ui.demo_seed --record`", path)
        return
    last = None
    for question, mode, resp in load_seed(path):
        state.add_message(state.Message(role="user", content=question))
        state.add_message(state.Message(role="assistant", response=resp, question=question, mode=mode))
        last = resp
    state.set_last_response(last)
    logger.info("Preloaded %d demo answers from %s", len(QUESTIONS), path)


def record(path: Path = SEED_FILE) -> None:
    """Answer the demo questions (cached LLM calls) and save the responses."""
    from backend.pipeline import run

    rows = []
    for question, mode in QUESTIONS:
        resp = run(question, mode=mode)
        logger.info("%s -> route=%s abstained=%s claims=%d", question, resp.route, resp.abstained, len(resp.claims))
        rows.append({"question": question, "mode": mode, "response": resp.model_dump(mode="json")})
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("Saved %s", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true", help="answer the demo questions and save docs/demo_seed.json")
    if ap.parse_args().record:
        record()
