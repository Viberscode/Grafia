"""Offline tests for the benchmark harness: resume, metrics, extraction audit and README table."""
from __future__ import annotations

import pandas as pd
import pytest

from eval import extraction_audit, make_questions_draft, metrics, run, write_readme_table

INDEX = {
    "doc_chunks": {"01_plan": {"c1"}, "02_blackout": {"c2"}, "03_wind": {"c3"}},
    "community_chunks": {"5": {"c1", "c2"}},
    "chunk_text": {"c1": "The plan text.", "c2": "The blackout text.", "c3": "Wind farm text."},
}
QUESTIONS = {
    "T1": {"id": "T1", "type": "thematic", "question": "Themes?", "reference": "x", "gold_docs": ["01_plan", "02_blackout"]},
    "F1": {"id": "F1", "type": "factual", "question": "Fact?", "reference": "y", "gold_chunk_ids": ["c3"]},
    "U1": {"id": "U1", "type": "unanswerable", "question": "Pizza?", "reference": "ABSTAIN", "gold_docs": []},
}


def _row(qid: str, qtype: str, system: str, route: str, evidence: list[str], abstained: bool = False,
         error: str | None = None) -> dict:
    """A raw.jsonl row with one claim per evidence chunk id."""
    claims = [{"text": "c", "evidence": [{"chunk_id": e, "quote": "q"}]} for e in evidence]
    row = {"qid": qid, "type": qtype, "system": system, "latency_ms": 100, "llm_calls": 2,
           "response": {"route": route, "abstained": abstained, "answer": "a", "claims": claims, "sources": [],
                        "stats": {"quotes_total": 2, "quotes_valid": 1}}}
    if error:
        row["error"] = error
    return row


def test_resume_skips_only_successful_pairs(tmp_path):
    raw = tmp_path / "raw.jsonl"
    run.append_row(_row("T1", "thematic", "graphrag_basic", "basic", ["c1"]), raw)
    run.append_row(_row("F1", "factual", "graphrag_basic", "abstain", [], True, error="RateLimitError: 429"), raw)
    raw.write_text(raw.read_text(encoding="utf-8") + '{"torn', encoding="utf-8")
    rows = run.read_rows(raw)
    assert run.finished_pairs(rows) == {("T1", "graphrag_basic")}
    assert run.is_quota_error(rows[1])


def test_metrics_use_gold_chunks_and_mixed_hybrid():
    rows = [
        _row("T1", "thematic", "themegraph_routed", "global", ["report_5"]),
        _row("F1", "factual", "themegraph_routed", "local", ["c3"]),
        _row("U1", "unanswerable", "themegraph_routed", "abstain", [], abstained=True),
        _row("F1", "factual", "graphrag_basic", "basic", ["c1"]),
        _row("F1", "factual", "graphrag_basic", "basic", [], error="boom"),
    ]
    out = {(r["system"], r["type"]): r for r in metrics.compute(rows, QUESTIONS, use_judge=False, index=INDEX)["rows"]}
    routed = out[("themegraph_routed", "ALL")]
    assert routed["evidence_recall"] == 1.0
    assert routed["router_acc"] == 1.0
    assert routed["abstain_precision"] == 1.0 and routed["abstain_recall"] == 1.0
    assert routed["verified_claims"] == pytest.approx(2 / 3)
    assert out[("graphrag_basic", "factual")]["evidence_recall"] == 0.0
    assert out[("graphrag_basic", "factual")]["n"] == 1
    mixed = {**QUESTIONS["T1"], "mixed": True}
    assert metrics.route_correct(_row("T1", "thematic", "themegraph_routed", "hybrid", []), mixed) == 1.0
    assert metrics.route_correct(_row("T1", "thematic", "themegraph_routed", "hybrid", []), QUESTIONS["T1"]) == 0.0


def test_baseline_sources_map_to_chunks_by_text():
    resp = {"claims": [{"text": "c", "evidence": [{"chunk_id": "7", "quote": "q"}]}],
            "sources": [{"chunk_id": "7", "doc": "03_wind.txt", "text": "Wind farm text."}]}
    assert metrics.cited_chunks(resp, INDEX) == {"c3"}


def test_draft_cleaning_validates_chunk_ids():
    drafts = [
        {"type": "factual", "question": "Who?", "reference": "A", "gold_chunk_ids": ["c1", "nope"]},
        {"type": "factual", "question": "Bad ids", "reference": "B", "gold_chunk_ids": ["nope"]},
        {"type": "unanswerable", "question": "Price?", "reference": "x", "gold_chunk_ids": ["c1"]},
        {"type": "thematic", "question": "Themes and who?", "reference": "C", "gold_chunk_ids": ["c2"], "mixed": True},
    ]
    rows = make_questions_draft.clean(drafts, {"c1": "01_plan", "c2": "02_blackout"})
    assert [r["id"] for r in rows] == ["F1", "U1", "T1"]
    assert rows[0]["gold_chunk_ids"] == ["c1"] and rows[0]["gold_docs"] == ["01_plan"]
    assert rows[1]["reference"] == "ABSTAIN" and rows[1]["gold_chunk_ids"] == []
    assert rows[2]["mixed"] is True


def test_extraction_audit_sample_and_score(tmp_path):
    tables = {
        "text_units": pd.DataFrame({"id": ["c1"], "text": ["Meridian Capital underwrote the bond."]}),
        "relationships": pd.DataFrame({"human_readable_id": [0, 1], "source": ["MERIDIAN CAPITAL", "A"],
                                       "target": ["GREEN BOND", "B"], "description": ["underwrote", "x"],
                                       "text_unit_ids": [["c1"], ["c1"]]}),
        "entities": pd.DataFrame({"human_readable_id": [0, 1], "title": ["MERIDIAN CAPITAL", "MERIDIAN"],
                                  "type": ["ORGANIZATION", "ORGANIZATION"], "description": ["bank", "bank"],
                                  "text_unit_ids": [["c1"], ["c1"]]}),
    }
    rel = extraction_audit.sample_relationships(tables, n=30)
    ent = extraction_audit.sample_entities(tables, n=30)
    assert len(rel) == 2 and rel["source_chunk"].str.contains("Meridian").all()
    assert "MERIDIAN" in ent.set_index("title").loc["MERIDIAN CAPITAL", "possible_duplicates"]
    rel["correct"] = ["y", "n"]
    ent["type_correct"] = ["yes", ""]
    ent["duplicate"] = ["y", "n"]
    rel.to_csv(tmp_path / "r.csv", index=False)
    ent.to_csv(tmp_path / "e.csv", index=False)
    s = extraction_audit.score(tmp_path / "r.csv", tmp_path / "e.csv")
    assert s == {"relationship_precision": 0.5, "relationships_judged": 2, "entity_type_precision": 1.0,
                 "entities_judged": 1, "duplicate_rate": 0.5}


def test_readme_block_is_replaced():
    readme = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nrest"
    out = write_readme_table.replace_block(readme, "| a |\n|---|")
    assert "old" not in out and "| a |" in out and out.endswith("rest")
    with pytest.raises(ValueError):
        write_readme_table.replace_block("no markers", "x")
