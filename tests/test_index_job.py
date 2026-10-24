"""Offline tests for upload validation, estimates, log parsing and the index lock."""
import pytest

from backend import index_job


def test_validate_uploads_limits_and_types():
    big = b"x" * (index_job.MAX_BYTES + 1)
    rep = index_job.validate_uploads([("a.txt", b"hello"), ("b.docx", b"x"), ("c.txt", big)])
    assert [n for n, _ in rep.accepted] == ["a.txt"]
    assert any("b.docx" in e for e in rep.errors) and any("10 MB" in e for e in rep.errors)
    too_many = index_job.validate_uploads([("x.txt", b"x")] * 3, existing=index_job.MAX_FILES - 1)
    assert too_many.errors and not too_many.accepted


def test_pdf_without_text_is_warned_and_skipped():
    rep = index_job.validate_uploads([("scan.pdf", b"%PDF-1.4 not really a pdf")])
    assert not rep.accepted and rep.warnings


def test_estimate_scales_with_corpus():
    small, large = index_job.estimate([2000]), index_job.estimate([2000] * 10)
    assert small["chunks"] >= 1 and large["chunks"] == 10 * small["chunks"]
    assert large["llm_calls"] > small["llm_calls"] > 0


def test_parse_stages_follows_workflow_log():
    lines = ["Workflow load_input_documents started", "Workflow load_input_documents finished",
             "Workflow create_base_text_units started", "Workflow create_base_text_units finished",
             "Workflow create_final_documents started", "Workflow create_final_documents finished",
             "Workflow extract_graph started"]
    assert index_job.parse_stages(lines) == (["chunking"], "entities")
    assert index_job.parse_stages(lines + ["Workflow extract_graph finished", "Workflow generate_text_embeddings started",
                                           "Workflow generate_text_embeddings finished"]) == (
        ["chunking", "entities", "communities", "reports", "embeddings"], None)
    assert index_job.parse_stages(["nothing yet"]) == ([], None)


def test_friendly_error_messages():
    assert "API key" in index_job.friendly_error(["ERROR ... API key not valid. Please pass a valid API key."])
    assert "rate limit" in index_job.friendly_error(["litellm.RateLimitError: 429 RESOURCE_EXHAUSTED"])
    assert index_job.friendly_error(["ERROR boom happened"]).startswith("Indexing failed")


def test_lock_blocks_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(index_job, "LOG_DIR", tmp_path)
    monkeypatch.setattr(index_job, "LOCK_FILE", tmp_path / "index.lock")
    with index_job.hold_lock():
        assert index_job.lock_active()
        with pytest.raises(index_job.IndexLocked):
            with index_job.hold_lock():
                pass
    assert not index_job.lock_active()
    (tmp_path / "index.lock").write_text('{"pid": 999999, "created": 0, "started": 0}', encoding="utf-8")
    assert not index_job.lock_active()


def test_status_reads_result_and_cancel_markers(tmp_path, monkeypatch):
    log = tmp_path / "index.log"
    monkeypatch.setattr(index_job, "LOG_FILE", log)
    monkeypatch.setattr(index_job, "LOCK_FILE", tmp_path / "index.lock")
    monkeypatch.setattr(index_job, "_PROC", None)
    monkeypatch.setattr(index_job, "_MOCK", None)
    log.write_text(f"{index_job.START_MARKER} 100.0 (x)\nWorkflow extract_graph started\n"
                   f"ERROR API_KEY_INVALID\n{index_job.RESULT_MARKER} failed\n", encoding="utf-8")
    job = index_job.status()
    assert job.state == "failed" and "API key" in job.error
    log.write_text(f"{index_job.START_MARKER} 100.0 (x)\n{index_job.RESULT_MARKER} success\n", encoding="utf-8")
    assert index_job.status().state == "success"
    log.write_text(f"{index_job.START_MARKER} 100.0 (x)\n{index_job.CANCEL_MARKER} by user\n", encoding="utf-8")
    assert index_job.status().state == "cancelled"
