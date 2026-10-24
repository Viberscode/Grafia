"""Documents panel: upload, list/delete corpus files, and build the knowledge graph in the background."""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from backend import index_job
from ui.data import DataSource, index_ready, mock_requested

logger = logging.getLogger(__name__)

STAGE_LABELS = {"starting": "Starting (loading GraphRAG)", "chunking": "Chunking documents",
                "entities": "Extracting entities and relationships", "communities": "Detecting communities",
                "reports": "Writing community reports", "embeddings": "Embedding text and entities"}
UPLOAD_KEY = "tg_upload_gen"
RECENT_S = 15 * 60
_REFRESHED_RUN: float | None = None


def _uploads() -> list[tuple[str, bytes]]:
    files = st.session_state.get(f"tg_upload_{st.session_state.get(UPLOAD_KEY, 0)}") or []
    return [(f.name, f.getvalue()) for f in files]


def refresh_after_index() -> None:
    """Drop every cache so graph, themes and counts reload from the new index."""
    from backend import pipeline

    pipeline.reset_graph()
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Caches cleared after indexing")


def _delete(path: str) -> None:
    Path(path).unlink(missing_ok=True)
    logger.info("Deleted corpus file %s", path)


def _start_build() -> None:
    """Ingest validated uploads, then start the background index run (or a simulation under MOCK=1)."""
    if mock_requested():
        index_job.start_mock()
        return
    report = index_job.validate_uploads(_uploads(), existing=len(index_job.input_files()))
    if report.errors:
        return
    try:
        if report.accepted:
            from backend.engine import ingest_files

            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for name, data in report.accepted:
                    p = Path(tmp) / name
                    p.write_bytes(data)
                    paths.append(p)
                ingest_files(paths)
            st.session_state[UPLOAD_KEY] = st.session_state.get(UPLOAD_KEY, 0) + 1
        index_job.start()
    except index_job.IndexLocked as err:
        st.session_state["tg_index_msg"] = str(err)
    except Exception as err:
        logger.exception("Could not start indexing")
        st.session_state["tg_index_msg"] = f"Could not start indexing: {err}"


def corpus_list() -> None:
    """Files in myproj/input with size and a delete button."""
    files = index_job.input_files()
    if not files:
        st.caption("No documents in the corpus yet.")
        return
    st.caption(f"{len(files)} document{'s' if len(files) != 1 else ''} in the corpus (changes apply after the next build)")
    with st.container(height=min(44 * len(files) + 8, 220), border=False):
        for f in files:
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(f":material/description: `{f.name}` <span class='tg-muted'>{f.stat().st_size / 1024:.1f} KB</span>",
                        unsafe_allow_html=True)
            c2.button("", key=f"del_{f.name}", icon=":material/delete:", help=f"Delete {f.name}",
                      on_click=_delete, args=(str(f),), disabled=index_job.running())


def upload_and_estimate() -> index_job.UploadReport:
    """Uploader, validation messages and the cost estimate."""
    st.file_uploader("Add documents (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True,
                     key=f"tg_upload_{st.session_state.get(UPLOAD_KEY, 0)}",
                     help=f"Up to {index_job.MAX_FILES} documents, 10 MB each.")
    existing = index_job.input_files()
    report = index_job.validate_uploads(_uploads(), existing=len(existing))
    for msg in report.errors:
        st.error(msg, icon=":material/error:")
    for msg in report.warnings:
        st.warning(msg, icon=":material/warning:")
    chars = [f.stat().st_size for f in existing] + report.chars
    est = index_job.estimate(chars)
    st.caption(f"Estimate for {est['docs']} documents: about **{est['chunks']} chunks** and **{est['llm_calls']} LLM calls**. "
               "The Gemini free tier is rate limited, so a build can take several minutes; chunks indexed "
               "before are reused from the cache.")
    return report


def _progress(job: index_job.JobStatus) -> None:
    """Stage checklist, elapsed time, log tail and Cancel while a run is active."""
    label = STAGE_LABELS.get(job.current or "", "Indexing")
    with st.status(f"{label}... ({job.elapsed_s:.0f} s)", state="running", expanded=True):
        for stage, _ in index_job.STAGES:
            mark = ":green[:material/check_circle:]" if stage in job.done else (
                ":blue[:material/progress_activity:]" if stage == job.current else ":gray[:material/radio_button_unchecked:]")
            st.markdown(f"{mark} {STAGE_LABELS[stage]}")
        st.code("\n".join(job.tail[-6:]) or "waiting for log...", language=None, wrap_lines=True)
        st.button("Cancel", icon=":material/stop_circle:", on_click=index_job.cancel, key="tg_index_cancel")


def _finished(job: index_job.JobStatus) -> None:
    """Result card for the last run; refreshes caches once per successful run."""
    global _REFRESHED_RUN
    if job.state == "success":
        if job.started != _REFRESHED_RUN:
            _REFRESHED_RUN = job.started
            refresh_after_index()
            st.rerun(scope="app")
        if job.started and time.time() - job.started - job.elapsed_s < RECENT_S:
            st.success(f"Knowledge graph built in {job.elapsed_s:.0f} s. Graph, themes and counts are up to date.",
                       icon=":material/check_circle:")
            if job.error:
                st.warning(job.error, icon=":material/warning:")
        else:
            st.caption(f":material/check_circle: Last build succeeded ({job.elapsed_s:.0f} s).")
    elif job.state == "failed":
        st.error(job.error or "Indexing failed.", icon=":material/error:")
        with st.expander("Show log"):
            st.code("\n".join(index_job.read_log(60)), language=None, wrap_lines=True)
    elif job.state == "cancelled":
        st.info("The last build was cancelled.", icon=":material/stop_circle:")


def job_panel(polling: bool = False) -> None:
    """Progress while running, else the last result; a polling fragment reruns the app when the run ends."""
    job = index_job.status()
    if job.state == "running":
        _progress(job)
    elif polling and job.state != "success":
        st.rerun(scope="app")
    else:
        _finished(job)


def _poll() -> None:
    job_panel(polling=True)


def render(data: DataSource) -> None:
    """The Documents expander at the top of the chat column."""
    busy = index_job.running()
    open_ = busy or (not index_ready() and not data.is_mock)
    with st.expander("Documents", expanded=open_, icon=":material/upload_file:"):
        if mock_requested():
            st.caption("Demo mode: Build simulates progress without changing files or calling the API.")
        corpus_list()
        report = upload_and_estimate()
        st.button("Build knowledge graph", type="primary", icon=":material/hub:", on_click=_start_build,
                  disabled=busy or bool(report.errors) or not (report.accepted or index_job.input_files() or mock_requested()))
        msg = st.session_state.pop("tg_index_msg", None)
        if msg:
            st.error(msg, icon=":material/lock:")
        if busy:
            st.fragment(run_every=2.0)(_poll)()
        else:
            job_panel()
