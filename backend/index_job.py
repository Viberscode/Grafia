"""Background GraphRAG index runs: validation, estimates, lock file, subprocess, log-parsed progress."""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import psutil
import yaml

from backend.config import GRAPH_ROOT, INPUT_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

LOG_DIR = GRAPH_ROOT / "logs"
LOG_FILE = LOG_DIR / "index.log"
LOCK_FILE = LOG_DIR / "index.lock"
RESULT_MARKER = "INDEX RESULT:"
START_MARKER = "INDEX STARTED"
CANCEL_MARKER = "INDEX CANCELLED"
MAX_FILES = 20
MAX_BYTES = 10 * 1024 * 1024
STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("chunking", ("load_input_documents", "create_base_text_units", "create_final_documents")),
    ("entities", ("extract_graph", "finalize_graph", "extract_covariates")),
    ("communities", ("create_communities", "create_final_text_units")),
    ("reports", ("create_community_reports",)),
    ("embeddings", ("generate_text_embeddings",)),
]
MOCK_STAGE_S = 1.5
ERROR_HINTS: list[tuple[str, str]] = [
    (r"API_KEY_INVALID|API key not valid|invalid api key|401|PERMISSION_DENIED|Unauthorized|AuthenticationError",
     "Gemini rejected the API key. Check GEMINI_API_KEY in .env, then build again."),
    (r"429|RESOURCE_EXHAUSTED|rate limit|quota",
     "Gemini's free-tier rate limit or daily quota was hit. Wait a few minutes (or until tomorrow) and build again; "
     "finished steps are cached."),
    (r"No input documents|No documents|input.*empty", "No input documents were found. Upload at least one file."),
    (r"ConnectError|Connection refused|getaddrinfo|Timeout", "Could not reach the Gemini API. Check your internet connection."),
]


class IndexLocked(RuntimeError):
    """Another index run holds the lock."""


@dataclass
class UploadReport:
    """Validation outcome for a batch of uploads."""

    accepted: list[tuple[str, bytes]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    chars: list[int] = field(default_factory=list)


@dataclass
class JobStatus:
    """Progress of the current or last index run."""

    state: Literal["idle", "running", "success", "failed", "cancelled"]
    done: list[str] = field(default_factory=list)
    current: str | None = None
    elapsed_s: float = 0.0
    tail: list[str] = field(default_factory=list)
    error: str | None = None
    started: float | None = None


# ---------- validation and estimates ----------

def pdf_text(data: bytes) -> str:
    """Extractable text of a PDF (empty string if none or unreadable)."""
    from pypdf import PdfReader

    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages).strip()
    except Exception:
        logger.warning("Could not read PDF", exc_info=True)
        return ""


def validate_uploads(files: list[tuple[str, bytes]], existing: int = 0) -> UploadReport:
    """Check count, size, type and PDF text; accepted files keep their bytes."""
    rep = UploadReport()
    if existing + len(files) > MAX_FILES:
        rep.errors.append(f"At most {MAX_FILES} documents are allowed ({existing} already in the corpus).")
        return rep
    for name, data in files:
        ext = Path(name).suffix.lower()
        if ext not in (".txt", ".pdf"):
            rep.errors.append(f"{name}: only .txt and .pdf files are supported.")
        elif len(data) > MAX_BYTES:
            rep.errors.append(f"{name}: {len(data) / 1e6:.1f} MB is over the 10 MB limit.")
        elif ext == ".pdf" and not (text := pdf_text(data)):
            rep.warnings.append(f"{name}: no extractable text (scanned PDF?). It will be skipped.")
        else:
            rep.accepted.append((name, data))
            rep.chars.append(len(text) if ext == ".pdf" else len(data.decode("utf-8", errors="ignore")))
    return rep


def chunk_settings() -> tuple[int, int, int]:
    """(chunk size, overlap, max gleanings) from settings.yaml, with GraphRAG-like defaults."""
    try:
        cfg = yaml.safe_load((GRAPH_ROOT / "settings.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    chunking = cfg.get("chunking") or {}
    gleanings = (cfg.get("extract_graph") or {}).get("max_gleanings", 1)
    return int(chunking.get("size", 1200)), int(chunking.get("overlap", 100)), int(gleanings)


def estimate(doc_chars: list[int]) -> dict[str, int]:
    """Rough chunk and LLM-call estimate for a corpus (about 4 characters per token)."""
    size, overlap, gleanings = chunk_settings()
    step = max(1, size - overlap)
    chunks = sum(max(1, math.ceil(max(0, c / 4 - overlap) / step)) for c in doc_chars if c > 0)
    extraction = chunks * (1 + gleanings)
    summaries = chunks
    reports = max(1, round(chunks * 1.2))
    embeddings = math.ceil(chunks * 8 / 16)
    return {"docs": len(doc_chars), "chunks": chunks, "llm_calls": extraction + summaries + reports + embeddings}


def input_files() -> list[Path]:
    """Documents currently in the GraphRAG input folder."""
    return sorted(p for p in INPUT_DIR.glob("*.txt") if p.is_file()) if INPUT_DIR.exists() else []


# ---------- lock ----------

def _read_lock() -> dict | None:
    try:
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _alive(lock: dict | None) -> bool:
    """True if the lock's process is still the one that took it."""
    if not lock:
        return False
    try:
        proc = psutil.Process(int(lock["pid"]))
        return proc.is_running() and abs(proc.create_time() - float(lock["created"])) < 2
    except (psutil.Error, KeyError, ValueError, TypeError):
        return False


def lock_active() -> bool:
    """An index run is in progress (stale locks are ignored)."""
    return _alive(_read_lock())


@contextmanager
def hold_lock() -> Iterator[None]:
    """Hold the index lock for this process; raises IndexLocked if another live run holds it."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if lock_active():
        raise IndexLocked("Another index run is already in progress.")
    me = psutil.Process(os.getpid())
    LOCK_FILE.write_text(json.dumps({"pid": me.pid, "created": me.create_time(), "started": time.time()}), encoding="utf-8")
    try:
        yield
    finally:
        if (_read_lock() or {}).get("pid") == me.pid:
            LOCK_FILE.unlink(missing_ok=True)


# ---------- run, cancel, status ----------

_PROC: subprocess.Popen | None = None
_MOCK: dict | None = None


def running() -> bool:
    """A real or simulated run is in progress."""
    return (_PROC is not None and _PROC.poll() is None) or lock_active() or (
        _MOCK is not None and time.time() - _MOCK["started"] < MOCK_STAGE_S * len(STAGES))


def start(env: dict[str, str] | None = None) -> int:
    """Start `python -m backend.engine --index` in the background, logging to myproj/logs/index.log."""
    global _PROC
    if running():
        raise IndexLocked("Another index run is already in progress.")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")  # noqa: SIM115 (handed to the child process)
    log.write(f"{START_MARKER} {time.time():.1f} ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
    log.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    _PROC = subprocess.Popen([sys.executable, "-m", "backend.engine", "--index"], cwd=PROJECT_ROOT,
                             stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                             creationflags=flags)
    log.close()
    logger.info("Started index run pid=%s", _PROC.pid)
    return _PROC.pid


def start_mock() -> None:
    """Simulated run for MOCK=1: no files, no LLM calls."""
    global _MOCK
    _MOCK = {"started": time.time()}


def cancel() -> None:
    """Stop the running index (and its children), release the lock and note it in the log."""
    global _MOCK
    if _MOCK is not None:
        _MOCK = None
        return
    pids = {int((_read_lock() or {}).get("pid", 0))} | ({_PROC.pid} if _PROC is not None else set())
    for pid in filter(None, pids):
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                child.kill()
            proc.kill()
        except psutil.Error:
            pass
    LOCK_FILE.unlink(missing_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"{CANCEL_MARKER} by user\n")
    logger.info("Index run cancelled")


def read_log(n: int = 400) -> list[str]:
    """Last n lines of the index log."""
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def parse_stages(lines: list[str]) -> tuple[list[str], str | None]:
    """(finished stages, current stage) from 'Workflow X started/finished' log lines."""
    names = [s for s, _ in STAGES]
    stage_of = {w: s for s, ws in STAGES for w in ws}
    started = [m.group(1) for ln in lines if (m := re.search(r"Workflow (\w+) started", ln))]
    finished = {m.group(1) for ln in lines if (m := re.search(r"Workflow (\w+) (?:finished|done)", ln))}
    open_ = [w for w in started if w not in finished and w in stage_of]
    if open_:
        idx = names.index(stage_of[open_[-1]])
        return names[:idx], names[idx]
    last = [w for w in started if w in stage_of]
    if not last:
        return [], None
    idx = names.index(stage_of[last[-1]])
    return names[:idx + 1], None


def friendly_error(lines: list[str]) -> str:
    """A readable reason for a failed run, from known error patterns in the log."""
    text = "\n".join(lines)
    for pattern, message in ERROR_HINTS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return message
    errors = [ln for ln in lines if " ERROR " in ln or "Error" in ln]
    return f"Indexing failed: {errors[-1][-300:]}" if errors else "Indexing failed. See the log for details."


def _mock_status() -> JobStatus:
    elapsed = time.time() - _MOCK["started"]
    k = min(len(STAGES), int(elapsed // MOCK_STAGE_S))
    names = [s for s, _ in STAGES]
    if k >= len(STAGES):
        return JobStatus("success", done=names, elapsed_s=MOCK_STAGE_S * len(STAGES), started=_MOCK["started"],
                         tail=["Simulated run (MOCK=1): no files changed, no API calls."])
    return JobStatus("running", done=names[:k], current=names[k], elapsed_s=elapsed, started=_MOCK["started"],
                     tail=[f"Simulating {names[k]}..."])


def _started_at() -> float | None:
    """Start time written on the log's first line."""
    first = read_log(n=10**6)[:1]
    m = re.search(rf"{START_MARKER} ([\d.]+)", first[0]) if first else None
    return float(m.group(1)) if m else None


def _elapsed(started: float | None, finished: bool) -> float:
    """Seconds since start; for finished runs, up to the last log write."""
    if not started:
        return 0.0
    try:
        end = LOG_FILE.stat().st_mtime if finished else time.time()
    except OSError:
        end = time.time()
    return max(0.0, end - started)


def status() -> JobStatus:
    """Current or last run, read from the process handle, lock file and log."""
    if _MOCK is not None:
        return _mock_status()
    lines = read_log()
    if not lines:
        return JobStatus("idle")
    done, current = parse_stages(lines)
    started = _started_at()
    is_running = running()
    elapsed = _elapsed(started, finished=not is_running)
    tail = lines[-12:]
    if any(CANCEL_MARKER in ln for ln in lines):
        return JobStatus("cancelled", done, None, elapsed, tail, started=started)
    result = next((ln for ln in reversed(lines) if RESULT_MARKER in ln), None)
    if result is None and is_running:
        names = [s for s, _ in STAGES]
        nxt = names[len(done)] if done and len(done) < len(names) else "starting"
        return JobStatus("running", done, current or nxt, elapsed, tail, started=started)
    if result is not None and result.rstrip().endswith("success"):
        errors = sum(1 for ln in lines if " ERROR " in ln)
        warn = f"{errors} errors were logged; some chunks may be missing from the graph." if errors else None
        return JobStatus("success", [s for s, _ in STAGES], None, elapsed, tail, error=warn, started=started)
    return JobStatus("failed", done, current, elapsed, tail, error=friendly_error(lines), started=started)
