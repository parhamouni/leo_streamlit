"""FastAPI backend for fence detection analysis.

Runs analysis jobs in background workers. Streamlit frontend communicates
via HTTP to submit jobs, poll progress, and fetch results.

    uvicorn api_server:app --host 127.0.0.1 --port 8503
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

# Load .env.local before importing config so SUPABASE_*, AUTH_MODE, etc.
# are visible to Config's field_factory env reads. Falls back silently if
# python-dotenv isn't installed (e.g. running the prod monolith only).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env.local")
except ImportError:
    pass

import job_registry
import uuid as _uuid
from backend.app import db
from backend.app.auth import get_current_user, require_supabase_jwt
import config as cfg_mod
from config import cfg
from pipeline import PipelineConfig, PipelineResult, run_analysis
from secrets_loader import load_api_keys as _load_api_keys


_MEASUREMENT_PDF_WORKER = Path(__file__).parent / "ops" / "measurement_pdf_worker.py"
_EXPORT_VECTOR_WORKER = Path(__file__).parent / "ops" / "export_vector_worker.py"
_PAGECOUNT_WORKER = Path(__file__).parent / "ops" / "pdf_pagecount_worker.py"
# Page-count peek is normally sub-second; a hang here means a malformed PDF,
# so cap it tight rather than waiting on the long export timeouts.
_PAGECOUNT_TIMEOUT = 30


class ExportWorkerError(RuntimeError):
    """Controlled export-worker failure safe to surface as an HTTP error."""


class ExportWorkerTimeout(TimeoutError):
    """Export worker exceeded its wall-clock cap."""


def _is_uuid(s: str) -> bool:
    """True if s is a well-formed UUID (Supabase user IDs always are).
    Legacy X-User-Id values like 'alice' or 'anonymous' return False."""
    try:
        _uuid.UUID(s)
        return True
    except Exception:
        return False


def _unix_to_iso(ts: Any) -> str | None:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _sqlite_job_to_document(job: dict) -> dict:
    """Present a legacy SQLite-only job as a document row.

    The Next.js app is document-oriented, but the worker queue and older
    runs are job-oriented. When a job has no Postgres mirror row, using the
    job_id as the synthetic document id lets `/documents/{id}` load and then
    fetch results through the existing `/api/jobs/{job_id}/results` path.
    """
    progress = job_registry.read_progress(job["job_id"]) or {}
    status = job.get("status")
    created_at = _unix_to_iso(job.get("created_at")) or datetime.now(timezone.utc).isoformat()
    return {
        "id": job["job_id"],
        "user_id": job.get("user_id"),
        "original_filename": job.get("filename") or "unknown.pdf",
        "storage_path": job.get("pdf_path") or "",
        "document_status": "uploaded",
        "total_pages": job.get("total_pages"),
        "created_at": created_at,
        "latest_job_id": job["job_id"],
        "job_status": status,
        "current_phase": progress.get("phase") if status in ("queued", "running") else ("done" if status == "completed" else None),
        "progress_percent": progress.get("pct") if status in ("queued", "running") else (100 if status in ("completed", "failed", "cancelled") else 0),
        "error_message": job.get("error_msg"),
        "job_started_at": _unix_to_iso(job.get("started_at")),
        "phase_started_at": None,
        "source": "sqlite",
    }


def _merge_sqlite_fallback_documents(user_id: str, documents: list[dict]) -> list[dict]:
    """Append SQLite jobs that are not represented by any Postgres job row."""
    seen_job_ids = {
        str(d.get("latest_job_id"))
        for d in documents
        if d.get("latest_job_id")
    }
    merged = [dict(d) for d in documents]
    for job in job_registry.get_user_jobs(user_id):
        job_id = str(job.get("job_id") or "")
        if not job_id or job_id in seen_job_ids:
            continue
        merged.append(_sqlite_job_to_document(job))
        seen_job_ids.add(job_id)
    merged.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
    return merged


def _delete_postgres_document_for_job(job_id: str, user_id: str) -> bool:
    """Delete the Postgres document owning `job_id`, if the user owns it.

    Older dashboard rows can outlive their SQLite job_registry row because
    local job results expire separately from the Postgres mirror. In that
    state the normal delete path cannot start from SQLite, so this helper
    deletes by Postgres job id and lets FK cascades remove page_results,
    artifacts, and the mirrored jobs row.
    """
    if not _is_uuid(user_id):
        return False
    with db.pool().connection() as conn:
        row = conn.execute(
            """
            delete from documents
            where user_id = %s
              and id in (select document_id from jobs where id = %s)
            returning id
            """,
            (user_id, job_id),
        ).fetchone()
    return row is not None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("api_server")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_shutdown_event = threading.Event()


def _worker_loop():
    """Poll for queued jobs and run them. Runs in a daemon thread."""
    keys = _load_api_keys()
    log.info(f"Worker started (max_concurrent={cfg.API_WORKER_COUNT})")

    active_threads: list[threading.Thread] = []

    while not _shutdown_event.is_set():
        active_threads = [t for t in active_threads if t.is_alive()]

        if len(active_threads) >= cfg.API_WORKER_COUNT:
            _shutdown_event.wait(2)
            continue

        try:
            job = job_registry.next_queued_job()
        except Exception as e:
            log.error(f"Failed to fetch next job: {e}")
            _shutdown_event.wait(5)
            continue

        if job is None:
            _shutdown_event.wait(2)
            continue

        t = threading.Thread(
            target=_run_job, args=(job, keys), daemon=True,
            name=f"worker-{job['job_id'][:8]}",
        )
        t.start()
        active_threads.append(t)
        log.info(f"Dispatched job {job['job_id'][:8]} ({job.get('filename')})")


def _run_job(job: dict, keys: dict):
    """Execute a single analysis job."""
    job_id = job["job_id"]
    pdf_path = job.get("pdf_path")

    if not pdf_path or not Path(pdf_path).exists():
        job_registry.update_job(job_id, status="failed",
                                 error_msg="PDF file not found on disk",
                                 completed_at=int(time.time()))
        return

    config_data = {}
    if job.get("config_json"):
        try:
            config_data = json.loads(job["config_json"])
        except Exception:
            pass

    # Contracting trade / mode the user picked at upload ("fence" | "electrical").
    # Keywords default to that trade's set when the client didn't send a custom
    # list (a blank/empty list also falls back to the trade default).
    trade = config_data.get("trade") or cfg_mod.DEFAULT_TRADE
    trade_keywords = list(cfg_mod.trade_profile(trade)["keywords"])

    config = PipelineConfig(
        openai_api_key=keys.get("openai_key", ""),
        ade_api_key=keys.get("ade_key", ""),
        google_cloud_config=keys.get("google_cloud_config"),
        analysis_model=config_data.get("analysis_model", cfg.ANALYSIS_MODEL),
        classifier_model=config_data.get("classifier_model", cfg.CLASSIFIER_MODEL),
        trade=trade,
        fence_keywords=config_data.get("fence_keywords") or trade_keywords,
        use_ade=config_data.get("use_ade", True),
        highlight_fence_text=config_data.get("highlight_fence_text", True),
        enable_unified_measurement=config_data.get("enable_unified_measurement", True),
        # When False, pipeline skips the LLM-guided fallback measurement on
        # pages with no fence layers — saves minutes per page on layerless
        # PDFs the user knows to be irrelevant.
        enable_nonlayer_suggestions=config_data.get("enable_nonlayer_suggestions", False),
        cache_scope=f"job_{job_id[:16]}",
    )

    def _progress(phase: str, pct: int, message: str):
        try:
            job_registry.write_progress(job_id, phase, pct, message)
        except Exception:
            pass
        # Best-effort Postgres mirror so the dashboard sees live progress.
        try:
            db.update_job_progress(
                job_id,
                current_phase=phase,
                progress_percent=pct,
            )
        except Exception:
            pass

    # Resolve the document this job belongs to so page_cb can stream
    # per-page rows as Phase 1c / Phase 3 finish each page. Legacy
    # X-User-Id jobs have no Postgres row → document_id stays None and
    # page_cb is a no-op.
    #
    # The Postgres mirror is written by a background task after the
    # upload response returns, so the row may not exist yet when the
    # worker grabs this job. Resolve lazily and cache once we see it.
    _doc_state: dict[str, Any] = {"id": None, "checked_legacy": False}

    def _resolve_document_id() -> str | None:
        if _doc_state["id"] is not None:
            return _doc_state["id"]
        if _doc_state["checked_legacy"]:
            return None
        try:
            did = db.get_document_id_by_job(job_id)
        except Exception:
            return None
        if did:
            _doc_state["id"] = did
            return did
        # Not in Postgres (yet). For legacy X-User-Id jobs it will never
        # arrive; user_id on the job row tells us which. Latch a sentinel
        # so we stop hammering the DB on every page callback.
        if not _is_uuid(job.get("user_id", "")):
            _doc_state["checked_legacy"] = True
        return None

    def _page_cb(page: dict) -> None:
        document_id = _resolve_document_id()
        if not document_id:
            return
        try:
            # Stamp the trade into each page payload. page_results persist in
            # Postgres long after the per-job results.json is cleaned up, so
            # this is what lets an expired document still reconstruct its
            # results *and* know which mode it ran in (see _results_from_pages).
            rj_in = page.get("result_json")
            if isinstance(rj_in, dict):
                rj_in = {**rj_in, "_trade": trade}
            db.upsert_page_result(
                document_id=document_id,
                page_number=int(page["page_number"]),
                is_fence_page=bool(page.get("is_fence_page", False)),
                result_json=rj_in,
            )
            rj = page.get("result_json") or {}
            phase = rj.get("phase") or (
                "phase3" if (rj.get("measurements") or rj.get("legend_entries")) else "rich?"
            )
            log.info(
                "page_cb job=%s page=%s fence=%s phase=%s",
                job_id[:8],
                page.get("page_number"),
                page.get("is_fence_page"),
                phase,
            )
        except Exception:
            log.exception(f"Job {job_id[:8]}: upsert_page_result failed")

    # Mark Postgres jobs row as running.
    try:
        db.update_job_progress(job_id, status="running", started_at_now=True)
    except Exception:
        pass

    try:
        _progress("start", 0, "Analysis starting...")
        result = run_analysis(pdf_path, config, progress_cb=_progress, page_cb=_page_cb)

        results_dir = Path(job.get("results_dir") or str(Path("~/.leo/results").expanduser() / job_id))
        results_dir.mkdir(parents=True, exist_ok=True)

        results_data = {
            "fence_pages": result.fence_pages,
            "non_fence_pages": result.non_fence_pages,
            "element_details": result.element_details,
            "per_page_scale_info": result.per_page_scale_info,
            "unified_measurements": result.unified_measurements,
            "page_categories": result.page_categories,
            "total_pages": result.total_pages,
            # trade drives the document view's mode labels/badge + the
            # re-analyze "current mode" detection; broken_pages drives the
            # damaged-pages banner. Both must be persisted or the frontend
            # can't tell which trade ran (defaults to fence).
            "trade": result.trade,
            "broken_pages": result.broken_pages,
            "timings": result.timings,
            "error": result.error,
        }
        job_registry.save_results(job_id, results_data)

        if result.highlighted_pdf_bytes:
            hl_path = results_dir / "highlighted.pdf"
            hl_path.write_bytes(result.highlighted_pdf_bytes)

        if result.error:
            job_registry.update_job(
                job_id, status="failed",
                error_msg=result.error[:500],
                completed_at=int(time.time()),
                total_pages=result.total_pages,
            )
            try:
                db.update_job_progress(
                    job_id, status="failed",
                    error_message=result.error[:500],
                    progress_percent=100,
                    finished_at_now=True,
                )
            except Exception:
                pass
            _progress("done", 100, f"Failed: {result.error[:100]}")
        else:
            job_registry.update_job(
                job_id, status="completed",
                completed_at=int(time.time()),
                total_pages=result.total_pages,
                fence_count=len(result.fence_pages),
                non_fence_count=len(result.non_fence_pages),
            )
            try:
                db.update_job_progress(
                    job_id, status="completed",
                    progress_percent=100,
                    current_phase="done",
                    finished_at_now=True,
                )
            except Exception:
                pass
            _progress("done", 100, "Analysis complete")

        log.info(f"Job {job_id[:8]} completed: "
                 f"{len(result.fence_pages)} fence, {len(result.non_fence_pages)} non-fence")

    except Exception as e:
        log.exception(f"Job {job_id[:8]} failed: {e}")
        job_registry.update_job(
            job_id, status="failed",
            error_msg=str(e)[:500],
            completed_at=int(time.time()),
        )
        try:
            db.update_job_progress(
                job_id, status="failed",
                error_message=str(e)[:500],
                progress_percent=100,
                finished_at_now=True,
            )
        except Exception:
            pass
        try:
            job_registry.write_progress(job_id, "error", 100, f"Failed: {e}")
        except Exception:
            pass


def _ttl_cleanup_loop():
    """Periodically clean up expired jobs and their results."""
    while not _shutdown_event.is_set():
        _shutdown_event.wait(3600)
        if _shutdown_event.is_set():
            break
        try:
            removed = job_registry.cleanup_expired_jobs()
            if removed:
                log.info(f"TTL cleanup: removed {removed} expired jobs")
        except Exception as e:
            log.error(f"TTL cleanup failed: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Re-queue jobs orphaned by previous API restart (clean recovery)
    startup_ts = int(time.time())
    requeued, poisoned = job_registry.requeue_orphaned_running()
    if requeued:
        log.info(f"Startup: re-queued {requeued} orphaned running jobs")
    if poisoned:
        log.warning(
            f"Startup: {poisoned} orphaned jobs exceeded the re-queue cap "
            "and were marked failed (poison-pill guard)"
        )
    # Catch ancient stale rows that pre-date the requeue logic
    stale = job_registry.mark_stale_running_as_failed(max_age_seconds=7200)
    if stale:
        log.info(f"Startup: marked {stale} stale running jobs as failed")
    # Best-effort Postgres mirror for jobs failed by the startup guards, so
    # the dashboard doesn't show them as running forever.
    if poisoned or stale:
        for j in job_registry.list_failed_since(startup_ts):
            try:
                db.update_job_progress(
                    j["job_id"],
                    status="failed",
                    error_message=j.get("error_msg"),
                    finished_at_now=True,
                )
            except Exception:
                pass

    global _worker_thread
    _shutdown_event.clear()

    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="job-dispatcher")
    _worker_thread.start()

    ttl_thread = threading.Thread(target=_ttl_cleanup_loop, daemon=True, name="ttl-cleanup")
    ttl_thread.start()

    log.info("API server started")
    yield

    _shutdown_event.set()
    log.info("API server shutting down")


app = FastAPI(title="Leo Fence Detection API", lifespan=lifespan)


# Allow the Next.js frontend (Vercel + local dev) to call us across origins.
#   FENCE_CORS_ORIGINS — comma-separated explicit origin list; defaults
#                        to local dev origins.
#   FENCE_CORS_ORIGIN_REGEX — optional regex (e.g. for Vercel preview
#                        deploys at https://leo-fence-git-*.vercel.app).
#                        Matched in addition to the explicit list.
_default_origins = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
_cors_origins = [o.strip() for o in os.environ.get("FENCE_CORS_ORIGINS", _default_origins).split(",") if o.strip()]
_cors_origin_regex = os.environ.get("FENCE_CORS_ORIGIN_REGEX") or None
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    # Range: the frontend downloads big artifacts in ranged chunks because
    # some client-side security software kills large single responses
    # mid-stream (2026-07-08: consistent cuts at ~14 MB).
    allow_headers=["Authorization", "Content-Type", "X-User-Id", "Range"],
    expose_headers=[
        "Content-Disposition", "Content-Range", "Accept-Ranges",
        "X-Page-Number", "X-Image-Source",
    ],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _mirror_document_to_postgres(
    *,
    document_id: str,
    job_id: str,
    user_id: str,
    original_filename: str,
    storage_path: str,
    total_pages: int,
    pdf_hash: str,
) -> None:
    """Background task: write the document + job row to Postgres after the
    upload response has been sent. On failure, mark the SQLite job failed
    so the worker won't pick it up and the user sees an error in the
    dashboard. The PDF file is kept for debugging.
    """
    log.info(f"Mirror starting for job {job_id[:8]} doc {document_id[:8]}")
    try:
        db.insert_document_and_job(
            user_id=user_id,
            original_filename=original_filename,
            storage_path=storage_path,
            total_pages=total_pages,
            job_id=job_id,
            pdf_hash=pdf_hash,
            document_id=document_id,
        )
        log.info(f"Mirror OK for job {job_id[:8]} doc {document_id[:8]}")
    except Exception as e:
        log.exception(f"Postgres mirror failed for job {job_id[:8]}; marking failed")
        try:
            job_registry.update_job(
                job_id,
                status="failed",
                error_msg=f"Database unavailable: {str(e)[:400]}",
                completed_at=int(time.time()),
            )
        except Exception:
            log.exception(f"Could not mark job {job_id[:8]} failed after mirror failure")


@app.post("/api/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    config: str = Form("{}"),
    user_id: str = Depends(get_current_user),
):
    """Submit a PDF for analysis. Returns 202 with the job + document IDs
    immediately after the file is on disk and the SQLite job row exists;
    the Postgres mirror runs as a background task after the response is
    sent (so a slow / flaky DB write doesn't push the response past the
    edge router's timeout — see incident notes for the 502 from sfo1)."""
    try:
        config_data = json.loads(config)
    except Exception:
        config_data = {}

    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty PDF file")

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > cfg.MAX_PDF_MB:
        raise HTTPException(
            413,
            f"File too large ({size_mb:.0f} MB). Max is {cfg.MAX_PDF_MB} MB.",
        )

    import hashlib
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Dedup: if this Supabase user already has a document with the same
    # pdf_hash, return the existing one. Re-process only if the previous
    # attempt is in a terminal-failed state. Done before the page-count
    # peek so duplicates skip that work entirely.
    if _is_uuid(user_id):
        try:
            existing = db.find_document_by_hash(user_id, pdf_hash)
        except Exception:
            log.exception("dedup lookup failed; falling through to fresh upload")
            existing = None
        if existing and existing.get("job_status") != "failed":
            return {
                "job_id": existing.get("latest_job_id"),
                "document_id": existing["id"],
                "status": "deduped",
                "existing_status": existing.get("job_status"),
                "original_filename": existing["original_filename"],
                "queue_position": None,
                "running_jobs": None,
            }

    upload_dir = Path(cfg.PDF_TMP_DIR) / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = f"job_{pdf_hash[:16]}.pdf"
    pdf_path = upload_dir / pdf_filename
    pdf_path.write_bytes(pdf_bytes)

    # Peek page count in a subprocess with a hard timeout. A malformed PDF
    # can drive PyMuPDF into a C-level recovery loop; doing this inline would
    # block the API event loop (and every concurrent request) until it gave
    # up. On failure we drop the just-written file so a rejected upload
    # doesn't leave an orphan in PDF_TMP_DIR.
    try:
        n_pages = _peek_page_count(str(pdf_path))
    except ExportWorkerTimeout:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            400, "Could not read PDF: page-count timed out (file may be damaged)."
        )
    except ExportWorkerError as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read PDF: {e}")

    if n_pages > cfg.MAX_PAGES:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            413,
            f"PDF has {n_pages} pages (max {cfg.MAX_PAGES}). "
            "Please split the document by page range.",
        )

    job_id = job_registry.create_job(
        user_id=user_id,
        filename=pdf.filename or "unknown.pdf",
        pdf_hash=pdf_hash,
        pdf_path=str(pdf_path),
        config_json=json.dumps(config_data, default=str),
    )

    # Mirror to Postgres so the dashboard list survives across sessions.
    # Only Supabase (UUID) users get a Postgres row — the legacy X-User-Id
    # path (e.g. the existing Streamlit fast-stack client) skips this and
    # continues to live in SQLite job_registry only.
    #
    # The mirror runs as a background task so the response can return
    # immediately. We pre-allocate the document_id here so the client gets
    # it in the response before the row is actually persisted. The worker
    # tolerates the mirror not yet being committed when it picks up the job
    # — see `_run_job._resolve_document_id`.
    document_id: str | None = None
    if _is_uuid(user_id):
        document_id = str(_uuid.uuid4())
        rel_storage_path = f"{user_id}/{pdf_filename}"
        background_tasks.add_task(
            _mirror_document_to_postgres,
            document_id=document_id,
            job_id=job_id,
            user_id=user_id,
            original_filename=pdf.filename or "unknown.pdf",
            storage_path=rel_storage_path,
            total_pages=n_pages,
            pdf_hash=pdf_hash,
        )

    queue_pos = job_registry.queue_position(job_id)
    running = job_registry.count_running_jobs()

    return {
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "queue_position": queue_pos,
        "running_jobs": running,
    }


@app.post("/api/jobs/{job_id}/reanalyze", status_code=202)
async def reanalyze_job(
    job_id: str,
    trade: str,
    user_id: str = Depends(get_current_user),
):
    """Re-run analysis on an already-uploaded document in a different trade
    mode (fence/electrical), reusing the PDF already on disk — no re-upload.

    Creates a NEW job for the same document with the requested trade. Because
    page_results are keyed per document (not per job), the new run replaces
    the prior trade's per-page rows and the document view switches to the new
    trade. The pipeline cache is keyed by (pdf_hash, trade), so a trade that
    was analysed before re-runs from cache — fast and no new LLM spend, which
    is what makes "do it again if not done already" cheap.
    """
    trade = (trade or "").strip().lower()
    if trade not in cfg_mod.TRADE_PROFILES:
        raise HTTPException(
            400,
            f"Unknown trade '{trade}'. Choose one of: {', '.join(cfg_mod.TRADE_PROFILES)}.",
        )

    # Source PDF + metadata: prefer the live SQLite job, but fall back to the
    # Postgres document when the job has aged out of the registry. That lets an
    # older (reconstructed) document still be re-analysed as long as its
    # uploaded PDF is on disk — the job row expiring no longer blocks it.
    old = job_registry.get_job(job_id)
    document_id: str | None = None
    if old is not None:
        if old.get("user_id") != user_id:
            raise HTTPException(404, "Job not found")
        pdf_path = old.get("pdf_path")
        filename = old.get("filename") or "document.pdf"
        pdf_hash = old.get("pdf_hash")
        try:
            new_cfg = json.loads(old.get("config_json") or "{}")
        except Exception:
            new_cfg = {}
    else:
        document_id = db.get_document_id_by_job(job_id) if _is_uuid(user_id) else None
        doc = db.get_document(document_id, user_id) if document_id else None
        if not doc:
            raise HTTPException(404, "Job not found")
        storage_path = doc.get("storage_path") or ""
        pdf_path = str(Path(cfg.PDF_TMP_DIR) / storage_path) if storage_path else None
        filename = doc.get("original_filename") or "document.pdf"
        pdf_hash = None
        new_cfg = {}

    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(
            409,
            "The original PDF is no longer on the server. Re-upload it to "
            f"analyse in {cfg_mod.trade_profile(trade)['label']} mode.",
        )

    # Swap the trade + its keyword set, keep the rest of the prior settings.
    new_cfg["trade"] = trade
    new_cfg["fence_keywords"] = list(cfg_mod.trade_profile(trade)["keywords"])

    new_job_id = job_registry.create_job(
        user_id=user_id,
        filename=filename,
        pdf_hash=pdf_hash,
        pdf_path=pdf_path,
        config_json=json.dumps(new_cfg, default=str),
    )

    # Mirror under the SAME Postgres document so the document view picks the
    # new job up as "latest". Legacy X-User-Id jobs have no Postgres doc.
    if _is_uuid(user_id):
        try:
            if document_id is None:
                document_id = db.get_document_id_by_job(job_id)
            if document_id:
                db.insert_job(document_id, user_id, status="queued", job_id=new_job_id)
        except Exception:
            log.exception("reanalyze: postgres job mirror failed")

    return {
        "job_id": new_job_id,
        "document_id": document_id,
        "trade": trade,
        "status": "queued",
        "queue_position": job_registry.queue_position(new_job_id),
        "running_jobs": job_registry.count_running_jobs(),
    }


@app.get("/api/jobs")
async def list_jobs(user_id: str = Depends(get_current_user)):
    """List the user's recent jobs."""
    jobs = job_registry.get_user_jobs(user_id)
    for job in jobs:
        if job.get("status") in ("queued", "running"):
            prog = job_registry.read_progress(job["job_id"])
            job["progress"] = prog
            if job["status"] == "queued":
                job["queue_position"] = job_registry.queue_position(job["job_id"])
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
async def get_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get job details. Includes results summary if completed."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    resp = dict(job)
    if job["status"] in ("queued", "running"):
        resp["progress"] = job_registry.read_progress(job_id)
        if job["status"] == "queued":
            resp["queue_position"] = job_registry.queue_position(job_id)
    elif job["status"] == "completed":
        results = job_registry.load_results(job_id)
        if results:
            resp["results_summary"] = {
                "fence_count": len(results.get("fence_pages", [])),
                "non_fence_count": len(results.get("non_fence_pages", [])),
                "total_pages": results.get("total_pages", 0),
                "has_element_details": bool(results.get("element_details")),
                "timings": results.get("timings", {}),
            }
    return resp


def _slim_results(results: dict) -> dict:
    """Strip megabytes of payload that the new web UI doesn't render but
    that bloat the response for large PDFs. For a 179-page run the raw
    JSON is ~780 MB; this trim brings it to ~5-15 MB.

    Removed:
      - unified_measurements (duplicate of fence_pages[].measurements,
        re-indexed by page_idx string — same data twice)
      - fence_pages[].ade_chunks (raw LandingAI response payload — not
        rendered anywhere; chunk metadata users care about is already
        in definitions/instances/legend_entries)
      - definition / instance .text bodies that are larger than 800
        chars (preserves indicator + bbox + a preview)

    Set ?full=1 on the request to get the unmodified payload (legacy
    Streamlit clients).
    """
    if not isinstance(results, dict):
        return results
    out = {k: v for k, v in results.items() if k != "unified_measurements"}

    fps = out.get("fence_pages")
    if isinstance(fps, list):
        slim_fps = []
        for p in fps:
            if not isinstance(p, dict):
                slim_fps.append(p)
                continue
            sp = {k: v for k, v in p.items() if k != "ade_chunks"}
            for field in ("definitions", "instances"):
                arr = sp.get(field)
                if isinstance(arr, list):
                    sp[field] = [_slim_chunk(c) for c in arr]
            # Strip the raw vector-line geometry from measurements — it's
            # internal pipeline data (used for the measurement calc, not
            # rendered in the UI). Can be ~4 MB per page on a 100-page run.
            m = sp.get("measurements")
            if isinstance(m, dict):
                sp["measurements"] = {
                    k: v for k, v in m.items() if k != "all_fence_lines"
                }
            slim_fps.append(sp)
        out["fence_pages"] = slim_fps
    return out


def _slim_chunk(c):
    if not isinstance(c, dict):
        return c
    out = dict(c)
    txt = out.get("text")
    if isinstance(txt, str) and len(txt) > 800:
        out["text"] = txt[:800] + "…"
        out["text_truncated"] = True
    return out


def _results_from_pages(job_id: str, user_id: str) -> dict | None:
    """Reconstruct a results payload from the persisted Postgres page_results
    when the per-job results.json is gone (the job TTL cleans up SQLite rows +
    results_dir, but page_results live on). Lets an old completed document
    still render its per-page results long after its job artifacts expired.

    Ownership is enforced by list_page_results (joins through documents.user_id).
    The cross-page extras that only existed in results.json (element specs,
    unified measurements, timings) come back empty — the per-page cards, which
    are the bulk of the view, are fully reconstructed."""
    try:
        document_id = db.get_document_id_by_job(job_id)
        rows = db.list_page_results(document_id, user_id) if document_id else None
    except Exception:
        log.exception("results reconstruction: page_results lookup failed")
        rows = None
    if not rows:
        return None

    fence_pages: list[dict] = []
    non_fence_pages: list[dict] = []
    broken_pages: list[dict] = []
    trade: str | None = None
    for r in rows:
        rj = r.get("result_json") or {}
        if isinstance(rj, dict):
            trade = trade or rj.get("_trade")
            if rj.get("phase") == "broken":
                broken_pages.append({
                    "page_idx": rj.get("page_idx"),
                    "page_num": rj.get("page_num") or r.get("page_number"),
                    "reason": rj.get("error") or "extraction failed",
                })
                continue
        (fence_pages if r.get("is_fence_page") else non_fence_pages).append(rj)

    return {
        "fence_pages": fence_pages,
        "non_fence_pages": non_fence_pages,
        "element_details": {},
        "per_page_scale_info": {},
        "unified_measurements": {},
        "page_categories": {},
        "total_pages": len(rows),
        "trade": trade or "fence",
        "broken_pages": broken_pages,
        "timings": {},
        "error": None,
        "reconstructed": True,
    }


@app.get("/api/jobs/{job_id}/results")
async def get_results(
    job_id: str,
    user_id: str = Depends(get_current_user),
    full: int = 0,
):
    """Get analysis results for a completed job.

    Returns a slim payload by default (drops `unified_measurements`,
    `ade_chunks`, and truncates oversized chunk text bodies). Pass
    ?full=1 for the unmodified original — needed by legacy Streamlit
    UMT and other older clients.

    Falls back to reconstructing from Postgres page_results when the job's
    on-disk results.json has been cleaned up by the TTL, so old documents
    don't render blank.
    """
    job = job_registry.get_job(job_id)
    if job is not None:
        if job["user_id"] != user_id:
            raise HTTPException(403, "Access denied")
        if job["status"] != "completed":
            raise HTTPException(409, f"Job is {job['status']}, not completed")
        results = job_registry.load_results(job_id)
        if results is not None:
            return results if full else _slim_results(results)

    # SQLite job row gone (TTL) or results.json missing — rebuild from the
    # persisted per-page rows in Postgres.
    rebuilt = _results_from_pages(job_id, user_id)
    if rebuilt is None:
        raise HTTPException(404, "Results not found")
    return rebuilt if full else _slim_results(rebuilt)


# A completed job's trade never changes, so cache it (avoids re-reading the
# multi-MB results.json on every /modes poll).
_job_trade_cache: dict[str, str] = {}


def _job_trade(job_id: str, user_id: str) -> str:
    """Best-effort analysis mode for a job. Cheapest source first: the SQLite
    job's config_json (small, present for any job within the registry TTL),
    then the saved results.json, then page_results reconstruction, then
    'fence'. Cached — a job's trade never changes."""
    if not job_id:
        return "fence"
    cached = _job_trade_cache.get(job_id)
    if cached:
        return cached
    trade = None
    # 1. cheap: SQLite job config_json
    try:
        j = job_registry.get_job(job_id)
        if j and j.get("config_json"):
            cfg = json.loads(j["config_json"])
            if cfg.get("trade"):
                trade = str(cfg["trade"])
    except Exception:
        pass
    # 2. saved results.json
    if not trade:
        res = job_registry.load_results(job_id)
        if isinstance(res, dict) and res.get("trade"):
            trade = str(res["trade"])
    # 3. reconstruct from page_results
    if not trade:
        rebuilt = _results_from_pages(job_id, user_id)
        if rebuilt and rebuilt.get("trade"):
            trade = str(rebuilt["trade"])
    trade = trade or "fence"
    _job_trade_cache[job_id] = trade
    return trade


@app.get("/api/documents/{document_id}/modes")
async def get_document_modes(
    document_id: str, user_id: str = Depends(get_current_user)
):
    """List the analysis modes a document has completed results for — the
    latest completed job per trade. Drives the Fence/Electrical tabs on the
    document view so both runs are viewable without re-analysing.

    No DB schema dependency: trades are read from each job's saved results
    (Path B). Returns e.g. {"modes": [{"trade":"fence","job_id":...}, ...]}.
    """
    try:
        jobs = db.list_document_jobs(document_id, user_id, only_completed=True)
    except Exception:
        log.exception("get_document_modes: list_document_jobs failed")
        jobs = []
    by_trade: dict[str, dict] = {}
    for j in jobs:  # newest first
        jid = str(j.get("id"))
        trade = _job_trade(jid, user_id)
        if trade not in by_trade:  # first seen per trade == newest
            by_trade[trade] = {
                "trade": trade,
                "job_id": jid,
                "created_at": j.get("created_at"),
            }
    modes = sorted(by_trade.values(), key=lambda m: (m["trade"] != "fence", m["trade"]))
    return {"modes": modes}


# Subprocess worker for safe PDF page rasterization. Some PDFs (especially
# those with very dense vector content) can crash MuPDF at the C level,
# which would take down the entire API server if rendered in-process.
# Running the render in a short-lived child means the worst case is a
# failed image, not a dead backend.
_PAGE_RENDER_SCRIPT = r"""
import sys, fitz
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass
pdf_path, page_num_str, dpi_str = sys.argv[1], sys.argv[2], sys.argv[3]
page_num = int(page_num_str)
dpi = int(dpi_str)
with fitz.open(pdf_path) as doc:
    if page_num < 1 or page_num > doc.page_count:
        sys.stderr.write(f'PAGE_OUT_OF_RANGE 1..{doc.page_count}')
        sys.exit(2)
    page = doc.load_page(page_num - 1)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    sys.stdout.buffer.write(pix.tobytes('png'))
"""


@app.get("/api/jobs/{job_id}/page-image/{page_num}")
async def get_page_image(
    job_id: str,
    page_num: int,
    dpi: int = 110,
    source: str = "auto",
    user_id: str = Depends(get_current_user),
):
    """Rasterize page `page_num` (1-indexed) and return it as a PNG.

    `source` selects which PDF to render from:
      - "auto" (default): highlighted PDF if the job has finished writing
        one, otherwise the original. Used by fence-page cards.
      - "original": always use the original uploaded PDF. Used by non-fence
        cards so the user sees the unannotated page.
      - "highlighted": always use the highlighted PDF (404 if missing).

    Renders in a subprocess (with timeout) so a MuPDF crash on a dense
    vector page doesn't kill the API.
    """
    if dpi < 50 or dpi > 200:
        raise HTTPException(400, "dpi must be between 50 and 200")
    if source not in ("auto", "original", "highlighted"):
        raise HTTPException(400, "source must be one of: auto, original, highlighted")
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    hl_path = job_registry.get_highlighted_pdf_path(job_id)
    orig_path = job.get("pdf_path")

    # The highlighted PDF only contains FENCE pages (renumbered sequentially).
    # Build an original-page-num → highlighted-page-index map so we can
    # render the correct page when serving from the highlighted PDF; without
    # this, requesting page 7 of the highlighted PDF would return the 7th
    # fence page in order, not the user's original page 7. For non-fence
    # pages there's nothing in the highlighted PDF — fall back to original.
    fence_page_pos: dict[int, int] = {}
    if hl_path is not None:
        results = job_registry.load_results(job_id)
        if results:
            fence_pages = sorted(
                (results.get("fence_pages") or []),
                key=lambda fp: fp.get("page_idx", fp.get("page_index_in_original_doc", 0)),
            )
            for i, fp in enumerate(fence_pages):
                pn = fp.get("page_num") or fp.get("page_number")
                if pn is not None:
                    fence_page_pos[int(pn)] = i

    if source == "highlighted":
        if hl_path is not None and page_num in fence_page_pos:
            src_pdf = hl_path
            page_in_pdf = fence_page_pos[page_num] + 1  # 1-indexed for renderer
        else:
            raise HTTPException(
                404,
                f"Page {page_num} is not in the highlighted PDF "
                "(only fence-classified pages have a highlighted version).",
            )
    elif source == "original":
        src_pdf = orig_path
        page_in_pdf = page_num
    else:  # "auto"
        if hl_path is not None and page_num in fence_page_pos:
            src_pdf = hl_path
            page_in_pdf = fence_page_pos[page_num] + 1
        else:
            src_pdf = orig_path
            page_in_pdf = page_num

    if not src_pdf or not Path(src_pdf).exists():
        raise HTTPException(
            404,
            f"PDF not available for source={source!r}"
            + (" — try again once upload finishes" if source == "auto" else ""),
        )

    import subprocess
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _PAGE_RENDER_SCRIPT,
                str(src_pdf),
                str(page_in_pdf),
                str(dpi),
            ],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            504,
            f"Page {page_num} took longer than 30s to render — likely a "
            "dense / damaged page. Try a lower DPI or skip this page.",
        )

    if proc.returncode == 2:
        raise HTTPException(404, proc.stderr.decode("utf-8", "ignore"))
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "ignore")[:200] or "unknown error"
        log.warning(
            "page-image render failed for job=%s page=%s rc=%s err=%s",
            job_id[:8], page_num, proc.returncode, err,
        )
        raise HTTPException(500, f"Page render failed: {err}")

    served_highlighted = src_pdf == hl_path
    return Response(
        content=proc.stdout,
        media_type="image/png",
        headers={
            # Don't cache long while the job is still running so the user
            # picks up the highlighted PDF once it lands. After completion
            # the response is stable so an hour is fine.
            "Cache-Control": (
                "private, max-age=3600" if served_highlighted else "private, max-age=10"
            ),
            "X-Page-Number": str(page_num),
            "X-Image-Source": "highlighted" if served_highlighted else "original",
        },
    )


def _clean_legend_entries(entries: list[dict]) -> list[dict]:
    """Drop the pipeline's "indicator code" placeholder rows (where the
    keyword is just the indicator number itself, or description is the
    literal "Indicator Code") and de-duplicate on (indicator, keyword)
    keeping the entry with the longer description."""
    filtered: list[dict] = []
    for le in entries:
        ind = (le.get("indicator") or "").strip()
        kw = (le.get("keyword") or "").strip()
        desc = (le.get("description") or "").strip()
        if ind and kw and ind == kw:
            continue
        if desc.lower() == "indicator code":
            continue
        filtered.append(le)
    by_key: dict[tuple[str, str], dict] = {}
    for le in filtered:
        ind = (le.get("indicator") or "").strip()
        kw = (le.get("keyword") or "").strip().upper()
        key = (ind, kw)
        prev = by_key.get(key)
        if prev is None or len((le.get("description") or "")) > len(
            prev.get("description") or ""
        ):
            by_key[key] = le
    return list(by_key.values())


@app.get("/api/jobs/{job_id}/page-vector-lines/{page_num}")
async def get_page_vector_lines(
    job_id: str,
    page_num: int,
    user_id: str = Depends(get_current_user),
):
    """Return all vector lines on a page (PDF display space) for the
    UMT canvas. Indices are stable for the lifetime of the job — frontend
    uses them as keys in `umt_state.line_assignments`."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    # Resolve a usable PDF on disk. /tmp gets reaped on reboot, so the
    # original may be gone; fall back to the highlighted PDF (lives under
    # ~/.leo/results/<job_id>/, persistent). Drawing-mode canvas still
    # works with the highlighted PDF; auto vector-line extraction may
    # pick up the cyan overlay strokes too — flagged via source_missing.
    pdf_path = job.get("pdf_path")
    source_missing = False
    if not pdf_path or not Path(pdf_path).exists():
        hl = job_registry.get_highlighted_pdf_path(job_id)
        if hl is not None and Path(hl).exists():
            pdf_path = str(hl)
            source_missing = True
        else:
            raise HTTPException(
                404,
                "Source PDF for this job is no longer on disk and no "
                "highlighted PDF was found. Re-upload the document to use "
                "the measurement canvas.",
            )

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available — job may still be running")

    fence_pages = (results or {}).get("fence_pages") or []
    target = next(
        (fp for fp in fence_pages
         if fp.get("page_num") == page_num or fp.get("page_number") == page_num),
        None,
    )
    # Resolve page_idx: prefer the entry from fence_pages (richer metadata for
    # auto-assignments); fall back to (page_num - 1) so the canvas works on
    # any page in the PDF, including ones the pipeline marked
    # measurement_skipped. Drawing mode is always useful even with no auto data.
    page_idx: int | None = None
    if target is not None:
        page_idx = target.get("page_idx")
        if page_idx is None:
            page_idx = target.get("page_index_in_original_doc")
    if page_idx is None:
        page_idx = page_num - 1

    import fitz
    from utils_vector import extract_vector_lines

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            raise HTTPException(404, f"Page index {page_idx} out of range")
        page = doc[page_idx]
        # page.rect already reflects display orientation (rotation applied)
        # in current PyMuPDF — matches the coord space `extract_vector_lines`
        # uses with `apply_rotation=True`.
        rect = page.rect
        rotation = page.rotation
        display_w, display_h = rect.width, rect.height
        vlines = extract_vector_lines(page, apply_rotation=True)
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    out_lines = []
    # Index vector lines by rounded endpoints (both orientations) so we can
    # match each pipeline-detected fence line to its source vector index.
    coord_to_idx: dict[tuple, int] = {}
    for i, ln in enumerate(vlines):
        sx, sy = float(ln.start[0]), float(ln.start[1])
        ex, ey = float(ln.end[0]), float(ln.end[1])
        out_lines.append({
            "idx": i,
            "start": [sx, sy],
            "end": [ex, ey],
            "length_pts": float(ln.length_pts),
            "layer": ln.layer or "",
        })
        # Round to 1dp — the same VectorLine that was extracted upstream
        # serialises through json.dumps without precision loss, so exact
        # match works for ~all lines that came out of extract_vector_lines.
        k1 = (round(sx, 1), round(sy, 1), round(ex, 1), round(ey, 1))
        k2 = (round(ex, 1), round(ey, 1), round(sx, 1), round(sy, 1))
        coord_to_idx.setdefault(k1, i)
        coord_to_idx.setdefault(k2, i)

    # Build auto-categories from legend entries (matches `umt.py` behaviour).
    PALETTE = [
        (0, 255, 0), (255, 165, 0), (0, 191, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 105, 180), (173, 255, 47),
    ]
    legend_entries = _clean_legend_entries((target or {}).get("legend_entries") or [])
    auto_categories: dict[str, dict] = {}
    for le in legend_entries:
        keyword = (le.get("keyword") or "").strip()
        indicator = (le.get("indicator") or "").strip()
        if not keyword:
            continue
        cat_name = f"{indicator}: {keyword}" if indicator else keyword
        if cat_name in auto_categories:
            continue
        c = PALETTE[len(auto_categories) % len(PALETTE)]
        auto_categories[cat_name] = {
            "indicator": indicator,
            "keyword": keyword,
            "color": [c[0], c[1], c[2]],
        }

    # Match pipeline auto fence lines → vector indices → category.
    measurements = (target or {}).get("measurements") or {}
    # Honour the pipeline's decision to skip auto-measurement on this page
    # (MAX_FENCE_LINES cap — see _auto_export_for_page / commit 3cca2bb).
    # Without this, opening the canvas on a skipped page seeds 18k+ noise
    # assignments that autosave persists into umt_state, and the UMT export
    # path re-creates the dense-page 504 the skip exists to prevent.
    measurement_skipped = measurements.get("measurement_method") == "skipped"
    all_fence_lines = [] if measurement_skipped else (
        measurements.get("all_fence_lines") or [])
    layer_to_cat: dict[str, str] = measurements.get("layer_to_category") or {}
    any_layer_mapped = bool(layer_to_cat)
    FALLBACK_CAT = "Auto-detected"

    auto_assignments: dict[str, str] = {}
    for fl in all_fence_lines:
        if not isinstance(fl, dict):
            continue
        s = fl.get("start"); e = fl.get("end")
        if not (isinstance(s, (list, tuple)) and isinstance(e, (list, tuple))):
            continue
        try:
            sx, sy = float(s[0]), float(s[1])
            ex, ey = float(e[0]), float(e[1])
        except (TypeError, ValueError):
            continue
        layer = fl.get("layer") or ""
        category = layer_to_cat.get(layer)
        if not category and layer and any_layer_mapped:
            for k, v in layer_to_cat.items():
                if k and (k in layer or layer in k):
                    category = v
                    break
        if not category:
            if any_layer_mapped:
                continue
            category = FALLBACK_CAT

        if category not in auto_categories:
            c = PALETTE[len(auto_categories) % len(PALETTE)]
            auto_categories[category] = {
                "indicator": "",
                "keyword": category,
                "color": [c[0], c[1], c[2]],
            }
        key = (round(sx, 1), round(sy, 1), round(ex, 1), round(ey, 1))
        vidx = coord_to_idx.get(key)
        if vidx is None:
            continue
        auto_assignments[str(vidx)] = category

    # When the source PDF is gone we read from the highlighted PDF, which
    # contains the cyan fence-overlay strokes. Auto-assignments computed
    # from coord-matching against `all_fence_lines` would be unreliable —
    # the user's *saved* assignments still apply because we always return
    # `lines` (indices match the highlighted PDF's vector order, which is
    # additive over the original).
    scale_info = (target or {}).get("scale_info") or {}
    verified_scale = scale_info.get("verified_scale") or scale_info.get("text_scale")
    return {
        "page_num": page_num,
        "page_idx": page_idx,
        "rotation": rotation,
        "pdf_width": float(display_w),
        "pdf_height": float(display_h),
        "verified_scale": float(verified_scale) if verified_scale else None,
        "lines": out_lines,
        "auto_categories": auto_categories,
        "auto_assignments": {} if source_missing else auto_assignments,
        "source_missing": source_missing,
        "measurement_skipped": measurement_skipped,
        "skip_reason": measurements.get("skip_reason") if measurement_skipped else None,
    }


@app.get("/api/jobs/{job_id}/page-vector-lines/{page_num}/smart-assign")
async def smart_assign_page(
    job_id: str,
    page_num: int,
    user_id: str = Depends(get_current_user),
    proximity_pts: float = 80.0,
    min_votes: int = 3,
    min_share: float = 0.5,
    min_participation: float = 0.05,
):
    """Compute *layer-level* category suggestions, not per-line.

    Strategy:
      1. For each line on a CAD layer, find the nearest indicator-bbox
         within `proximity_pts`. Cast one vote for that indicator on
         the line's layer.
      2. Per layer, the indicator with the most votes wins. All lines
         on that layer are assigned to the corresponding category.
      3. Layers with zero votes fall back to a *strict* layer-token
         match: a category wins only if exactly one category has a
         5+ char keyword token that appears in the layer name.
         Ambiguous matches are skipped (left unassigned) to keep the
         output reliable.

    Returns {assignments: {line_idx: category_name},
             layer_assignments: {layer_name: {indicator, category, votes, line_count}},
             stats: {...}}.
    """
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        hl = job_registry.get_highlighted_pdf_path(job_id)
        if hl is not None and Path(hl).exists():
            pdf_path = str(hl)
        else:
            raise HTTPException(404, "No PDF on disk for this job")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available")
    fence_pages = (results or {}).get("fence_pages") or []
    target = next(
        (fp for fp in fence_pages
         if fp.get("page_num") == page_num or fp.get("page_number") == page_num),
        None,
    )
    page_idx: int | None = None
    if target is not None:
        page_idx = target.get("page_idx")
        if page_idx is None:
            page_idx = target.get("page_index_in_original_doc")
    if page_idx is None:
        page_idx = page_num - 1

    import fitz
    from utils_vector import extract_vector_lines

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            raise HTTPException(404, "Page out of range")
        page = doc[page_idx]
        rmat = page.rotation_matrix  # native -> display
        vlines = extract_vector_lines(page, apply_rotation=True)
        # Transform instance bboxes from native to display space.
        instances_disp: list[dict] = []
        for inst in (target or {}).get("instances") or []:
            ind = (inst.get("indicator") or "").strip()
            if not ind:
                continue
            try:
                rect = fitz.Rect(
                    float(inst["x0"]), float(inst["y0"]),
                    float(inst["x1"]), float(inst["y1"]),
                ) * rmat
                rect.normalize()
            except (KeyError, TypeError, ValueError):
                continue
            instances_disp.append({
                "indicator": ind,
                "cx": (rect.x0 + rect.x1) / 2,
                "cy": (rect.y0 + rect.y1) / 2,
            })
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    # Build category lookup tables from legend.
    legend_entries = _clean_legend_entries((target or {}).get("legend_entries") or [])
    indicator_to_category: dict[str, str] = {}
    # token -> set of categories that contain it (for ambiguity check)
    token_to_categories: dict[str, set[str]] = {}
    for le in legend_entries:
        keyword = (le.get("keyword") or "").strip()
        indicator = (le.get("indicator") or "").strip()
        if not keyword:
            continue
        cat_name = f"{indicator}: {keyword}" if indicator else keyword
        if indicator:
            indicator_to_category.setdefault(indicator, cat_name)
        seen_in_kw: set[str] = set()
        for raw in keyword.upper().split():
            tok = "".join(ch for ch in raw if ch.isalnum())
            # Strict: 5+ chars, not seen twice in the same keyword (avoid
            # double-counting); we'll only use unambiguous tokens later.
            if len(tok) >= 5 and tok not in seen_in_kw:
                seen_in_kw.add(tok)
                token_to_categories.setdefault(tok, set()).add(cat_name)

    # Group lines by layer.
    from collections import Counter, defaultdict
    layer_lines: dict[str, list[tuple[int, "object"]]] = defaultdict(list)
    for i, ln in enumerate(vlines):
        layer_lines[(ln.layer or "")].append((i, ln))

    prox_sq = proximity_pts * proximity_pts

    # Per-layer voting: for each line, find its nearest indicator within
    # the threshold; tally per-layer votes by indicator.
    layer_votes: dict[str, Counter] = {}
    for layer_name, items in layer_lines.items():
        votes: Counter = Counter()
        for _, ln in items:
            sx, sy = float(ln.start[0]), float(ln.start[1])
            ex, ey = float(ln.end[0]), float(ln.end[1])
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            best_d2 = prox_sq
            best_ind: str | None = None
            for inst in instances_disp:
                dx = inst["cx"] - mx
                dy = inst["cy"] - my
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_ind = inst["indicator"]
            if best_ind is not None:
                votes[best_ind] += 1
        layer_votes[layer_name] = votes

    # Decide each layer's category.
    layer_assignments: dict[str, dict] = {}
    for layer_name, items in layer_lines.items():
        if not layer_name:
            continue
        line_count = len(items)
        votes = layer_votes.get(layer_name) or Counter()
        winner_ind = None
        winner_votes = 0
        cat: str | None = None
        decided_by = "none"

        total_v = sum(votes.values()) if votes else 0
        participation = total_v / line_count if line_count else 0.0
        if votes:
            (winner_ind, winner_votes), *_ = votes.most_common(1)
            share = winner_votes / total_v if total_v else 0.0
            if (
                winner_votes >= min_votes
                and share >= min_share
                and participation >= min_participation
                and indicator_to_category.get(winner_ind)
            ):
                cat = indicator_to_category[winner_ind]
                decided_by = "indicator-vote"

        # Strict layer-token fallback when voting was inconclusive.
        if cat is None:
            up = layer_name.upper()
            unique_hit: str | None = None
            ambiguous = False
            for tok, cats in token_to_categories.items():
                if tok in up:
                    if len(cats) > 1:
                        ambiguous = True
                        unique_hit = None
                        break
                    only = next(iter(cats))
                    if unique_hit is None:
                        unique_hit = only
                    elif unique_hit != only:
                        ambiguous = True
                        unique_hit = None
                        break
            if unique_hit and not ambiguous:
                cat = unique_hit
                decided_by = "layer-token"

        layer_assignments[layer_name] = {
            "indicator": winner_ind,
            "category": cat,
            "votes": winner_votes,
            "total_votes": total_v,
            "line_count": line_count,
            "participation": round(participation, 3),
            "decided_by": decided_by,
        }

    # Materialize per-line assignments from layer assignments.
    assignments: dict[str, str] = {}
    by_indicator = 0
    by_layer = 0
    for layer_name, items in layer_lines.items():
        info = layer_assignments.get(layer_name)
        if not info or not info.get("category"):
            continue
        cat = info["category"]
        for i, _ in items:
            assignments[str(i)] = cat
        if info["decided_by"] == "indicator-vote":
            by_indicator += len(items)
        elif info["decided_by"] == "layer-token":
            by_layer += len(items)

    return {
        "assignments": assignments,
        "layer_assignments": layer_assignments,
        "stats": {
            "by_indicator": by_indicator,
            "by_layer": by_layer,
            "unassigned": len(vlines) - len(assignments),
            "total": len(vlines),
            "instance_count": len(instances_disp),
            "layer_count": len([n for n in layer_lines if n]),
            "layers_assigned": sum(
                1 for v in layer_assignments.values() if v.get("category")
            ),
            "proximity_pts": proximity_pts,
            "min_votes": min_votes,
            "min_share": min_share,
            "min_participation": min_participation,
        },
    }


@app.get("/api/jobs/{job_id}/highlighted-pdf")
async def get_highlighted_pdf(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Download the highlighted PDF for a completed job."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    hl_path = job_registry.get_highlighted_pdf_path(job_id)
    if hl_path is None:
        raise HTTPException(404, "Highlighted PDF not available")

    return FileResponse(
        str(hl_path),
        media_type="application/pdf",
        filename=f"fence_{job.get('filename', 'document')}.pdf",
    )


# ---------------------------------------------------------------------------
# Sprint 3 / C5 + C6 — Measurement export endpoints
# ---------------------------------------------------------------------------

# Default category palette mirrors prod's CATEGORY_COLORS at
# app_ade_prod.py:4844 so that lacking-UMT exports still get distinguishable
# line colors per fence type.
_CATEGORY_PALETTE = [
    (0, 255, 0),     (255, 165, 0),  (0, 191, 255),  (255, 0, 255),
    (255, 255, 0),   (0, 255, 255),  (255, 105, 180), (173, 255, 47),
]

# Ceiling on lines exported from one page via saved UMT state — matches
# MAX_FENCE_LINES in utils_ade/measure.py. See _umt_export_for_page.
MAX_EXPORT_LINES_PER_PAGE = 5000


def _scale_inches_to_points_per_foot(scale_inches: Any) -> float:
    """Convert architectural scale inches to PDF points per real foot.

    Scale detection returns the real-world inches represented by 1 drawing
    inch, e.g. 360 for `1" = 30'`. PDF coordinates are points, where
    1 drawing inch is 72 points. Therefore:

        feet = points / 72 * scale_inches / 12
             = points / (864 / scale_inches)
    """
    try:
        scale = float(scale_inches)
    except (TypeError, ValueError):
        scale = 360.0
    if scale <= 0:
        scale = 360.0
    return 864.0 / scale


def _auto_export_for_page(fp: dict) -> tuple[dict, dict, list]:
    """Auto-only export shape for one fence page (no UMT edits).

    Returns (cats, page_assignments, auto_lines) keyed for this page.
    Mirrors prod's partial-layer-match + "Auto-detected" fallback.
    """
    cats: dict[str, dict] = {}
    for le in _clean_legend_entries(fp.get("legend_entries") or []):
        keyword = (le.get("keyword") or "").strip()
        indicator = (le.get("indicator") or "").strip()
        if not keyword:
            continue
        cat_name = f"{indicator}: {keyword}" if indicator else keyword
        if cat_name in cats:
            continue
        cats[cat_name] = {
            "indicator": indicator,
            "keyword": keyword,
            "color": _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)],
        }

    measurements = fp.get("measurements") or {}
    # Honour the pipeline's decision to skip auto-measurement on this page
    # (utils_ade/measure.py caps at MAX_FENCE_LINES=5000). The skipped result
    # still carries the raw all_fence_lines list — 18k+ segments of dense
    # hatch/site-plan noise on real decks — and exporting it made the
    # measurement-PDF render chew >10 min and 504 (2026-07-13, 425-page GMP3
    # set). Skipped pages contribute categories only, no lines.
    if measurements.get("measurement_method") == "skipped":
        return cats, {}, []
    all_lines = measurements.get("all_fence_lines") or []
    layer_to_cat = measurements.get("layer_to_category") or {}
    any_layer_mapped = bool(layer_to_cat)
    FALLBACK_CAT = "Auto-detected"

    page_assignments: dict[str, str] = {}
    auto_lines: list[dict] = []
    for line in all_lines:
        if not isinstance(line, dict):
            continue
        layer = line.get("layer") or ""
        category = layer_to_cat.get(layer)
        if not category and layer and any_layer_mapped:
            for k, v in layer_to_cat.items():
                if k and (k in layer or layer in k):
                    category = v
                    break
        if not category:
            if any_layer_mapped:
                continue
            category = FALLBACK_CAT
        if category not in cats:
            cats[category] = {
                "indicator": "",
                "keyword": category,
                "color": _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)],
            }
        # Key by the auto_lines index (post-filter), not the all_lines
        # index — exports.generate_measurement_pdf indexes into auto_lines.
        page_assignments[str(len(auto_lines))] = category
        auto_lines.append(line)

    return cats, page_assignments, auto_lines


def _umt_export_for_page(
    page_state: dict,
    vlines_dicts: list[dict],
) -> tuple[dict, dict, list, list]:
    """UMT-driven export shape for one fence page.

    Returns (cats, page_assignments, auto_lines, user_drawn_lines).
    Drops indicator-code placeholder categories and orphaned assignments
    (saved indices that fall outside the current vector-line list, e.g.
    after a re-run). Re-keys assignments by auto_lines index to match
    exports.generate_measurement_pdf's iteration.
    """
    raw_cats = page_state.get("categories") or {}
    cats: dict[str, dict] = {}
    for cat_name, cat_info in raw_cats.items():
        ind = (cat_info.get("indicator") or "").strip()
        kw = (cat_info.get("keyword") or "").strip()
        if ind and kw and ind == kw:
            continue  # indicator-code placeholder
        color = cat_info.get("color")
        if isinstance(color, list) and len(color) == 3:
            color_t = (int(color[0]), int(color[1]), int(color[2]))
        else:
            color_t = _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)]
        cats[cat_name] = {
            "indicator": ind,
            "keyword": kw,
            "color": color_t,
        }

    page_assignments: dict[str, str] = {}
    auto_lines: list[dict] = []
    saved_assignments = page_state.get("line_assignments") or {}
    for idx_str, cat in saved_assignments.items():
        # Cap what a single page can export. umt_state files saved before the
        # page-vector-lines skipped-page guard can carry 18k+ noise
        # assignments (dense hatch seeded on a measurement-skipped page);
        # exporting them re-creates the dense-page 504 that commit 3cca2bb
        # fixed on the auto path. 5000 matches MAX_FENCE_LINES in
        # utils_ade/measure.py.
        if len(auto_lines) >= MAX_EXPORT_LINES_PER_PAGE:
            log.warning(
                "UMT export: capping page assignments at %d (saved %d)",
                MAX_EXPORT_LINES_PER_PAGE, len(saved_assignments),
            )
            break
        try:
            i = int(idx_str)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= len(vlines_dicts):
            continue
        if cat not in cats:
            cats[cat] = {
                "indicator": "",
                "keyword": cat,
                "color": _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)],
            }
        page_assignments[str(len(auto_lines))] = cat
        auto_lines.append(vlines_dicts[i])

    user_drawn = []
    for ul in page_state.get("user_drawn_lines") or []:
        cat = ul.get("category")
        if not cat:
            continue
        try:
            sx = float(ul["start"][0]); sy = float(ul["start"][1])
            ex = float(ul["end"][0]); ey = float(ul["end"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if cat not in cats:
            cats[cat] = {
                "indicator": "",
                "keyword": cat,
                "color": _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)],
            }
        user_drawn.append({"start": [sx, sy], "end": [ex, ey], "category": cat})

    return cats, page_assignments, auto_lines, user_drawn


def _line_assignment_export_pages(results: dict, pages_node: dict) -> list[tuple[int, int, dict]]:
    """Return pages whose saved UMT line assignments require PDF vector lookup."""
    out: list[tuple[int, int, dict]] = []
    for fp in (results or {}).get("fence_pages") or []:
        if not isinstance(fp, dict):
            continue
        page_num = fp.get("page_num") or fp.get("page_number")
        page_idx = fp.get("page_idx")
        if page_idx is None:
            page_idx = fp.get("page_index_in_original_doc")
        if page_num is None or page_idx is None:
            continue
        page_state = pages_node.get(f"page_{page_num}") or {}
        if page_state.get("line_assignments"):
            out.append((int(page_idx), int(page_num), page_state))
    return out


def _extract_export_vector_lines(
    *,
    job_id: str,
    pdf_path: str | None,
    page_indices: list[int],
) -> dict[int, list[dict]]:
    """Resolve UMT line indices via a subprocess-isolated PyMuPDF worker."""
    unique_indices = sorted({int(pi) for pi in page_indices})
    if not unique_indices:
        return {}
    if not pdf_path or not Path(pdf_path).exists():
        raise ExportWorkerError("Source PDF is required to resolve saved UMT edits")
    if not _EXPORT_VECTOR_WORKER.exists():
        raise ExportWorkerError(f"Export vector worker missing: {_EXPORT_VECTOR_WORKER}")

    t0 = time.perf_counter()
    log.info(
        "export vector extraction start job=%s pages=%d",
        job_id[:8],
        len(unique_indices),
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(_EXPORT_VECTOR_WORKER)],
            input=json.dumps(
                {"pdf_path": pdf_path, "page_indices": unique_indices},
                default=str,
            ),
            capture_output=True,
            text=True,
            timeout=cfg.HIGHLIGHT_PDF_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        log.warning(
            "export vector extraction timeout job=%s pages=%d wall=%.1fs cap=%ss",
            job_id[:8],
            len(unique_indices),
            elapsed,
            cfg.HIGHLIGHT_PDF_TIMEOUT,
        )
        raise ExportWorkerTimeout(
            f"Vector extraction exceeded {cfg.HIGHLIGHT_PDF_TIMEOUT}s"
        )

    payload = None
    for line in (proc.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue

    if proc.returncode != 0 or not payload or not payload.get("ok"):
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        if not err:
            err = (proc.stderr or "export vector worker failed").strip()[:500]
        log.warning(
            "export vector extraction failed job=%s pages=%d rc=%s error=%s",
            job_id[:8],
            len(unique_indices),
            proc.returncode,
            err,
        )
        raise ExportWorkerError(str(err))

    raw = payload.get("lines_by_page") or {}
    lines_by_page = {int(k): (v or []) for k, v in raw.items()}
    log.info(
        "export vector extraction done job=%s pages=%d wall=%ss",
        job_id[:8],
        len(unique_indices),
        payload.get("wall_s"),
    )
    return lines_by_page


def _peek_page_count(pdf_path: str) -> int:
    """Count a freshly-uploaded PDF's pages in a subprocess with a hard
    timeout. Isolating this keeps a malformed PDF that hangs PyMuPDF from
    wedging the API event loop (the same C-level recovery-loop hazard the
    pipeline already isolates in Phase 1a). Raises ExportWorkerTimeout on
    hang and ExportWorkerError on any read failure — both map to a 4xx in
    create_job."""
    if not _PAGECOUNT_WORKER.exists():
        raise ExportWorkerError(f"Page-count worker missing: {_PAGECOUNT_WORKER}")
    try:
        proc = subprocess.run(
            [sys.executable, str(_PAGECOUNT_WORKER)],
            input=json.dumps({"pdf_path": pdf_path}, default=str),
            capture_output=True,
            text=True,
            timeout=_PAGECOUNT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise ExportWorkerTimeout(
            f"page-count exceeded {_PAGECOUNT_TIMEOUT}s (file may be damaged)"
        )

    payload = None
    for line in (proc.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue

    if proc.returncode != 0 or not payload or not payload.get("ok"):
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        if not err:
            err = (proc.stderr or "page-count worker failed").strip()[:500]
        raise ExportWorkerError(str(err))

    return int(payload.get("page_count") or 0)


def _build_export_state(
    job_id: str,
    results: dict,
    pdf_path: str | None,
) -> tuple[list, dict, dict, dict, dict, dict]:
    """Build the export tuple for `exports.generate_measurement_*`.

    Per page: if `umt_state.json` has saved edits, recompute the page's
    categories / line_assignments / user_drawn_lines from saved state
    (re-extracting vector lines from `pdf_path` to resolve indices).
    Otherwise fall back to pipeline auto detection.

    Returns: (fence_pages_with_auto_lines, line_assignments,
    user_drawn_lines, page_categories, per_page_scale_info, lines_by_page).
    """
    from backend.app import umt_state as umt_state_mod

    fence_pages_in = (results or {}).get("fence_pages") or []
    per_page_scale = (results or {}).get("per_page_scale_info") or {}
    umt = umt_state_mod.load(job_id)
    pages_node = (umt or {}).get("pages") or {}

    # Pre-extract vector lines for any page that has UMT edits. This must
    # stay out of the FastAPI process: PyMuPDF can hang in C on malformed
    # content and freeze every endpoint, including /api/healthz.
    umt_lines_by_idx: dict[int, list[dict]] = {}
    needs_extraction = _line_assignment_export_pages(results, pages_node)
    if needs_extraction:
        umt_lines_by_idx = _extract_export_vector_lines(
            job_id=job_id,
            pdf_path=pdf_path,
            page_indices=[page_idx for page_idx, _page_num, _state in needs_extraction],
        )

    line_assignments: dict[str, dict[str, str]] = {}
    user_drawn_lines: dict[str, list] = {}
    page_categories: dict[str, dict] = {}
    lines_by_page: dict[str, list] = {}
    fence_pages_out: list[dict] = []

    for fp in fence_pages_in:
        if not isinstance(fp, dict):
            continue
        page_num = fp.get("page_num") or fp.get("page_number")
        if page_num is None:
            continue
        page_key = f"page_{page_num}"
        page_state = pages_node.get(page_key)
        has_user_edits = bool(
            page_state
            and (page_state.get("line_assignments") or page_state.get("user_drawn_lines"))
        )

        if has_user_edits:
            page_idx = fp.get("page_idx")
            if page_idx is None:
                page_idx = fp.get("page_index_in_original_doc")
            vlines_dicts = (
                umt_lines_by_idx.get(int(page_idx)) if page_idx is not None else []
            ) or []
            cats, page_assignments, auto_lines, udl = _umt_export_for_page(
                page_state, vlines_dicts
            )
            if udl:
                user_drawn_lines[page_key] = udl
        else:
            cats, page_assignments, auto_lines = _auto_export_for_page(fp)

        page_categories[page_key] = cats
        if page_assignments:
            line_assignments[page_key] = page_assignments
        if auto_lines:
            lines_by_page[page_key] = auto_lines

        # Carry the per-page scale override from umt_state if set.
        if page_state and page_state.get("scale_override"):
            override = page_state.get("scale_override")
            if isinstance(override, (int, float)) and override > 0:
                ps = dict(per_page_scale.get(page_key) or {})
                ps["verified_scale"] = float(override)
                per_page_scale = {**per_page_scale, page_key: ps}

        fp_out = dict(fp)
        fp_out["auto_lines"] = auto_lines
        fence_pages_out.append(fp_out)

    return (
        fence_pages_out,
        line_assignments,
        user_drawn_lines,
        page_categories,
        per_page_scale,
        lines_by_page,
    )


def _generate_measurement_pdf_in_subprocess(
    *,
    pdf_path: str,
    fence_pages: list,
    line_assignments: dict,
    user_drawn_lines: dict,
    page_categories: dict,
    uploaded_pdf_name: str,
    out_path: str,
) -> None:
    """Run measurement-PDF generation outside uvicorn, writing to out_path.

    PyMuPDF can enter unbounded C-level loops on malformed content streams.
    If that happens in the API process, even `/api/healthz` stops responding.
    The worker process gives us an OS-level kill switch.

    The worker writes the finished PDF to out_path atomically (.tmp +
    rename), so out_path either doesn't exist or is complete — the bytes
    never pass through this process's memory (a 224-page deck produces a
    100+ MB artifact).
    """
    from exports import MIN_LINE_PTS

    if not _MEASUREMENT_PDF_WORKER.exists():
        raise RuntimeError(f"Measurement PDF worker missing: {_MEASUREMENT_PDF_WORKER}")

    task = {
        "pdf_path": pdf_path,
        "out_path": out_path,
        "fence_pages": fence_pages,
        "line_assignments": line_assignments,
        "user_drawn_lines": user_drawn_lines,
        "page_categories": page_categories,
        "uploaded_pdf_name": uploaded_pdf_name,
        "min_line_pts": MIN_LINE_PTS,
        "max_labels_per_page": 150,
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(_MEASUREMENT_PDF_WORKER)],
            input=json.dumps(task, default=str),
            capture_output=True,
            text=True,
            timeout=cfg.HIGHLIGHT_PDF_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        # Only the .tmp can be partial — the rename to out_path is atomic,
        # so if the worker got that far the artifact is complete and kept.
        try:
            Path(out_path + ".tmp").unlink(missing_ok=True)
        except Exception:
            pass
        raise TimeoutError(
            f"Measurement PDF generation exceeded {cfg.HIGHLIGHT_PDF_TIMEOUT}s"
        )

    payload = None
    for line in (proc.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue

    if proc.returncode != 0 or not payload or not payload.get("ok"):
        err = (payload or {}).get("error") if isinstance(payload, dict) else None
        if not err:
            err = (proc.stderr or "measurement worker failed").strip()[:500]
        try:
            Path(out_path + ".tmp").unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(err)

    out = Path(out_path)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("Measurement PDF worker produced no output file")


# Returned (as HTTP 422, not 500) when a document has no fence lines to
# export — neither auto-detected nor manually drawn. This is a normal
# outcome (e.g. a report whose "fence" pages have no CAD fence layers), not
# a server fault, so it must not be retried as a transient 5xx by the client.
NO_MEASUREMENTS_MSG = (
    "No measurements to export — no fence lines were detected on this "
    "document. Draw or assign lines in the measurement editor to populate "
    "this export."
)


# Measurement-PDF builds run in a background thread and land as an on-disk
# artifact (<results_dir>/measurement.pdf) served via FileResponse. The old
# build-inside-the-GET design could not work for big decks: nginx cuts the
# connection at proxy_read_timeout (600s) while the build legitimately runs
# longer, the browser surfaced "Failed to fetch", and every retry spawned
# another full rebuild (2026-07-17: a 224-page set left two orphaned ~20 GB
# workers building PDFs nobody could receive). Now the GET returns 202 while
# a build is running, the frontend polls, and the finished artifact is served
# in ranged chunks and cached for later clicks.
#
# job_id -> {"status": "building", "stale": bool}
#         | {"status": "failed", "code": int, "detail": str}
# The dict is process-local, which is safe because uvicorn runs one worker.
_measurement_builds: dict[str, dict] = {}
_measurement_builds_lock = threading.Lock()


def _measurement_pdf_artifact(job: dict) -> Path | None:
    """Where this job's measurement-PDF artifact lives (may not exist yet)."""
    if not job.get("results_dir"):
        return None
    return Path(job["results_dir"]) / "measurement.pdf"


def invalidate_measurement_pdf(job_id: str) -> None:
    """Drop the cached measurement-PDF artifact after a UMT edit.

    If a build is currently running it is marked stale instead — its output
    would bake in the pre-edit line set, so the builder discards it on
    completion and the next poll starts a fresh build.
    """
    with _measurement_builds_lock:
        state = _measurement_builds.get(job_id)
        if state and state.get("status") == "building":
            state["stale"] = True
        else:
            _measurement_builds.pop(job_id, None)
    job = job_registry.get_job(job_id)
    artifact = _measurement_pdf_artifact(job) if job else None
    if artifact is not None:
        try:
            artifact.unlink(missing_ok=True)
        except OSError as e:
            log.warning("measurement-pdf invalidate failed job=%s: %s", job_id[:8], e)


def _measurement_pdf_build(
    job_id: str,
    results: dict,
    pdf_path: str,
    uploaded_name: str,
    artifact: Path,
) -> None:
    """Build the measurement-PDF artifact (runs on a daemon thread)."""
    code, detail = 500, "Measurement PDF generation failed"
    try:
        fence_pages_out, line_assignments, user_drawn, page_cats, _scale, _lines = (
            _build_export_state(job_id, results, pdf_path)
        )
        # No auto-detected or user-drawn lines anywhere → nothing to overlay.
        # Short-circuit before the subprocess, which would otherwise spend
        # minutes re-rendering dense pages just to produce a blank overlay.
        if not line_assignments and not user_drawn:
            log.info("measurement-pdf no-lines job=%s", job_id[:8])
            code, detail = 422, NO_MEASUREMENTS_MSG
            raise _MeasurementBuildFailed()
        _generate_measurement_pdf_in_subprocess(
            pdf_path=pdf_path,
            fence_pages=fence_pages_out,
            line_assignments=line_assignments,
            user_drawn_lines=user_drawn,
            page_categories=page_cats,
            uploaded_pdf_name=uploaded_name,
            out_path=str(artifact),
        )
    except _MeasurementBuildFailed:
        pass
    except ExportWorkerTimeout as e:
        log.warning("measurement-pdf vector timeout job=%s: %s", job_id[:8], e)
        code, detail = 504, str(e)
    except ExportWorkerError as e:
        log.warning("measurement-pdf vector failed job=%s: %s", job_id[:8], e)
        code, detail = 500, f"Measurement PDF export state failed: {e}"
    except TimeoutError as e:
        log.warning("measurement-pdf timed out job=%s: %s", job_id[:8], e)
        code, detail = 504, str(e)
    except Exception as e:
        log.warning("measurement-pdf failed job=%s: %s", job_id[:8], e)
        code, detail = 500, f"Measurement PDF generation failed: {e}"
    else:
        with _measurement_builds_lock:
            state = _measurement_builds.pop(job_id, None)
            stale = bool(state and state.get("stale"))
        if stale:
            # A UMT edit landed mid-build; this output no longer matches.
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                pass
            log.info("measurement-pdf stale discard job=%s", job_id[:8])
        else:
            log.info(
                "measurement-pdf done job=%s bytes=%d",
                job_id[:8],
                artifact.stat().st_size if artifact.exists() else 0,
            )
        return

    # Failure path: keep the error for the next poll to surface — unless the
    # state went stale, in which case just clear so the next poll rebuilds.
    with _measurement_builds_lock:
        state = _measurement_builds.get(job_id)
        if state and state.get("stale"):
            _measurement_builds.pop(job_id, None)
        else:
            _measurement_builds[job_id] = {
                "status": "failed", "code": code, "detail": detail,
            }


class _MeasurementBuildFailed(Exception):
    """Internal: jump to the failure path with code/detail already set."""


@app.get("/api/jobs/{job_id}/measurement-pdf")
async def get_measurement_pdf(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Measurement-overlay PDF (Sprint 3 / C5).

    Serves the cached artifact when present (FileResponse → ranged chunks
    work). Otherwise starts a background build and answers 202; the client
    polls until the artifact is ready or the build records a failure."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    artifact = _measurement_pdf_artifact(job)
    if artifact is not None and artifact.exists():
        base = (job.get("filename") or "document.pdf").rsplit(".", 1)[0]
        return FileResponse(
            str(artifact),
            media_type="application/pdf",
            filename=f"{base}_measurement.pdf",
            headers={"Cache-Control": "private, no-store"},
        )

    with _measurement_builds_lock:
        state = _measurement_builds.get(job_id)
        if state and state["status"] == "building":
            return JSONResponse(
                {"status": "building", "detail": "Measurement PDF is being prepared"},
                status_code=202,
            )
        if state and state["status"] == "failed":
            # Surface once, then clear so the next request retries the build.
            _measurement_builds.pop(job_id, None)
            raise HTTPException(state["code"], state["detail"])

    # PDF generation overlays measurement lines on top of the page's
    # native vector content. If the original PDF is gone we can't
    # cleanly fall back to the highlighted PDF — it would carry the cyan
    # fence overlays into the output. Ask the user to re-upload instead.
    # (Excel works either way — it only consumes coords, not the visual
    # content.)
    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(
            404,
            "Source PDF for this job is no longer on disk. "
            "Re-upload the document to generate a measurement PDF "
            "(the Measurement Excel still works without the source).",
        )

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available — job may still be running")

    if artifact is None:
        raise HTTPException(500, "Job has no results directory for the artifact")

    with _measurement_builds_lock:
        # Re-check under the lock — another request may have won the race.
        if _measurement_builds.get(job_id, {}).get("status") == "building":
            return JSONResponse(
                {"status": "building", "detail": "Measurement PDF is being prepared"},
                status_code=202,
            )
        _measurement_builds[job_id] = {"status": "building", "stale": False}

    log.info("measurement-pdf build start job=%s", job_id[:8])
    threading.Thread(
        target=_measurement_pdf_build,
        args=(job_id, results, pdf_path,
              job.get("filename") or "document.pdf", artifact),
        name=f"measurement-pdf-{job_id[:8]}",
        daemon=True,
    ).start()
    return JSONResponse(
        {"status": "building", "detail": "Measurement PDF is being prepared"},
        status_code=202,
    )


@app.get("/api/jobs/{job_id}/measurement-excel")
async def get_measurement_excel(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Measurement Excel (Sprint 3 / C6, UMT-aware after Sprint 4 4a.8).

    Pulls saved UMT edits from `umt_state.json` per page when present,
    otherwise falls back to auto-detected fence lines from the pipeline.
    """
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        hl = job_registry.get_highlighted_pdf_path(job_id)
        if hl is not None and Path(hl).exists():
            pdf_path = str(hl)
        # Excel doesn't need the PDF unless UMT edits exist; if none of
        # the pages have edits, _build_export_state will skip extraction.

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available — job may still be running")

    log.info("measurement-excel start job=%s", job_id[:8])
    try:
        fence_pages_out, line_assignments, user_drawn, page_cats, scale_info, lines_by_page = (
            await asyncio.to_thread(_build_export_state, job_id, results, pdf_path)
        )
    except ExportWorkerTimeout as e:
        log.warning("measurement-excel vector timeout job=%s: %s", job_id[:8], e)
        raise HTTPException(504, str(e))
    except ExportWorkerError as e:
        log.warning("measurement-excel vector failed job=%s: %s", job_id[:8], e)
        raise HTTPException(500, f"Measurement Excel export state failed: {e}")

    # No auto-detected or user-drawn lines anywhere → no rows to write.
    if not line_assignments and not user_drawn:
        log.info("measurement-excel no-lines job=%s", job_id[:8])
        raise HTTPException(422, NO_MEASUREMENTS_MSG)

    element_details = (results or {}).get("element_details") or {}

    from exports import generate_measurement_spreadsheet
    xlsx_bytes = await asyncio.to_thread(
        generate_measurement_spreadsheet,
        fence_pages=fence_pages_out,
        line_assignments=line_assignments,
        user_drawn_lines=user_drawn,
        page_categories=page_cats,
        per_page_scale_info=scale_info,
        element_details=element_details,
        lines_by_page=lines_by_page,
    )
    if not xlsx_bytes:
        # Belt-and-suspenders: the early no-lines check above normally
        # catches this, but generate_measurement_spreadsheet can also drop
        # every candidate row (e.g. all lines below min length). Still a
        # "nothing to export" outcome, not a server fault.
        raise HTTPException(422, NO_MEASUREMENTS_MSG)

    base = (job.get("filename") or "document.pdf").rsplit(".", 1)[0]
    fname = f"{base}_measurements.xlsx"
    log.info(
        "measurement-excel done job=%s pages=%d bytes=%d",
        job_id[:8],
        len(fence_pages_out),
        len(xlsx_bytes),
    )
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control": "private, no-store",
        },
    )


def _summary_for_page(
    page_idx: int,
    fp: dict,
    page_state: dict | None,
    vlines: list[dict] | None = None,
) -> dict:
    """Compute per-category measurement totals for one fence page.

    If `page_state` (from `umt_state.json`) carries any saved
    line_assignments or user_drawn_lines, totals are recomputed from
    those edits. Otherwise we fall back to the pipeline's auto totals
    (`measurements.all_fence_lines` + `layer_to_category`).

    Returns: {scale, has_user_edits, per_category: {cat: {pts, ft, auto, manual}}}.
    """
    scale_info = (fp or {}).get("scale_info") or {}
    page_scale_override = (page_state or {}).get("scale_override")
    scale_inches = (
        page_scale_override
        if page_scale_override
        else scale_info.get("verified_scale") or scale_info.get("text_scale") or 360.0
    )
    page_scale = _scale_inches_to_points_per_foot(scale_inches)

    per_cat: dict[str, dict] = {}
    has_user_edits = bool(
        (page_state or {}).get("line_assignments")
        or (page_state or {}).get("user_drawn_lines")
    )

    if has_user_edits:
        # Recompute from saved UMT state. Re-extract vector lines so we
        # can map saved indices -> length_pts.
        vlines = vlines or []

        line_assignments = (page_state or {}).get("line_assignments") or {}
        for idx_str, cat in line_assignments.items():
            try:
                i = int(idx_str)
            except (TypeError, ValueError):
                continue
            if i < 0 or i >= len(vlines):
                continue
            ln = vlines[i]
            entry = per_cat.setdefault(
                cat, {"pts": 0.0, "auto": 0, "manual": 0}
            )
            if isinstance(ln, dict):
                length_pts = ln.get("length_pts")
                if not isinstance(length_pts, (int, float)):
                    try:
                        sx, sy = float(ln["start"][0]), float(ln["start"][1])
                        ex, ey = float(ln["end"][0]), float(ln["end"][1])
                        length_pts = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                entry["pts"] += float(length_pts)
            else:
                entry["pts"] += float(ln.length_pts)
            entry["auto"] += 1

        user_drawn = (page_state or {}).get("user_drawn_lines") or []
        for ud in user_drawn:
            cat = ud.get("category")
            if not cat:
                continue
            try:
                sx, sy = float(ud["start"][0]), float(ud["start"][1])
                ex, ey = float(ud["end"][0]), float(ud["end"][1])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            entry = per_cat.setdefault(
                cat, {"pts": 0.0, "auto": 0, "manual": 0}
            )
            entry["pts"] += ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            entry["manual"] += 1
    else:
        # Auto-only: mirror `_build_auto_export_state` per-line bucketing
        # so the summary matches exports.
        measurements = (fp or {}).get("measurements") or {}
        all_fence_lines = measurements.get("all_fence_lines") or []
        layer_to_cat = measurements.get("layer_to_category") or {}
        any_layer_mapped = bool(layer_to_cat)
        FALLBACK = "Auto-detected"
        for fl in all_fence_lines:
            if not isinstance(fl, dict):
                continue
            try:
                sx, sy = float(fl["start"][0]), float(fl["start"][1])
                ex, ey = float(fl["end"][0]), float(fl["end"][1])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            layer = fl.get("layer") or ""
            cat = layer_to_cat.get(layer)
            if not cat and layer and any_layer_mapped:
                for k, v in layer_to_cat.items():
                    if k and (k in layer or layer in k):
                        cat = v
                        break
            if not cat:
                if any_layer_mapped:
                    continue
                cat = FALLBACK
            length_pts = (
                float(fl["length_pts"])
                if isinstance(fl.get("length_pts"), (int, float))
                else ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            )
            entry = per_cat.setdefault(
                cat, {"pts": 0.0, "auto": 0, "manual": 0}
            )
            entry["pts"] += length_pts
            entry["auto"] += 1

    for cat, entry in per_cat.items():
        entry["ft"] = round(entry["pts"] / page_scale, 2)
        entry["pts"] = round(entry["pts"], 1)

    return {
        "scale": page_scale,
        "has_user_edits": has_user_edits,
        "per_category": per_cat,
    }


@app.get("/api/jobs/{job_id}/measurement-summary")
async def measurement_summary(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Cross-page totals by category. Falls back to auto-only on pages
    with no UMT edits, recomputes from umt_state on pages that do."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        hl = job_registry.get_highlighted_pdf_path(job_id)
        if hl is not None and Path(hl).exists():
            pdf_path = str(hl)
        else:
            raise HTTPException(404, "No PDF on disk for this job")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available")

    from backend.app import umt_state
    umt = umt_state.load(job_id)
    pages_node = (umt or {}).get("pages") or {}

    fence_pages = (results or {}).get("fence_pages") or []
    edited_pages = _line_assignment_export_pages(results, pages_node)
    edited_lines_by_idx: dict[int, list[dict]] = {}
    if edited_pages:
        log.info(
            "measurement-summary vector extraction job=%s edited_pages=%d",
            job_id[:8],
            len(edited_pages),
        )
        try:
            edited_lines_by_idx = await asyncio.to_thread(
                _extract_export_vector_lines,
                job_id=job_id,
                pdf_path=pdf_path,
                page_indices=[page_idx for page_idx, _page_num, _state in edited_pages],
            )
        except ExportWorkerTimeout as e:
            log.warning("measurement-summary vector timeout job=%s: %s", job_id[:8], e)
            raise HTTPException(504, str(e))
        except ExportWorkerError as e:
            log.warning("measurement-summary vector failed job=%s: %s", job_id[:8], e)
            raise HTTPException(500, f"Measurement summary export state failed: {e}")

    log.info("measurement-summary start job=%s pages=%d", job_id[:8], len(fence_pages))
    pages_data: list[dict] = []
    grand: dict[str, dict] = {}
    for fp in fence_pages:
        page_num = fp.get("page_num") or fp.get("page_number")
        page_idx = fp.get("page_idx")
        if page_idx is None:
            page_idx = fp.get("page_index_in_original_doc")
        if page_idx is None or page_num is None:
            continue
        page_state = pages_node.get(f"page_{page_num}")
        ps = _summary_for_page(
            int(page_idx),
            fp,
            page_state,
            edited_lines_by_idx.get(int(page_idx)),
        )
        pages_data.append({
            "page_num": page_num,
            "scale": ps["scale"],
            "has_user_edits": ps["has_user_edits"],
            "per_category": ps["per_category"],
        })
        for cat, entry in ps["per_category"].items():
            g = grand.setdefault(
                cat, {"pts": 0.0, "ft": 0.0, "auto": 0, "manual": 0}
            )
            g["pts"] += entry["pts"]
            g["ft"] += entry["ft"]
            g["auto"] += entry["auto"]
            g["manual"] += entry["manual"]
    for cat, g in grand.items():
        g["pts"] = round(g["pts"], 1)
        g["ft"] = round(g["ft"], 2)

    sorted_grand = dict(
        sorted(grand.items(), key=lambda kv: -kv[1]["ft"])
    )
    log.info(
        "measurement-summary done job=%s pages=%d categories=%d",
        job_id[:8],
        len(pages_data),
        len(sorted_grand),
    )
    return {
        "pages": pages_data,
        "grand_total": sorted_grand,
        "page_count": len(pages_data),
        "category_count": len(sorted_grand),
    }


def _job_for_user(job_id: str, user_id: str) -> dict:
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")
    return job


@app.get("/api/jobs/{job_id}/umt-state")
async def get_umt_state(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Return the user's saved UMT edits for this job. Returns the empty
    skeleton (`{"version":1,"pages":{}}`) when nothing has been saved yet."""
    _job_for_user(job_id, user_id)
    from backend.app import umt_state
    return umt_state.load(job_id)


@app.put("/api/jobs/{job_id}/umt-state/{page_num}")
async def put_umt_page_state(
    job_id: str,
    page_num: int,
    page_state: dict,
    user_id: str = Depends(get_current_user),
):
    """Upsert one page's UMT state (categories, line assignments,
    user-drawn lines, scale override, min-length filter)."""
    _job_for_user(job_id, user_id)
    from backend.app import umt_state
    try:
        full = umt_state.update_page(job_id, page_num, page_state)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    # Edits change the exported line set — the cached measurement PDF (and
    # its Line # labels, which must stay in sync with the Excel) is stale.
    invalidate_measurement_pdf(job_id)
    return full


@app.delete("/api/jobs/{job_id}/umt-state/{page_num}")
async def delete_umt_page_state(
    job_id: str,
    page_num: int,
    user_id: str = Depends(get_current_user),
):
    """Clear UMT edits for one page (used by the 'Clear auto' button)."""
    _job_for_user(job_id, user_id)
    from backend.app import umt_state
    full = umt_state.delete_page(job_id, page_num)
    invalidate_measurement_pdf(job_id)
    return full


@app.get("/api/jobs/{job_id}/progress")
async def stream_progress(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """SSE stream of progress updates for a running job."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    async def event_generator() -> AsyncIterator[str]:
        last_pct = -1
        while True:
            current = job_registry.get_job(job_id)
            if current is None:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            prog = job_registry.read_progress(job_id) or {}
            pct = prog.get("pct", 0)

            if pct != last_pct or current["status"] not in ("queued", "running"):
                payload = {
                    "status": current["status"],
                    "phase": prog.get("phase", ""),
                    "pct": pct,
                    "message": prog.get("message", ""),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_pct = pct

            if current["status"] not in ("queued", "running"):
                break

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Cancel a running/queued job, or hard-delete a terminal one.

    - queued/running → mark cancelled (job loop will skip it)
    - cancelled/failed/completed → hard-delete row + remove disk results
    """
    job = job_registry.get_job(job_id)
    if job is None:
        try:
            if _delete_postgres_document_for_job(job_id, user_id):
                return {"status": "deleted"}
        except Exception:
            log.exception("Postgres-only cleanup failed for delete_job %s", job_id)
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    status = job.get("status")
    if status in ("queued", "running"):
        job_registry.update_job(job_id, status="cancelled",
                                completed_at=int(time.time()))
        # Mirror to Postgres so the dashboard reflects it immediately.
        try:
            db.update_job_progress(job_id, status="cancelled", finished_at_now=True)
        except Exception:
            pass
        return {"status": "cancelled"}

    # Terminal state → hard delete (SQLite row, disk artifacts, Postgres document).
    rdir = job.get("results_dir")
    if rdir:
        try:
            shutil.rmtree(rdir, ignore_errors=True)
        except Exception:
            pass
    pdf_path = job.get("pdf_path")
    if pdf_path:
        try:
            Path(pdf_path).unlink(missing_ok=True)
        except Exception:
            pass
    job_registry.delete_job(job_id)
    with _measurement_builds_lock:
        _measurement_builds.pop(job_id, None)

    # Also remove the Postgres document row (cascades to its jobs row,
    # page_results, artifacts via FK ON DELETE CASCADE). Skip silently
    # for legacy non-UUID users or if the document was never mirrored.
    if _is_uuid(user_id):
        try:
            _delete_postgres_document_for_job(job_id, user_id)
        except Exception:
            log.exception("Postgres cleanup failed for delete_job %s", job_id)

    return {"status": "deleted"}


@app.get("/api/me")
async def me(user_id: str = Depends(require_supabase_jwt)):
    """Identity check — verifies a Supabase JWT and echoes its `sub` claim.

    Always demands a valid Bearer token regardless of API_AUTH_MODE.
    Used by the Next.js frontend to confirm the backend can verify its tokens.
    """
    return {"user_id": user_id}


def _slim_page_result(row: dict) -> dict:
    """Strip the heaviest arrays from `result_json` before sending to the
    frontend. The dashboard's live "Pages so far" ticker only needs
    summary metrics; the line-level arrays balloon /pages responses to
    8-12 MB and block the event loop while psycopg streams them.
    Pages with full detail are still available via
    `/api/jobs/{id}/results?full=1` when the user opens the document.
    """
    rj = row.get("result_json")
    if not isinstance(rj, dict):
        return row
    # Don't mutate the caller's dict.
    rj_slim = dict(rj)
    measurements = rj_slim.get("measurements")
    if isinstance(measurements, dict):
        m_slim = dict(measurements)
        # `all_fence_lines` is the dominant size — thousands of {start,end,
        # length_pts,layer} dicts per dense page.
        m_slim.pop("all_fence_lines", None)
        m_slim.pop("layer_measurements", None)
        rj_slim["measurements"] = m_slim
    # Keep `legend_entries` (needed for chips) but drop these heavy ones:
    rj_slim.pop("keyword_matches", None)
    rj_slim.pop("ade_chunks", None)
    return {**row, "result_json": rj_slim}


# NOTE: these three routes are *sync* (def, not async def). They call into
# psycopg synchronously, which would block the asyncio event loop when
# declared async — every other request (including /api/healthz) queues
# behind a slow query. FastAPI dispatches sync handlers to a worker
# thread-pool (via anyio), so the event loop stays free.
@app.get("/api/documents")
def list_documents(user_id: str = Depends(require_supabase_jwt)):
    """List the user's uploaded documents (from Postgres mirror).

    Each row includes the latest job's status/progress for dashboard rendering.
    Requires a Supabase JWT — the legacy X-User-Id path doesn't have a
    Postgres mirror.
    """
    try:
        documents = db.list_documents(user_id)
    except Exception:
        log.exception("Postgres document list failed; falling back to SQLite jobs")
        documents = []
    docs = _merge_sqlite_fallback_documents(user_id, documents)
    # Tag each row with the analysis mode it was run in, so the dashboard can
    # show Fence/Electrical per file. Cheap + cached (see _job_trade).
    for d in docs:
        jid = d.get("latest_job_id")
        if jid:
            try:
                d["trade"] = _job_trade(str(jid), user_id)
            except Exception:
                d["trade"] = None
    return {"documents": docs}


@app.get("/api/documents/{document_id}")
def get_document(
    document_id: str,
    user_id: str = Depends(require_supabase_jwt),
):
    """Get one document, ownership-checked. Requires Supabase JWT."""
    if not _is_uuid(document_id):
        raise HTTPException(400, "document_id must be a UUID")
    try:
        doc = db.get_document(document_id, user_id)
    except Exception:
        log.exception("Postgres document lookup failed; trying SQLite fallback")
        doc = None
    if doc is None:
        job = job_registry.get_job(document_id)
        if job is None or job.get("user_id") != user_id:
            raise HTTPException(404, "Document not found")
        return _sqlite_job_to_document(job)
    return doc


@app.get("/api/documents/{document_id}/pages")
def list_document_pages(
    document_id: str,
    user_id: str = Depends(require_supabase_jwt),
    slim: bool = True,
):
    """Stream per-page results as the worker writes them. Used by the detail
    page to render pages incrementally while a job runs (Sprint 2 / A7).

    `slim=true` (default) drops heavy line-level arrays from result_json
    so the response stays small enough that polling every 3 s doesn't
    saturate the network or block other requests.
    """
    if not _is_uuid(document_id):
        raise HTTPException(400, "document_id must be a UUID")
    # Confirm ownership before listing — prevents existence-leak via empty []
    try:
        doc = db.get_document(document_id, user_id)
    except Exception:
        log.exception("Postgres document lookup failed for pages; trying SQLite fallback")
        doc = None
    if doc is None:
        job = job_registry.get_job(document_id)
        if job is None or job.get("user_id") != user_id:
            raise HTTPException(404, "Document not found")
        return {"pages": []}
    rows = db.list_page_results(document_id, user_id)
    if slim:
        rows = [_slim_page_result(r) for r in rows]
    return {"pages": rows}


@app.get("/api/healthz")
async def healthz():
    """Health check endpoint."""
    import psutil
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    running = job_registry.count_running_jobs()
    queued = job_registry.count_queued_jobs()

    keys = _load_api_keys()

    return {
        "status": "ok",
        "rss_mb": round(rss_mb, 1),
        "running_jobs": running,
        "queued_jobs": queued,
        "max_workers": cfg.API_WORKER_COUNT,
        "has_openai_key": bool(keys.get("openai_key")),
        "has_ade_key": bool(keys.get("ade_key")),
        "has_google_config": keys.get("google_cloud_config") is not None,
        "results_ttl_hours": cfg.RESULTS_TTL_HOURS,
    }
