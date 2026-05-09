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
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
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
from config import cfg
from pipeline import PipelineConfig, PipelineResult, run_analysis
from secrets_loader import load_api_keys as _load_api_keys


def _is_uuid(s: str) -> bool:
    """True if s is a well-formed UUID (Supabase user IDs always are).
    Legacy X-User-Id values like 'alice' or 'anonymous' return False."""
    try:
        _uuid.UUID(s)
        return True
    except Exception:
        return False

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

    config = PipelineConfig(
        openai_api_key=keys.get("openai_key", ""),
        ade_api_key=keys.get("ade_key", ""),
        google_cloud_config=keys.get("google_cloud_config"),
        analysis_model=config_data.get("analysis_model", cfg.ANALYSIS_MODEL),
        classifier_model=config_data.get("classifier_model", cfg.CLASSIFIER_MODEL),
        fence_keywords=config_data.get("fence_keywords", list(cfg.DEFAULT_FENCE_KEYWORDS)),
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
    document_id: str | None = None
    try:
        document_id = db.get_document_id_by_job(job_id)
    except Exception:
        log.exception(f"Job {job_id[:8]}: document lookup failed; live page updates disabled")

    def _page_cb(page: dict) -> None:
        if not document_id:
            return
        try:
            db.upsert_page_result(
                document_id=document_id,
                page_number=int(page["page_number"]),
                is_fence_page=bool(page.get("is_fence_page", False)),
                result_json=page.get("result_json"),
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
    requeued = job_registry.requeue_orphaned_running()
    if requeued:
        log.info(f"Startup: re-queued {requeued} orphaned running jobs")
    # Catch ancient stale rows that pre-date the requeue logic
    stale = job_registry.mark_stale_running_as_failed(max_age_seconds=7200)
    if stale:
        log.info(f"Startup: marked {stale} stale running jobs as failed")

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
    allow_headers=["Authorization", "Content-Type", "X-User-Id"],
    expose_headers=["Content-Disposition", "X-Page-Number", "X-Image-Source"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    pdf: UploadFile = File(...),
    config: str = Form("{}"),
    user_id: str = Depends(get_current_user),
):
    """Submit a PDF for analysis. Returns the job ID immediately."""
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

    # Peek page count before queueing
    try:
        import fitz as _fitz
        _doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        n_pages = _doc.page_count
        _doc.close()
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")

    if n_pages > cfg.MAX_PAGES:
        raise HTTPException(
            413,
            f"PDF has {n_pages} pages (max {cfg.MAX_PAGES}). "
            "Please split the document by page range.",
        )

    import hashlib
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Dedup: if this Supabase user already has a document with the same
    # pdf_hash, return the existing one. Re-process only if the previous
    # attempt is in a terminal-failed state.
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
    document_id: str | None = None
    if _is_uuid(user_id):
        rel_storage_path = f"{user_id}/{pdf_filename}"
        try:
            document_id, _ = db.insert_document_and_job(
                user_id=user_id,
                original_filename=pdf.filename or "unknown.pdf",
                storage_path=rel_storage_path,
                total_pages=n_pages,
                job_id=job_id,
                pdf_hash=pdf_hash,
            )
        except Exception as e:
            log.exception("Postgres mirror failed for upload")
            try:
                job_registry.delete_job(job_id)
            except Exception:
                pass
            try:
                Path(pdf_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(503, f"Database unavailable: {e}")

    queue_pos = job_registry.queue_position(job_id)
    running = job_registry.count_running_jobs()

    return {
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "queue_position": queue_pos,
        "running_jobs": running,
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
    """
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}, not completed")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not found on disk")
    return results if full else _slim_results(results)


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
    all_fence_lines = measurements.get("all_fence_lines") or []
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
    return {
        "page_num": page_num,
        "page_idx": page_idx,
        "rotation": rotation,
        "pdf_width": float(display_w),
        "pdf_height": float(display_h),
        "lines": out_lines,
        "auto_categories": auto_categories,
        "auto_assignments": {} if source_missing else auto_assignments,
        "source_missing": source_missing,
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
    for idx_str, cat in (page_state.get("line_assignments") or {}).items():
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

    # Pre-extract vector lines for any page that has UMT edits, sharing
    # one fitz.Doc handle across pages to avoid repeated opens.
    umt_lines_by_idx: dict[int, list[dict]] = {}
    needs_extraction = [
        fp for fp in fence_pages_in
        if isinstance(fp, dict)
        and pages_node.get(f"page_{fp.get('page_num') or fp.get('page_number')}")
        and (
            (pages_node.get(f"page_{fp.get('page_num') or fp.get('page_number')}", {}).get("line_assignments"))
            or (pages_node.get(f"page_{fp.get('page_num') or fp.get('page_number')}", {}).get("user_drawn_lines"))
        )
    ]
    if needs_extraction and pdf_path and Path(pdf_path).exists():
        import fitz
        from utils_vector import extract_vector_lines

        doc = None
        try:
            doc = fitz.open(pdf_path)
            for fp in needs_extraction:
                page_idx = fp.get("page_idx")
                if page_idx is None:
                    page_idx = fp.get("page_index_in_original_doc")
                if page_idx is None:
                    continue
                if not (0 <= page_idx < len(doc)):
                    continue
                vlines = extract_vector_lines(doc[page_idx], apply_rotation=True)
                umt_lines_by_idx[page_idx] = [
                    {
                        "start": [float(v.start[0]), float(v.start[1])],
                        "end": [float(v.end[0]), float(v.end[1])],
                        "length_pts": float(v.length_pts),
                        "layer": v.layer or "",
                    }
                    for v in vlines
                ]
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

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


@app.get("/api/jobs/{job_id}/measurement-pdf")
async def get_measurement_pdf(
    job_id: str,
    user_id: str = Depends(get_current_user),
):
    """Measurement-overlay PDF (Sprint 3 / C5).

    Wraps `exports.generate_measurement_pdf`. Until UMT (Sprint 4) ships,
    this only contains auto-detected lines pre-assigned via the layer →
    category map. User edits are not yet captured."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    # PDF generation overlays measurement lines on top of the page's
    # native vector content. If the original PDF is gone we can't
    # cleanly fall back to the highlighted PDF — `insert_pdf` would
    # carry the cyan fence overlays into the output and the resulting
    # doubly-decorated, deflate-compressed PDF takes minutes on dense
    # pages. Ask the user to re-upload instead. (Excel works either
    # way — it only consumes coords, not the visual content.)
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

    fence_pages_out, line_assignments, user_drawn, page_cats, _scale, _lines = (
        _build_export_state(job_id, results, pdf_path)
    )

    from exports import generate_measurement_pdf
    pdf_bytes, fname = generate_measurement_pdf(
        pdf_path=pdf_path,
        fence_pages=fence_pages_out,
        line_assignments=line_assignments,
        user_drawn_lines=user_drawn,
        page_categories=page_cats,
        uploaded_pdf_name=job.get("filename") or "document.pdf",
    )
    if not pdf_bytes:
        raise HTTPException(500, "Measurement PDF generation returned no bytes")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control": "private, no-store",
        },
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

    fence_pages_out, line_assignments, user_drawn, page_cats, scale_info, lines_by_page = (
        _build_export_state(job_id, results, pdf_path)
    )
    element_details = (results or {}).get("element_details") or {}

    from exports import generate_measurement_spreadsheet
    xlsx_bytes = generate_measurement_spreadsheet(
        fence_pages=fence_pages_out,
        line_assignments=line_assignments,
        user_drawn_lines=user_drawn,
        page_categories=page_cats,
        per_page_scale_info=scale_info,
        element_details=element_details,
        lines_by_page=lines_by_page,
    )
    if not xlsx_bytes:
        raise HTTPException(
            500,
            "No measurement rows were generated — likely because no fence "
            "lines were auto-detected. After UMT (Sprint 4) ships, manual "
            "edits will populate this export.",
        )

    base = (job.get("filename") or "document.pdf").rsplit(".", 1)[0]
    fname = f"{base}_measurements.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control": "private, no-store",
        },
    )


def _summary_for_page(
    pdf_path: str,
    page_idx: int,
    fp: dict,
    page_state: dict | None,
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
    page_scale = (
        float(page_scale_override)
        if page_scale_override
        else float(scale_info.get("verified_scale") or 360.0)
    )
    if page_scale <= 0:
        page_scale = 360.0

    per_cat: dict[str, dict] = {}
    has_user_edits = bool(
        (page_state or {}).get("line_assignments")
        or (page_state or {}).get("user_drawn_lines")
    )

    if has_user_edits:
        # Recompute from saved UMT state. Re-extract vector lines so we
        # can map saved indices -> length_pts.
        import fitz
        from utils_vector import extract_vector_lines

        doc = None
        vlines = []
        try:
            doc = fitz.open(pdf_path)
            if 0 <= page_idx < len(doc):
                vlines = extract_vector_lines(doc[page_idx], apply_rotation=True)
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

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
        ps = _summary_for_page(pdf_path, int(page_idx), fp, page_state)
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
    return umt_state.delete_page(job_id, page_num)


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

    # Also remove the Postgres document row (cascades to its jobs row,
    # page_results, artifacts via FK ON DELETE CASCADE). Skip silently
    # for legacy non-UUID users or if the document was never mirrored.
    if _is_uuid(user_id):
        try:
            with db.pool().connection() as conn:
                conn.execute(
                    """
                    delete from documents
                    where user_id = %s
                      and id in (select document_id from jobs where id = %s)
                    """,
                    (user_id, job_id),
                )
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
    return {"documents": db.list_documents(user_id)}


@app.get("/api/documents/{document_id}")
def get_document(
    document_id: str,
    user_id: str = Depends(require_supabase_jwt),
):
    """Get one document, ownership-checked. Requires Supabase JWT."""
    if not _is_uuid(document_id):
        raise HTTPException(400, "document_id must be a UUID")
    doc = db.get_document(document_id, user_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
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
    if db.get_document(document_id, user_id) is None:
        raise HTTPException(404, "Document not found")
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
