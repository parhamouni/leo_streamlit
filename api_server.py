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
# Comma-separated env override; defaults to local dev origins.
_default_origins = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
_cors_origins = [o.strip() for o in os.environ.get("FENCE_CORS_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-Id"],
    expose_headers=["Content-Disposition"],
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
    user_id: str = Depends(get_current_user),
):
    """Rasterize page `page_num` (1-indexed) from the job's highlighted PDF
    and return it as a PNG. Used by the detail page's per-page cards so the
    user can see the colored fence overlays inline without scrolling the
    whole embedded PDF.

    Renders in a subprocess (with timeout) so a MuPDF crash on a dense
    vector page doesn't kill the API. DPI defaults to 110 (fast) — matches
    the "Low-DPI preview" toggle in app_ade_prod.py:2077-2081.
    """
    if dpi < 50 or dpi > 200:
        raise HTTPException(400, "dpi must be between 50 and 200")
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    # Prefer the highlighted PDF (with overlays). It's only written at the
    # very end of run_analysis(), so during a running/queued job we fall
    # back to the original uploaded PDF — the user can still see the page,
    # just without the colored boxes yet. Once the job completes the same
    # endpoint serves the highlighted version automatically.
    hl_path = job_registry.get_highlighted_pdf_path(job_id)
    src_pdf = hl_path if hl_path is not None else job.get("pdf_path")
    if not src_pdf or not Path(src_pdf).exists():
        raise HTTPException(
            404,
            "Source PDF not available — try again once upload finishes",
        )

    import subprocess
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _PAGE_RENDER_SCRIPT,
                str(src_pdf),
                str(page_num),
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

    is_highlighted = hl_path is not None
    return Response(
        content=proc.stdout,
        media_type="image/png",
        headers={
            # Don't cache long while the job is running — we want the
            # highlighted version once it lands. After completion the
            # response is stable so an hour is fine.
            "Cache-Control": (
                "private, max-age=3600" if is_highlighted else "private, max-age=10"
            ),
            "X-Page-Number": str(page_num),
            "X-Image-Source": "highlighted" if is_highlighted else "original",
        },
    )


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


def _build_auto_export_state(results: dict) -> tuple[list, dict, dict, dict]:
    """Reshape pipeline results into the (line_assignments, page_categories,
    auto-lines-per-page) tuple that exports.py wants. UMT (Sprint 4) hasn't
    landed yet, so user_drawn_lines is always empty and we treat every
    auto-detected fence line as accepted.

    Returns: (fence_pages_with_auto_lines, line_assignments, user_drawn_lines,
    page_categories, per_page_scale_info, lines_by_page).
    """
    fence_pages_in = (results or {}).get("fence_pages") or []
    per_page_scale = (results or {}).get("per_page_scale_info") or {}

    line_assignments: dict[str, dict[str, str]] = {}
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

        # 1) Build category palette from this page's legend entries (the
        # LLM-extracted (indicator, keyword, description) tuples). Note: the
        # `definitions` field on a fence page is the raw ADE chunk text, not
        # the legend rows — those live on `legend_entries`.
        cats: dict[str, dict] = {}
        for le in fp.get("legend_entries") or []:
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
        page_categories[page_key] = cats

        # 2) Auto-detected fence lines + layer→category assignment.
        measurements = fp.get("measurements") or {}
        all_lines = measurements.get("all_fence_lines") or []
        layer_to_cat = measurements.get("layer_to_category") or {}

        page_assignments: dict[str, str] = {}
        auto_lines: list[dict] = []
        # Last-resort bucket so users get *something* when the LLM didn't
        # produce a layer→category mapping (this happens when legend
        # extraction returns no matches but Phase 3 still detected lines
        # geometrically — the prod UMT relies on the user assigning these
        # by hand, which we can't do until Sprint 4).
        FALLBACK_CAT = "Auto-detected"
        any_layer_mapped = bool(layer_to_cat)

        for idx, line in enumerate(all_lines):
            if not isinstance(line, dict):
                continue
            layer = line.get("layer") or ""
            category = layer_to_cat.get(layer)
            # Partial layer match fallback (mirrors prod logic at
            # app_ade_prod.py:4799-4804).
            if not category and layer and any_layer_mapped:
                for k, v in layer_to_cat.items():
                    if k and (k in layer or layer in k):
                        category = v
                        break
            if not category:
                # If we have NO layer mapping at all, group the page's auto
                # lines under a single fallback category. Skip lines that
                # have no geometry though — they wouldn't render anyway.
                if any_layer_mapped:
                    continue
                category = FALLBACK_CAT
            # Only record an assignment if the named category actually exists
            # in this page's palette — otherwise the export drops the row.
            if category not in cats:
                cats[category] = {
                    "indicator": "",
                    "keyword": category,
                    "color": _CATEGORY_PALETTE[len(cats) % len(_CATEGORY_PALETTE)],
                }
            page_assignments[str(idx)] = category
            auto_lines.append(line)

        if page_assignments:
            line_assignments[page_key] = page_assignments
        if auto_lines:
            lines_by_page[page_key] = auto_lines

        # 3) Carry the page through with `auto_lines` populated so the PDF
        # exporter can index into it.
        fp_out = dict(fp)
        fp_out["auto_lines"] = auto_lines
        fence_pages_out.append(fp_out)

    return fence_pages_out, line_assignments, {}, page_categories, per_page_scale, lines_by_page


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

    pdf_path = job.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "Source PDF not on disk")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available — job may still be running")

    fence_pages_out, line_assignments, user_drawn, page_cats, _scale, _lines = (
        _build_auto_export_state(results)
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
    """Measurement Excel (Sprint 3 / C6).

    Wraps `exports.generate_measurement_spreadsheet`. Same auto-only caveat
    as the PDF export."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != user_id:
        raise HTTPException(403, "Access denied")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available — job may still be running")

    fence_pages_out, line_assignments, user_drawn, page_cats, scale_info, lines_by_page = (
        _build_auto_export_state(results)
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


@app.get("/api/documents")
async def list_documents(user_id: str = Depends(require_supabase_jwt)):
    """List the user's uploaded documents (from Postgres mirror).

    Each row includes the latest job's status/progress for dashboard rendering.
    Requires a Supabase JWT — the legacy X-User-Id path doesn't have a
    Postgres mirror.
    """
    return {"documents": db.list_documents(user_id)}


@app.get("/api/documents/{document_id}")
async def get_document(
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
async def list_document_pages(
    document_id: str,
    user_id: str = Depends(require_supabase_jwt),
):
    """Stream per-page results as the worker writes them. Used by the detail
    page to render pages incrementally while a job runs (Sprint 2 / A7)."""
    if not _is_uuid(document_id):
        raise HTTPException(400, "document_id must be a UUID")
    # Confirm ownership before listing — prevents existence-leak via empty []
    if db.get_document(document_id, user_id) is None:
        raise HTTPException(404, "Document not found")
    return {"pages": db.list_page_results(document_id, user_id)}


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
