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
import threading
import time
import toml
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import job_registry
from config import cfg
from pipeline import PipelineConfig, PipelineResult, run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# API key loading (same source as Streamlit: secrets.toml or env vars)
# ---------------------------------------------------------------------------

def _load_api_keys() -> dict:
    secrets: dict = {}
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            secrets = toml.load(str(secrets_path))
        except Exception as e:
            log.warning(f"Failed to load secrets.toml: {e}")

    openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    ade_key = secrets.get("LANDINGAI_API_KEY", os.getenv("LANDINGAI_API_KEY", ""))

    google_cloud_config = None
    try:
        if "google_cloud" in secrets and "gcp_service_account" in secrets:
            google_cloud_config = {
                "project_number": secrets["google_cloud"]["project_number"],
                "location": secrets["google_cloud"]["location"],
                "processor_id": secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(secrets["gcp_service_account"]),
            }
    except Exception:
        pass

    return {
        "openai_key": openai_key,
        "ade_key": ade_key,
        "google_cloud_config": google_cloud_config,
    }


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

    try:
        _progress("start", 0, "Analysis starting...")
        result = run_analysis(pdf_path, config, progress_cb=_progress)

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
            _progress("done", 100, f"Failed: {result.error[:100]}")
        else:
            job_registry.update_job(
                job_id, status="completed",
                completed_at=int(time.time()),
                total_pages=result.total_pages,
                fence_count=len(result.fence_pages),
                non_fence_count=len(result.non_fence_pages),
            )
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(
    pdf: UploadFile = File(...),
    config: str = Form("{}"),
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
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

    upload_dir = Path(cfg.PDF_TMP_DIR) / x_user_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    import hashlib
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    pdf_filename = f"job_{pdf_hash[:16]}.pdf"
    pdf_path = upload_dir / pdf_filename
    pdf_path.write_bytes(pdf_bytes)

    job_id = job_registry.create_job(
        user_id=x_user_id,
        filename=pdf.filename or "unknown.pdf",
        pdf_hash=pdf_hash,
        pdf_path=str(pdf_path),
        config_json=json.dumps(config_data, default=str),
    )

    queue_pos = job_registry.queue_position(job_id)
    running = job_registry.count_running_jobs()

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_pos,
        "running_jobs": running,
    }


@app.get("/api/jobs")
async def list_jobs(x_user_id: str = Header("anonymous", alias="X-User-Id")):
    """List the user's recent jobs."""
    jobs = job_registry.get_user_jobs(x_user_id)
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
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
):
    """Get job details. Includes results summary if completed."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != x_user_id:
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


@app.get("/api/jobs/{job_id}/results")
async def get_results(
    job_id: str,
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
):
    """Get full analysis results for a completed job."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != x_user_id:
        raise HTTPException(403, "Access denied")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}, not completed")

    results = job_registry.load_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not found on disk")
    return results


@app.get("/api/jobs/{job_id}/highlighted-pdf")
async def get_highlighted_pdf(
    job_id: str,
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
):
    """Download the highlighted PDF for a completed job."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != x_user_id:
        raise HTTPException(403, "Access denied")

    hl_path = job_registry.get_highlighted_pdf_path(job_id)
    if hl_path is None:
        raise HTTPException(404, "Highlighted PDF not available")

    return FileResponse(
        str(hl_path),
        media_type="application/pdf",
        filename=f"fence_{job.get('filename', 'document')}.pdf",
    )


@app.get("/api/jobs/{job_id}/progress")
async def stream_progress(
    job_id: str,
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
):
    """SSE stream of progress updates for a running job."""
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != x_user_id:
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
    x_user_id: str = Header("anonymous", alias="X-User-Id"),
):
    """Cancel a running/queued job, or hard-delete a terminal one.

    - queued/running → mark cancelled (job loop will skip it)
    - cancelled/failed/completed → hard-delete row + remove disk results
    """
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != x_user_id:
        raise HTTPException(403, "Access denied")

    status = job.get("status")
    if status in ("queued", "running"):
        job_registry.update_job(job_id, status="cancelled",
                                completed_at=int(time.time()))
        return {"status": "cancelled"}

    # Terminal state → hard delete
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
    return {"status": "deleted"}


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
