"""Background analysis worker — Phases 1a → 1b → 1c → 2 → 3a.

Runs the expensive analysis phases in a daemon=False thread so the work
survives browser disconnects.  Phase 3b (interactive Streamlit render)
stays in the UI session.

Entry point
-----------
    from ops.analysis_worker import run_job
    t = threading.Thread(target=run_job, args=(job_id, config), daemon=False)
    t.start()

Config dict keys
----------------
    job_id                  str
    pdf_path                str     disk path to the uploaded PDF
    pdf_hash                str     SHA-256 hex of the file
    filename                str     original upload name (for display)
    user_id                 str     hashed Google email
    openai_key              str
    ade_key                 str | None
    google_cloud_config     dict | None
    analysis_model          str     e.g. "gpt-5.1"
    classifier_model        str     e.g. "gpt-5-mini"
    fence_keywords          list[str]
    use_ade                 bool
    highlight_fence_text    bool
    enable_unified_measurement  bool
    broken_pages            list[int]   pre-known damaged page indices
    cache_params            str     fence_cache.params_hash(...) — pre-computed

Writes
------
    ~/.leo/results/<job_id>/progress.json     polled by the Streamlit UI
    ~/.leo/results/<job_id>/results_summary.json  read by Phase 3b on reconnect
Job registry is updated throughout (running → phases_ready or failed).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Make the repo root importable (this file lives in ops/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fitz  # PyMuPDF
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

import psutil

import utils_ade as ade
import fence_cache
import telemetry
import job_registry
from utils_vector import verify_scale_with_bar_fast
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Constants / helpers (mirror what app_ade_fast.py reads from env)
# ---------------------------------------------------------------------------

def _workers(name: str, default: int, cap: int = 16) -> int:
    try:
        return min(max(1, int(os.environ.get(name, default))), cap)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return default


FENCE_WORKERS_PHASE1A = _workers("FENCE_WORKERS_PHASE1A", 4, cap=8)
FENCE_WORKERS_PHASE1B = _workers("FENCE_WORKERS_PHASE1B", 6)
FENCE_WORKERS_PHASE1C = _workers("FENCE_WORKERS_PHASE1C", 16)
FENCE_WORKERS_PHASE2  = _workers("FENCE_WORKERS_PHASE2",  5, cap=8)

_default_phase3_workers = 1 if _env_bool("FENCE_LOW_MEMORY") else 2
FENCE_WORKERS_PHASE3 = _workers("FENCE_WORKERS_PHASE3", _default_phase3_workers, cap=12)

FENCE_CLASSIFY_BATCH_SIZE = _workers("FENCE_CLASSIFY_BATCH_SIZE", 10, cap=25)
FENCE_OCR_BATCH_SIZE      = _workers("FENCE_OCR_BATCH_SIZE", 15, cap=15)
FENCE_PHASE3_EAGER        = _env_bool("FENCE_PHASE3_EAGER", True)   # pre-compute ALL fence pages so reconnect is instant
FENCE_PHASE3_PREVIEW      = _workers("FENCE_PHASE3_PREVIEW", 5, cap=40)
FENCE_PHASE3_PAGE_TIMEOUT = _workers("FENCE_PHASE3_PAGE_TIMEOUT", 300, cap=600)
FENCE_PHASE3_USE_SUBPROCESS = _env_bool("FENCE_PHASE3_USE_SUBPROCESS", True)

FENCE_PHASE1A_BATCH_SIZE  = _workers("FENCE_PHASE1A_BATCH_SIZE", 5, cap=25)
PHASE_1A_PAGE_TIMEOUT     = 20  # seconds per page

_PHASE3_WORKER_SCRIPT = os.path.join(_REPO_ROOT, "ops", "phase3_worker.py")
_EXTRACTOR_SCRIPT     = os.path.join(_REPO_ROOT, "ops", "page_extractor.py")

# ---------------------------------------------------------------------------
# LLM factory (no @st.cache_resource — worker creates its own instances)
# ---------------------------------------------------------------------------

def _make_llm(api_key: str, model: str, timeout: int = 60, max_retries: int = 1) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress(job_id: str, phase: str, pct: int, message: str, **extra: Any) -> None:
    job_registry.write_progress(job_id, phase, pct, message, **extra)
    print(f"[worker {job_id[:8]}] {phase} {pct:3d}% — {message}", flush=True)


def _mem_mb() -> float:
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _log(job_id: str, msg: str) -> None:
    print(f"[worker {job_id[:8]}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Cache helpers (scope = job_<job_id>)
# ---------------------------------------------------------------------------

def _cget(scope: str, phase: str, pdf_sha: str, params: str, page_idx=None):
    return fence_cache.get(phase, pdf_sha, params, page_idx=page_idx, user_scope=scope)


def _cput(scope: str, phase: str, pdf_sha: str, params: str, value, page_idx=None):
    return fence_cache.put(phase, pdf_sha, params, value, page_idx=page_idx, user_scope=scope)


# ---------------------------------------------------------------------------
# Phase 1a — native text extraction via batched subprocesses
# ---------------------------------------------------------------------------

def _extract_via_subprocess(pdf_path: str, page_idx: int, timeout: int) -> tuple:
    """Returns (lines, error|None)."""
    try:
        result = subprocess.run(
            [sys.executable, _EXTRACTOR_SCRIPT, pdf_path, str(page_idx)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, TimeoutError(f"page {page_idx+1} extraction exceeded {timeout}s")
    except Exception as e:
        return None, RuntimeError(f"subprocess launch failed: {e}")
    if not result.stdout:
        return None, RuntimeError(f"worker no output (rc={result.returncode}): {result.stderr[:200]}")
    try:
        resp = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        return None, RuntimeError(f"parse error: {e}; raw={result.stdout[:200]}")
    if resp.get("ok"):
        return resp.get("lines", []), None
    return None, RuntimeError(resp.get("error", "unknown worker failure"))


def _extract_batch_via_subprocess(pdf_path: str, batch_indices: list, per_page_timeout: int) -> dict | None:
    """Returns {page_idx: (lines, err)} for pages that finished before timeout."""
    import threading as _th

    pages_csv = ",".join(str(pi) for pi in batch_indices)
    batch_timeout = 3 + per_page_timeout * max(1, len(batch_indices))

    proc = subprocess.Popen(
        [sys.executable, _EXTRACTOR_SCRIPT, pdf_path, "--pages", pages_csv],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    collected: dict = {}

    def _drain():
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pr = obj.get("page_result")
                if pr and pr.get("page_idx") is not None:
                    pi = pr["page_idx"]
                    if pr.get("ok"):
                        collected[pi] = (pr.get("lines", []), None)
                    else:
                        collected[pi] = (None, RuntimeError(pr.get("error", "extract failed")))
        except Exception:
            pass

    t = _th.Thread(target=_drain, daemon=True)
    t.start()
    try:
        proc.wait(timeout=batch_timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    t.join(timeout=1)
    return collected


def _run_phase1a(
    job_id: str, scope: str, pdf_path: str, pdf_sha: str, cache_params: str,
    total_pages: int, broken: set,
) -> tuple[dict, dict, dict, set]:
    """Run Phase 1a. Returns (pdf_lines_by_page, page_dims, page_byte_estimates, broken)."""
    _progress(job_id, "1a", 0, f"extracting native text from {total_pages} pages…")

    pdf_lines_by_page: dict = {}
    page_dims: dict = {}
    page_byte_estimates: dict = {}
    pages_needing_extract: list = []
    cache_hits = 0

    # Page-dims pass — open fitz doc serially, collect dims + cache hits.
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        _log(job_id, f"Phase 1a: could not open PDF: {e}")
        return {}, {}, {}, broken

    for page_idx in range(total_pages):
        if page_idx in broken:
            page_dims[page_idx] = (0, 0)
            pdf_lines_by_page[page_idx] = []
            page_byte_estimates[page_idx] = 0
            continue
        try:
            page = doc[page_idx]
            page_dims[page_idx] = (page.rect.width, page.rect.height)
            _est = 0
            try:
                for img in page.get_images(full=True):
                    try:
                        s = doc.xref_stream_raw(img[0])
                        if s:
                            _est += len(s)
                    except Exception:
                        pass
                contents = page.get_contents()
                if contents:
                    for cx in contents:
                        try:
                            cs = doc.xref_stream_raw(cx)
                            if cs:
                                _est += len(cs)
                        except Exception:
                            pass
            except Exception:
                pass
            page_byte_estimates[page_idx] = _est
        except Exception as e:
            _log(job_id, f"Phase 1a page {page_idx+1} open failed: {e} — marking damaged")
            broken.add(page_idx)
            page_dims[page_idx] = (0, 0)
            pdf_lines_by_page[page_idx] = []
            page_byte_estimates[page_idx] = 0
            continue

        cached = _cget(scope, "phase1a", pdf_sha, cache_params, page_idx=page_idx)
        if cached is not None:
            pdf_lines_by_page[page_idx] = cached
            cache_hits += 1
            continue
        pages_needing_extract.append(page_idx)

    try:
        doc.close()
    except Exception:
        pass

    _log(job_id, f"Phase 1a: {len(pages_needing_extract)} pages need extraction, {cache_hits} cache hits")

    # Extraction pass — batched subprocesses.
    if pages_needing_extract:
        batches = [
            pages_needing_extract[i:i + FENCE_PHASE1A_BATCH_SIZE]
            for i in range(0, len(pages_needing_extract), FENCE_PHASE1A_BATCH_SIZE)
        ]

        def _run_batch(batch):
            partial = _extract_batch_via_subprocess(pdf_path, batch, PHASE_1A_PAGE_TIMEOUT) or {}
            missing = [pi for pi in batch if pi not in partial]
            for pi in missing:
                partial[pi] = _extract_via_subprocess(pdf_path, pi, PHASE_1A_PAGE_TIMEOUT)
            return partial

        complete = 0
        with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1A) as pool:
            futs = {pool.submit(_run_batch, b): b for b in batches}
            for fut in as_completed(futs):
                batch = futs[fut]
                try:
                    batch_results = fut.result()
                except Exception as fe:
                    batch_results = {pi: (None, fe) for pi in batch}
                for pi in batch:
                    lines, err = batch_results.get(pi, (None, RuntimeError("no result")))
                    complete += 1
                    if err is not None:
                        _log(job_id, f"Phase 1a page {pi+1}: {err} — marking damaged")
                        broken.add(pi)
                        pdf_lines_by_page[pi] = []
                    else:
                        pdf_lines_by_page[pi] = lines
                        _cput(scope, "phase1a", pdf_sha, cache_params, lines, page_idx=pi)
                pct = int(complete / max(len(pages_needing_extract), 1) * 15)
                _progress(job_id, "1a", pct, f"extracting text: {complete}/{len(pages_needing_extract)} pages")

    _progress(job_id, "1a", 15, f"Phase 1a done — {cache_hits}/{total_pages-len(broken)} cache hits")
    return pdf_lines_by_page, page_dims, page_byte_estimates, broken


# ---------------------------------------------------------------------------
# Phase 1b — Google OCR
# ---------------------------------------------------------------------------

def _run_phase1b(
    job_id: str, scope: str, pdf_path: str, pdf_sha: str, cache_params: str,
    total_pages: int, broken: set,
    pdf_lines_by_page: dict, page_dims: dict, page_byte_estimates: dict,
    google_cloud_config: dict | None,
) -> dict:
    """Run Phase 1b OCR. Returns ocr_lines_by_page."""
    ocr_lines: dict = {i: [] for i in range(total_pages)}

    if not google_cloud_config:
        _progress(job_id, "1b", 22, "Phase 1b skipped — no Google Cloud config")
        return ocr_lines

    _progress(job_id, "1b", 15, f"running OCR on {total_pages} pages (batched)…")

    ocr_page_indices = []
    cache_hits = 0
    for pi in range(total_pages):
        if pi in broken:
            continue
        cached = _cget(scope, "phase1b", pdf_sha, cache_params, page_idx=pi)
        if cached is not None:
            ocr_lines[pi] = cached
            cache_hits += 1
        else:
            ocr_page_indices.append(pi)

    if not ocr_page_indices:
        _progress(job_id, "1b", 22, f"Phase 1b done — all {cache_hits} pages cached")
        return ocr_lines

    # Size-aware batch packing.
    _OCR_BATCH_TARGET_BYTES = int(os.environ.get("FENCE_OCR_BATCH_TARGET_BYTES", str(30 * 1024 * 1024)))
    _OCR_PER_PAGE_OVERHEAD  = 50 * 1024
    _OCR_TOTAL_MAX_BYTES    = int(os.environ.get("FENCE_OCR_TOTAL_MAX_BYTES", str(35 * 1024 * 1024)))

    ocr_batches: list = []
    current_batch: list = []
    current_size = 0
    for pi in ocr_page_indices:
        est = page_byte_estimates.get(pi, 0) + _OCR_PER_PAGE_OVERHEAD
        if current_batch and (
            current_size + est > _OCR_BATCH_TARGET_BYTES
            or len(current_batch) >= FENCE_OCR_BATCH_SIZE
        ):
            ocr_batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(pi)
        current_size += est
    if current_batch:
        ocr_batches.append(current_batch)

    _log(job_id, f"Phase 1b: {len(ocr_batches)} OCR batches from {len(ocr_page_indices)} pages")

    # Lazy single-page PDF builder (fallback when disk path unavailable).
    _single_page_pdfs: dict = {}
    _single_page_misses = [0]

    def _get_single_page_pdf(page_idx):
        if page_idx in _single_page_pdfs:
            return _single_page_pdfs[page_idx]
        if not os.path.exists(pdf_path):
            return None
        _single_page_misses[0] += 1
        try:
            src = fitz.open(pdf_path)
            try:
                tmp = fitz.open()
                tmp.insert_pdf(src, from_page=page_idx, to_page=page_idx)
                data = tmp.tobytes()
                tmp.close()
            finally:
                src.close()
            _single_page_pdfs[page_idx] = data
            return data
        except Exception as e:
            _log(job_id, f"lazy single-page PDF build failed for page {page_idx+1}: {e}")
            return None

    def _run_ocr_batch(batch_indices):
        try:
            if os.path.exists(pdf_path):
                batch_pdf = ade.safe_multi_page_pdf_from_path(
                    pdf_path, batch_indices, per_page_max_bytes=_OCR_TOTAL_MAX_BYTES,
                )
                if len(batch_pdf) > _OCR_TOTAL_MAX_BYTES:
                    tight_cap = max(256 * 1024, _OCR_TOTAL_MAX_BYTES // max(len(batch_indices), 1))
                    batch_pdf = ade.safe_multi_page_pdf_from_path(
                        pdf_path, batch_indices, per_page_max_bytes=tight_cap,
                    )
            else:
                tmpdoc = fitz.open()
                for pi in batch_indices:
                    data = _get_single_page_pdf(pi)
                    if data is None:
                        continue
                    sp = fitz.open(stream=data, filetype="pdf")
                    tmpdoc.insert_pdf(sp, from_page=0, to_page=0)
                    sp.close()
                batch_pdf = tmpdoc.tobytes()
                tmpdoc.close()
            page_dims_by_local = {
                local: page_dims[orig] for local, orig in enumerate(batch_indices)
            }
            result_by_local = ade.run_google_ocr_blocks_multipage(
                batch_pdf, google_cloud_config, page_dims_by_local,
            )
            return {orig: result_by_local.get(local, []) for local, orig in enumerate(batch_indices)}
        except Exception as be:
            _log(job_id, f"OCR batch ({len(batch_indices)}p) failed: {be}; falling back per-page")
            out = {}
            for pi in batch_indices:
                try:
                    pdf_w, pdf_h = page_dims[pi]
                    spd = _get_single_page_pdf(pi)
                    if spd is None:
                        out[pi] = []
                        continue
                    out[pi] = ade.run_google_ocr_blocks(spd, google_cloud_config, pdf_w, pdf_h)
                except Exception as pe:
                    _log(job_id, f"OCR fallback page {pi+1} failed: {pe}")
                    out[pi] = []
            return out

    pages_done = 0
    batches_done = 0
    with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1B) as executor:
        futures = {executor.submit(_run_ocr_batch, b): bi for bi, b in enumerate(ocr_batches)}
        for future in as_completed(futures):
            batches_done += 1
            try:
                batch_result = future.result()
            except Exception as e:
                bi = futures[future]
                _log(job_id, f"OCR batch {bi} worker crashed: {e}")
                batch_result = {pi: [] for pi in ocr_batches[bi]}
            for pi, result in batch_result.items():
                ocr_lines[pi] = result
                _cput(scope, "phase1b", pdf_sha, cache_params, result, page_idx=pi)
                pages_done += 1
            pct = 15 + int(pages_done / max(len(ocr_page_indices), 1) * 7)
            _progress(job_id, "1b", pct, f"OCR: {pages_done}/{len(ocr_page_indices)} pages")

    _single_page_pdfs.clear()
    gc.collect()
    _progress(job_id, "1b", 22,
              f"Phase 1b done — {len(ocr_batches)} batches, {cache_hits}/{total_pages-len(broken)} cached")
    return ocr_lines


# ---------------------------------------------------------------------------
# Phase 1c — keyword scan + batched LLM fence classification
# ---------------------------------------------------------------------------

def _run_phase1c(
    job_id: str, scope: str, pdf_sha: str, cache_params: str,
    total_pages: int, broken: set,
    pdf_lines_by_page: dict, ocr_lines_by_page: dict,
    fence_keywords: list,
    llm_analysis_instance,
    classifier_llm_instance,
) -> tuple[dict, list]:
    """Run Phase 1c. Returns (page_cache, fence_page_indices)."""
    _progress(job_id, "1c", 22, "classifying pages…")

    page_cache: dict = {}
    fence_page_indices: list = []
    pending_indices: list = []
    cache_hits = 0

    _HIGH_SIGNAL_KW = {
        'fence', 'fencing', 'gate', 'gates', 'chain link', 'guardrail',
        'railing', 'handrail', 'bollard', 'barrier',
    }

    for page_idx in range(total_pages):
        if page_idx in broken:
            page_cache[page_idx] = {
                'pdf_lines': [], 'ocr_lines': [],
                'prefilter_result': {"fence_found": False, "method": "skipped_damaged", "matched_lines": []},
                'skipped_damaged': True,
            }
            continue
        cached = _cget(scope, "phase1c", pdf_sha, cache_params, page_idx=page_idx)
        if cached is not None:
            page_cache[page_idx] = {
                'pdf_lines': pdf_lines_by_page.get(page_idx, []),
                'ocr_lines': ocr_lines_by_page.get(page_idx, []),
                'prefilter_result': cached,
            }
            if cached.get("fence_found"):
                fence_page_indices.append(page_idx)
            cache_hits += 1
            continue
        pending_indices.append(page_idx)

    # Keyword scan — fast, no network.
    needs_llm: list = []
    scan_results: dict = {}
    for page_idx in pending_indices:
        kres = ade.scan_page_for_keywords_fast(
            pdf_lines_by_page.get(page_idx, []),
            ocr_lines_by_page.get(page_idx, []),
            fence_keywords,
        )
        scan_results[page_idx] = kres
        if not kres["has_keywords"]:
            continue
        matched_lower = {kw.lower() for kw in kres["matched_keywords"]}
        if matched_lower & _HIGH_SIGNAL_KW:
            continue
        all_lines = pdf_lines_by_page.get(page_idx, []) + ocr_lines_by_page.get(page_idx, [])
        page_text = " ".join(line.get("text", "") for line in all_lines)
        needs_llm.append((page_idx, page_text, kres["matched_keywords"]))

    # Batched LLM classification.
    llm_by_page: dict = {}
    classifier = classifier_llm_instance or llm_analysis_instance
    if needs_llm and classifier:
        batches = [needs_llm[i:i + FENCE_CLASSIFY_BATCH_SIZE]
                   for i in range(0, len(needs_llm), FENCE_CLASSIFY_BATCH_SIZE)]

        def _run_batch(batch):
            return ade.llm_classify_pages_batch(
                classifier, batch, fence_keywords, batch_size=FENCE_CLASSIFY_BATCH_SIZE,
            )

        done = 0
        with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1C) as pool:
            futs = {pool.submit(_run_batch, b): bi for bi, b in enumerate(batches)}
            for fut in as_completed(futs):
                done += 1
                try:
                    llm_by_page.update(fut.result())
                except Exception as be:
                    _log(job_id, f"Phase 1c LLM batch {futs[fut]} failed: {be}")
                pct = 22 + int(done / max(len(batches), 1) * 4)
                _progress(job_id, "1c", pct, f"classifying: {done}/{len(batches)} LLM batches")

    # Merge decisions.
    for page_idx in pending_indices:
        kres = scan_results[page_idx]
        if not kres["has_keywords"]:
            prefilter = {"fence_found": False, "method": "keyword_scan",
                         "matched_keywords": [], "matched_lines": [], "llm_result": None}
        elif {kw.lower() for kw in kres["matched_keywords"]} & _HIGH_SIGNAL_KW:
            prefilter = {"fence_found": True, "method": "keyword_high_signal",
                         "matched_keywords": kres["matched_keywords"],
                         "matched_lines": kres["matched_lines"], "llm_result": None}
        else:
            llm_r = llm_by_page.get(page_idx)
            if llm_r and llm_r.get("confidence", 0.0) >= 0.5:
                prefilter = {"fence_found": bool(llm_r.get("is_fence_related", False)),
                             "method": "llm_confirmed",
                             "matched_keywords": kres["matched_keywords"],
                             "matched_lines": kres["matched_lines"], "llm_result": llm_r}
            else:
                prefilter = {"fence_found": True, "method": "keyword_only",
                             "matched_keywords": kres["matched_keywords"],
                             "matched_lines": kres["matched_lines"], "llm_result": llm_r}

        page_cache[page_idx] = {
            'pdf_lines': pdf_lines_by_page.get(page_idx, []),
            'ocr_lines': ocr_lines_by_page.get(page_idx, []),
            'prefilter_result': prefilter,
        }
        if prefilter.get("fence_found"):
            fence_page_indices.append(page_idx)
        _cput(scope, "phase1c", pdf_sha, cache_params, prefilter, page_idx=page_idx)

    fence_page_indices.sort()
    _log(job_id, f"Phase 1c done: {len(fence_page_indices)}/{total_pages} fence pages, {cache_hits} cached")
    _progress(job_id, "1c", 26,
              f"Phase 1c done — {len(fence_page_indices)} fence pages detected")
    return page_cache, fence_page_indices


# ---------------------------------------------------------------------------
# Phase 2 — ADE (LandingAI) for fence pages
# ---------------------------------------------------------------------------

def _run_phase2(
    job_id: str, scope: str, pdf_path: str, pdf_sha: str, cache_params: str,
    fence_page_indices: list, broken: set, page_dims: dict,
    ade_key: str | None, use_ade: bool,
) -> dict:
    """Run Phase 2 ADE. Returns ade_chunks_by_page."""
    ade_chunks: dict = {}

    if not (use_ade and ade_key and fence_page_indices):
        if not use_ade:
            _progress(job_id, "2", 35, "Phase 2 skipped — ADE disabled")
        elif not ade_key:
            _progress(job_id, "2", 35, "Phase 2 skipped — no ADE key")
        else:
            _progress(job_id, "2", 35, "Phase 2 skipped — no fence pages")
        return ade_chunks

    # Filter out already-broken pages.
    fence_pages_filtered = [i for i in fence_page_indices if i not in broken]

    _progress(job_id, "2", 26, f"{len(fence_pages_filtered)} fence pages queued for ADE…")

    cache_hits = 0
    pages_to_fetch: list = []
    for pi in fence_pages_filtered:
        cached = _cget(scope, "phase2", pdf_sha, cache_params, page_idx=pi)
        if cached is not None:
            ade_chunks[pi] = cached
            cache_hits += 1
        else:
            pages_to_fetch.append(pi)

    _log(job_id, f"Phase 2: {cache_hits} cached, {len(pages_to_fetch)} need ADE")

    if not pages_to_fetch:
        _progress(job_id, "2", 35, f"Phase 2 done — all {cache_hits} pages cached")
        return ade_chunks

    _ADE_BATCH_MAX_BYTES = int(os.environ.get("FENCE_ADE_BATCH_MAX_BYTES", str(15 * 1024 * 1024)))
    _ADE_PAGE_MAX_BYTES  = int(os.environ.get("FENCE_ADE_PAGE_MAX_BYTES",  str(12 * 1024 * 1024)))

    if os.path.exists(pdf_path):
        batches = ade.create_page_batches_from_path(
            pdf_path, pages_to_fetch,
            max_batch_bytes=_ADE_BATCH_MAX_BYTES,
            max_pages_per_batch=1,
        ) if pages_to_fetch else []
    else:
        batches = []

    _ade_consecutive_timeouts = [0]
    _ade_degraded = [False]
    _ADE_DEGRADE_THRESHOLD = int(os.environ.get("FENCE_ADE_DEGRADE_THRESHOLD", "3"))

    def _retry_single(orig_idx, result):
        try:
            if os.path.exists(pdf_path):
                single_pdf = ade.safe_single_page_pdf_from_path(
                    pdf_path, orig_idx, max_bytes=_ADE_PAGE_MAX_BYTES,
                )
            else:
                return
            single_resp = ade.ade_parse_document(single_pdf, ade_key)
            del single_pdf
            if single_resp["success"]:
                pdf_w, pdf_h = page_dims.get(orig_idx, (0.0, 0.0))
                chunks = ade.align_ade_chunks_to_page(single_resp, 0, pdf_w, pdf_h)
                result[orig_idx] = chunks
                _cput(scope, "phase2", pdf_sha, cache_params, chunks, page_idx=orig_idx)
                return bool(chunks)
            result[orig_idx] = None
        except Exception as e:
            _log(job_id, f"ADE page {orig_idx+1} retry error: {e}")
            result[orig_idx] = None

    def _run_phase2_batch(batch_idx, batch):
        result = {}
        if len(batch) == 1:
            _retry_single(batch[0], result)
            return result
        # Defensive multi-page path (batch size == 1 above handles normal case).
        try:
            if os.path.exists(pdf_path):
                batch_pdf = ade.safe_multi_page_pdf_from_path(
                    pdf_path, batch, per_page_max_bytes=_ADE_PAGE_MAX_BYTES,
                )
            else:
                for orig_idx in batch:
                    result[orig_idx] = None
                return result
        except Exception as e:
            _log(job_id, f"ADE batch {batch_idx+1} PDF-build failed: {e}")
            for orig_idx in batch:
                result[orig_idx] = None
            return result
        resp = ade.ade_parse_document(batch_pdf, ade_key)
        del batch_pdf
        if resp["success"]:
            for local_idx, orig_idx in enumerate(batch):
                try:
                    pdf_w, pdf_h = page_dims.get(orig_idx, (0.0, 0.0))
                    chunks = ade.align_ade_chunks_to_page(resp, local_idx, pdf_w, pdf_h)
                    result[orig_idx] = chunks
                    _cput(scope, "phase2", pdf_sha, cache_params, chunks, page_idx=orig_idx)
                except Exception as e:
                    result[orig_idx] = None
        else:
            for orig_idx in batch:
                _retry_single(orig_idx, result)
        return result

    batch_done = 0
    with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE2) as pool:
        futs = {pool.submit(_run_phase2_batch, bi, b): bi for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            batch_done += 1
            try:
                ade_chunks.update(fut.result())
            except Exception as e:
                bi = futs[fut]
                _log(job_id, f"Phase 2 batch {bi} crashed: {e}")
                for oi in batches[bi]:
                    ade_chunks.setdefault(oi, None)
            pct = 26 + int(batch_done / max(len(batches), 1) * 9)
            _progress(job_id, "2", pct, f"ADE: {batch_done}/{len(batches)} pages")

    ok = sum(1 for v in ade_chunks.values() if v is not None)
    _log(job_id, f"Phase 2 done: ADE results for {ok}/{len(fence_pages_filtered)} fence pages")
    _progress(job_id, "2", 35, f"Phase 2 done — {ok}/{len(fence_pages_filtered)} fence pages processed")
    return ade_chunks


# ---------------------------------------------------------------------------
# Phase 3a — pre-compute legend / scale / measure for fence pages
# ---------------------------------------------------------------------------

def _run_phase3a(
    job_id: str, scope: str, pdf_path: str, pdf_sha: str, cache_params: str,
    fence_page_indices: list, broken: set,
    pdf_lines_by_page: dict, ocr_lines_by_page: dict, ade_chunks_by_page: dict,
    fence_keywords: list,
    llm_analysis_instance,
    classifier_llm_instance,
    highlight_fence_text: bool,
    enable_unified_measurement: bool,
    analysis_model: str,
    classifier_model: str,
    openai_key: str,
) -> None:
    """Run Phase 3a pre-compute. Results written to fence_cache."""
    fence_pages = [i for i in fence_page_indices if i not in broken]

    if not fence_pages or not os.path.exists(pdf_path):
        _progress(job_id, "3a", 90, "Phase 3a skipped — no fence pages or PDF missing")
        return

    _progress(job_id, "3a", 35, f"pre-computing {len(fence_pages)} fence pages…")

    # --- Legend pre-batch --------------------------------------------------
    prefill_legend: dict = {}
    segmented_by_page: dict = {}

    if fence_pages and llm_analysis_instance:
        to_batch_legend: dict = {}
        for pi in fence_pages:
            if _cget(scope, "phase3_legend", pdf_sha, cache_params, page_idx=pi) is not None:
                continue
            ade_ch = ade_chunks_by_page.get(pi) or []
            if not ade_ch:
                continue
            leg, fig = ade.segment_chunks(ade_ch)
            segmented_by_page[pi] = (leg, fig)
            for ci, chunk in enumerate(leg):
                txt = chunk.get("text", "")
                if txt:
                    to_batch_legend[(pi, ci)] = txt

        skip_prebatch = _env_bool("FENCE_SKIP_LEGEND_PREBATCH", False)
        if skip_prebatch:
            _log(job_id, "Phase 3a: legend pre-batch SKIPPED (FENCE_SKIP_LEGEND_PREBATCH=true)")
            to_batch_legend = {}

        if to_batch_legend:
            _legend_batch_size = int(os.environ.get("FENCE_LEGEND_BATCH_SIZE", "6"))
            _legend_workers    = _workers("FENCE_LEGEND_PREBATCH_WORKERS", 4, cap=8)
            _SHARD_TIMEOUT     = _workers("FENCE_LEGEND_SHARD_TIMEOUT", 120, cap=600)

            legend_keys = list(to_batch_legend.keys())

            def _shards(keys, n):
                if n <= 1 or len(keys) <= _legend_batch_size:
                    return [keys]
                chunk = max(_legend_batch_size, (len(keys) + n - 1) // n)
                return [keys[i:i + chunk] for i in range(0, len(keys), chunk)]

            k_shards = _shards(legend_keys, _legend_workers)
            _log(job_id, f"Phase 3a: legend pre-batch — {len(to_batch_legend)} chunks, {len(k_shards)} shards")
            _progress(job_id, "3a", 36, f"legend pre-batch ({len(to_batch_legend)} chunks)…")

            def _run_legend_shard(shard_keys):
                shard_dict = {k: to_batch_legend[k] for k in shard_keys}
                return ade.llm_extract_fence_elements_batch(
                    llm_analysis_instance, shard_dict, fence_keywords,
                    batch_size=_legend_batch_size,
                )

            items_by_id: dict = {}
            import concurrent.futures as _cf
            try:
                with ThreadPoolExecutor(max_workers=len(k_shards)) as lp:
                    lfuts = {lp.submit(_run_legend_shard, s): i for i, s in enumerate(k_shards)}
                    for lf in list(lfuts.keys()):
                        try:
                            part = lf.result(timeout=_SHARD_TIMEOUT)
                            items_by_id.update(part or {})
                        except _cf.TimeoutError:
                            _log(job_id, f"legend shard {lfuts[lf]} timed out after {_SHARD_TIMEOUT}s — skipping")
                        except Exception as se:
                            _log(job_id, f"legend shard {lfuts[lf]} failed: {se}")
                    lp.shutdown(wait=False, cancel_futures=True)
                for key, items in items_by_id.items():
                    prefill_legend[key] = items
            except Exception as be:
                _log(job_id, f"Phase 3a legend pre-batch failed: {be}")

    # --- Per-page precompute workers ---------------------------------------

    def _phase3_precompute_thread(page_idx):
        """Thread-based worker: legend, scale, measurement for one page."""
        worker_doc = None
        try:
            worker_doc = fitz.open(pdf_path)
            try:
                worker_page = worker_doc[page_idx]
            except Exception:
                return
            pdf_lines = pdf_lines_by_page.get(page_idx, [])
            ocr_lines_ = ocr_lines_by_page.get(page_idx, [])
            ade_ch = ade_chunks_by_page.get(page_idx) or []

            if page_idx in segmented_by_page:
                legend_chunks, figure_chunks = segmented_by_page[page_idx]
            else:
                legend_chunks, figure_chunks = ade.segment_chunks(ade_ch) if ade_ch else ([], [])

            page_legend_prefill = {
                pci: items for (ppi, pci), items in prefill_legend.items() if ppi == page_idx
            }

            # 1. Legend entries.
            definitions = []
            if highlight_fence_text and legend_chunks:
                cached = _cget(scope, "phase3_legend", pdf_sha, cache_params, page_idx=page_idx)
                if cached is not None:
                    definitions = cached
                else:
                    try:
                        definitions = ade.extract_legend_entries(
                            legend_chunks=legend_chunks,
                            pdf_lines=pdf_lines,
                            ocr_lines=ocr_lines_,
                            fence_keywords=fence_keywords,
                            llm=llm_analysis_instance,
                            figure_chunks=figure_chunks,
                            prefilled_legend_items=page_legend_prefill or None,
                        )
                        _cput(scope, "phase3_legend", pdf_sha, cache_params, definitions, page_idx=page_idx)
                    except Exception as le:
                        _log(job_id, f"page {page_idx+1} legend error: {le}")

            # 2. Page tokens.
            try:
                rotation = worker_page.rotation
                mediabox_w = worker_page.mediabox.width
                mediabox_h = worker_page.mediabox.height
                native_words = worker_page.get_text("words")
            except Exception:
                native_words, rotation, mediabox_w, mediabox_h = [], 0, 0, 0

            def _xform(x0, y0, x1, y1):
                if rotation == 90:
                    return mediabox_h - y1, x0, mediabox_h - y0, x1
                if rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                if rotation == 270:
                    return y0, mediabox_w - x1, y1, mediabox_w - x0
                return x0, y0, x1, y1

            all_page_tokens = []
            for w in native_words:
                nx0, ny0, nx1, ny1 = _xform(w[0], w[1], w[2], w[3])
                all_page_tokens.append({"text": w[4], "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1})

            # 3. Instances.
            instances = []
            if definitions and figure_chunks:
                try:
                    instances = ade.find_instances_in_figures_fast(
                        definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines_,
                    )
                except Exception as ie:
                    _log(job_id, f"page {page_idx+1} instances error: {ie}")

            # 4. Scale detection.
            detected_scale = None
            scale_cached = _cget(scope, "phase3_scale", pdf_sha, cache_params, page_idx=page_idx)
            if scale_cached is not None:
                detected_scale = scale_cached.get('verified_scale')
            else:
                try:
                    # Use same model as analysis; worker creates own LLM instance for scale.
                    scale_llm = _make_llm(openai_key, analysis_model, timeout=90, max_retries=1)
                    scale_info = verify_scale_with_bar_fast(worker_page, llm=scale_llm)
                    if scale_info.get('success') is not False or scale_info.get('verified_scale'):
                        _cput(scope, "phase3_scale", pdf_sha, cache_params, scale_info, page_idx=page_idx)
                    if scale_info.get('success') and scale_info.get('verified_scale'):
                        detected_scale = scale_info['verified_scale']
                except Exception as se:
                    _log(job_id, f"page {page_idx+1} scale error: {se}")

            # 5. Measurement.
            if enable_unified_measurement and (definitions or instances):
                if _cget(scope, "phase3_measure", pdf_sha, cache_params, page_idx=page_idx) is None:
                    try:
                        ocr_full_text = "\n".join(line.get('text', '') for line in ocr_lines_) if ocr_lines_ else None
                        meas = ade.measure_fence_elements(
                            worker_page, definitions, instances,
                            figure_chunks=figure_chunks,
                            llm=llm_analysis_instance,
                            light_llm=classifier_llm_instance,
                            scale_factor=detected_scale or 1.0,
                            ocr_text=ocr_full_text,
                        )
                        if meas:
                            _cput(scope, "phase3_measure", pdf_sha, cache_params, meas, page_idx=page_idx)
                    except Exception as me:
                        _log(job_id, f"page {page_idx+1} measure error: {me}")
        finally:
            if worker_doc is not None:
                try:
                    worker_doc.close()
                except Exception:
                    pass

    def _phase3_precompute_subprocess(page_idx):
        """Subprocess-isolated worker (same as app_ade_fast.py version)."""
        page_prefill = {
            pci: items for (ppi, pci), items in prefill_legend.items() if ppi == page_idx
        }
        task = {
            "pdf_path":                   pdf_path,
            "page_idx":                   int(page_idx),
            "pdf_sha":                    pdf_sha,
            "cache_params":               cache_params,
            "user_scope":                 scope,
            "fence_cache_dir":            os.environ.get("FENCE_CACHE_DIR", ""),
            "openai_api_key":             openai_key or "",
            "analysis_model":             analysis_model,
            "classifier_model":           classifier_model,
            "scale_model":                analysis_model,
            "fence_keywords":             list(fence_keywords),
            "pdf_lines":                  pdf_lines_by_page.get(page_idx, []),
            "ocr_lines":                  ocr_lines_by_page.get(page_idx, []),
            "ade_chunks":                 ade_chunks_by_page.get(page_idx) or [],
            "legend_prefill":             page_prefill,
            "highlight_fence_text_app":   bool(highlight_fence_text),
            "enable_unified_measurement": bool(enable_unified_measurement),
        }
        task_json = json.dumps(task, default=str)
        wallclock_cap = FENCE_PHASE3_PAGE_TIMEOUT + 15

        proc = subprocess.Popen(
            [sys.executable, _PHASE3_WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            out, err = proc.communicate(task_json, timeout=wallclock_cap)
            if proc.returncode != 0:
                try:
                    rec = json.loads(out.strip().splitlines()[-1])
                    raise RuntimeError(f"subprocess exit={proc.returncode}: {rec.get('error', 'unknown')}")
                except Exception:
                    raise RuntimeError(f"phase3 worker exit={proc.returncode}; stderr={err[:200]}")
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            raise TimeoutError(f"phase3 worker page {page_idx+1} exceeded {wallclock_cap}s")

    def _precompute_timed(page_idx):
        t0 = time.perf_counter()
        status = "ok"
        try:
            if FENCE_PHASE3_USE_SUBPROCESS:
                _phase3_precompute_subprocess(page_idx)
            else:
                _phase3_precompute_thread(page_idx)
        except Exception as pe:
            status = f"error:{type(pe).__name__}"
            _log(job_id, f"phase3_precompute page {page_idx+1} raised: {pe}")
        dt = time.perf_counter() - t0
        try:
            telemetry.event(
                "phase3_page_done_worker",
                job_id=job_id,
                page_idx=page_idx,
                wall_s=round(dt, 3),
                status=status,
                worker=("subprocess" if FENCE_PHASE3_USE_SUBPROCESS else "thread"),
            )
        except Exception:
            pass

    # Choose lazy vs. eager.
    if FENCE_PHASE3_EAGER:
        p3_pages = list(fence_pages)
        p3_mode = "eager"
    else:
        p3_pages = list(fence_pages[:FENCE_PHASE3_PREVIEW])
        p3_mode = f"lazy (preview {len(p3_pages)}/{len(fence_pages)})"

    _log(job_id, f"Phase 3a: {p3_mode}, {FENCE_WORKERS_PHASE3} workers")

    import concurrent.futures as _cf2
    p3timeouts = 0
    p3done = 0

    if p3_pages:
        with ThreadPoolExecutor(max_workers=max(1, min(FENCE_WORKERS_PHASE3, len(p3_pages)))) as p3pool:
            p3futs = {p3pool.submit(_precompute_timed, pi): pi for pi in p3_pages}
            for pf, pi in list(p3futs.items()):
                try:
                    pf.result(timeout=FENCE_PHASE3_PAGE_TIMEOUT)
                except _cf2.TimeoutError:
                    p3timeouts += 1
                    _log(job_id, f"phase3_precompute page {pi+1} exceeded {FENCE_PHASE3_PAGE_TIMEOUT}s — skipping")
                except Exception as re_:
                    _log(job_id, f"phase3_precompute page {pi+1} raised: {re_}")
                p3done += 1
                pct = 35 + int(p3done / max(len(p3_pages), 1) * 55)
                _progress(job_id, "3a", pct, f"pre-compute: {p3done}/{len(p3_pages)} pages")
            p3pool.shutdown(wait=False, cancel_futures=True)

    timeout_note = f", {p3timeouts} timeouts" if p3timeouts else ""
    _log(job_id, f"Phase 3a done: {p3_mode}, {FENCE_WORKERS_PHASE3} workers{timeout_note}")
    _progress(job_id, "3a", 90, f"Phase 3a done — {len(p3_pages)} fence pages pre-computed")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_job(job_id: str, config: dict) -> None:
    """Run the full analysis (Phases 1a–3a) for a job.

    This function is designed to run as a daemon=False thread.  It writes
    progress to the job registry throughout and saves a results_summary.json
    that the Streamlit session reads to kick off Phase 3b.
    """
    pdf_path           = config["pdf_path"]
    pdf_hash           = config.get("pdf_hash", "")
    openai_key         = config.get("openai_key", "")
    ade_key_           = config.get("ade_key")
    google_cloud_config = config.get("google_cloud_config")
    analysis_model     = config.get("analysis_model", "gpt-5.1")
    classifier_model   = config.get("classifier_model", "gpt-5-mini")
    fence_keywords     = config.get("fence_keywords", [])
    use_ade            = bool(config.get("use_ade", True))
    highlight_fence_text = bool(config.get("highlight_fence_text", True))
    enable_unified_measurement = bool(config.get("enable_unified_measurement", True))
    broken_pages_input = set(config.get("broken_pages", []))
    cache_params       = config.get("cache_params", "")

    # Job-scoped cache namespace — never collides with a Streamlit session.
    scope = f"job_{job_id}"

    # Recompute pdf_hash from file if not provided.
    if not pdf_hash:
        try:
            import hashlib as _hl
            with open(pdf_path, "rb") as _f:
                pdf_hash = _hl.sha256(_f.read()).hexdigest()
        except Exception as e:
            _log(job_id, f"could not hash PDF: {e} — using empty string")
            pdf_hash = ""

    # Discover total_pages.
    total_pages = config.get("total_pages", 0)
    if not total_pages:
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
        except Exception as e:
            _log(job_id, f"could not open PDF to count pages: {e}")
            job_registry.update_job(job_id, status="failed", error_msg=f"PDF unreadable: {e}",
                                    completed_at=int(time.time()))
            _progress(job_id, "failed", 0, f"Failed: {e}")
            return

    # Build cache_params if not supplied.
    if not cache_params:
        try:
            cache_params = fence_cache.params_hash(
                model=analysis_model,
                keywords=tuple(sorted(fence_keywords)),
                use_ade=bool(use_ade and ade_key_),
                highlight_fence_text=bool(highlight_fence_text),
                unified_measurement=bool(enable_unified_measurement),
                dpi=150,
            )
        except Exception as e:
            _log(job_id, f"could not build cache_params: {e}")
            cache_params = ""

    _log(job_id, f"starting — {total_pages} pages, cache key: {pdf_hash[:8]}/{cache_params}")

    # Mark job as running.
    job_registry.update_job(
        job_id,
        status="running",
        started_at=int(time.time()),
        total_pages=total_pages,
    )
    _progress(job_id, "starting", 0, f"Analysis started — {total_pages} pages")

    # Build LLM instances.
    llm_analysis = None
    classifier_llm = None
    if openai_key:
        try:
            llm_analysis   = _make_llm(openai_key, analysis_model, timeout=60, max_retries=1)
            classifier_llm = _make_llm(openai_key, classifier_model, timeout=60, max_retries=2)
        except Exception as e:
            _log(job_id, f"LLM init failed: {e} — classification will degrade to keyword-only")

    phase_timings: dict = {}

    try:
        # ---- Phase 1a ----
        t0 = time.perf_counter()
        broken = set(broken_pages_input)
        pdf_lines, page_dims, page_byte_estimates, broken = _run_phase1a(
            job_id, scope, pdf_path, pdf_hash, cache_params, total_pages, broken,
        )
        phase_timings["1a"] = time.perf_counter() - t0
        _log(job_id, f"Phase 1a done in {phase_timings['1a']:.1f}s")
        # Touch PDF to prevent cleanup daemon from sweeping it.
        try:
            os.utime(pdf_path)
        except Exception:
            pass

        # ---- Phase 1b ----
        t0 = time.perf_counter()
        ocr_lines = _run_phase1b(
            job_id, scope, pdf_path, pdf_hash, cache_params, total_pages, broken,
            pdf_lines, page_dims, page_byte_estimates, google_cloud_config,
        )
        phase_timings["1b"] = time.perf_counter() - t0
        _log(job_id, f"Phase 1b done in {phase_timings['1b']:.1f}s")
        try:
            os.utime(pdf_path)
        except Exception:
            pass

        # ---- Phase 1c ----
        t0 = time.perf_counter()
        page_cache, fence_page_indices = _run_phase1c(
            job_id, scope, pdf_hash, cache_params, total_pages, broken,
            pdf_lines, ocr_lines, fence_keywords, llm_analysis, classifier_llm,
        )
        phase_timings["1c"] = time.perf_counter() - t0
        _log(job_id, f"Phase 1c done in {phase_timings['1c']:.1f}s")
        try:
            os.utime(pdf_path)
        except Exception:
            pass

        # Compact non-fence text (mirror what app_ade_fast.py does).
        fence_set = set(fence_page_indices)
        for pi in range(total_pages):
            if pi in fence_set or pi in broken:
                continue
            if pi in page_cache:
                page_cache[pi]['pdf_lines'] = []
                page_cache[pi]['ocr_lines'] = []
            pdf_lines[pi] = []
            ocr_lines[pi] = []
        gc.collect()

        # Update registry with fence count.
        job_registry.update_job(
            job_id,
            fence_count=len(fence_page_indices),
            non_fence_count=total_pages - len(fence_page_indices) - len(broken),
        )

        # ---- Phase 2 ----
        t0 = time.perf_counter()
        ade_chunks = _run_phase2(
            job_id, scope, pdf_path, pdf_hash, cache_params,
            fence_page_indices, broken, page_dims, ade_key_, use_ade,
        )
        phase_timings["2"] = time.perf_counter() - t0
        _log(job_id, f"Phase 2 done in {phase_timings['2']:.1f}s")

        # Drop file_bytes analogue — nothing to do here since we never loaded them.
        gc.collect()
        try:
            os.utime(pdf_path)
        except Exception:
            pass

        # ---- Phase 3a ----
        t0 = time.perf_counter()
        _run_phase3a(
            job_id, scope, pdf_path, pdf_hash, cache_params,
            fence_page_indices, broken,
            pdf_lines, ocr_lines, ade_chunks,
            fence_keywords, llm_analysis, classifier_llm,
            highlight_fence_text, enable_unified_measurement,
            analysis_model, classifier_model, openai_key,
        )
        phase_timings["3a"] = time.perf_counter() - t0
        _log(job_id, f"Phase 3a done in {phase_timings['3a']:.1f}s")

        # ---- Write results summary ----------------------------------------
        results_dir = Path(job_registry.get_job(job_id).get("results_dir") or
                           Path("~/.leo/results").expanduser() / job_id)
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "job_id":             job_id,
            "pdf_sha":            pdf_hash,
            "cache_params":       cache_params,
            "cache_scope":        scope,
            "fence_page_indices": fence_page_indices,
            "total_pages":        total_pages,
            "broken_pages":       sorted(broken),
            "phase_timings":      {k: round(v, 2) for k, v in phase_timings.items()},
            "phase3_pending":     [i for i in fence_page_indices
                                   if i not in (fence_page_indices[:FENCE_PHASE3_PREVIEW] if not FENCE_PHASE3_EAGER else fence_page_indices)],
        }
        summary_path = results_dir / "results_summary.json"
        tmp_path     = results_dir / "results_summary.json.tmp"
        tmp_path.write_bytes(json.dumps(summary, default=str).encode())
        os.replace(str(tmp_path), str(summary_path))
        _log(job_id, f"results_summary.json written to {summary_path}")

        # ---- Mark complete ------------------------------------------------
        job_registry.update_job(
            job_id,
            status="phases_ready",
            completed_at=int(time.time()),
        )
        _progress(job_id, "phases_ready", 90,
                  f"Ready — {len(fence_page_indices)} fence pages detected",
                  fence_page_indices=fence_page_indices,
                  total_pages=total_pages,
                  broken_pages=sorted(broken))
        _log(job_id, "analysis complete — status=phases_ready")

    except Exception as exc:
        tb = traceback.format_exc()
        _log(job_id, f"FATAL: {exc}\n{tb}")
        try:
            job_registry.update_job(
                job_id,
                status="failed",
                error_msg=str(exc)[:500],
                completed_at=int(time.time()),
            )
            _progress(job_id, "failed", 0, f"Failed: {exc}")
        except Exception:
            pass

    finally:
        # ── Auto-start next queued job for this user ──────────────────────
        # When this job finishes (either phases_ready or failed), check if
        # the user has more queued jobs and start the oldest one.  This
        # makes multi-file uploads self-draining without needing the browser
        # to be open.
        try:
            this_job = job_registry.get_job(job_id)
            if this_job:
                _user_id = this_job.get("user_id")
                if _user_id:
                    next_job = job_registry.get_next_queued_job(_user_id)
                    if next_job:
                        _next_id = next_job["job_id"]
                        _next_config = {
                            "job_id":       _next_id,
                            "pdf_path":     next_job.get("pdf_path"),
                            "pdf_hash":     next_job.get("pdf_hash"),
                            "filename":     next_job.get("filename", "unknown.pdf"),
                            "user_id":      _user_id,
                            # Reuse the same API keys and settings as the
                            # just-finished job (stored in config dict).
                            "openai_key":               config.get("openai_key"),
                            "ade_key":                  config.get("ade_key"),
                            "google_cloud_config":      config.get("google_cloud_config"),
                            "analysis_model":           config.get("analysis_model", "gpt-5.1"),
                            "classifier_model":         config.get("classifier_model", "gpt-5-mini"),
                            "fence_keywords":           config.get("fence_keywords", []),
                            "use_ade":                  config.get("use_ade", True),
                            "highlight_fence_text":     config.get("highlight_fence_text", True),
                            "enable_unified_measurement": config.get("enable_unified_measurement", True),
                            "broken_pages":             [],
                            "cache_params":             next_job.get("cache_params_hint", ""),
                        }
                        _t = threading.Thread(
                            target=run_job,
                            args=(_next_id, _next_config),
                            daemon=False,
                            name=f"fence-job-{_next_id[:8]}",
                        )
                        _t.start()
                        _log(job_id, f"auto-started next queued job {_next_id[:8]}")
        except Exception as _ae:
            # Best-effort: never crash in finally
            try:
                _log(job_id, f"auto-start-next failed (non-fatal): {_ae}")
            except Exception:
                pass
