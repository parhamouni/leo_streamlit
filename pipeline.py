"""Standalone analysis pipeline for fence detection.

Runs the full Phase 1a → 1b → 1c → 2 → 3 pipeline without any Streamlit
dependency. Called by api_server.py background workers.

Usage:
    from pipeline import run_analysis, PipelineConfig, PipelineResult
    result = run_analysis(pdf_path="/path/to.pdf", config=PipelineConfig(...),
                          progress_cb=my_callback)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import fitz
import psutil

try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

from langchain_openai import ChatOpenAI

import fence_cache
import telemetry
import spend_tracker
import utils_ade as ade
from utils_vector import (
    measure_lines_in_selection,
    measure_at_click_point,
    infer_scale_from_page,
    verify_scale_with_bar,
    verify_scale_with_bar_fast,
    transform_coords_for_rotation,
)
from config import cfg

log = logging.getLogger("pipeline")

ProgressCallback = Callable[[str, int, str], None]

# Fired once per page when classification is known (Phase 1c) and again per
# fence page when full enrichment is available (Phase 3). The dict carries:
#   page_number   — 1-indexed page (matches the page_results schema)
#   is_fence_page — bool from classification
#   result_json   — partial or full per-page payload, or None for the
#                   minimal Phase 1c stub
# Workers can upsert this straight into Postgres `page_results`.
PageCallback = Callable[[dict], None]


def _noop_page(_page: dict) -> None:
    pass


@dataclass
class PipelineConfig:
    openai_api_key: str = ""
    ade_api_key: str = ""
    google_cloud_config: dict | None = None
    analysis_model: str = field(default_factory=lambda: cfg.ANALYSIS_MODEL)
    classifier_model: str = field(default_factory=lambda: cfg.CLASSIFIER_MODEL)
    fence_keywords: list[str] = field(default_factory=lambda: list(cfg.DEFAULT_FENCE_KEYWORDS))
    use_ade: bool = True
    highlight_fence_text: bool = True
    enable_unified_measurement: bool = True
    enable_nonlayer_suggestions: bool = False
    display_image_dpi: int = 150
    debug_mode: bool = False
    cache_scope: str = ""


@dataclass
class FencePageResult:
    page_idx: int
    page_num: int
    width: float
    height: float
    rotation: int
    fence_text: str = ""
    definitions: list[dict] = field(default_factory=list)
    instances: list[dict] = field(default_factory=list)
    keyword_matches: list[dict] = field(default_factory=list)
    ade_chunks: list[dict] = field(default_factory=list)
    legend_entries: list[dict] = field(default_factory=list)
    scale_info: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    fence_pages: list[dict] = field(default_factory=list)
    non_fence_pages: list[dict] = field(default_factory=list)
    element_details: dict = field(default_factory=dict)
    highlighted_pdf_bytes: bytes | None = None
    per_page_scale_info: dict = field(default_factory=dict)
    unified_measurements: dict = field(default_factory=dict)
    page_categories: dict = field(default_factory=dict)
    total_pages: int = 0
    timings: dict = field(default_factory=dict)
    error: str | None = None


def _noop_progress(phase: str, pct: int, message: str) -> None:
    pass


def _line_to_dict(line: Any) -> dict:
    """Coerce a `VectorLine` (or similar) into a JSON-friendly dict so that
    `all_fence_lines` survives `json.dumps` to disk. Plain dicts pass
    through; everything else gets best-effort attribute extraction."""
    if isinstance(line, dict):
        return line
    if hasattr(line, "__dataclass_fields__"):
        from dataclasses import asdict
        try:
            return asdict(line)
        except Exception:
            pass
    out: dict[str, Any] = {}
    for attr in ("start", "end", "length_pts", "color", "width", "dashes", "layer"):
        if hasattr(line, attr):
            out[attr] = getattr(line, attr)
    return out


def _normalize_measurements(measurements: dict) -> dict:
    """Make a measurement payload JSON-safe before it lands in
    `result.fence_pages` / `job_registry.save_results` / fence_cache.

    Today's only offender is `all_fence_lines`, a list of `VectorLine`
    dataclasses that `json.dumps(..., default=str)` was silently turning
    into repr strings — defeating downstream exporters and the highlight
    overlay that wants to read line geometry."""
    if not isinstance(measurements, dict):
        return measurements
    out = dict(measurements)
    afl = out.get("all_fence_lines")
    if isinstance(afl, list):
        out["all_fence_lines"] = [_line_to_dict(ln) for ln in afl]
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_llm_clients(config: PipelineConfig) -> dict[str, ChatOpenAI | None]:
    if not config.openai_api_key:
        return {"analysis": None, "scale": None, "classifier": None}
    return {
        "analysis": ChatOpenAI(
            model=config.analysis_model,
            temperature=0,
            openai_api_key=config.openai_api_key,
            timeout=60,
            max_retries=1,
        ),
        "scale": ChatOpenAI(
            model=config.analysis_model,
            temperature=0,
            openai_api_key=config.openai_api_key,
            timeout=90,
            max_retries=1,
        ),
        "classifier": ChatOpenAI(
            model=config.classifier_model,
            temperature=0,
            openai_api_key=config.openai_api_key,
            timeout=60,
            max_retries=2,
        ),
    }


def _mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _text_to_lines(text: str, source: str = "pdf") -> list[dict]:
    """Convert plain text into structured line dicts for scan_page_for_keywords_fast."""
    return [
        {"text": line, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "source": source}
        for line in text.split("\n") if line.strip()
    ]


# Phase 1a hard-timeout per page (seconds). Healthy pages finish in <100 ms;
# subtly-damaged pages can drive MuPDF into an internal C-level recovery
# loop that holds the GIL and never returns. The only escape hatch is to
# run extraction in a subprocess and SIGKILL on timeout.
_PHASE1A_PAGE_TIMEOUT = 20

# Subprocess startup is ~300 ms (Python import + fitz.open). Batching
# amortises this; one bad page in a batch hangs the whole subprocess but
# we recover via per-page fallback. Matches the legacy app_ade_prod cap.
_PHASE1A_BATCH_SIZE = 5

_PAGE_EXTRACTOR_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ops", "page_extractor.py"
)


def _phase1a_extract_one(pdf_path: str, page_idx: int, timeout_s: float) -> dict:
    """Single-page Phase 1a extraction in a subprocess. SIGKILL on timeout.

    Returns one of:
        {"ok": True, "text": str, "dims": {"width": ..., "height": ..., "rotation": ...}}
        {"ok": False, "error": str}
    Always returns within ~timeout_s + a small overhead, regardless of
    what MuPDF does internally.
    """
    if not os.path.exists(pdf_path):
        return {"ok": False, "error": "PDF disk file missing"}
    try:
        proc = subprocess.run(
            [sys.executable, _PAGE_EXTRACTOR_SCRIPT, pdf_path,
             "--phase1a-batch", str(page_idx)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error":
                f"page extraction exceeded {timeout_s}s — MuPDF recovery loop on damaged page"}
    except Exception as e:
        return {"ok": False, "error": f"subprocess launch failed: {e}"}
    # The phase1a-batch protocol streams JSON lines; the page result is
    # one line, followed by {"done": true}. Pick out the page_result.
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        pr = obj.get("page_result")
        if pr and pr.get("page_idx") == page_idx:
            if pr.get("ok"):
                return {"ok": True, "text": pr.get("text", ""), "dims": pr.get("dims", {})}
            return {"ok": False, "error": pr.get("error", "extract failed")}
    return {"ok": False, "error":
            f"worker produced no page_result (rc={proc.returncode}): {(proc.stderr or '')[:200]}"}


def _phase3_extract_lines(pdf_path: str, page_idx: int,
                          timeout_s: float) -> tuple[list, str | None]:
    """Subprocess-isolated `get_native_pdf_lines` call for a single
    fence page. Returns (lines, error_or_None).

    Same root cause as the Phase 1a hang we already isolated: PyMuPDF's
    text/line extraction can hold the GIL inside C code on certain
    pages, starving every other thread in the process — including the
    Postgres I/O thread that's trying to read a COMMIT response from
    the local DB. Verified live: 40 bytes from PG sat unread in our
    recv buffer for 12+ minutes while ThreadPoolExecutor-4 burned the
    GIL inside `get_native_pdf_lines`. The whole API stopped serving
    HTTP requests until we forcibly killed the worker.

    Routing the call through ops/page_extractor.py's existing legacy
    single-page mode means each fence page's heavy fitz call lives in
    its own OS process — its GIL is irrelevant to ours, and a SIGKILL
    after `timeout_s` is the hard-stop on damaged pages.
    """
    if not os.path.exists(pdf_path):
        return [], "PDF disk file missing"
    try:
        proc = subprocess.run(
            [sys.executable, _PAGE_EXTRACTOR_SCRIPT, pdf_path, str(page_idx)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return [], (
            f"page extraction exceeded {timeout_s}s — "
            f"MuPDF recovery loop on damaged page"
        )
    except Exception as e:
        return [], f"subprocess launch failed: {e}"
    if not proc.stdout:
        return [], (
            f"worker produced no output (rc={proc.returncode}): "
            f"{(proc.stderr or '')[:200]}"
        )
    try:
        # Legacy single-page mode emits one JSON object on stdout.
        obj = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return [], f"could not parse worker output: {e}"
    if obj.get("ok"):
        return obj.get("lines", []) or [], None
    return [], obj.get("error", "extract failed")


def _phase1a_extract_batch(pdf_path: str, page_indices: list[int],
                           per_page_timeout_s: float) -> dict[int, dict]:
    """Batch Phase 1a extraction. Returns {page_idx: result_dict} for
    pages that finished before SIGKILL; missing entries mean the
    subprocess hung on that page (caller should retry per-page).

    The subprocess streams one JSON line per page so partial progress
    survives a kill on a later hung page.
    """
    if not os.path.exists(pdf_path) or not page_indices:
        return {}
    pages_csv = ",".join(str(pi) for pi in page_indices)
    # 3s startup budget + per-page slack.
    batch_timeout = 3 + per_page_timeout_s * len(page_indices)
    proc = subprocess.Popen(
        [sys.executable, _PAGE_EXTRACTOR_SCRIPT, pdf_path,
         "--phase1a-batch", pages_csv],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    collected: dict[int, dict] = {}

    import threading

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
                        collected[pi] = {"ok": True, "text": pr.get("text", ""),
                                         "dims": pr.get("dims", {})}
                    else:
                        collected[pi] = {"ok": False,
                                         "error": pr.get("error", "extract failed")}
        except Exception:
            pass

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()
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
    drain_thread.join(timeout=1)
    return collected


def run_analysis(
    pdf_path: str,
    config: PipelineConfig,
    progress_cb: ProgressCallback | None = None,
    page_cb: PageCallback | None = None,
) -> PipelineResult:
    """Run the full fence detection pipeline on a PDF.

    This is the main entry point for the FastAPI background worker.
    Returns a PipelineResult with all analysis data.
    """
    progress = progress_cb or _noop_progress
    emit_page = page_cb or _noop_page
    result = PipelineResult()
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    try:
        pdf_path_str = str(pdf_path)
        pdf_hash = _sha256_file(pdf_path_str)
        cache_scope = config.cache_scope or f"pipeline_{pdf_hash[:16]}"

        params = fence_cache.params_hash(
            model=config.analysis_model,
            classifier=config.classifier_model,
            keywords=sorted(config.fence_keywords),
            use_ade=config.use_ade,
        )

        llms = _create_llm_clients(config)
        llm_analysis = llms["analysis"]
        llm_scale = llms["scale"]
        llm_classifier = llms["classifier"]

        progress("init", 2, "Opening PDF...")

        doc = fitz.open(pdf_path_str)
        total_pages = doc.page_count
        result.total_pages = total_pages

        broken_pages: set[int] = set()
        for pi in range(total_pages):
            try:
                _ = doc[pi].rect.width
            except Exception:
                broken_pages.add(pi)

        valid_pages = [i for i in range(total_pages) if i not in broken_pages]
        if not valid_pages:
            result.error = "No readable pages found in PDF"
            return result

        progress("init", 5, f"PDF loaded: {total_pages} pages ({len(broken_pages)} damaged)")

        # ---- Phase 1a: Native text extraction ----
        t1a = time.perf_counter()
        progress("phase1a", 8, "Phase 1a: Extracting native text...")

        cached_1a = fence_cache.get("phase1a", pdf_hash, params, user_scope=cache_scope)
        if cached_1a:
            page_texts = cached_1a.get("page_texts", {})
            page_dims = cached_1a.get("page_dims", {})
            log.info("Phase 1a: cache hit")
        else:
            # Phase 1a runs in a subprocess per batch with hard SIGKILL on
            # timeout. Subtly-damaged PDFs can drive MuPDF into a C-level
            # recovery loop that holds the GIL forever — see legacy
            # app_ade_prod.py for the original rationale. In-process
            # try/except never catches that; only OS-level kill does.
            page_texts = {}
            page_dims = {}
            batches = [
                valid_pages[i:i + _PHASE1A_BATCH_SIZE]
                for i in range(0, len(valid_pages), _PHASE1A_BATCH_SIZE)
            ]
            for batch in batches:
                results = _phase1a_extract_batch(
                    pdf_path_str, batch, _PHASE1A_PAGE_TIMEOUT
                )
                # Pages the batch subprocess never reported on (most
                # likely it hung on one of them and got SIGKILL'd) are
                # retried single-page so one bad page can't poison the
                # rest of the batch.
                missing = [pi for pi in batch if pi not in results]
                for pi in missing:
                    results[pi] = _phase1a_extract_one(
                        pdf_path_str, pi, _PHASE1A_PAGE_TIMEOUT
                    )
                for pi in batch:
                    r = results.get(pi) or {"ok": False, "error": "no result"}
                    if r.get("ok"):
                        page_texts[str(pi)] = r.get("text", "") or ""
                        page_dims[str(pi)] = r.get("dims", {})
                    else:
                        log.warning(
                            f"Phase 1a page {pi}: {r.get('error')} — "
                            f"marking damaged and skipping"
                        )
                        broken_pages.add(pi)

            cache_data = {
                "page_texts": page_texts,
                "page_dims": page_dims,
            }
            fence_cache.put("phase1a", pdf_hash, params, cache_data, user_scope=cache_scope)

        timings["phase1a"] = time.perf_counter() - t1a
        progress("phase1a", 15, f"Phase 1a done ({timings['phase1a']:.1f}s)")

        # ---- Phase 1b: OCR (Google Document AI) ----
        t1b = time.perf_counter()
        progress("phase1b", 18, "Phase 1b: Running OCR on scanned pages...")

        ocr_texts_by_page: dict[str, str] = {}
        # Per-page OCR lines WITH bboxes — mirrors app_ade_prod's
        # `_ocr_lines_by_page`. Required for keyword/legend/instance
        # highlighting on scanned pages whose native PDF text is missing
        # or unmappable (CID fonts without ToUnicode CMap). Cache key was
        # bumped to phase1b_v2 because the previous version stored only
        # the joined text — we want the original line dicts back.
        ocr_lines_by_page: dict[int, list[dict]] = {}
        if config.google_cloud_config:
            pages_needing_ocr = []
            for pi in valid_pages:
                if pi in broken_pages:
                    continue
                native_text = page_texts.get(str(pi), "")
                if len(native_text.strip()) < 50:
                    cached_ocr = fence_cache.get("phase1b_v2", pdf_hash, params,
                                                  page_idx=pi, user_scope=cache_scope)
                    if cached_ocr is not None:
                        ocr_texts_by_page[str(pi)] = cached_ocr.get("ocr_text", "")
                        ocr_lines_by_page[pi] = cached_ocr.get("lines") or []
                    else:
                        pages_needing_ocr.append(pi)

            if pages_needing_ocr:
                batch_size = cfg.OCR_BATCH_SIZE
                total_ocr = max(len(pages_needing_ocr), 1)
                processed_ocr = 0
                for batch_start in range(0, len(pages_needing_ocr), batch_size):
                    batch = pages_needing_ocr[batch_start:batch_start + batch_size]
                    try:
                        batch_pdf = ade.safe_multi_page_pdf_from_path(
                            pdf_path_str, batch,
                            per_page_max_bytes=cfg.OCR_BATCH_TARGET_BYTES,
                        )
                        dims_by_local = {}
                        for local_idx, pi in enumerate(batch):
                            d = page_dims.get(str(pi), {})
                            dims_by_local[local_idx] = (
                                d.get("width", 612), d.get("height", 792))
                        result_by_local = ade.run_google_ocr_blocks_multipage(
                            batch_pdf, config.google_cloud_config,
                            dims_by_local,
                        )
                        for local_idx, pi in enumerate(batch):
                            lines = result_by_local.get(local_idx, [])
                            text = " ".join(
                                ln.get("text", "") for ln in lines).strip()
                            ocr_texts_by_page[str(pi)] = text
                            ocr_lines_by_page[pi] = lines
                            fence_cache.put("phase1b_v2", pdf_hash, params,
                                           {"ocr_text": text, "lines": lines},
                                           page_idx=pi, user_scope=cache_scope)
                        processed_ocr += len(batch)
                        # Phase 1b spans 18% → 30% on the overall bar.
                        # Bump per batch so the user sees movement during
                        # Google DocAI work.
                        pct = 18 + int(12 * processed_ocr / total_ocr)
                        progress("phase1b", pct,
                                 f"Phase 1b: OCR {processed_ocr}/{total_ocr} pages")
                    except Exception as e:
                        log.warning(f"Phase 1b batch OCR failed: {e}")
                        for pi in batch:
                            try:
                                page_pdf = ade.create_single_page_pdf_from_path(
                                    pdf_path_str, pi)
                                d = page_dims.get(str(pi), {})
                                ocr_lines = ade.run_google_ocr_blocks(
                                    page_pdf, config.google_cloud_config,
                                    d.get("width", 612), d.get("height", 792))
                                text = " ".join(
                                    ln.get("text", "") for ln in ocr_lines
                                ).strip()
                                ocr_texts_by_page[str(pi)] = text
                                ocr_lines_by_page[pi] = ocr_lines
                                fence_cache.put("phase1b_v2", pdf_hash, params,
                                               {"ocr_text": text, "lines": ocr_lines},
                                               page_idx=pi, user_scope=cache_scope)
                            except Exception:
                                pass

        timings["phase1b"] = time.perf_counter() - t1b
        progress("phase1b", 30, f"Phase 1b done ({timings['phase1b']:.1f}s)")

        # Merge texts: native + OCR
        merged_texts: dict[str, str] = {}
        for pi in valid_pages:
            native = page_texts.get(str(pi), "")
            ocr = ocr_texts_by_page.get(str(pi), "")
            merged_texts[str(pi)] = (native + "\n" + ocr).strip() if ocr else native

        # ---- Phase 1c: Page classification ----
        t1c = time.perf_counter()
        progress("phase1c", 33, "Phase 1c: Classifying pages...")

        fence_page_indices: list[int] = []
        non_fence_page_indices: list[int] = []
        # page_idx -> list of {keyword, text, x0, y0, x1, y1, source} dicts
        # produced by scan_page_for_keywords_fast. Surfaced into
        # page_result["keyword_matches"] so the highlight worker can draw
        # orange rectangles around fence-keyword text (stage-1 overlay).
        keyword_matches_by_page: dict[int, list[dict]] = {}
        # page_idx -> {method, reason, confidence, signals, keyword_count}
        # Captured during classification so the user can see WHY a page was
        # marked non-fence. Cached entries (re-runs) only have method;
        # fresh classifications get the rich payload.
        classification_meta: dict[int, dict] = {}

        pages_to_classify: list[int] = []
        for pi in valid_pages:
            if pi in broken_pages:
                continue
            cached_cls = fence_cache.get("phase1c", pdf_hash, params,
                                         page_idx=pi, user_scope=cache_scope)
            if cached_cls is not None:
                if cached_cls.get("is_fence"):
                    fence_page_indices.append(pi)
                else:
                    non_fence_page_indices.append(pi)
                # Pull whatever rich fields the cache happens to have. Old
                # cache entries only have {is_fence, method}; new ones
                # carry reason/confidence/signals/keyword_count.
                classification_meta[pi] = {
                    k: cached_cls.get(k)
                    for k in ("method", "reason", "confidence", "signals", "keyword_count")
                    if cached_cls.get(k) is not None
                }
            else:
                pages_to_classify.append(pi)

        if pages_to_classify:
            keyword_hits: dict[int, bool] = {}
            llm_needed: list[int] = []
            for pi in pages_to_classify:
                pdf_lines = _text_to_lines(page_texts.get(str(pi), ""), "pdf")
                ocr_lines = _text_to_lines(ocr_texts_by_page.get(str(pi), ""), "ocr")
                kw_result = ade.scan_page_for_keywords_fast(pdf_lines, ocr_lines, config.fence_keywords)
                # Keep matched_lines so the per-page result can render orange
                # keyword-match rectangles in the highlighted PDF (parity with
                # app_ade_prod's stage-1 overlay).
                keyword_matches_by_page[pi] = kw_result.get("matched_lines", []) or []
                kw_count = len(kw_result.get("matched_lines") or [])
                if kw_result.get("has_keywords"):
                    keyword_hits[pi] = True
                    fence_page_indices.append(pi)
                    cls_payload = {
                        "is_fence": True,
                        "method": "keyword",
                        "keyword_count": kw_count,
                    }
                    classification_meta[pi] = {
                        "method": "keyword",
                        "keyword_count": kw_count,
                    }
                    fence_cache.put("phase1c", pdf_hash, params, cls_payload,
                                   page_idx=pi, user_scope=cache_scope)
                else:
                    llm_needed.append(pi)

            if llm_needed and llm_classifier:
                batch_size = cfg.CLASSIFY_BATCH_SIZE
                for batch_start in range(0, len(llm_needed), batch_size):
                    batch = llm_needed[batch_start:batch_start + batch_size]
                    batch_pages = [
                        (pi, merged_texts.get(str(pi), "")[:2000], [])
                        for pi in batch
                    ]
                    try:
                        results = ade.llm_classify_pages_batch(
                            llm_classifier, batch_pages, config.fence_keywords)
                        for pi in batch:
                            page_cls = results.get(pi, {})
                            is_fence = page_cls.get("is_fence_related", False)
                            if is_fence:
                                fence_page_indices.append(pi)
                            else:
                                non_fence_page_indices.append(pi)
                            meta = {
                                "method": "llm",
                                "reason": page_cls.get("reason"),
                                "confidence": page_cls.get("confidence"),
                                "signals": page_cls.get("signals"),
                            }
                            # Strip Nones to keep payload lean.
                            meta = {k: v for k, v in meta.items() if v is not None}
                            classification_meta[pi] = meta
                            fence_cache.put("phase1c", pdf_hash, params,
                                           {"is_fence": is_fence, **meta},
                                           page_idx=pi, user_scope=cache_scope)
                    except Exception as e:
                        log.warning(f"Phase 1c LLM batch failed: {e}")
                        for pi in batch:
                            non_fence_page_indices.append(pi)
                            classification_meta[pi] = {
                                "method": "error",
                                "reason": f"LLM batch failed: {e}",
                            }
                            fence_cache.put("phase1c", pdf_hash, params,
                                           {"is_fence": False, "method": "error",
                                            "reason": f"LLM batch failed: {e}"},
                                           page_idx=pi, user_scope=cache_scope)
            elif llm_needed:
                for pi in llm_needed:
                    non_fence_page_indices.append(pi)
                    classification_meta[pi] = {
                        "method": "no_llm",
                        "reason": "no LLM classifier configured; defaulted to non-fence",
                    }

        fence_page_indices.sort()
        non_fence_page_indices.sort()

        # Emit Phase 1c stubs so the frontend can render placeholder rows
        # (and the dashboard's "pages so far" counter) before Phase 3
        # finishes enriching them. Fence pages get overwritten with the full
        # payload below; non-fence pages keep their stub.
        for pi in fence_page_indices:
            try:
                emit_page({
                    "page_number": pi + 1,
                    "is_fence_page": True,
                    "result_json": {"page_idx": pi, "page_num": pi + 1, "phase": "phase1c"},
                })
            except Exception:
                log.exception("page_cb (phase1c fence) failed")
        for pi in non_fence_page_indices:
            try:
                emit_page({
                    "page_number": pi + 1,
                    "is_fence_page": False,
                    "result_json": {"page_idx": pi, "page_num": pi + 1, "phase": "phase1c"},
                })
            except Exception:
                log.exception("page_cb (phase1c non-fence) failed")

        timings["phase1c"] = time.perf_counter() - t1c
        progress("phase1c", 45, f"Phase 1c done: {len(fence_page_indices)} fence, "
                 f"{len(non_fence_page_indices)} non-fence ({timings['phase1c']:.1f}s)")

        if not fence_page_indices:
            progress("done", 100, "No fence pages found")
            result.non_fence_pages = [
                {
                    "page_idx": pi,
                    "page_num": pi + 1,
                    **(classification_meta.get(pi) or {}),
                }
                for pi in non_fence_page_indices
            ]
            result.timings = timings
            doc.close()
            return result

        # ---- Phase 2: ADE detection ----
        t2 = time.perf_counter()
        progress("phase2", 48, "Phase 2: Running ADE detection on fence pages...")

        ade_chunks_by_page: dict[int, list] = {}

        if config.use_ade and config.ade_api_key:
            pages_needing_ade: list[int] = []
            for pi in fence_page_indices:
                cached_ade = fence_cache.get("phase2", pdf_hash, params,
                                            page_idx=pi, user_scope=cache_scope)
                if cached_ade is not None:
                    ade_chunks_by_page[pi] = cached_ade.get("chunks", [])
                else:
                    pages_needing_ade.append(pi)

            if pages_needing_ade:
                def _ade_single_page(pi: int) -> tuple[int, list]:
                    try:
                        page_pdf = ade.safe_single_page_pdf_from_path(
                            pdf_path_str, pi,
                            max_bytes=cfg.ADE_PAGE_MAX_BYTES,
                        )
                        if page_pdf is None:
                            page_pdf = ade.create_single_page_pdf(pdf_path_str, pi)
                        resp = ade.ade_parse_document(
                            page_pdf, config.ade_api_key)
                        dims = page_dims.get(str(pi), {})
                        chunks = []
                        if resp and resp.get("success") and dims:
                            chunks = ade.align_ade_chunks_to_page(
                                resp, 0,
                                dims.get("width", 612),
                                dims.get("height", 792))
                        fence_cache.put("phase2", pdf_hash, params,
                                       {"chunks": chunks}, page_idx=pi,
                                       user_scope=cache_scope)
                        return pi, chunks
                    except Exception as e:
                        log.warning(f"Phase 2 ADE page {pi}: {e}")
                        fence_cache.put("phase2", pdf_hash, params,
                                       {"chunks": []}, page_idx=pi,
                                       user_scope=cache_scope)
                        return pi, []

                with ThreadPoolExecutor(max_workers=cfg.WORKERS_PHASE2) as pool:
                    futures = {pool.submit(_ade_single_page, pi): pi
                               for pi in pages_needing_ade}
                    completed = 0
                    total = max(len(pages_needing_ade), 1)
                    for fut in as_completed(futures):
                        try:
                            pi, chunks = fut.result(timeout=120)
                            ade_chunks_by_page[pi] = chunks
                        except Exception as e:
                            pi = futures[fut]
                            log.warning(f"Phase 2 ADE future {pi}: {e}")
                            ade_chunks_by_page[pi] = []
                        completed += 1
                        # Phase 2 spans 48% → 60% on the overall bar.
                        # Interpolate so the dashboard sees movement
                        # instead of a 12-min freeze between Phase 1c
                        # and Phase 3.
                        pct = 48 + int(12 * completed / total)
                        progress("phase2", pct,
                                 f"Phase 2: {completed}/{total} pages")
                        # Also emit a page_cb stub so the per-page ticker
                        # on the detail page shows the page transitioning
                        # through Phase 2. Phase 3 will overwrite with
                        # the rich payload.
                        try:
                            emit_page({
                                "page_number": pi + 1,
                                "is_fence_page": True,
                                "result_json": {
                                    "page_idx": pi,
                                    "page_num": pi + 1,
                                    "phase": "phase2",
                                },
                            })
                        except Exception:
                            log.exception(f"page_cb (phase2 pi={pi}) failed")

        timings["phase2"] = time.perf_counter() - t2
        progress("phase2", 60, f"Phase 2 done ({timings['phase2']:.1f}s)")

        # ---- Phase 3: Legend, scale, measurement ----
        t3 = time.perf_counter()
        progress("phase3", 63, "Phase 3: Extracting legends, scales, measurements...")

        fence_results: list[dict] = []
        per_page_scale: dict[str, dict] = {}
        per_page_measurements: dict[str, dict] = {}
        per_page_categories: dict[str, dict] = {}

        def _process_fence_page(pi: int) -> dict:
            page_result: dict[str, Any] = {
                "page_idx": pi,
                "page_num": pi + 1,
            }
            dims = page_dims.get(str(pi), {})
            page_result["width"] = dims.get("width", 612)
            page_result["height"] = dims.get("height", 792)
            page_result["rotation"] = dims.get("rotation", 0)

            text = merged_texts.get(str(pi), "")
            page_result["fence_text"] = text[:500]

            chunks = ade_chunks_by_page.get(pi, [])
            page_result["ade_chunks"] = chunks

            # ----- Stage-1 highlight data (matches app_ade_prod.py) -----
            # Three-layer overlay: green = legend entries (per-row), purple =
            # figure instances (per-token indicator hits), orange = keyword
            # fallback. Mirrors app_ade_prod's generate_combined_highlighted_pdf
            # at lines 1562-1670 and the orchestration at 4654-4703 / 4887-4896.
            if chunks:
                legend_chunks, figure_chunks = ade.segment_chunks(chunks)
            else:
                legend_chunks, figure_chunks = [], []

            # Per-line PDF and OCR text dicts WITH real bboxes in display
            # space — mirrors app_ade_prod's _pdf_lines_by_page +
            # _ocr_lines_by_page pipeline. Using these gives all three
            # highlight layers (definitions / instances / keyword_matches)
            # the same geometry source prod uses, so highlights render
            # correctly on scanned pages whose native PDF text is missing
            # or unmappable (CID fonts without ToUnicode CMap).
            # Subprocess-isolated to avoid GIL starvation under
            # concurrent Phase 3 workers — see _phase3_extract_lines
            # for the full rationale. SIGKILL on damaged pages instead
            # of wedging the whole event loop.
            pdf_lines, _err = _phase3_extract_lines(
                pdf_path_str, pi, _PHASE1A_PAGE_TIMEOUT
            )
            if _err:
                log.warning(f"Phase 3 native pdf lines page {pi}: {_err}")
            ocr_lines = ocr_lines_by_page.get(pi, []) or []

            # Legend entries — one dict per legend row, tight bbox around the
            # legend item's text. Used both for the detail-page table
            # (`legend_entries`) and the green highlight layer (`definitions`).
            cached_legend = fence_cache.get("phase3_legend", pdf_hash, params,
                                           page_idx=pi, user_scope=cache_scope)
            if cached_legend is not None:
                legend_entries = cached_legend.get("entries", []) or []
            elif llm_analysis and chunks:
                try:
                    legend_entries = ade.extract_legend_entries(
                        legend_chunks, pdf_lines, ocr_lines,
                        config.fence_keywords, llm_analysis,
                        figure_chunks=figure_chunks,
                    )
                    fence_cache.put("phase3_legend", pdf_hash, params,
                                   {"entries": legend_entries}, page_idx=pi,
                                   user_scope=cache_scope)
                except Exception as e:
                    log.warning(f"Phase 3 legend page {pi}: {e}")
                    legend_entries = []
            else:
                legend_entries = []
            page_result["legend_entries"] = legend_entries

            # Figure instances — search figure-chunk tokens for indicator
            # strings from legend_entries, return one dict per matched token.
            # Mirrors app_ade_prod.py:4670-4703.
            cached_instances = fence_cache.get("phase3_instances", pdf_hash, params,
                                               page_idx=pi, user_scope=cache_scope)
            if cached_instances is not None:
                figure_instances = cached_instances.get("instances", []) or []
            elif legend_entries and figure_chunks:
                try:
                    page_obj = doc[pi]
                    rotation_deg = page_obj.rotation
                    mbox_w = page_obj.mediabox.width
                    mbox_h = page_obj.mediabox.height
                    all_page_tokens: list[dict] = []
                    for w in page_obj.get_text("words"):
                        tx0, ty0 = transform_coords_for_rotation(
                            w[0], w[1], rotation_deg, mbox_w, mbox_h)
                        tx1, ty1 = transform_coords_for_rotation(
                            w[2], w[3], rotation_deg, mbox_w, mbox_h)
                        all_page_tokens.append({
                            "text": w[4],
                            "x0": min(tx0, tx1), "y0": min(ty0, ty1),
                            "x1": max(tx0, tx1), "y1": max(ty0, ty1),
                        })
                    figure_instances = ade.find_instances_in_figures_fast(
                        legend_entries, figure_chunks, all_page_tokens,
                        ocr_lines=ocr_lines,
                    )
                    fence_cache.put("phase3_instances", pdf_hash, params,
                                   {"instances": figure_instances}, page_idx=pi,
                                   user_scope=cache_scope)
                except Exception as e:
                    log.warning(f"Phase 3 instances page {pi}: {e}")
                    figure_instances = []
            else:
                figure_instances = []

            page_result["definitions"] = legend_entries
            page_result["instances"] = figure_instances

            # Orange = fence-keyword text. Always populate so keyword
            # rectangles render alongside green/purple even when ADE
            # succeeded. pdf_lines / ocr_lines already carry real bboxes
            # (built above), so matches land at the right location.
            kw_result = ade.scan_page_for_keywords_fast(
                pdf_lines, ocr_lines, config.fence_keywords)
            page_result["keyword_matches"] = kw_result.get("matched_lines", []) or []

            # Scale detection
            cached_scale = fence_cache.get("phase3_scale", pdf_hash, params,
                                          page_idx=pi, user_scope=cache_scope)
            if cached_scale is not None:
                page_result["scale_info"] = cached_scale
            else:
                try:
                    page_obj = doc[pi]
                    scale_info = verify_scale_with_bar_fast(
                        page_obj,
                        llm=llm_scale or llm_analysis,
                    )
                    page_result["scale_info"] = scale_info or {}
                    fence_cache.put("phase3_scale", pdf_hash, params,
                                   scale_info or {}, page_idx=pi,
                                   user_scope=cache_scope)
                except Exception as e:
                    log.warning(f"Phase 3 scale page {pi}: {e}")
                    page_result["scale_info"] = {}

            # Measurement
            cached_measure = fence_cache.get("phase3_measure", pdf_hash, params,
                                            page_idx=pi, user_scope=cache_scope)
            if cached_measure is not None:
                page_result["measurements"] = _normalize_measurements(cached_measure)
            elif config.enable_unified_measurement:
                try:
                    page_obj = doc[pi]
                    scale_val = (page_result.get("scale_info") or {}).get("verified_scale") or \
                                (page_result.get("scale_info") or {}).get("text_scale")
                    measurements = ade.measure_fence_elements(
                        page_obj, legend_chunks, figure_chunks,
                        figure_chunks=figure_chunks,
                        llm=llm_analysis,
                        scale_factor=scale_val,
                        ocr_text=text,
                        light_llm=llm_classifier,
                        enable_nonlayer_suggestions=config.enable_nonlayer_suggestions,
                    )
                    measurements = _normalize_measurements(measurements or {})
                    page_result["measurements"] = measurements
                    fence_cache.put("phase3_measure", pdf_hash, params,
                                   measurements, page_idx=pi,
                                   user_scope=cache_scope)
                except Exception as e:
                    log.warning(f"Phase 3 measure page {pi}: {e}")
                    page_result["measurements"] = {}
            else:
                page_result["measurements"] = {}

            return page_result

        completed_count = 0
        with ThreadPoolExecutor(max_workers=cfg.WORKERS_PHASE3) as pool:
            futures = {pool.submit(_process_fence_page, pi): pi
                       for pi in fence_page_indices}
            for fut in as_completed(futures):
                pi = futures[fut]
                try:
                    page_result = fut.result(timeout=600)
                    fence_results.append(page_result)
                    if page_result.get("scale_info"):
                        per_page_scale[str(pi)] = page_result["scale_info"]
                    if page_result.get("measurements"):
                        per_page_measurements[str(pi)] = page_result["measurements"]
                except Exception as e:
                    log.warning(f"Phase 3 page {pi} failed: {e}")
                    page_result = {
                        "page_idx": pi, "page_num": pi + 1,
                        "error": str(e),
                    }
                    fence_results.append(page_result)
                completed_count += 1
                pct = 63 + int(30 * completed_count / max(len(fence_page_indices), 1))
                progress("phase3", pct, f"Phase 3: {completed_count}/{len(fence_page_indices)} pages")
                # Stream the enriched page row out so the frontend can render
                # measurements / scale / legends as soon as each page finishes.
                try:
                    emit_page({
                        "page_number": pi + 1,
                        "is_fence_page": True,
                        "result_json": page_result,
                    })
                except Exception:
                    log.exception(f"page_cb (phase3 fence pi={pi}) failed")

        fence_results.sort(key=lambda x: x.get("page_idx", 0))

        timings["phase3"] = time.perf_counter() - t3
        progress("phase3", 93, f"Phase 3 done ({timings['phase3']:.1f}s)")

        # ---- Highlighted PDF generation ----
        progress("highlight", 95, "Generating highlighted PDF...")
        highlighted_pdf_bytes = None
        try:
            highlighted_pdf_bytes = _generate_highlighted_pdf(
                pdf_path_str, fence_results, config)
        except Exception as e:
            log.warning(f"Highlighted PDF generation failed: {e}")

        # ---- Element details extraction ----
        element_details = {}
        if llm_analysis and fence_results:
            try:
                progress("details", 97, "Extracting element details...")
                element_names = set()
                for r in fence_results:
                    for defn in r.get("definitions", []):
                        name = defn.get("text", "")[:80]
                        if name:
                            element_names.add(name)
                if element_names:
                    all_page_texts = {}
                    for r in fence_results:
                        pi = r.get("page_idx", 0)
                        all_page_texts[pi] = merged_texts.get(str(pi), "")
                    element_details = ade.extract_element_details(
                        llm_analysis, list(element_names)[:20], all_page_texts)
            except Exception as e:
                log.warning(f"Element details extraction failed: {e}")

        # ---- Assemble result ----
        result.fence_pages = fence_results
        result.non_fence_pages = [
            {
                "page_idx": pi,
                "page_num": pi + 1,
                **(classification_meta.get(pi) or {}),
            }
            for pi in non_fence_page_indices
        ]
        result.element_details = element_details or {}
        result.highlighted_pdf_bytes = highlighted_pdf_bytes
        result.per_page_scale_info = per_page_scale
        result.unified_measurements = per_page_measurements
        result.page_categories = per_page_categories
        result.timings = timings

        timings["total"] = time.perf_counter() - t_start
        progress("done", 100, f"Analysis complete in {timings['total']:.1f}s")

        doc.close()
        gc.collect()

        return result

    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        result.error = str(e)
        result.timings = timings
        return result


def _generate_highlighted_pdf(
    pdf_path: str,
    fence_results: list[dict],
    config: PipelineConfig,
) -> bytes | None:
    """Generate a highlighted PDF with fence annotations overlaid."""
    try:
        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ops", "highlight_pdf_worker.py",
        )

        import tempfile, uuid
        out_path = os.path.join(
            tempfile.gettempdir(),
            f"highlight_{uuid.uuid4().hex}.pdf",
        )

        if os.path.exists(worker_script):
            # Reshape fence_results for the worker. The stage-1 highlighted
            # PDF carries only the keyword/instance/legend rectangles
            # (green/purple/orange). Fence-line strokes belong to the stage-2
            # measurement PDF (`/api/jobs/{id}/measurement-pdf`) — drawing
            # them here floods the page with cyan and visually drowns the
            # stage-1 overlay. The worker still accepts a `fence_lines`
            # field; we just don't populate it from this code path.
            worker_pages = []
            for r in fence_results:
                pi = r.get("page_idx")
                if pi is None:
                    continue
                worker_pages.append({
                    "page_index_in_original_doc": pi,
                    "definitions": r.get("definitions") or [],
                    "instances": r.get("instances") or [],
                    "keyword_matches": r.get("keyword_matches") or [],
                })
            task = {
                "pdf_path": pdf_path,
                "out_path": out_path,
                "fence_pages": worker_pages,
                "highlight_fence_text": config.highlight_fence_text,
            }
            proc = subprocess.run(
                [sys.executable, worker_script],
                input=json.dumps(task, default=str).encode(),
                capture_output=True,
                timeout=cfg.HIGHLIGHT_PDF_TIMEOUT,
            )
            if proc.returncode == 0 and os.path.exists(out_path):
                try:
                    with open(out_path, "rb") as f:
                        return f.read()
                finally:
                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass
            log.warning(
                "Highlight worker failed (rc=%s): %s",
                proc.returncode,
                proc.stderr[:500] if proc.stderr else "no stderr",
            )
            try:
                os.unlink(out_path)
            except Exception:
                pass

        doc = fitz.open(pdf_path)
        fence_page_set = {r["page_idx"] for r in fence_results if "page_idx" in r}

        for page_result in fence_results:
            pi = page_result.get("page_idx")
            if pi is None or pi >= doc.page_count:
                continue
            page = doc[pi]

            for defn in page_result.get("definitions", []):
                bbox = defn.get("bbox")
                if bbox and len(bbox) == 4:
                    rect = fitz.Rect(bbox)
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=cfg.HIGHLIGHT_COLOR_UI)
                    annot.set_border(width=cfg.HIGHLIGHT_WIDTH_UI)
                    annot.update()

            # Stage-1 highlighter only draws keyword/instance/legend
            # rectangles. Fence-line strokes are stage 2 (measurement PDF).

            for inst in page_result.get("instances", []):
                bbox = inst.get("bbox")
                if bbox and len(bbox) == 4:
                    rect = fitz.Rect(bbox)
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=cfg.HIGHLIGHT_COLOR_INSTANCE)
                    annot.set_border(width=cfg.HIGHLIGHT_WIDTH_UI)
                    annot.update()

        out = BytesIO()
        doc.save(out)
        doc.close()
        return out.getvalue()

    except Exception as e:
        log.warning(f"Highlighted PDF generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI entry point: `python -m pipeline <pdf> --out <dir>`
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse
    from dataclasses import asdict

    from secrets_loader import load_api_keys

    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Run the fence detection pipeline on a PDF without Streamlit.",
    )
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory; results.json + highlighted.pdf written here",
    )
    parser.add_argument(
        "--secrets",
        default=".streamlit/secrets.toml",
        help="Path to secrets.toml (default: .streamlit/secrets.toml)",
    )
    parser.add_argument(
        "--no-ade",
        action="store_true",
        help="Skip LandingAI ADE parsing (faster, lower quality)",
    )
    parser.add_argument(
        "--no-measurement",
        action="store_true",
        help="Skip Phase 3 measurement (faster, no fence lengths)",
    )
    parser.add_argument(
        "--keywords",
        help="Comma-separated fence keywords (overrides default)",
    )
    parser.add_argument(
        "--analysis-model",
        help="Override analysis model (e.g. gpt-5.1, gpt-5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress phase progress output",
    )
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = load_api_keys(args.secrets)
    if not keys["openai_key"]:
        print("error: no OpenAI key found (set OPENAI_API_KEY or .streamlit/secrets.toml)", file=sys.stderr)
        return 2

    cfg_kwargs: dict = {
        "openai_api_key": keys["openai_key"],
        "ade_api_key": keys["ade_key"] or "",
        "google_cloud_config": keys["google_cloud_config"],
        "use_ade": not args.no_ade,
        "enable_unified_measurement": not args.no_measurement,
    }
    if args.keywords:
        cfg_kwargs["fence_keywords"] = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if args.analysis_model:
        cfg_kwargs["analysis_model"] = args.analysis_model

    pipeline_cfg = PipelineConfig(**cfg_kwargs)

    def _print_progress(phase: str, pct: int, msg: str) -> None:
        if not args.quiet:
            print(f"[{phase:<12}] {pct:3d}%  {msg}", flush=True)

    print(f"Running pipeline on {pdf_path} → {out_dir}", flush=True)
    t0 = time.time()
    result = run_analysis(str(pdf_path), pipeline_cfg, progress_cb=_print_progress)
    elapsed = time.time() - t0

    if result.error:
        print(f"\nERROR: {result.error}", file=sys.stderr)
        return 1

    # Serialize result (drop bytes; write PDF separately)
    summary = asdict(result)
    pdf_bytes = summary.pop("highlighted_pdf_bytes", None)

    summary_path = out_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    if pdf_bytes:
        pdf_out = out_dir / "highlighted.pdf"
        pdf_out.write_bytes(pdf_bytes)

    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{len(result.fence_pages)} fence pages / "
        f"{len(result.non_fence_pages)} non-fence pages. "
        f"Results: {summary_path}"
        + (f" + {out_dir / 'highlighted.pdf'}" if pdf_bytes else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
