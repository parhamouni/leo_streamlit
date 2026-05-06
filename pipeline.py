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
    extract_vector_lines,
    verify_scale_with_bar,
    verify_scale_with_bar_fast,
)
from config import cfg

log = logging.getLogger("pipeline")

ProgressCallback = Callable[[str, int, str], None]


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


def run_analysis(
    pdf_path: str,
    config: PipelineConfig,
    progress_cb: ProgressCallback | None = None,
) -> PipelineResult:
    """Run the full fence detection pipeline on a PDF.

    This is the main entry point for the FastAPI background worker.
    Returns a PipelineResult with all analysis data.
    """
    progress = progress_cb or _noop_progress
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
            page_texts = {}
            page_dims = {}
            for pi in valid_pages:
                try:
                    page = doc[pi]
                    page_texts[str(pi)] = page.get_text("text") or ""
                    rect = page.rect
                    page_dims[str(pi)] = {
                        "width": rect.width,
                        "height": rect.height,
                        "rotation": page.rotation,
                    }
                except Exception as e:
                    log.warning(f"Phase 1a page {pi}: {e}")
                    broken_pages.add(pi)

            try:
                vector_lines_by_page = {}
                for pi in valid_pages:
                    if pi in broken_pages:
                        continue
                    try:
                        lines = extract_vector_lines(doc, pi)
                        vector_lines_by_page[str(pi)] = lines
                    except Exception:
                        vector_lines_by_page[str(pi)] = []
            except Exception:
                vector_lines_by_page = {}

            cache_data = {
                "page_texts": page_texts,
                "page_dims": page_dims,
                "vector_lines": vector_lines_by_page,
            }
            fence_cache.put("phase1a", pdf_hash, params, cache_data, user_scope=cache_scope)

        timings["phase1a"] = time.perf_counter() - t1a
        progress("phase1a", 15, f"Phase 1a done ({timings['phase1a']:.1f}s)")

        # ---- Phase 1b: OCR (Google Document AI) ----
        t1b = time.perf_counter()
        progress("phase1b", 18, "Phase 1b: Running OCR on scanned pages...")

        ocr_texts_by_page: dict[str, str] = {}
        if config.google_cloud_config:
            pages_needing_ocr = []
            for pi in valid_pages:
                if pi in broken_pages:
                    continue
                native_text = page_texts.get(str(pi), "")
                if len(native_text.strip()) < 50:
                    cached_ocr = fence_cache.get("phase1b", pdf_hash, params,
                                                  page_idx=pi, user_scope=cache_scope)
                    if cached_ocr is not None:
                        ocr_texts_by_page[str(pi)] = cached_ocr.get("ocr_text", "")
                    else:
                        pages_needing_ocr.append(pi)

            if pages_needing_ocr:
                batch_size = cfg.OCR_BATCH_SIZE
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
                            fence_cache.put("phase1b", pdf_hash, params,
                                           {"ocr_text": text}, page_idx=pi,
                                           user_scope=cache_scope)
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
                                fence_cache.put("phase1b", pdf_hash, params,
                                               {"ocr_text": text}, page_idx=pi,
                                               user_scope=cache_scope)
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
            else:
                pages_to_classify.append(pi)

        if pages_to_classify:
            keyword_hits: dict[int, bool] = {}
            llm_needed: list[int] = []
            for pi in pages_to_classify:
                pdf_lines = _text_to_lines(page_texts.get(str(pi), ""), "pdf")
                ocr_lines = _text_to_lines(ocr_texts_by_page.get(str(pi), ""), "ocr")
                kw_result = ade.scan_page_for_keywords_fast(pdf_lines, ocr_lines, config.fence_keywords)
                if kw_result.get("has_keywords"):
                    keyword_hits[pi] = True
                    fence_page_indices.append(pi)
                    fence_cache.put("phase1c", pdf_hash, params,
                                   {"is_fence": True, "method": "keyword"},
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
                            fence_cache.put("phase1c", pdf_hash, params,
                                           {"is_fence": is_fence, "method": "llm"},
                                           page_idx=pi, user_scope=cache_scope)
                    except Exception as e:
                        log.warning(f"Phase 1c LLM batch failed: {e}")
                        for pi in batch:
                            non_fence_page_indices.append(pi)
                            fence_cache.put("phase1c", pdf_hash, params,
                                           {"is_fence": False, "method": "error"},
                                           page_idx=pi, user_scope=cache_scope)
            elif llm_needed:
                for pi in llm_needed:
                    non_fence_page_indices.append(pi)

        fence_page_indices.sort()
        non_fence_page_indices.sort()

        timings["phase1c"] = time.perf_counter() - t1c
        progress("phase1c", 45, f"Phase 1c done: {len(fence_page_indices)} fence, "
                 f"{len(non_fence_page_indices)} non-fence ({timings['phase1c']:.1f}s)")

        if not fence_page_indices:
            progress("done", 100, "No fence pages found")
            result.non_fence_pages = [
                {"page_idx": pi, "page_num": pi + 1} for pi in non_fence_page_indices
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
                    for fut in as_completed(futures):
                        try:
                            pi, chunks = fut.result(timeout=120)
                            ade_chunks_by_page[pi] = chunks
                        except Exception as e:
                            pi = futures[fut]
                            log.warning(f"Phase 2 ADE future {pi}: {e}")
                            ade_chunks_by_page[pi] = []

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

            if chunks:
                legend_chunks, figure_chunks = ade.segment_chunks(chunks)
                page_result["definitions"] = legend_chunks
                page_result["instances"] = figure_chunks
                page_result["keyword_matches"] = []
            else:
                page_result["definitions"] = []
                page_result["instances"] = []
                page_result["keyword_matches"] = []

            legend_chunks = page_result.get("definitions", [])
            figure_chunks = page_result.get("instances", [])
            pdf_lines = _text_to_lines(page_texts.get(str(pi), ""), "pdf")
            ocr_lines = _text_to_lines(ocr_texts_by_page.get(str(pi), ""), "ocr")

            # Legend extraction
            cached_legend = fence_cache.get("phase3_legend", pdf_hash, params,
                                           page_idx=pi, user_scope=cache_scope)
            if cached_legend is not None:
                page_result["legend_entries"] = cached_legend.get("entries", [])
            elif llm_analysis and chunks:
                try:
                    legend_entries = ade.extract_legend_entries(
                        legend_chunks, pdf_lines, ocr_lines,
                        config.fence_keywords, llm_analysis,
                        figure_chunks=figure_chunks,
                    )
                    page_result["legend_entries"] = legend_entries
                    fence_cache.put("phase3_legend", pdf_hash, params,
                                   {"entries": legend_entries}, page_idx=pi,
                                   user_scope=cache_scope)
                except Exception as e:
                    log.warning(f"Phase 3 legend page {pi}: {e}")
                    page_result["legend_entries"] = []
            else:
                page_result["legend_entries"] = []

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
                page_result["measurements"] = cached_measure
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
                    )
                    page_result["measurements"] = measurements or {}
                    fence_cache.put("phase3_measure", pdf_hash, params,
                                   measurements or {}, page_idx=pi,
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
                    fence_results.append({
                        "page_idx": pi, "page_num": pi + 1,
                        "error": str(e),
                    })
                completed_count += 1
                pct = 63 + int(30 * completed_count / max(len(fence_page_indices), 1))
                progress("phase3", pct, f"Phase 3: {completed_count}/{len(fence_page_indices)} pages")

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
            {"page_idx": pi, "page_num": pi + 1} for pi in non_fence_page_indices
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

        if os.path.exists(worker_script):
            task = {
                "pdf_path": pdf_path,
                "fence_pages": fence_results,
                "highlight_fence_text": config.highlight_fence_text,
            }
            proc = subprocess.run(
                [sys.executable, worker_script],
                input=json.dumps(task, default=str).encode(),
                capture_output=True,
                timeout=cfg.HIGHLIGHT_PDF_TIMEOUT,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
            log.warning(f"Highlight worker failed (rc={proc.returncode}): "
                       f"{proc.stderr[:500] if proc.stderr else 'no stderr'}")

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
