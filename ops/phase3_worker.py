#!/usr/bin/env python3
"""Phase 3 per-page worker run as a short-lived subprocess.

Processes one fence page: legend extraction, scale detection, and
measurement. Writes results directly to fence_cache on disk (keyed by
pdf_sha + params + user_scope, atomic via `*.tmp` + os.replace). When
the subprocess exits the OS reclaims ALL its memory — that's the whole
point vs. threads in the parent process.

IPC
---
Input: one JSON object on stdin describing the task.
Output: one JSON object on stdout with status, timings, error (if any).
Exit code: 0 on success, non-zero on fatal error.

The parent process (app_ade_fast.py) launches this with
`subprocess.Popen(..., stdin=PIPE, stdout=PIPE)` then writes the JSON
task to stdin and closes it, reads stdout, waits with timeout, and
SIGKILLs on timeout.

Task schema
-----------
{
  "pdf_path":                 "/tmp/fence_pdfs/user/file.pdf",
  "page_idx":                 42,
  "pdf_sha":                  "f6d2a197...",
  "cache_params":             "b0ec39df9a64",
  "user_scope":               "session_abc...",
  "fence_cache_dir":          "/home/ubuntu/.cache/fence_ade",
  "openai_api_key":           "sk-proj-...",
  "analysis_model":           "gpt-5.1",
  "classifier_model":         "gpt-5-mini",
  "scale_model":              "gpt-5.1",
  "fence_keywords":           ["fence", "gate", ...],
  "pdf_lines":                [...],       # from _pdf_lines_by_page[page_idx]
  "ocr_lines":                [...],       # from _ocr_lines_by_page[page_idx]
  "ade_chunks":               [...],       # from _ade_chunks_by_page[page_idx]
  "legend_prefill":           {...},       # {chunk_idx: [items]} for this page
  "highlight_fence_text_app": true,
  "enable_unified_measurement": true
}

Result schema
-------------
{
  "page_idx":   42,
  "ok":         true,
  "wrote":      ["phase3_legend", "phase3_scale", "phase3_measure"],
  "wall_s":     17.3,
  "error":      null
}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

# Make the repo root importable so we can reuse utils_ade / fence_cache /
# utils_vector without duplicating their logic here.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def _emit(obj) -> None:
    """Write a single JSON line to stdout + flush."""
    try:
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _load_task() -> dict:
    """Read ONE JSON object from stdin (may span multiple lines)."""
    raw = sys.stdin.read()
    if not raw.strip():
        raise RuntimeError("empty stdin — expected JSON task")
    return json.loads(raw)


def main() -> int:
    t0 = time.perf_counter()
    wrote: list[str] = []
    page_idx = -1
    try:
        task = _load_task()
        page_idx = int(task["page_idx"])

        # Silence MuPDF's C-level stderr spam.
        import fitz
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        # Optionally override the fence_cache dir — same env var the
        # parent uses, so parent and child hit the same files.
        if task.get("fence_cache_dir"):
            os.environ["FENCE_CACHE_DIR"] = task["fence_cache_dir"]

        import fence_cache
        import utils_ade as ade
        from utils_vector import verify_scale_with_bar_fast

        # OpenAI / LangChain LLMs — created fresh per subprocess.
        # Startup cost (~50-200ms for imports, ~0ms for the constructor
        # — the first actual request does the TLS handshake) is the
        # price we pay for subprocess isolation. Acceptable for the
        # memory win.
        from langchain_openai import ChatOpenAI
        api_key = task.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("no OpenAI API key provided")

        analysis_model   = task.get("analysis_model")   or "gpt-5.1"
        classifier_model = task.get("classifier_model") or "gpt-5-mini"
        scale_model      = task.get("scale_model")      or analysis_model

        llm_analysis = ChatOpenAI(
            model=analysis_model, temperature=0, openai_api_key=api_key,
            timeout=60, max_retries=1,
        )
        llm_classifier = ChatOpenAI(
            model=classifier_model, temperature=0, openai_api_key=api_key,
            timeout=60, max_retries=1,
        )
        llm_scale = ChatOpenAI(
            model=scale_model, temperature=0, openai_api_key=api_key,
            timeout=90, max_retries=1,
        )

        pdf_sha       = task["pdf_sha"]
        cache_params  = task["cache_params"]
        user_scope    = task.get("user_scope") or fence_cache.SHARED_SCOPE
        keywords      = task.get("fence_keywords", [])
        pdf_lines     = task.get("pdf_lines", [])
        ocr_lines     = task.get("ocr_lines", [])
        ade_chunks    = task.get("ade_chunks", [])
        legend_prefill = task.get("legend_prefill", {}) or {}
        # Prefill keys come over JSON as strings — coerce back to int.
        legend_prefill = {int(k): v for k, v in legend_prefill.items()}

        highlight_fence_text    = bool(task.get("highlight_fence_text_app", True))
        enable_unified_measure  = bool(task.get("enable_unified_measurement", True))

        # Open the PDF page. The parent already verified the page is
        # non-broken (part of the fence_page_indices it passed us).
        worker_doc = fitz.open(task["pdf_path"])
        try:
            worker_page = worker_doc[page_idx]
        except Exception as e:
            raise RuntimeError(f"page {page_idx} not accessible: {e}")

        # --- 1. Legend entries -------------------------------------------------
        legend_chunks, figure_chunks = ade.segment_chunks(ade_chunks) if ade_chunks else ([], [])
        definitions = []
        if highlight_fence_text and legend_chunks:
            cached = fence_cache.get("phase3_legend", pdf_sha, cache_params,
                                     page_idx=page_idx, user_scope=user_scope)
            if cached is not None:
                definitions = cached
            else:
                try:
                    definitions = ade.extract_legend_entries(
                        legend_chunks=legend_chunks,
                        pdf_lines=pdf_lines,
                        ocr_lines=ocr_lines,
                        fence_keywords=keywords,
                        llm=llm_analysis,
                        figure_chunks=figure_chunks,
                        prefilled_legend_items=legend_prefill or None,
                    )
                    fence_cache.put("phase3_legend", pdf_sha, cache_params,
                                    definitions, page_idx=page_idx, user_scope=user_scope)
                    wrote.append("phase3_legend")
                except Exception as le:
                    print(f"[phase3_worker] page {page_idx + 1} legend error: {le}",
                          file=sys.stderr)

        # --- 2. Page tokens for instance finding ------------------------------
        try:
            rotation    = worker_page.rotation
            mediabox_w  = worker_page.mediabox.width
            mediabox_h  = worker_page.mediabox.height
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
            all_page_tokens.append({
                "text": w[4], "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1,
            })

        # --- 3. Instance detection (numpy, no cache) --------------------------
        instances = []
        if definitions and figure_chunks:
            try:
                instances = ade.find_instances_in_figures_fast(
                    definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines,
                )
            except Exception as ie:
                print(f"[phase3_worker] page {page_idx + 1} instances error: {ie}",
                      file=sys.stderr)

        # --- 4. Scale detection ------------------------------------------------
        detected_scale = None
        scale_cached = fence_cache.get("phase3_scale", pdf_sha, cache_params,
                                       page_idx=page_idx, user_scope=user_scope)
        if scale_cached is not None:
            detected_scale = scale_cached.get("verified_scale")
        else:
            try:
                scale_info = verify_scale_with_bar_fast(worker_page, llm=llm_scale)
                if scale_info.get("success") is not False or scale_info.get("verified_scale"):
                    fence_cache.put("phase3_scale", pdf_sha, cache_params,
                                    scale_info, page_idx=page_idx, user_scope=user_scope)
                    wrote.append("phase3_scale")
                if scale_info.get("success") and scale_info.get("verified_scale"):
                    detected_scale = scale_info["verified_scale"]
            except Exception as se:
                print(f"[phase3_worker] page {page_idx + 1} scale error: {se}",
                      file=sys.stderr)

        # --- 5. Measurement ----------------------------------------------------
        if enable_unified_measure and (definitions or instances):
            if fence_cache.get("phase3_measure", pdf_sha, cache_params,
                               page_idx=page_idx, user_scope=user_scope) is None:
                try:
                    ocr_full_text = "\n".join(line.get("text", "") for line in ocr_lines) if ocr_lines else None
                    measurement_result = ade.measure_fence_elements(
                        worker_page, definitions, instances,
                        figure_chunks=figure_chunks,
                        llm=llm_analysis,
                        light_llm=llm_classifier,
                        scale_factor=detected_scale or 1.0,
                        ocr_text=ocr_full_text,
                    )
                    if measurement_result:
                        fence_cache.put("phase3_measure", pdf_sha, cache_params,
                                        measurement_result, page_idx=page_idx,
                                        user_scope=user_scope)
                        wrote.append("phase3_measure")
                except Exception as me:
                    print(f"[phase3_worker] page {page_idx + 1} measure error: {me}",
                          file=sys.stderr)

        try:
            worker_doc.close()
        except Exception:
            pass

        _emit({
            "page_idx": page_idx,
            "ok": True,
            "wrote": wrote,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": None,
        })
        return 0

    except Exception as e:
        _emit({
            "page_idx": page_idx,
            "ok": False,
            "wrote": wrote,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
