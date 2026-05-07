"""Submodule of utils_ade — split from the original 3,153-line monolith.

The public surface is preserved via utils_ade/__init__.py (re-export shim).
External callers should keep importing as `import utils_ade as ade` and
calling `ade.<function>` — the shim makes that work unchanged.
"""

from __future__ import annotations

import re
import json
import os
import time
import functools
import requests
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

try:
    from google.cloud import documentai_v1 as documentai
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None

try:
    from utils_vector import (
        extract_vector_lines,
        extract_layer_names,
        extract_lines_by_layers,
        group_lines_by_layer,
        group_connected_lines,
        calculate_total_length,
        find_lines_near_bbox,
        find_fence_run_from_indicator,
        infer_scale_from_page,
        detect_scale_with_vision,
        VectorLine,
    )
    VECTOR_UTILS_AVAILABLE = True
except ImportError:
    VECTOR_UTILS_AVAILABLE = False

# ADE API endpoint constant (used by ade_api.py; harmless elsewhere).
ADE_PARSE_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"

# Module-level cache for Google Doc AI client (used by ocr.py).
_DOCAI_CLIENT_CACHE = None

def ade_parse_document(pdf_bytes: bytes, api_key: str, zdr: bool = False) -> Dict:
    print(f"[DEBUG] Starting ADE Parsing for document ({len(pdf_bytes)} bytes)...")
    if not api_key:
        return {"success": False, "error": "Missing ADE API Key"}

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {"options": json.dumps({"zdr": zdr})} if zdr else {}

    max_attempts = 2
    connect_timeout_s = 10
    read_timeout_s = 120

    for attempt in range(max_attempts):
        try:
            print(f"[DEBUG] ADE API Request - Attempt {attempt+1}")
            response = requests.post(
                ADE_PARSE_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=(connect_timeout_s, read_timeout_s)
            )
            response.raise_for_status()
            result = response.json()

            chunks = result.get("chunks", [])
            pages = {c.get("grounding", {}).get("page", 0) for c in chunks}
            total_pages = max(pages) + 1 if pages else 0
            print(f"[DEBUG] ADE Success! Found {len(chunks)} chunks across {total_pages} pages.")

            return {
                "success": True,
                "data": {
                    "chunks": chunks,
                    "total_pages": total_pages,
                    "raw": result
                }
            }
        except Exception as e:
            print(f"[DEBUG] ADE Attempt {attempt+1} Failed: {e}")
            if attempt == (max_attempts - 1):
                return {"success": False, "error": str(e)}
            time.sleep(2 * (attempt + 1))

    return {"success": False, "error": "Unknown error after retries"}


def align_ade_chunks_to_page(ade_result: Dict, page_idx: int, page_width: float, page_height: float) -> List[Dict]:
    """Filter the flat ADE chunk list down to chunks belonging to `page_idx`.

    Defensive about page-index base: LandingAI ADE docs don't commit to 0- vs
    1-based `grounding.page`, and we saw cases on dense detail sheets where
    pages appeared to get 0 chunks despite ADE returning data. Check both
    candidates (page_idx and page_idx+1) — if the chunks' observed min page
    is >0, treat the API as 1-based for THIS response.
    """
    chunks = ade_result.get("data", {}).get("chunks", [])
    if not chunks:
        return []

    # Determine base: if the smallest grounding.page we see is >= 1 AND
    # there are no page-0 chunks, the API is 1-based for this call.
    pages_seen = set()
    for c in chunks:
        p = c.get("grounding", {}).get("page")
        if isinstance(p, int):
            pages_seen.add(p)
    is_one_based = bool(pages_seen) and 0 not in pages_seen and min(pages_seen) >= 1
    target_page = page_idx + 1 if is_one_based else page_idx

    if is_one_based:
        # One-line log helps verify next run. Safe to keep — fires at most
        # once per ADE response.
        print(f"[DEBUG] ADE response is 1-based (pages seen: {sorted(pages_seen)[:10]}); "
              f"mapping requested page_idx={page_idx} → target_page={target_page}")

    page_chunks = []
    for chunk in chunks:
        grounding = chunk.get("grounding", {})
        if grounding.get("page") != target_page:
            continue

        box = grounding.get("box", {})
        x0 = float(box.get("left", 0.0)) * page_width
        y0 = float(box.get("top", 0.0)) * page_height
        x1 = float(box.get("right", 1.0)) * page_width
        y1 = float(box.get("bottom", 1.0)) * page_height

        page_chunks.append({
            "id": chunk.get("id", ""),
            "type": chunk.get("type", "unknown"),
            "text": chunk.get("text") or chunk.get("markdown") or "",
            "markdown": chunk.get("markdown", ""),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "bbox": (x0, y0, x1, y1)
        })

    # Extra diagnostics when we come up empty — helps find cases where
    # ADE returned chunks but none matched our page_idx filter.
    if not page_chunks and pages_seen:
        print(f"[DEBUG] align: NO chunks matched page_idx={page_idx} "
              f"(target_page={target_page}, pages_seen={sorted(pages_seen)}, "
              f"total_chunks={len(chunks)})")

    return page_chunks


def segment_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Segment chunks into legend-like (for definition extraction) and figure-like (for instance finding).
    
    A chunk is figure-like if:
    - It's typed as 'figure' or 'architectural_drawing'
    - AND it doesn't have strong legend indicators at the START of its text
    
    We check only the first 200 chars to avoid false positives from ADE's descriptive text.
    """
    legend_like = []
    figure_like = []
    for chunk in chunks:
        raw_type = (chunk.get("type") or "").lower()
        text = chunk.get("text") or ""
        # Only check the beginning of text for legend hints (avoid ADE descriptions)
        text_start = text[:200].lower()
        
        is_figure = raw_type in {"figure", "architectural_drawing"}
        # Check for explicit legend section headers
        has_legend_hint = any(token in text_start for token in {"legend", "keynote", "abbreviation", "symbols"})

        if is_figure and not has_legend_hint:
            figure_like.append(chunk)
        else:
            legend_like.append(chunk)
    print(f"[DEBUG] Segmented: {len(legend_like)} Legend-like chunks, {len(figure_like)} Figure-like chunks.")
    return legend_like, figure_like
