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

def get_native_pdf_lines(page: fitz.Page) -> List[Dict]:
    """
    Extract text lines from PDF with coordinates in display space.
    
    IMPORTANT: Transforms MediaBox coords to display coords for rotated pages.
    PDF text coordinates are in MediaBox space, but we need display space
    to match the rendered image coordinates.
    """
    structure = page.get_text("dict")
    rotation = page.rotation
    mediabox_w = page.mediabox.width
    mediabox_h = page.mediabox.height
    
    def transform_for_rotation(x0, y0, x1, y1):
        """Transform MediaBox coords to display coords based on page rotation"""
        if rotation == 0:
            return x0, y0, x1, y1
        elif rotation == 90:
            return mediabox_h - y1, x0, mediabox_h - y0, x1
        elif rotation == 180:
            return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
        elif rotation == 270:
            return y0, mediabox_w - x1, y1, mediabox_w - x0
        return x0, y0, x1, y1
    
    lines = []
    for block in structure.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                bbox = line["bbox"]
                if text:
                    # Transform from MediaBox to display coordinates
                    nx0, ny0, nx1, ny1 = transform_for_rotation(bbox[0], bbox[1], bbox[2], bbox[3])
                    lines.append({
                        "text": text,
                        "x0": nx0, "y0": ny0, 
                        "x1": nx1, "y1": ny1,
                        "source": "pdf_native"
                    })
    print(f"[DEBUG] PDF Native extraction: Found {len(lines)} lines.")
    return lines


def create_single_page_pdf(pdf_bytes: bytes, page_index: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def create_multi_page_pdf(full_pdf_bytes: bytes, page_indices: List[int]) -> bytes:
    """Create a PDF containing only the specified pages (preserving order).

    In the resulting PDF, pages are numbered 0..len(page_indices)-1,
    so callers must map local index back to the original page index.
    """
    doc = fitz.open(stream=full_pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    for idx in page_indices:
        new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def create_single_page_pdf_from_path(pdf_path: str, page_index: int) -> bytes:
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def create_multi_page_pdf_from_path(pdf_path: str, page_indices: List[int]) -> bytes:
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    for idx in page_indices:
        new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def measure_page_bytes(pdf_path: str, page_index: int) -> int:
    """Return the serialized byte length of a one-page PDF containing
    just `page_index`. Used by size-aware batching.

    Uses garbage=4 + clean=True so the returned size reflects only the
    page's own contribution. Without garbage cleanup, fitz's `insert_pdf`
    keeps cross-references to ALL source-doc resources for any page that
    references a shared global resource dictionary — and `tobytes()`
    materialises them, making a single-page PDF report the FULL source
    PDF size (e.g. one page reading as 378 MB on a 378 MB source). That
    bogus reading made the batch packer think the page exceeded the
    per-page hard cap and put it alone in a batch unnecessarily.
    """
    try:
        doc = fitz.open(pdf_path)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        n = len(new_doc.tobytes(garbage=4, deflate=True, clean=True))
        doc.close()
        new_doc.close()
        return n
    except Exception:
        return 0


def _render_page_as_image_pdf(doc: 'fitz.Document', page_index: int, dpi: int,
                              jpeg_quality: int = 85) -> bytes:
    """Render one page as a JPEG and wrap it in a single-page PDF of the
    same physical size. Vector → raster; uses JPEG because PNG balloons
    on complex line art (we've seen PNG at 250 DPI outgrow the original
    vector PDF). JPEG at q=85 preserves engineering text/lines while
    cutting size by ~10× vs PNG.

    Physical-pixel safety: if the requested DPI would produce an image
    larger than MAX_EDGE_PX on either axis (e.g. poster-size drawings),
    scale DPI down so the pixmap stays bounded. Avoids 100 MB images
    from huge page dimensions regardless of DPI.
    """
    MAX_EDGE_PX = 6000  # ~40in × 150dpi; keeps JPEG under ~3 MB on typical drawings

    src_page = doc.load_page(page_index)
    # Target pixel size at the requested DPI
    w_pts = src_page.rect.width
    h_pts = src_page.rect.height
    w_px = (w_pts / 72.0) * dpi
    h_px = (h_pts / 72.0) * dpi
    effective_dpi = int(dpi)
    if max(w_px, h_px) > MAX_EDGE_PX:
        effective_dpi = max(36, int(dpi * (MAX_EDGE_PX / max(w_px, h_px))))

    pix = src_page.get_pixmap(dpi=effective_dpi, alpha=False)
    # fitz's .tobytes("jpeg") accepts a jpg_quality kwarg in recent
    # builds; fall back to PIL if the installed fitz is older.
    try:
        img_bytes = pix.tobytes("jpeg", jpg_quality=jpeg_quality)
    except TypeError:
        from PIL import Image
        import io as _io
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        img_bytes = buf.getvalue()
    del pix

    new_doc = fitz.open()
    new_page = new_doc.new_page(width=w_pts, height=h_pts)
    new_page.insert_image(new_page.rect, stream=img_bytes)
    out = new_doc.tobytes()
    new_doc.close()
    return out


def safe_single_page_pdf_from_path(
    pdf_path: str,
    page_index: int,
    max_bytes: int = 15 * 1024 * 1024,
    # DPI floor 150 preserves readable text/lines for both DocAI (reads
    # text) and LandingAI ADE (detects symbols). Below 150 the risk of
    # misreading small callouts and numbers is real. If a page can't fit
    # the caller's size budget at 150 DPI, we still return the result —
    # caller can log / skip as appropriate.
    dpi_ladder: tuple = (250, 200, 175, 150),
) -> bytes:
    """Build a single-page PDF ≤ max_bytes.

    Strategy:
      1. Try a naive page-copy (fast path — preserves vectors).
      2. If the copy exceeds max_bytes, re-render the page as PNG at
         progressively lower DPI until the result fits. The last DPI
         in `dpi_ladder` is returned even if it still exceeds the cap
         (caller can decide to give up vs. send it).
    """
    doc = fitz.open(pdf_path)
    try:
        return _safe_single_page_pdf_from_doc(doc, page_index, max_bytes, dpi_ladder)
    finally:
        doc.close()


def _safe_single_page_pdf_from_doc(
    doc: 'fitz.Document',
    page_index: int,
    max_bytes: int,
    dpi_ladder: tuple,
) -> bytes:
    """Internal: same as safe_single_page_pdf_from_path but reuses a doc
    handle (avoids re-opening the PDF for every page in a batch)."""
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    # garbage=4 + clean=True prunes resources `insert_pdf` dragged in by
    # xref but the page itself doesn't reference. Without this, pages
    # that participate in a shared global resource dict (common in
    # CAD-exported PDFs) extract as a single-page document that still
    # carries the FULL source PDF size — e.g. one page reporting
    # native=378.2MB even though the page itself is only a few MB.
    # That bogus size triggered spurious downsampling. With garbage
    # collection the reported size matches the actual page contribution.
    native = new_doc.tobytes(garbage=4, deflate=True, clean=True)
    new_doc.close()
    if len(native) <= max_bytes:
        return native
    last_out = native
    for dpi in dpi_ladder:
        try:
            out = _render_page_as_image_pdf(doc, page_index, dpi)
        except Exception as e:
            print(f"[downsample] page {page_index+1} render@{dpi}dpi failed: {e}")
            continue
        print(f"[downsample] page {page_index+1}: native={len(native)/1024/1024:.1f}MB "
              f"→ @{dpi}dpi={len(out)/1024/1024:.1f}MB")
        last_out = out
        if len(out) <= max_bytes:
            return out
    return last_out


def safe_multi_page_pdf_from_path(
    pdf_path: str,
    page_indices: List[int],
    per_page_max_bytes: int = 15 * 1024 * 1024,
    # DPI floor 150 preserves readable text/lines for both DocAI (reads
    # text) and LandingAI ADE (detects symbols). Below 150 the risk of
    # misreading small callouts and numbers is real. If a page can't fit
    # the caller's size budget at 150 DPI, we still return the result —
    # caller can log / skip as appropriate.
    dpi_ladder: tuple = (250, 200, 175, 150),
) -> bytes:
    """Multi-page variant of safe_single_page_pdf_from_path.

    For each requested page: naive-copy if it fits the per-page cap;
    otherwise render at decreasing DPI until it does. The resulting
    multi-page PDF has deterministic page order (same as page_indices).
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()
    try:
        for pi in page_indices:
            safe_bytes = _safe_single_page_pdf_from_doc(
                doc, pi, per_page_max_bytes, dpi_ladder,
            )
            tmp = fitz.open(stream=safe_bytes, filetype="pdf")
            new_doc.insert_pdf(tmp, from_page=0, to_page=0)
            tmp.close()
        return new_doc.tobytes()
    finally:
        new_doc.close()
        doc.close()


def create_page_batches(
    full_pdf_bytes: bytes,
    page_indices: List[int],
    max_batch_bytes: int = 15 * 1024 * 1024,
    max_pages_per_batch: int = 10
) -> List[List[int]]:
    """Split page indices into batches respecting estimated size and page count limits.
    
    Uses average page size from the full PDF as a heuristic. A 1.3x safety factor
    accounts for PDF structure overhead. This avoids the expense of creating
    temporary single-page PDFs just to measure their size.
    
    Args:
        full_pdf_bytes: Complete PDF file bytes
        page_indices: 0-indexed page numbers to include
        max_batch_bytes: Max estimated PDF size per batch (default 15MB)
        max_pages_per_batch: Hard cap on pages per batch (default 10)
    
    Returns:
        List of page-index lists, one per batch
    """
    if not page_indices:
        return []
    
    total_pdf_size = len(full_pdf_bytes)
    try:
        doc = fitz.open(stream=full_pdf_bytes, filetype="pdf")
        total_pages_in_doc = len(doc)
        doc.close()
    except Exception:
        total_pages_in_doc = max(page_indices) + 1
    
    avg_page_size = (total_pdf_size / max(total_pages_in_doc, 1)) * 1.3
    max_by_size = max(1, int(max_batch_bytes / max(avg_page_size, 1)))
    effective_max = min(max_by_size, max_pages_per_batch)
    
    print(f"[BATCH] PDF: {total_pdf_size/1024/1024:.1f}MB, {total_pages_in_doc} pages, "
          f"~{avg_page_size/1024:.0f}KB/page -> max {effective_max} pages/batch")
    
    batches = []
    for i in range(0, len(page_indices), effective_max):
        batches.append(page_indices[i:i + effective_max])
    
    print(f"[BATCH] {len(batches)} batch(es) for {len(page_indices)} fence pages: "
          + ", ".join(f"[{','.join(str(p+1) for p in b)}]" for b in batches))
    return batches


def create_page_batches_from_path(
    pdf_path: str,
    page_indices: List[int],
    max_batch_bytes: int = 15 * 1024 * 1024,
    max_pages_per_batch: int = 10,
    per_page_hard_cap: int = 40 * 1024 * 1024,
) -> List[List[int]]:
    """Size-aware batching.

    Earlier versions used (total PDF size / page count) × 1.3 as the
    estimate and capped batches at 10 pages. That fails when one page
    embeds a massive raster — a single 145 MB page would sit in a batch
    alongside 9 small pages and the whole batch would be 150 MB, well
    past LandingAI's limit.

    This version measures each page individually (cheap: fitz open +
    single-page copy), then greedily packs pages into batches with a
    hard byte cap. Any page whose native size exceeds
    `per_page_hard_cap` gets its OWN batch so the caller knows to
    downsample it.

    Return value still mirrors the old API: list[list[int]].
    """
    import os as _os
    if not page_indices:
        return []
    try:
        total_pdf_size = _os.path.getsize(pdf_path)
    except Exception:
        total_pdf_size = 0
    try:
        doc_probe = fitz.open(pdf_path)
        total_pages_in_doc = len(doc_probe)
        doc_probe.close()
    except Exception:
        total_pages_in_doc = max(page_indices) + 1

    # Measure per-page size. This opens the source PDF once per page —
    # expensive O(P) but fence-page counts are tens, not thousands.
    page_bytes: dict = {}
    oversized: list = []
    for pi in page_indices:
        sz = measure_page_bytes(pdf_path, pi)
        page_bytes[pi] = sz
        if sz > per_page_hard_cap:
            oversized.append((pi, sz))

    if oversized:
        pretty = ", ".join(f"p{pi+1}={sz/1024/1024:.1f}MB" for pi, sz in oversized)
        print(f"[BATCH-PATH] ⚠ oversized pages (each in own batch, will be downsampled): {pretty}")

    batches: List[List[int]] = []
    current: List[int] = []
    current_bytes = 0
    for pi in page_indices:
        sz = page_bytes.get(pi, 0)
        if sz > per_page_hard_cap:
            # Flush the running batch; the giant page gets its own batch.
            if current:
                batches.append(current); current = []; current_bytes = 0
            batches.append([pi])
            continue
        if (current_bytes + sz > max_batch_bytes
                or len(current) >= max_pages_per_batch):
            if current:
                batches.append(current)
            current = [pi]; current_bytes = sz
        else:
            current.append(pi); current_bytes += sz
    if current:
        batches.append(current)

    print(f"[BATCH-PATH] PDF: {total_pdf_size/1024/1024:.1f}MB, {total_pages_in_doc} pages; "
          f"{len(page_indices)} target pages → {len(batches)} batch(es). "
          f"Sizes: " + ", ".join(
              f"{sum(page_bytes.get(p, 0) for p in b)/1024/1024:.1f}MB"
              for b in batches))
    return batches
