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

def get_docai_client(google_cloud_config: Dict):
    global _DOCAI_CLIENT_CACHE
    if _DOCAI_CLIENT_CACHE:
        return _DOCAI_CLIENT_CACHE
    if not GOOGLE_CLOUD_AVAILABLE:
        print("[DEBUG] Google Cloud libraries missing.")
        return None

    try:
        service_info = google_cloud_config.get("service_account_info")
        if service_info:
            creds = service_account.Credentials.from_service_account_info(service_info)
            _DOCAI_CLIENT_CACHE = documentai.DocumentProcessorServiceClient(credentials=creds)
            return _DOCAI_CLIENT_CACHE
    except Exception as e:
        print(f"[DEBUG] ❌ Error creating DocAI client: {e}")
        return None


def run_google_ocr_blocks_multipage(
    multipage_pdf_bytes: bytes,
    google_cloud_config: Dict,
    page_dims_by_local_idx: Dict[int, Tuple[float, float]],
) -> Dict[int, List[Dict]]:
    """Run Google Document AI OCR on a multi-page PDF in one API call.

    Instead of N calls × 1 page, sends one request with up to 15 pages
    as a PDF (DocAI's sync per-request page cap). Returns a dict keyed
    by the LOCAL page index within the PDF (0, 1, …, N-1). The caller
    is responsible for mapping local → original page index.

    page_dims_by_local_idx: {local_idx: (pdf_width, pdf_height)} — used
    to convert normalized vertices to PDF coordinate space per page.

    Each value in the returned dict is the same shape as
    run_google_ocr_blocks() returned for a single page.
    """
    print(f"[DEBUG] Starting multi-page Google OCR ({len(page_dims_by_local_idx)} pages)...")
    client = get_docai_client(google_cloud_config)
    if not client:
        return {i: [] for i in page_dims_by_local_idx}

    project_id = google_cloud_config.get("project_number")
    location = google_cloud_config.get("location")
    processor_id = google_cloud_config.get("processor_id")
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    raw_document = documentai.RawDocument(content=multipage_pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    try:
        result = client.process_document(request=request, timeout=180)
        doc_result = result.document
        print(f"[DEBUG] Multi-page OCR returned; total text length: {len(doc_result.text)}, "
              f"pages: {len(doc_result.pages)}")
    except Exception as e:
        print(f"[DEBUG] ❌ DocAI Multi-page OCR failed: {e}")
        return {i: [] for i in page_dims_by_local_idx}

    out: Dict[int, List[Dict]] = {i: [] for i in page_dims_by_local_idx}
    for local_idx, ocr_page in enumerate(doc_result.pages):
        if local_idx not in page_dims_by_local_idx:
            continue
        pdf_width, pdf_height = page_dims_by_local_idx[local_idx]
        lines: List[Dict] = []

        for paragraph in ocr_page.paragraphs:
            text_content = ""
            layout = paragraph.layout
            for segment in layout.text_anchor.text_segments:
                text_content += doc_result.text[segment.start_index:segment.end_index]
            text_content = text_content.strip()
            if not text_content:
                continue
            vertices = layout.bounding_poly.normalized_vertices
            if not vertices:
                continue
            xs = [v.x * pdf_width for v in vertices if v.x is not None]
            ys = [v.y * pdf_height for v in vertices if v.y is not None]
            if not xs or not ys:
                continue
            lines.append({
                "text": text_content,
                "x0": min(xs), "y0": min(ys),
                "x1": max(xs), "y1": max(ys),
                "source": "ocr_paragraph",
            })

        # Token-level (same dedup logic as the single-page variant).
        existing_positions = set((round(l['x0']), round(l['y0']), l['text']) for l in lines)
        token_source = list(getattr(ocr_page, 'tokens', None) or [])
        if not token_source:
            for line in getattr(ocr_page, 'lines', []):
                for word in getattr(line, 'words', []):
                    token_source.append(word)
        for token in token_source:
            layout = token.layout
            text_content = ""
            for segment in layout.text_anchor.text_segments:
                text_content += doc_result.text[segment.start_index:segment.end_index]
            text_content = text_content.strip()
            if not text_content:
                continue
            vertices = layout.bounding_poly.normalized_vertices
            if not vertices:
                continue
            xs = [v.x * pdf_width for v in vertices if v.x is not None]
            ys = [v.y * pdf_height for v in vertices if v.y is not None]
            if not xs or not ys:
                continue
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            pos_key = (round(x0), round(y0), text_content)
            if pos_key in existing_positions:
                continue
            existing_positions.add(pos_key)
            lines.append({
                "text": text_content,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "source": "ocr_token",
            })

        out[local_idx] = lines
    return out


def run_google_ocr_blocks(page_bytes: bytes, google_cloud_config: Dict, pdf_width: float, pdf_height: float) -> List[Dict]:
    """
    Run Google Document AI OCR on a PDF page and return text blocks with PDF coordinates.
    
    IMPORTANT: normalized_vertices from Document AI are relative to the IMAGE dimensions.
    We must convert them to PDF coordinate space using the correct scale factors.
    """
    print("[DEBUG] Starting Google OCR...")
    client = get_docai_client(google_cloud_config)
    if not client:
        return []

    project_id = google_cloud_config.get("project_number")
    location = google_cloud_config.get("location")
    processor_id = google_cloud_config.get("processor_id")
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]

    # --- DYNAMIC ZOOM ---
    max_dimension = 4000.0
    current_max = max(page.rect.width, page.rect.height)
    zoom = min(3.0, max_dimension / current_max)
    print(f"[DEBUG] Rendering page image with Zoom: {zoom:.2f}...")

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Store actual image dimensions for coordinate conversion
    img_width = pix.width
    img_height = pix.height
    print(f"[DEBUG] Image dimensions: {img_width} x {img_height} pixels")
    print(f"[DEBUG] PDF dimensions: {pdf_width:.1f} x {pdf_height:.1f} points")

    # --- JPEG COMPRESSION ---
    image_content = pix.tobytes("jpeg", jpg_quality=85)
    img_size_mb = len(image_content) / (1024 * 1024)
    print(f"[DEBUG] Image rendered. Size: {img_size_mb:.2f} MB. Sending to Google API...")

    raw_document = documentai.RawDocument(content=image_content, mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    try:
        result = client.process_document(request=request, timeout=60)
        doc_result = result.document
        print(f"[DEBUG] Google API returned successfully. Text length: {len(doc_result.text)}")
    except Exception as e:
        print(f"[DEBUG] ❌ DocAI Processing Failed: {e}")
        doc.close()
        return []

    if not doc_result.pages:
        doc.close()
        return []
    ocr_page = doc_result.pages[0]

    # Calculate scale factors to convert from image pixels to PDF points
    # normalized_vertices are 0-1 relative to image, so:
    # pixel_coord = normalized * img_dimension
    # pdf_coord = pixel_coord * (pdf_dimension / img_dimension)
    # Simplified: pdf_coord = normalized * pdf_dimension (since aspect ratio is preserved)
    
    # But to be safe, let's use explicit scale factors
    scale_x = pdf_width / img_width if img_width > 0 else 1.0
    scale_y = pdf_height / img_height if img_height > 0 else 1.0
    print(f"[DEBUG] Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

    ocr_lines = []

    for paragraph in ocr_page.paragraphs:
        text_content = ""
        layout = paragraph.layout
        for segment in layout.text_anchor.text_segments:
            text_content += doc_result.text[segment.start_index:segment.end_index]

        text_content = text_content.strip()
        if not text_content:
            continue

        vertices = layout.bounding_poly.normalized_vertices
        if not vertices:
            continue

        # Convert normalized (0-1) to pixel coordinates, then to PDF coordinates
        xs = [v.x * img_width * scale_x for v in vertices if v.x is not None]
        ys = [v.y * img_height * scale_y for v in vertices if v.y is not None]
        
        if not xs or not ys:
            continue

        ocr_lines.append({
            "text": text_content,
            "x0": min(xs), "y0": min(ys),
            "x1": max(xs), "y1": max(ys),
            "source": "ocr_paragraph"
        })

    # Also extract at token/word level — captures small individual indicator callouts
    # (like circled "2.09") that paragraph-level extraction misses or groups
    token_count = 0
    existing_positions = set((round(l['x0']), round(l['y0']), l['text']) for l in ocr_lines)
    
    # Try page-level tokens first, fall back to words within lines
    token_source = list(getattr(ocr_page, 'tokens', None) or [])
    if not token_source:
        for line in getattr(ocr_page, 'lines', []):
            for word in getattr(line, 'words', []):
                token_source.append(word)
    
    for token in token_source:
        layout = token.layout
        text_content = ""
        for segment in layout.text_anchor.text_segments:
            text_content += doc_result.text[segment.start_index:segment.end_index]
        text_content = text_content.strip()
        if not text_content:
            continue
        
        vertices = layout.bounding_poly.normalized_vertices
        if not vertices:
            continue
        xs = [v.x * img_width * scale_x for v in vertices if v.x is not None]
        ys = [v.y * img_height * scale_y for v in vertices if v.y is not None]
        if not xs or not ys:
            continue
        
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pos_key = (round(x0), round(y0), text_content)
        if pos_key in existing_positions:
            continue
        existing_positions.add(pos_key)
        
        ocr_lines.append({
            "text": text_content,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "source": "ocr_token"
        })
        token_count += 1

    print(f"[DEBUG] OCR extraction complete. Found {len(ocr_lines)} items ({len(ocr_lines) - token_count} paragraphs + {token_count} tokens).")
    doc.close()
    return ocr_lines
