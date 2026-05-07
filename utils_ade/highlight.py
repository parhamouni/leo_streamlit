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

def highlight_page_image(page_image_bytes: bytes, definitions: List[Dict], instances: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    print("[DEBUG] Generating Highlighted Image...")
    print(f"[DEBUG] Highlighting {len(definitions)} definitions and {len(instances)} instances")
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        print(f"[DEBUG] Image size: {img_w} x {img_h}")
        print(f"[DEBUG] PDF size: {pdf_width} x {pdf_height}")
        print(f"[DEBUG] Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
        # Sanity check: warn if scaled coordinates would be outside image
        for i in instances[:3]:
            sx = i.get('x0', 0) * scale_x
            sy = i.get('y0', 0) * scale_y
            if sx > img_w or sy > img_h:
                print(f"[DEBUG] ⚠️ Instance '{i.get('indicator')}' scaled coords ({sx:.1f}, {sy:.1f}) OUTSIDE image bounds!")

        def scale_box(box_dict):
            scaled = [
                box_dict.get("x0", 0) * scale_x,
                box_dict.get("y0", 0) * scale_y,
                box_dict.get("x1", 0) * scale_x,
                box_dict.get("y1", 0) * scale_y
            ]
            return scaled

        for d in definitions:
            box = scale_box(d)
            print(f"[DEBUG] Definition box: orig=({d.get('x0'):.1f}, {d.get('y0'):.1f}) -> scaled=({box[0]:.1f}, {box[1]:.1f})")
            draw.rectangle(box, outline=(0, 255, 0, 255), width=3)
            draw.rectangle(box, fill=(0, 255, 0, 40))
        
        for idx, i in enumerate(instances[:5]):  # Log first 5 instances
            box = scale_box(i)
            print(f"[DEBUG] Instance {idx} '{i.get('indicator')}' box: orig=({i.get('x0'):.1f}, {i.get('y0'):.1f}, {i.get('x1'):.1f}, {i.get('y1'):.1f}) -> scaled=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        
        for i in instances:
            box = scale_box(i)
            draw.rectangle(box, outline=(255, 0, 255, 255), width=3)

        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Visualization Error: {e}")
        return page_image_bytes


def highlight_keyword_matches(page_image_bytes: bytes, matched_lines: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    """
    Highlight keyword matches on the page image.
    Uses orange color to distinguish from definition (green) and instance (purple) highlights.
    """
    if not matched_lines:
        return page_image_bytes
    
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        for line in matched_lines:
            box = [
                line.get("x0", 0) * scale_x,
                line.get("y0", 0) * scale_y,
                line.get("x1", 0) * scale_x,
                line.get("y1", 0) * scale_y
            ]
            # Orange outline for keyword matches
            draw.rectangle(box, outline=(255, 165, 0, 255), width=2)
            draw.rectangle(box, fill=(255, 165, 0, 40))
        
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Keyword highlight error: {e}")
        return page_image_bytes


def debug_visualize_coordinates(page_bytes: bytes, ade_chunks: List[Dict], pdf_lines: List[Dict], ocr_lines: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    print("[DEBUG] Generating Layer Visualization...")
    try:
        img = Image.open(BytesIO(page_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0

        def get_rect(item):
            return [
                item.get("x0", 0) * scale_x,
                item.get("y0", 0) * scale_y,
                item.get("x1", 0) * scale_x,
                item.get("y1", 0) * scale_y
            ]

        for line in pdf_lines:
            draw.rectangle(get_rect(line), outline=(0, 0, 255, 128), width=1)
        for line in ocr_lines:
            draw.rectangle(get_rect(line), outline=(255, 165, 0, 128), width=1)
        for chunk in ade_chunks:
            draw.rectangle(get_rect(chunk), outline=(255, 0, 0, 255), width=3)

        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Debug Viz Error: {e}")
        return page_bytes


def highlight_fence_lines(
    page_image_bytes: bytes,
    fence_lines: List,  # List of VectorLine
    pdf_width: float,
    pdf_height: float
) -> bytes:
    """
    Highlight measured fence lines on the page image.
    
    Args:
        page_image_bytes: Original page image
        fence_lines: List of VectorLine objects to highlight
        pdf_width, pdf_height: PDF page dimensions
    
    Returns:
        Image bytes with highlighted lines
    """
    if not fence_lines:
        return page_image_bytes
    
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        for line in fence_lines:
            x1 = line.start[0] * scale_x
            y1 = line.start[1] * scale_y
            x2 = line.end[0] * scale_x
            y2 = line.end[1] * scale_y
            
            # Draw line in cyan color
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 255, 200), width=3)
        
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Fence line highlight error: {e}")
        return page_image_bytes
