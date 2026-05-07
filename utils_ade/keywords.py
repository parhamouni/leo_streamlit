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

# Cross-submodule imports.
from .llm import llm_classify_page

def scan_page_for_keywords(pdf_lines: List[Dict], ocr_lines: List[Dict], fence_keywords: List[str]) -> Dict:
    """
    Scan page text for fence-related keywords using word boundary matching.
    Returns dict with matched keywords and their locations.

    Uses regex word boundaries to avoid false positives like "gate" in "aggregate".
    Also handles common plural forms (gate -> gates, fence -> fences, etc.)
    """
    all_lines = pdf_lines + ocr_lines
    combined_text = " ".join(line.get("text", "") for line in all_lines).lower()

    matches = []
    matched_lines = []

    for keyword in fence_keywords:
        kw_lower = keyword.lower()
        # Use word boundary matching with optional plural suffix (s, es, ing)
        # This matches: gate, gates, gating; fence, fences, fencing; etc.
        pattern = r'\b' + re.escape(kw_lower) + r'(?:s|es|ing)?\b'
        if re.search(pattern, combined_text):
            matches.append(keyword)
            # Find lines containing this keyword (with word boundary)
            for line in all_lines:
                line_text = line.get("text", "").lower()
                if re.search(pattern, line_text):
                    matched_lines.append({
                        "keyword": keyword,
                        "text": line.get("text", ""),
                        "x0": line.get("x0", 0),
                        "y0": line.get("y0", 0),
                        "x1": line.get("x1", 0),
                        "y1": line.get("y1", 0),
                        "source": line.get("source", "unknown")
                    })

    return {
        "has_keywords": len(matches) > 0,
        "matched_keywords": list(set(matches)),
        "matched_lines": matched_lines,
        "total_text_length": len(combined_text)
    }


@functools.lru_cache(maxsize=8)
def _compile_keyword_patterns(keyword_tuple):
    """Return [(keyword, compiled_pattern), ...] for a tuple of keywords.
    LRU-cached by the tuple so the same config doesn't re-compile."""
    out = []
    for kw in keyword_tuple:
        kw_lower = kw.lower()
        pat = re.compile(r'\b' + re.escape(kw_lower) + r'(?:s|es|ing)?\b')
        out.append((kw, pat))
    return out


def scan_page_for_keywords_fast(pdf_lines: List[Dict], ocr_lines: List[Dict], fence_keywords: List[str]) -> Dict:
    """Drop-in replacement for scan_page_for_keywords with precompiled regex.
    Keep behavior identical — callers can swap transparently."""
    all_lines = pdf_lines + ocr_lines
    combined_text = " ".join(line.get("text", "") for line in all_lines).lower()

    patterns = _compile_keyword_patterns(tuple(fence_keywords))

    matches = []
    matched_lines = []

    for keyword, pat in patterns:
        if pat.search(combined_text):
            matches.append(keyword)
            for line in all_lines:
                line_text = line.get("text", "").lower()
                if pat.search(line_text):
                    matched_lines.append({
                        "keyword": keyword,
                        "text": line.get("text", ""),
                        "x0": line.get("x0", 0),
                        "y0": line.get("y0", 0),
                        "x1": line.get("x1", 0),
                        "y1": line.get("y1", 0),
                        "source": line.get("source", "unknown")
                    })

    return {
        "has_keywords": len(matches) > 0,
        "matched_keywords": list(set(matches)),
        "matched_lines": matched_lines,
        "total_text_length": len(combined_text)
    }


def fallback_fence_detection(
    pdf_lines: List[Dict],
    ocr_lines: List[Dict],
    fence_keywords: List[str],
    llm=None,
    use_llm_confirmation: bool = True
) -> Dict:
    """
    Fallback detection when ADE doesn't find structured definitions.
    1. Scan for keywords
    2. If keywords found and LLM available, confirm with LLM
    3. Return detection result with matched items
    """
    print("[DEBUG] Running fallback fence detection...")
    
    # Step 1: Keyword scan
    keyword_result = scan_page_for_keywords(pdf_lines, ocr_lines, fence_keywords)
    
    if not keyword_result["has_keywords"]:
        print("[DEBUG] No keywords found in fallback scan.")
        return {
            "fence_found": False,
            "method": "keyword_scan",
            "matched_keywords": [],
            "matched_lines": [],
            "llm_result": None
        }
    
    print(f"[DEBUG] Keywords found: {keyword_result['matched_keywords']}")
    
    # FIX 2: High-signal keywords that should NOT be overridden by LLM rejection
    # These are specific fence-related terms that strongly indicate fence content
    HIGH_SIGNAL_KEYWORDS = {
        'fence', 'fencing', 'gate', 'gates', 'chain link', 'guardrail', 
        'railing', 'handrail', 'bollard', 'barrier'
    }
    
    matched_lower = {kw.lower() for kw in keyword_result["matched_keywords"]}
    has_high_signal = bool(matched_lower & HIGH_SIGNAL_KEYWORDS)
    
    if has_high_signal:
        print(f"[DEBUG] High-signal keywords found: {matched_lower & HIGH_SIGNAL_KEYWORDS} - trusting keywords over LLM")
        return {
            "fence_found": True,
            "method": "keyword_high_signal",
            "matched_keywords": keyword_result["matched_keywords"],
            "matched_lines": keyword_result["matched_lines"],
            "llm_result": None
        }
    
    # Step 2: LLM confirmation (if enabled and available) for lower-signal keywords
    llm_result = None
    if use_llm_confirmation and llm:
        all_lines = pdf_lines + ocr_lines
        page_text = " ".join(line.get("text", "") for line in all_lines)
        llm_result = llm_classify_page(llm, page_text, fence_keywords)
        print(f"[DEBUG] LLM classification: {llm_result}")
        
        # Use LLM decision if confident
        if llm_result["confidence"] >= 0.5:
            return {
                "fence_found": llm_result["is_fence_related"],
                "method": "llm_confirmed",
                "matched_keywords": keyword_result["matched_keywords"],
                "matched_lines": keyword_result["matched_lines"],
                "llm_result": llm_result
            }
    
    # Step 3: Fall back to keyword-only decision
    # If we found keywords, consider it fence-related
    return {
        "fence_found": True,
        "method": "keyword_only",
        "matched_keywords": keyword_result["matched_keywords"],
        "matched_lines": keyword_result["matched_lines"],
        "llm_result": llm_result
    }


def fallback_fence_detection_fast(
    pdf_lines: List[Dict],
    ocr_lines: List[Dict],
    fence_keywords: List[str],
    llm=None,
    use_llm_confirmation: bool = True
) -> Dict:
    """Same behavior as fallback_fence_detection, but uses precompiled
    regex for the keyword scan. Drop-in replacement for the fast build."""
    keyword_result = scan_page_for_keywords_fast(pdf_lines, ocr_lines, fence_keywords)

    if not keyword_result["has_keywords"]:
        return {
            "fence_found": False,
            "method": "keyword_scan",
            "matched_keywords": [],
            "matched_lines": [],
            "llm_result": None,
        }

    HIGH_SIGNAL_KEYWORDS = {
        'fence', 'fencing', 'gate', 'gates', 'chain link', 'guardrail',
        'railing', 'handrail', 'bollard', 'barrier'
    }
    matched_lower = {kw.lower() for kw in keyword_result["matched_keywords"]}
    if matched_lower & HIGH_SIGNAL_KEYWORDS:
        return {
            "fence_found": True,
            "method": "keyword_high_signal",
            "matched_keywords": keyword_result["matched_keywords"],
            "matched_lines": keyword_result["matched_lines"],
            "llm_result": None,
        }

    llm_result = None
    if use_llm_confirmation and llm:
        all_lines = pdf_lines + ocr_lines
        page_text = " ".join(line.get("text", "") for line in all_lines)
        llm_result = llm_classify_page(llm, page_text, fence_keywords)
        if llm_result["confidence"] >= 0.5:
            return {
                "fence_found": llm_result["is_fence_related"],
                "method": "llm_confirmed",
                "matched_keywords": keyword_result["matched_keywords"],
                "matched_lines": keyword_result["matched_lines"],
                "llm_result": llm_result,
            }

    return {
        "fence_found": True,
        "method": "keyword_only",
        "matched_keywords": keyword_result["matched_keywords"],
        "matched_lines": keyword_result["matched_lines"],
        "llm_result": llm_result,
    }
