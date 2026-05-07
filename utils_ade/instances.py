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

def is_center_inside(item_box: Dict, chunk_box: Tuple[float, float, float, float], tolerance: float = 15.0) -> bool:
    item_cx = (item_box["x0"] + item_box["x1"]) / 2
    item_cy = (item_box["y0"] + item_box["y1"]) / 2
    cx0, cy0, cx1, cy1 = chunk_box

    inside_x = (cx0 - tolerance) <= item_cx <= (cx1 + tolerance)
    inside_y = (cy0 - tolerance) <= item_cy <= (cy1 + tolerance)
    return inside_x and inside_y


def find_best_bbox(search_text: str, pdf_lines: List[Dict], ocr_lines: List[Dict], chunk_bbox: Tuple[float, float, float, float], **kwargs) -> Optional[Dict]:
    if not search_text:
        return None
    target = re.sub(r'[^0-9a-zA-Z]', '', search_text).lower()
    if not target:
        return None

    def check_candidates(candidates):
        for line in candidates:
            line_clean = re.sub(r'[^0-9a-zA-Z]', '', line["text"]).lower()
            if target not in line_clean:
                continue
            # Global search override (if chunk is huge) or local check
            if chunk_bbox[2] > 5000 or is_center_inside(line, chunk_bbox):
                return line
        return None

    match = check_candidates(pdf_lines)
    if match:
        return match
    match = check_candidates(ocr_lines)
    if match:
        return match
    return None


def find_instances_in_figures(legend_entries: List[Dict], figure_chunks: List[Dict], all_tokens: List[Dict], ocr_lines: List[Dict] = None) -> List[Dict]:
    """
    Find instances of legend indicators ONLY within figure/architectural_drawing chunks.
    
    Args:
        legend_entries: List of definitions with 'indicator' field
        figure_chunks: List of figure region bounding boxes (type=figure or architectural_drawing)
        all_tokens: All text tokens from the page
        ocr_lines: OCR text blocks as fallback when page has no embedded text
    """
    print("[DEBUG] Finding Instances in Figure Chunks...")
    print(f"[DEBUG] Figure chunks count: {len(figure_chunks)}")
    
    # Always merge OCR tokens into the search pool alongside native PDF tokens.
    # Native PDF text often misses indicator callouts in drawings (they're graphical),
    # but OCR can read them. We deduplicate by position to avoid double-counting.
    if ocr_lines:
        existing_positions = set()
        for t in all_tokens:
            existing_positions.add((round(t['x0']), round(t['y0']), t.get('text', '')))
        
        ocr_added = 0
        for ocr_line in ocr_lines:
            text = ocr_line.get('text', '').strip()
            if not text:
                continue
            words = text.split()
            if len(words) <= 1:
                pos_key = (round(ocr_line.get('x0', 0)), round(ocr_line.get('y0', 0)), text)
                if pos_key not in existing_positions:
                    all_tokens.append({
                        'text': text,
                        'x0': ocr_line.get('x0', 0),
                        'y0': ocr_line.get('y0', 0),
                        'x1': ocr_line.get('x1', 0),
                        'y1': ocr_line.get('y1', 0),
                    })
                    existing_positions.add(pos_key)
                    ocr_added += 1
            else:
                # For multi-word OCR blocks, split into words with approximate positions
                total_len = sum(len(w) for w in words)
                x0 = ocr_line.get('x0', 0)
                x1 = ocr_line.get('x1', 0)
                y0 = ocr_line.get('y0', 0)
                y1 = ocr_line.get('y1', 0)
                span = x1 - x0
                cursor = x0
                for w in words:
                    w_span = (len(w) / max(total_len, 1)) * span
                    pos_key = (round(cursor), round(y0), w)
                    if pos_key not in existing_positions:
                        all_tokens.append({
                            'text': w,
                            'x0': cursor,
                            'y0': y0,
                            'x1': cursor + w_span,
                            'y1': y1,
                        })
                        existing_positions.add(pos_key)
                        ocr_added += 1
                    cursor += w_span
        print(f"[DEBUG] Merged {ocr_added} OCR tokens into {len(all_tokens)} total tokens (deduped)")
    for i, fc in enumerate(figure_chunks):
        print(f"[DEBUG]   Figure {i}: type={fc.get('type')} bbox=({fc['x0']:.1f}, {fc['y0']:.1f}) - ({fc['x1']:.1f}, {fc['y1']:.1f})")
    
    instances = []
    
    def normalize_indicator(s):
        """Normalize dotted indicator numbers by stripping leading zeros from each segment.
        E.g. '2.09' -> '2.9', '02.03' -> '2.3', '2.9' -> '2.9'.
        Non-numeric parts are left unchanged."""
        parts = s.split('.')
        normalized_parts = []
        for p in parts:
            # Strip leading zeros from numeric segments, but keep at least one digit
            if p.lstrip('0').isdigit():
                normalized_parts.append(p.lstrip('0') or '0')
            elif p.isdigit() or (p and p[0].isdigit()):
                normalized_parts.append(p.lstrip('0') or '0')
            else:
                normalized_parts.append(p)
        return '.'.join(normalized_parts)
    
    # Collect all indicators to search for
    indicators_to_find = set()
    # Map from normalized form -> original indicator string
    norm_to_original = {}
    for item in legend_entries:
        ind = item.get("indicator", "").strip()
        if ind:
            indicators_to_find.add(ind)
            # Also add cleaned version (remove special chars like parentheses)
            # but NOT for dotted numeric indicators (e.g. "2.8" -> "28" would
            # match random numbers like dimensions/grid refs)
            clean_ind = re.sub(r'[^\w]', '', ind)
            if clean_ind and clean_ind != ind and not clean_ind.isdigit():
                indicators_to_find.add(clean_ind)
            # Build normalized lookup: normalized form -> original indicator
            norm_key = normalize_indicator(ind)
            norm_to_original.setdefault(norm_key, ind)
            if clean_ind and clean_ind != ind and not clean_ind.isdigit():
                norm_clean = normalize_indicator(clean_ind)
                norm_to_original.setdefault(norm_clean, ind)
    
    print(f"[DEBUG] Looking for indicators: {indicators_to_find}")
    print(f"[DEBUG] Normalized indicator map: {norm_to_original}")
    
    if not indicators_to_find:
        return []
    
    if not figure_chunks:
        print("[DEBUG] No figure chunks to search in!")
        return []
    
    # Get legend bounding boxes to exclude (don't match indicators in legend area)
    # ONLY use legend-sourced definitions for exclusion — figure/OCR-sourced definitions
    # are IN the drawing area and should NOT create exclusion zones
    legend_bboxes = []
    for entry in legend_entries:
        if entry.get("extraction_pass", "legend") != "legend":
            continue
        if all(k in entry for k in ['x0', 'y0', 'x1', 'y1']):
            legend_bboxes.append((entry['x0'], entry['y0'], entry['x1'], entry['y1']))
    
    def is_in_legend_area(token):
        """Check if token is inside any legend bounding box (with margin)"""
        margin = 20  # PDF units margin
        tx, ty = (token['x0'] + token['x1']) / 2, (token['y0'] + token['y1']) / 2
        for lx0, ly0, lx1, ly1 in legend_bboxes:
            if lx0 - margin <= tx <= lx1 + margin and ly0 - margin <= ty <= ly1 + margin:
                return True
        return False
    
    # Filter tokens to only those inside figure chunks
    figure_tokens = []
    for chunk in figure_chunks:
        cx0, cy0, cx1, cy1 = chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]
        for t in all_tokens:
            # Check if token center is inside chunk
            tx, ty = (t['x0'] + t['x1']) / 2, (t['y0'] + t['y1']) / 2
            if cx0 <= tx <= cx1 and cy0 <= ty <= cy1:
                figure_tokens.append(t)
    
    print(f"[DEBUG] Tokens inside figure chunks: {len(figure_tokens)} (out of {len(all_tokens)} total)")
    
    # Search for indicator matches
    # For short numeric indicators (1-2 digits), require EXACT match to avoid false positives
    # from dimensions, grid references, etc.
    found_positions = set()
    
    for token in figure_tokens:
        token_text = token.get("text", "").strip()
        if not token_text:
            continue
        
        # Check if this token matches any indicator
        matched_indicator = None
        
        # For exact matching, the token text should match the indicator exactly
        # (not be part of a larger number like "180" matching "18")
        for ind in indicators_to_find:
            # Check for exact match
            if token_text == ind:
                matched_indicator = ind
                break
            # Check if token is the indicator with parentheses like "(20)"
            if token_text == f"({ind})" or token_text == f"[{ind}]":
                matched_indicator = ind
                break
            # For indicators with special chars like "(20)", match the token
            if ind.startswith("(") and token_text == ind:
                matched_indicator = ind
                break
        
        # Fallback: normalized numeric matching (handles "2.09" == "2.9")
        if not matched_indicator:
            # Strip parentheses/brackets from token for normalization
            stripped = token_text.strip('()[] ')
            norm_token = normalize_indicator(stripped)
            if norm_token in norm_to_original:
                matched_indicator = norm_to_original[norm_token]
                print(f"[DEBUG] Normalized match: token '{token_text}' -> '{norm_token}' matched indicator '{matched_indicator}'")
        
        if matched_indicator:
            # Skip if in legend area (we already have it as definition)
            if is_in_legend_area(token):
                print(f"[DEBUG] Skipping '{matched_indicator}' at ({token['x0']:.1f}, {token['y0']:.1f}) - in legend area")
                continue
            
            # Proximity-based dedup: skip if we already found the same indicator
            # within 30 PDF units (native + OCR often see the same callout at
            # slightly different coordinates)
            tx, ty = (token['x0'] + token['x1']) / 2, (token['y0'] + token['y1']) / 2
            is_dup = False
            for prev_ind, prev_x, prev_y in found_positions:
                if prev_ind == matched_indicator and abs(tx - prev_x) < 30 and abs(ty - prev_y) < 30:
                    is_dup = True
                    break
            if is_dup:
                continue
            found_positions.add((matched_indicator, tx, ty))
            
            instances.append({
                "indicator": matched_indicator,
                "x0": token["x0"], "y0": token["y0"], 
                "x1": token["x1"], "y1": token["y1"],
                "source": "figure_instance"
            })
            print(f"[DEBUG] ✓ Found instance '{matched_indicator}' at ({token['x0']:.1f}, {token['y0']:.1f}) token='{token_text}'")
    
    print(f"[DEBUG] Total instances found: {len(instances)}")
    return instances


def find_instances_in_figures_fast(legend_entries: List[Dict], figure_chunks: List[Dict], all_tokens: List[Dict], ocr_lines: List[Dict] = None) -> List[Dict]:
    """Performance variant of find_instances_in_figures.

    Semantically identical — matches and ordering are bit-equivalent to the
    original. The only difference is that the "which tokens are inside
    which figure chunks?" step uses a single numpy pass instead of
    O(chunks × tokens) Python-level comparisons.

    On a page with ~500 tokens and ~100 figure chunks the prefilter alone
    is 10-50x faster; overall speedup depends on how many tokens survive
    the prefilter.
    """
    import numpy as np

    print("[DEBUG-FAST] Finding Instances in Figure Chunks (numpy prefilter)...")
    print(f"[DEBUG-FAST] Figure chunks count: {len(figure_chunks)}")

    # --- OCR merge: identical to the original ---
    if ocr_lines:
        existing_positions = set()
        for t in all_tokens:
            existing_positions.add((round(t['x0']), round(t['y0']), t.get('text', '')))

        ocr_added = 0
        for ocr_line in ocr_lines:
            text = ocr_line.get('text', '').strip()
            if not text:
                continue
            words = text.split()
            if len(words) <= 1:
                pos_key = (round(ocr_line.get('x0', 0)), round(ocr_line.get('y0', 0)), text)
                if pos_key not in existing_positions:
                    all_tokens.append({
                        'text': text,
                        'x0': ocr_line.get('x0', 0),
                        'y0': ocr_line.get('y0', 0),
                        'x1': ocr_line.get('x1', 0),
                        'y1': ocr_line.get('y1', 0),
                    })
                    existing_positions.add(pos_key)
                    ocr_added += 1
            else:
                total_len = sum(len(w) for w in words)
                x0 = ocr_line.get('x0', 0)
                x1 = ocr_line.get('x1', 0)
                y0 = ocr_line.get('y0', 0)
                y1 = ocr_line.get('y1', 0)
                span = x1 - x0
                cursor = x0
                for w in words:
                    w_span = (len(w) / max(total_len, 1)) * span
                    pos_key = (round(cursor), round(y0), w)
                    if pos_key not in existing_positions:
                        all_tokens.append({
                            'text': w,
                            'x0': cursor,
                            'y0': y0,
                            'x1': cursor + w_span,
                            'y1': y1,
                        })
                        existing_positions.add(pos_key)
                        ocr_added += 1
                    cursor += w_span
        print(f"[DEBUG-FAST] Merged {ocr_added} OCR tokens into {len(all_tokens)} total tokens (deduped)")

    instances = []

    def normalize_indicator(s):
        parts = s.split('.')
        normalized_parts = []
        for p in parts:
            if p.lstrip('0').isdigit():
                normalized_parts.append(p.lstrip('0') or '0')
            elif p.isdigit() or (p and p[0].isdigit()):
                normalized_parts.append(p.lstrip('0') or '0')
            else:
                normalized_parts.append(p)
        return '.'.join(normalized_parts)

    indicators_to_find = set()
    norm_to_original = {}
    for item in legend_entries:
        ind = item.get("indicator", "").strip()
        if ind:
            indicators_to_find.add(ind)
            clean_ind = re.sub(r'[^\w]', '', ind)
            if clean_ind and clean_ind != ind and not clean_ind.isdigit():
                indicators_to_find.add(clean_ind)
            norm_key = normalize_indicator(ind)
            norm_to_original.setdefault(norm_key, ind)
            if clean_ind and clean_ind != ind and not clean_ind.isdigit():
                norm_clean = normalize_indicator(clean_ind)
                norm_to_original.setdefault(norm_clean, ind)

    if not indicators_to_find or not figure_chunks:
        return []

    legend_bboxes = []
    for entry in legend_entries:
        if entry.get("extraction_pass", "legend") != "legend":
            continue
        if all(k in entry for k in ['x0', 'y0', 'x1', 'y1']):
            legend_bboxes.append((entry['x0'], entry['y0'], entry['x1'], entry['y1']))

    def is_in_legend_area(token):
        margin = 20
        tx, ty = (token['x0'] + token['x1']) / 2, (token['y0'] + token['y1']) / 2
        for lx0, ly0, lx1, ly1 in legend_bboxes:
            if lx0 - margin <= tx <= lx1 + margin and ly0 - margin <= ty <= ly1 + margin:
                return True
        return False

    # --- FAST bbox prefilter: numpy instead of nested Python loop ---
    # Vectorize token centers once; iterate chunks and accumulate per-chunk
    # hits in the SAME order as the original (chunks outer, tokens inner)
    # so downstream behavior (found_positions proximity dedup) is
    # bit-identical.
    if not all_tokens:
        return []
    tok_cx = np.empty(len(all_tokens), dtype=np.float64)
    tok_cy = np.empty(len(all_tokens), dtype=np.float64)
    for _i, _t in enumerate(all_tokens):
        tok_cx[_i] = (_t['x0'] + _t['x1']) / 2.0
        tok_cy[_i] = (_t['y0'] + _t['y1']) / 2.0

    figure_tokens = []
    for chunk in figure_chunks:
        cx0, cy0, cx1, cy1 = chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]
        mask = (tok_cx >= cx0) & (tok_cx <= cx1) & (tok_cy >= cy0) & (tok_cy <= cy1)
        # np.nonzero returns sorted indices → preserves original token order
        for _i in np.nonzero(mask)[0]:
            figure_tokens.append(all_tokens[int(_i)])

    print(f"[DEBUG-FAST] Tokens inside figure chunks: {len(figure_tokens)} (out of {len(all_tokens)} total)")

    # --- Match loop: identical to the original ---
    found_positions = set()
    for token in figure_tokens:
        token_text = token.get("text", "").strip()
        if not token_text:
            continue

        matched_indicator = None
        for ind in indicators_to_find:
            if token_text == ind:
                matched_indicator = ind
                break
            if token_text == f"({ind})" or token_text == f"[{ind}]":
                matched_indicator = ind
                break
            if ind.startswith("(") and token_text == ind:
                matched_indicator = ind
                break

        if not matched_indicator:
            stripped = token_text.strip('()[] ')
            norm_token = normalize_indicator(stripped)
            if norm_token in norm_to_original:
                matched_indicator = norm_to_original[norm_token]

        if matched_indicator:
            if is_in_legend_area(token):
                continue
            tx, ty = (token['x0'] + token['x1']) / 2, (token['y0'] + token['y1']) / 2
            is_dup = False
            for prev_ind, prev_x, prev_y in found_positions:
                if prev_ind == matched_indicator and abs(tx - prev_x) < 30 and abs(ty - prev_y) < 30:
                    is_dup = True
                    break
            if is_dup:
                continue
            found_positions.add((matched_indicator, tx, ty))
            instances.append({
                "indicator": matched_indicator,
                "x0": token["x0"], "y0": token["y0"],
                "x1": token["x1"], "y1": token["y1"],
                "source": "figure_instance",
            })

    print(f"[DEBUG-FAST] Total instances found: {len(instances)}")
    return instances
