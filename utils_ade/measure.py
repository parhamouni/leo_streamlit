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
from .llm import (
    llm_identify_fence_layers,
    llm_match_layers_to_definitions,
    llm_suggest_filter_params,
)

def detect_dimension_lines(
    page: fitz.Page,
    scale_factor: float = 30.0,
    search_radius: float = 150.0,
    min_length_ft: float = 2.0,
    max_length_ft: float = 500.0
) -> Dict:
    """
    Detect fence measurements by finding numeric dimension text and matching to nearby lines.
    
    This method looks for dimension-style annotations (numbers with units like "45'" or "120 LF")
    and finds the vector line whose length best matches the annotated measurement.
    
    Args:
        page: PDF page object
        scale_factor: Drawing scale (e.g., 30 means 1" = 30')
        search_radius: Max distance (pts) to search for matching lines
        min_length_ft: Minimum fence length to consider (feet)
        max_length_ft: Maximum fence length to consider (feet)
    
    Returns:
        Dictionary with detected dimension lines and measurements
    """
    if not VECTOR_UTILS_AVAILABLE:
        return {"success": False, "error": "Vector utilities not available", "measurements": []}
    
    print("[DEBUG] Starting dimension line detection...")
    
    # Extract text and find numeric measurements
    text_dict = page.get_text('dict')
    measurements = []
    
    for block in text_dict.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip()
                bbox = span['bbox']
                
                # Look for patterns like: "45", "45'", "45'-0\"", "45 LF", "120'-6\""
                match = re.match(r'^(\d+\.?\d*)\s*[\'\"\-LF]*', text)
                if match:
                    try:
                        num = float(match.group(1))
                        # Filter to reasonable fence lengths
                        if min_length_ft <= num <= max_length_ft:
                            measurements.append({
                                'value_ft': num,
                                'text': text,
                                'bbox': bbox,
                                'center': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                            })
                    except:
                        pass
    
    print(f"[DEBUG] Found {len(measurements)} potential dimension texts")
    
    # Extract all lines from page
    all_lines = extract_vector_lines(page)
    # Filter to minimum length
    all_lines = [l for l in all_lines if l.length_pts >= 5.0]
    print(f"[DEBUG] Extracted {len(all_lines)} vector lines (min 5 pts)")
    
    matched_dimensions = []
    matched_lines = []
    
    for meas in measurements:
        px, py = meas['center']
        # Convert expected length from feet to points
        expected_pts = meas['value_ft'] * 12.0 / scale_factor * 72.0
        
        best_line = None
        best_score = float('inf')
        
        for line in all_lines:
            sx, sy = line.start
            ex, ey = line.end
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            
            # Distance from measurement text to line (any point)
            dist = min(
                ((sx-px)**2 + (sy-py)**2)**0.5,
                ((ex-px)**2 + (ey-py)**2)**0.5,
                ((mx-px)**2 + (my-py)**2)**0.5
            )
            
            if dist > search_radius:
                continue
            
            # Score: prioritize length match, penalize distance
            length_diff_pct = abs(line.length_pts - expected_pts) / expected_pts if expected_pts > 0 else 1
            score = length_diff_pct + (dist / search_radius) * 0.3
            
            if score < best_score:
                best_score = score
                best_line = line
        
        # Only include good matches (score < 1.0 means length match is within 70%)
        if best_line and best_score < 1.0:
            actual_ft = (best_line.length_pts / 72.0) * scale_factor / 12.0
            matched_dimensions.append({
                'expected_ft': meas['value_ft'],
                'actual_ft': actual_ft,
                'measurement_text': meas['text'],
                'text_bbox': meas['bbox'],
                'line_start': best_line.start,
                'line_end': best_line.end,
                'match_score': best_score,
                'error_pct': abs(actual_ft - meas['value_ft']) / meas['value_ft'] * 100 if meas['value_ft'] > 0 else 0
            })
            matched_lines.append(best_line)
    
    # Calculate totals
    total_expected_ft = sum(m['expected_ft'] for m in matched_dimensions)
    total_actual_ft = sum(m['actual_ft'] for m in matched_dimensions)
    
    print(f"[DEBUG] Matched {len(matched_dimensions)} dimension lines")
    print(f"[DEBUG] Total: expected={total_expected_ft:.0f}ft, actual={total_actual_ft:.0f}ft")
    
    return {
        'success': True,
        'method': 'dimension_line',
        'measurements': matched_dimensions,
        'matched_lines': matched_lines,
        'total_expected_ft': total_expected_ft,
        'total_actual_ft': total_actual_ft,
        'dimension_count': len(matched_dimensions)
    }


def measure_fence_elements(
    page: fitz.Page,
    fence_definitions: List[Dict],
    fence_instances: List[Dict],
    figure_chunks: List[Dict] = None,
    llm=None,
    scale_factor: Optional[float] = None,
    ocr_text: str = None,
    light_llm=None,  # optional: cheaper/faster model for layer-name matching
    # Additive kwarg: when False, skip the expensive LLM-guided fallback
    # path on pages that have no detectable fence layers. Default True
    # preserves the existing prod behaviour (legacy app filtered the
    # results post-hoc on the UI side rather than at the pipeline level).
    enable_nonlayer_suggestions: bool = True,
) -> Dict:
    """
    Measure fence-related vector elements based on detected indicators.
    
    This function:
    1. Extracts all layer names from the page
    2. Uses LLM to identify fence-related layers
    3. Extracts vector lines from those layers (constrained to figure areas)
    4. Matches lines to detected indicator instances
    5. Calculates measurements
    
    Args:
        page: PDF page object
        fence_definitions: Detected fence definitions from legend
        fence_instances: Detected fence instances in figures
        figure_chunks: ADE-detected figure/drawing areas for boundary constraint
        llm: Language model for layer identification (optional)
        scale_factor: Drawing scale override (if None, auto-inferred)
    
    Returns:
        Dictionary with measurement results
    """
    if not VECTOR_UTILS_AVAILABLE:
        return {"error": "Vector utilities not available", "measurements": {}}
    
    print("[DEBUG] Starting fence element measurement...")
    
    # Get page info
    page_width = page.rect.width
    page_height = page.rect.height
    rotation = page.rotation

    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, default)))
        except (TypeError, ValueError):
            return default

    def _skip_measurement(reason: str, *, layer_to_category: Optional[Dict] = None) -> Dict:
        print(f"[DEBUG] Skipping measurement: {reason}")
        return {
            "measurements": {},
            "all_fence_lines": [],
            "layer_measurements": {},
            "layer_to_category": layer_to_category or {},
            "measurement_method": "skipped",
            "page_info": {
                "width_pts": page_width,
                "height_pts": page_height,
                "rotation": rotation,
                "scale_factor": scale_factor,
            },
            "skip_reason": reason,
        }

    # Measurement is optional enrichment. If ADE found definitions but no
    # figure instances, broad vector-layer scans are both less useful and the
    # most common source of Phase 3 timeouts on dense plan sheets.
    require_instances = os.environ.get("FENCE_MEASURE_REQUIRE_INSTANCES", "true").lower() == "true"
    if require_instances and fence_definitions and not fence_instances:
        return _skip_measurement(
            "No detected figure instances; skipping automatic vector measurement"
        )

    max_figure_chunks = _env_int("FENCE_MEASURE_MAX_FIGURE_CHUNKS", 12)
    if figure_chunks and max_figure_chunks and len(figure_chunks) > max_figure_chunks:
        return _skip_measurement(
            f"Page has {len(figure_chunks)} figure chunks (limit: {max_figure_chunks})"
        )
    
    # Auto-infer scale if not provided
    if scale_factor is None:
        # 1) Try regex on embedded text / OCR text
        scale_factor = infer_scale_from_page(page, ocr_text=ocr_text)
        if scale_factor:
            print(f"[DEBUG] Auto-detected scale factor (regex): {scale_factor}")
        # 2) Fall back to vision model (GPT-4V analyzes page image)
        if scale_factor is None and llm is not None:
            print("[DEBUG] Regex scale detection failed, trying vision model...")
            try:
                vision_result = detect_scale_with_vision(page, llm)
                if vision_result.get('success') and vision_result.get('verified_scale'):
                    scale_factor = vision_result['verified_scale']
                    print(f"[DEBUG] Auto-detected scale factor (vision): {scale_factor}")
                else:
                    print(f"[DEBUG] Vision scale detection failed: {vision_result.get('message', 'unknown')}")
            except Exception as e:
                print(f"[DEBUG] Vision scale detection error: {e}")
        if scale_factor is None:
            scale_factor = 1.0
            print("[DEBUG] Could not detect scale, using 1.0")
    layer_names = extract_layer_names(page)
    print(f"[DEBUG] Found {len(layer_names)} layers on page")

    max_layers = _env_int("FENCE_MEASURE_MAX_LAYERS", 200)
    if max_layers and len(layer_names) > max_layers:
        return _skip_measurement(
            f"Page has {len(layer_names)} PDF layers (limit: {max_layers})"
        )
    
    # Use LLM to identify fence-related layers. These are simple
    # "does this layer name sound like fencing?" tasks — route them to
    # the lighter model if caller provided one.
    _light = light_llm or llm
    fence_layers = []
    if _light:
        fence_layers = llm_identify_fence_layers(_light, layer_names, fence_definitions)

    # If LLM didn't find layers, fall back to keyword matching
    if not fence_layers:
        print("[DEBUG] Falling back to keyword-based layer detection")
        fence_keywords = ['FENC', 'WALL', 'BARRIER', 'GUARD', 'RAIL', 'GATE', 'PERIM', 'BNDY', 'BOUND']
        for layer in layer_names:
            layer_upper = layer.upper()
            if any(kw in layer_upper for kw in fence_keywords):
                fence_layers.append(layer)
        print(f"[DEBUG] Keyword-matched fence layers: {fence_layers}")

    # Match fence layers to definition categories — also routed to light.
    layer_to_category = {}
    if fence_layers and _light and fence_definitions:
        layer_to_category = llm_match_layers_to_definitions(_light, fence_layers, fence_definitions)

    # Short-circuit: if the user disabled non-layer suggestions and we
    # found no fence layers on this page, skip the LLM-guided fallback
    # entirely. That fallback is the slow path — extracts every vector
    # line, runs filter_params LLM, asks LLM to pick fence-like lines.
    # On pages with no real fence content (annotation/title-block pages
    # the classifier flagged on keyword hits), this saves minutes per page.
    if not fence_layers and not enable_nonlayer_suggestions:
        return _skip_measurement(
            "non-layer suggestions disabled and no fence layers found",
            layer_to_category=layer_to_category,
        )

    # Extract lines from fence-related layers
    if fence_layers:
        fence_lines = extract_lines_by_layers(page, fence_layers)
    else:
        fence_lines = []
    print(f"[DEBUG] Extracted {len(fence_lines)} lines from fence layers")

    # Skip measurement for pages with extremely dense vector content
    # (e.g. complex site plans with thousands of line segments per layer)
    MAX_FENCE_LINES = 5000
    if len(fence_lines) > MAX_FENCE_LINES:
        print(f"[DEBUG] Skipping measurement: {len(fence_lines)} lines exceeds limit of {MAX_FENCE_LINES}")
        return {
            "measurements": {},
            "all_fence_lines": fence_lines,
            "layer_measurements": {},
            "layer_to_category": layer_to_category,
            "measurement_method": "skipped",
            "page_info": {
                "width_pts": page_width,
                "height_pts": page_height,
                "rotation": rotation,
                "scale_factor": scale_factor,
            },
            "skip_reason": f"Page has {len(fence_lines)} fence-layer lines (limit: {MAX_FENCE_LINES})",
        }

    # Group lines by layer for per-layer measurements
    lines_by_layer = group_lines_by_layer(fence_lines)
    
    # Calculate measurements per layer
    layer_measurements = {}
    for layer, lines in lines_by_layer.items():
        # Group connected lines
        connected_groups = group_connected_lines(lines)
        
        # Calculate total measurement
        total = calculate_total_length(lines, scale_factor)
        
        # Calculate per-group measurements
        group_measurements = []
        for group in connected_groups:
            group_total = calculate_total_length(group, scale_factor)
            group_measurements.append({
                'segment_count': group_total['segment_count'],
                'length_feet': round(group_total['total_feet'], 2)
            })
        
        # Sort groups by length (largest first)
        group_measurements.sort(key=lambda x: x['length_feet'], reverse=True)
        
        layer_measurements[layer] = {
            'total_segments': total['segment_count'],
            'total_length_feet': round(total['total_feet'], 2),
            'connected_runs': len(connected_groups),
            'runs': group_measurements[:10]  # Top 10 runs
        }
    
    # =========================================================================
    # FIGURE-CONSTRAINED MEASUREMENT:
    # Only measure lines that are INSIDE ADE-detected figure/drawing areas
    # =========================================================================
    
    # Get figure bounding boxes from ADE figure_chunks (the actual drawing areas)
    figure_bboxes = []
    if figure_chunks:
        for chunk in figure_chunks:
            x0 = chunk.get("x0", 0)
            y0 = chunk.get("y0", 0)
            x1 = chunk.get("x1", 0)
            y1 = chunk.get("y1", 0)
            # Only use chunks that are large enough to be actual drawing areas
            if x1 - x0 > 100 and y1 - y0 > 100:
                figure_bboxes.append((x0, y0, x1, y1))
    
    print(f"[DEBUG] Found {len(figure_bboxes)} figure/drawing area bounding boxes")
    
    # Filter fence_lines to only those inside figure bboxes
    def line_in_any_bbox(line, bboxes, margin=50.0):
        """Check if a line is inside or near any bounding box."""
        for (x0, y0, x1, y1) in bboxes:
            # Expand bbox by margin
            x0m, y0m = x0 - margin, y0 - margin
            x1m, y1m = x1 + margin, y1 + margin
            # Check if either endpoint is inside
            sx, sy = line.start
            ex, ey = line.end
            if (x0m <= sx <= x1m and y0m <= sy <= y1m) or \
               (x0m <= ex <= x1m and y0m <= ey <= y1m):
                return True
        return False
    
    # If we have figure bboxes, filter fence_lines
    if figure_bboxes and fence_lines:
        filtered_fence_lines = [l for l in fence_lines if line_in_any_bbox(l, figure_bboxes)]
        print(f"[DEBUG] Filtered to {len(filtered_fence_lines)} lines inside figures (from {len(fence_lines)})")
    else:
        filtered_fence_lines = fence_lines
    
    # =========================================================================
    # LAYER-FIRST APPROACH:
    # Use layer-based lines if we found any, otherwise fall back to proximity
    # =========================================================================
    
    indicator_measurements = {}
    final_fence_lines = []
    measurement_method = "none"
    
    if filtered_fence_lines:
        # PRIMARY: Use layer-based lines (already filtered to figures)
        measurement_method = "layer"
        final_fence_lines = filtered_fence_lines
        
        # Calculate per-indicator measurements by proximity within filtered lines
        for item in list(fence_instances) + list(fence_definitions):
            ind = item.get("indicator", "") or item.get("keyword", "")
            if not ind:
                continue
            
            bbox = (item.get("x0", 0), item.get("y0", 0), 
                    item.get("x1", 0), item.get("y1", 0))
            
            if bbox[2] - bbox[0] < 1:
                continue
            
            # Find lines near this indicator (within filtered lines)
            nearby = find_lines_near_bbox(filtered_fence_lines, bbox, margin=80.0)
            
            if nearby:
                total = calculate_total_length(nearby, scale_factor)
                if ind not in indicator_measurements:
                    indicator_measurements[ind] = {
                        'instance_count': 0,
                        'run_segment_count': 0,
                        'run_length_feet': 0.0,
                        'run_length_pts': 0.0
                    }
                indicator_measurements[ind]['instance_count'] += 1
                indicator_measurements[ind]['run_segment_count'] += total['segment_count']
                indicator_measurements[ind]['run_length_feet'] += total['total_feet']
                indicator_measurements[ind]['run_length_pts'] += total['total_pts']
        
        print(f"[DEBUG] Layer-based measurement: {len(final_fence_lines)} lines")
    
    else:
        # ALTERNATIVE: For layerless PDFs, use LLM-guided filtering
        measurement_method = "llm_guided"
        print(f"[DEBUG] No fence layers found - using LLM-guided filtering")
        
        all_page_lines = extract_vector_lines(page)
        max_all_page_lines = _env_int("FENCE_MEASURE_MAX_ALL_PAGE_LINES", 15000)
        if max_all_page_lines and len(all_page_lines) > max_all_page_lines:
            return _skip_measurement(
                f"Page has {len(all_page_lines)} vector lines (limit: {max_all_page_lines})"
            )
        
        # Compute line statistics for LLM
        total_lines = len(all_page_lines)
        under_10 = sum(1 for l in all_page_lines if l.length_pts < 10)
        range_10_50 = sum(1 for l in all_page_lines if 10 <= l.length_pts < 50)
        range_50_100 = sum(1 for l in all_page_lines if 50 <= l.length_pts < 100)
        over_100 = sum(1 for l in all_page_lines if l.length_pts >= 100)
        
        line_stats = {
            'total': total_lines,
            'under_10': under_10,
            'range_10_50': range_10_50,
            'range_50_100': range_50_100,
            'over_100': over_100,
            'pct_under_10': (under_10 / total_lines * 100) if total_lines > 0 else 0,
            'pct_10_50': (range_10_50 / total_lines * 100) if total_lines > 0 else 0,
            'pct_50_100': (range_50_100 / total_lines * 100) if total_lines > 0 else 0,
            'pct_over_100': (over_100 / total_lines * 100) if total_lines > 0 else 0,
            'layers': 'None detected',
            'indicators': [d.get('indicator', '') for d in fence_definitions][:5]
        }
        
        # Get LLM-suggested parameters
        filter_params = llm_suggest_filter_params(llm, line_stats)
        MIN_LINE_LENGTH = filter_params['min_length']
        PROXIMITY_MARGIN = filter_params['proximity_margin']
        
        print(f"[DEBUG] LLM params: min_length={MIN_LINE_LENGTH}, margin={PROXIMITY_MARGIN}")
        
        # Filter 1: Apply LLM-suggested length filter
        candidate_lines = [l for l in all_page_lines if l.length_pts > MIN_LINE_LENGTH]
        print(f"[DEBUG] Lines > {MIN_LINE_LENGTH} pts: {len(candidate_lines)} (from {len(all_page_lines)})")
        
        # Filter 2: Apply figure bbox constraint
        if figure_bboxes:
            candidate_lines = [l for l in candidate_lines if line_in_any_bbox(l, figure_bboxes)]
            print(f"[DEBUG] After figure constraint: {len(candidate_lines)}")
        
        # Find lines near each indicator using LLM-suggested proximity
        seen_line_ids = set()
        
        for item in list(fence_instances) + list(fence_definitions):
            ind = item.get("indicator", "") or item.get("keyword", "")
            if not ind:
                continue
            
            bbox = (item.get("x0", 0), item.get("y0", 0), 
                    item.get("x1", 0), item.get("y1", 0))
            
            if bbox[2] - bbox[0] < 1:
                continue
            
            # Find lines near this indicator using LLM-suggested margin
            nearby = find_lines_near_bbox(candidate_lines, bbox, margin=PROXIMITY_MARGIN)
            
            if nearby:
                new_lines = [l for l in nearby if id(l) not in seen_line_ids]
                final_fence_lines.extend(new_lines)
                for l in new_lines:
                    seen_line_ids.add(id(l))
                
                total = calculate_total_length(nearby, scale_factor)
                if ind not in indicator_measurements:
                    indicator_measurements[ind] = {
                        'instance_count': 0,
                        'run_segment_count': 0,
                        'run_length_feet': 0.0,
                        'run_length_pts': 0.0
                    }
                indicator_measurements[ind]['instance_count'] += 1
                indicator_measurements[ind]['run_segment_count'] += total['segment_count']
                indicator_measurements[ind]['run_length_feet'] += total['total_feet']
                indicator_measurements[ind]['run_length_pts'] += total['total_pts']
        
        print(f"[DEBUG] Length-filtered measurement: {len(final_fence_lines)} lines")
    
    # Round indicator measurements
    for ind in indicator_measurements:
        indicator_measurements[ind]['run_length_feet'] = round(
            indicator_measurements[ind]['run_length_feet'], 2
        )
        indicator_measurements[ind]['run_length_pts'] = round(
            indicator_measurements[ind]['run_length_pts'], 1
        )
    
    # Calculate totals
    grand_total_segments = sum(m['total_segments'] for m in layer_measurements.values())
    grand_total_feet = sum(m['total_length_feet'] for m in layer_measurements.values())
    
    final_total = calculate_total_length(final_fence_lines, scale_factor) if final_fence_lines else {}
    
    # Run dimension line detection (find measurement text and match to lines)
    dimension_result = detect_dimension_lines(page, scale_factor)
    
    # Add dimension lines to final fence lines if found
    if dimension_result.get('success') and dimension_result.get('matched_lines'):
        dim_lines = dimension_result['matched_lines']
        existing_endpoints = set()
        for line in final_fence_lines:
            existing_endpoints.add((round(line.start[0], 1), round(line.start[1], 1), 
                                   round(line.end[0], 1), round(line.end[1], 1)))
        
        new_dim_lines = []
        for line in dim_lines:
            key = (round(line.start[0], 1), round(line.start[1], 1),
                   round(line.end[0], 1), round(line.end[1], 1))
            if key not in existing_endpoints:
                new_dim_lines.append(line)
                existing_endpoints.add(key)
        
        if new_dim_lines:
            final_fence_lines.extend(new_dim_lines)
            final_total = calculate_total_length(final_fence_lines, scale_factor)
    
    result = {
        'page_info': {
            'width': page_width,
            'height': page_height,
            'rotation': rotation,
            'scale_factor': scale_factor,
            'scale_detected': scale_factor != 1.0
        },
        'measurement_method': measurement_method,
        'fence_layers': fence_layers,
        'layer_to_category': layer_to_category,
        'all_fence_lines': final_fence_lines,
        'layer_measurements': layer_measurements,
        'indicator_measurements': indicator_measurements,
        'dimension_measurements': dimension_result.get('measurements', []),
        'proximity_totals': {
            'total_segments': final_total.get('segment_count', 0),
            'total_length_feet': round(final_total.get('total_feet', 0), 2),
            'total_length_pts': round(final_total.get('total_pts', 0), 1)
        },
        'totals': {
            'total_layers': len(layer_measurements),
            'total_segments': grand_total_segments,
            'total_length_feet': round(grand_total_feet, 2)
        }
    }
    
    print(f"[DEBUG] Measurement complete ({measurement_method}): {final_total.get('total_feet', 0):.1f} ft from {len(final_fence_lines)} lines")
    return result
