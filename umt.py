"""Unified Measurement Tool (UMT).

Per-page interactive canvas for selecting auto-detected vector fence
lines, drawing custom lines, assigning categories, and exporting
measurements to PDF / Excel.

Adapted from the monolithic Streamlit app for the API-backed
architecture: no shared session-wide PDF caches; the original PDF path
and pre-computed analysis results are passed in by the caller.

Public API: ``render_umt(...)`` (see signature below).
"""
from __future__ import annotations

import gc
import math
import os
from io import BytesIO

import fitz
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from utils_vector import extract_vector_lines, verify_scale_with_bar_fast
from exports import generate_measurement_pdf, generate_measurement_spreadsheet

try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore
    _HAS_IMG_COORDS = True
except Exception:  # pragma: no cover - optional dep
    streamlit_image_coordinates = None  # type: ignore
    _HAS_IMG_COORDS = False


CATEGORY_COLORS = [
    (0, 255, 0),      # Green
    (255, 165, 0),    # Orange
    (0, 191, 255),    # Deep sky blue
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 105, 180),  # Hot pink
    (173, 255, 47),   # Green yellow
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pg_idx(page_data):
    """Pick a page index from API or monolith-shaped dicts."""
    return page_data.get("page_idx",
                         page_data.get("page_index_in_original_doc", 0))


def _pg_num(page_data):
    return page_data.get("page_num",
                         page_data.get("page_number",
                                       _pg_idx(page_data) + 1))


def _pg_dims(page_data):
    w = page_data.get("width", page_data.get("pdf_width", 612))
    h = page_data.get("height", page_data.get("pdf_height", 792))
    return float(w), float(h)


def _scale_lookup(per_page_scale_info, page_num):
    """Look up scale info from results. Tolerates page-key OR str-index keys."""
    if not isinstance(per_page_scale_info, dict):
        return {}
    candidates = [
        f"page_{page_num}",
        str(page_num),
        str(int(page_num) - 1),
    ]
    for c in candidates:
        v = per_page_scale_info.get(c)
        if v:
            return v
    return {}


def _meas_lookup(unified_measurements, page_num):
    if not isinstance(unified_measurements, dict):
        return {}
    candidates = [
        f"page_{page_num}",
        str(page_num),
        str(int(page_num) - 1),
    ]
    for c in candidates:
        v = unified_measurements.get(c)
        if v:
            return v
    return {}


def _get_fitz_doc(pdf_path, job_id):
    """Lazy fitz.open with per-job session cache. Returns None on failure."""
    cache_key = f"_umt_fitz_doc_{job_id or 'default'}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        try:
            # Cheap sanity probe
            _ = cached.page_count
            return cached
        except Exception:
            try:
                cached.close()
            except Exception:
                pass
            st.session_state.pop(cache_key, None)
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        doc = fitz.open(pdf_path)
        st.session_state[cache_key] = doc
        return doc
    except Exception as e:
        print(f"[UMT] fitz.open failed: {e}")
        return None


def _ensure_per_page_scale(per_page_scale_state, pdf_path, page_num,
                           page_idx, job_id):
    """Resolve scale info for a page. Prefers cached results; falls back
    to verify_scale_with_bar_fast(llm=None) on demand. Avoids LLM calls.
    """
    cache_key = f"page_{page_num}"
    if cache_key in per_page_scale_state and per_page_scale_state[cache_key]:
        return per_page_scale_state[cache_key]

    scale_info = {}
    doc = _get_fitz_doc(pdf_path, job_id)
    if doc is not None and 0 <= page_idx < doc.page_count:
        try:
            scale_info = verify_scale_with_bar_fast(doc[page_idx], llm=None) or {}
        except Exception as e:
            scale_info = {
                "success": False,
                "verified_scale": None,
                "message": str(e),
            }
    else:
        scale_info = {
            "success": False,
            "verified_scale": None,
            "message": "PDF unavailable",
        }
    per_page_scale_state[cache_key] = scale_info
    return scale_info


def _evict_other_umt_pages(active_pg_num):
    """Purge heavy per-page state for every page except active_pg_num.
    Mirrors the monolith helper. User edits (line_assignments,
    user_drawn_lines, page_categories) are *not* touched.
    """
    heavy_prefixes = (
        "base_img_", "base_img_size_", "drawn_img_", "orig_img_size_",
        "line_stats_", "lines_", "auto_synced_", "auto_matched_indices_",
        "click_key_",
    )
    purged = 0
    for k in list(st.session_state.keys()):
        if not any(k.startswith(p) for p in heavy_prefixes):
            continue
        # These keys embed the page number, e.g. base_img_3_1200,
        # drawn_img_5_1200_<hash>, line_stats_2_30_360.0. Keep entries
        # whose name has _{active}_ inside or ends with _{active}.
        if (f"_{active_pg_num}_" in k) or k.endswith(f"_{active_pg_num}"):
            continue
        try:
            del st.session_state[k]
            purged += 1
        except Exception:
            pass
    if purged:
        gc.collect()
        try:
            import ctypes as _ct
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _render_base_image(pdf_path, page_idx, dpi):
    """Render a page to PNG bytes. Returns None on any failure."""
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        with fitz.open(pdf_path) as doc:
            if not (0 <= page_idx < doc.page_count):
                return None
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            data = pix.tobytes("png")
            del pix
            return data
    except Exception as e:
        print(f"[UMT] _render_base_image page {page_idx}: {e}")
        return None


def _extract_lines_for_page(pdf_path, page_idx, job_id):
    """Extract VectorLine list for a page using shared cached fitz doc."""
    doc = _get_fitz_doc(pdf_path, job_id)
    if doc is None:
        return []
    if not (0 <= page_idx < doc.page_count):
        return []
    try:
        return extract_vector_lines(doc[page_idx])
    except Exception as e:
        print(f"[UMT] vector lines page {page_idx}: {e}")
        return []


def _match_auto_lines(auto_lines, accepted_auto, vector_lines, page_key):
    """Pre-populate line_assignments by matching auto-detected lines to
    selectable vector lines by endpoint distance. Returns (matched_count,
    matched_indices_set)."""
    matched = 0
    matched_indices = set()
    if not (auto_lines and accepted_auto and vector_lines):
        return matched, matched_indices
    for ai in accepted_auto:
        if ai >= len(auto_lines):
            continue
        auto_line = auto_lines[ai]
        a_sx, a_sy = auto_line["start"]
        a_ex, a_ey = auto_line["end"]
        category = auto_line.get("category")
        if not category:
            continue
        best_idx = None
        best_dist = float("inf")
        for vi, vline in enumerate(vector_lines):
            v_sx, v_sy = vline["start"]
            v_ex, v_ey = vline["end"]
            d1 = math.hypot(a_sx - v_sx, a_sy - v_sy) + math.hypot(a_ex - v_ex, a_ey - v_ey)
            d2 = math.hypot(a_sx - v_ex, a_sy - v_ey) + math.hypot(a_ex - v_sx, a_ey - v_sy)
            d = min(d1, d2)
            if d < best_dist:
                best_dist = d
                best_idx = vi
        if best_idx is not None and best_dist < 2.0:
            st.session_state.line_assignments[page_key][best_idx] = category
            matched_indices.add(best_idx)
            matched += 1
    return matched, matched_indices


def _point_to_line_distance(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
    t = max(0, min(1, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def _initial_categories_from_definitions(definitions):
    categories = {}
    for d in (definitions or []):
        indicator = d.get("indicator", "")
        keyword = d.get("keyword", "")
        if not keyword:
            # Synthesize from the chunk text for ADE-shaped definitions
            text = d.get("text", "") or d.get("markdown", "")
            keyword = (text or "").strip().split("\n", 1)[0][:60]
        if not keyword:
            continue
        cat_name = f"{indicator}: {keyword}" if indicator else keyword
        if cat_name in categories:
            continue
        color_idx = len(categories)
        categories[cat_name] = {
            "indicator": indicator,
            "keyword": keyword,
            "color": CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)],
        }
    return categories


def _render_page_panel(page_data, zoom_level, min_line_pts, pdf_path,
                      per_page_scale_info, unified_measurements,
                      enable_nonlayer, job_id):
    """Render a single page's measurement canvas + side panel + per-page
    summary. Mirrors render_page_fragment in the monolith."""
    page_num = _pg_num(page_data)
    page_key = f"page_{page_num}"
    page_idx = _pg_idx(page_data)
    pdf_width, pdf_height = _pg_dims(page_data)

    # On-demand sync of auto_lines from API-provided unified_measurements.
    # When the page's auto_lines slot is empty but the API delivered a
    # measurement result with all_fence_lines, build auto_lines so the
    # canvas shows them. Mirrors the monolith's phase3_measure re-sync.
    sync_meas = st.session_state.unified_measurements.get(page_key, {})
    api_meas = _meas_lookup(unified_measurements, page_num)
    if not sync_meas.get("auto_lines") and api_meas and api_meas.get("all_fence_lines"):
        mm = api_meas.get("measurement_method", "none")
        sync_accept = (mm == "layer") or (
            enable_nonlayer and mm in ("llm_guided", "skipped")
        )
        if sync_accept:
            l2c = api_meas.get("layer_to_category", {}) or {}
            sf = (api_meas.get("page_info", {}) or {}).get("scale_factor", 1.0) or 1.0
            fallback_cat = (
                "Auto-detected (LLM-guided)" if mm == "llm_guided"
                else "Auto-detected (high-density, layer-skipped)" if mm == "skipped"
                else None
            )
            auto_now = []
            for ln in api_meas["all_fence_lines"]:
                # Lines from the API are dicts (json-serialized), but
                # tolerate VectorLine-style attribute access too.
                if isinstance(ln, dict):
                    lyr = ln.get("layer", "") or ""
                    start = tuple(ln.get("start", (0, 0)))
                    end = tuple(ln.get("end", (0, 0)))
                    lpts = ln.get("length_pts",
                                  math.hypot(end[0] - start[0], end[1] - start[1]))
                else:
                    lyr = getattr(ln, "layer", "") or ""
                    start = ln.start
                    end = ln.end
                    lpts = ln.length_pts
                cat = l2c.get(lyr)
                if not cat and lyr:
                    for ml, c in l2c.items():
                        if ml in lyr or lyr in ml:
                            cat = c
                            break
                if not cat:
                    cat = fallback_cat
                if not cat:
                    continue
                auto_now.append({
                    "start": start,
                    "end": end,
                    "length_pts": lpts,
                    "length_feet": ((lpts / 72.0) * sf) / 12.0,
                    "layer": lyr,
                    "category": cat,
                    "source": "auto",
                    "method": mm,
                })
            if auto_now:
                if page_key not in st.session_state.unified_measurements:
                    st.session_state.unified_measurements[page_key] = {
                        "auto_lines": [], "manual_lines": [],
                        "drawn_lines": [], "accepted_auto": set(),
                    }
                st.session_state.unified_measurements[page_key]["auto_lines"] = auto_now
                st.session_state.unified_measurements[page_key]["accepted_auto"] = set(range(len(auto_now)))

    # Extract lines from PDF page (cached)
    lines_cache_key = f"lines_{page_num}_{min_line_pts}"
    if lines_cache_key not in st.session_state:
        # Evict previous min-length variants for this page
        for k in [k for k in list(st.session_state.keys())
                  if k.startswith(f"lines_{page_num}_") and k != lines_cache_key]:
            del st.session_state[k]
        all_lines = _extract_lines_for_page(pdf_path, page_idx, job_id)
        filtered_lines = [l for l in all_lines if l.length_pts >= min_line_pts]
        filtered_lines.sort(key=lambda l: l.length_pts, reverse=True)
        st.session_state[lines_cache_key] = [{
            "start": (float(l.start[0]), float(l.start[1])),
            "end": (float(l.end[0]), float(l.end[1])),
            "length_pts": float(l.length_pts),
            "layer": l.layer or "default",
        } for l in filtered_lines]

    lines = st.session_state.get(lines_cache_key, [])

    if not lines:
        st.warning(f"No lines found on this page (min length: {min_line_pts} pts)")
        return

    # Initialize per-page assignment dict
    if page_key not in st.session_state.line_assignments:
        st.session_state.line_assignments[page_key] = {}

    unified_page = st.session_state.unified_measurements.get(page_key, {})
    auto_lines_data = unified_page.get("auto_lines", [])
    accepted_auto = unified_page.get("accepted_auto", set())
    auto_sync_key = f"auto_synced_{page_key}_{len(auto_lines_data)}"

    if auto_lines_data and accepted_auto and lines and auto_sync_key not in st.session_state:
        matched, matched_indices = _match_auto_lines(
            auto_lines_data, accepted_auto, lines, page_key)
        st.session_state[f"auto_matched_indices_{page_key}"] = matched_indices
        st.session_state[auto_sync_key] = matched

    # Initialize categories for this page from its definitions
    if page_key not in st.session_state.page_categories:
        st.session_state.page_categories[page_key] = _initial_categories_from_definitions(
            page_data.get("definitions", []))

    page_categories = st.session_state.page_categories[page_key]

    if page_key not in st.session_state.active_category_per_page:
        cats = list(page_categories.keys())
        st.session_state.active_category_per_page[page_key] = cats[0] if cats else None

    # ---- Per-page scale ----
    page_scale_info = _ensure_per_page_scale(
        st.session_state.per_page_scale_info, pdf_path, page_num, page_idx, job_id)
    page_scale = (page_scale_info.get("verified_scale")
                  or page_scale_info.get("text_scale") or 360.0)
    scale_min = 1.0
    scale_max = 1200.0
    page_scale_clamped = max(scale_min, min(float(page_scale), scale_max))

    scale_col1, scale_col2 = st.columns([2, 1])
    with scale_col1:
        page_scale_input = st.number_input(
            f"Scale (Page {page_num})",
            min_value=scale_min,
            max_value=scale_max,
            value=page_scale_clamped,
            step=12.0,
            help=f"1\" = {page_scale_clamped/12:.1f}' actual",
            key=f"scale_input_{page_num}",
        )
        if float(page_scale) != page_scale_clamped:
            st.caption(
                f"Detected scale {float(page_scale):.1f} was outside allowed range "
                f"({scale_min:.0f}-{scale_max:.0f}) and was clamped."
            )
    with scale_col2:
        if page_scale_info.get("success"):
            confidence = page_scale_info.get("confidence", "low")
            scale_text = page_scale_info.get("scale_text", "")
            display_text = f"OK {scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'"
            if confidence == "high":
                st.success(display_text)
            elif confidence == "medium":
                st.warning(f"{scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'")
            else:
                st.info(scale_text if scale_text else f"1\"={page_scale/12:.0f}'")
        else:
            st.warning("Not detected")

    with st.expander("Scale Detection Details", expanded=False):
        page_size = page_scale_info.get("page_size", {})
        if page_size:
            size_str = (
                f"{page_size.get('width_inches', 0):.1f}\" x "
                f"{page_size.get('height_inches', 0):.1f}\""
            )
            detected = page_size.get("detected_size", "Unknown")
            st.markdown(f"**Page size:** {size_str} ({detected})")
        method = page_scale_info.get("method", "unknown")
        st.markdown(f"**Detection method:** {method}")
        scale_text = page_scale_info.get("scale_text", "")
        st.markdown(f"**Detected scale text:** {scale_text if scale_text else 'None'}")
        st.markdown(f"**Confidence:** {page_scale_info.get('confidence', 'N/A')}")
        st.markdown(f"**Message:** {page_scale_info.get('message', 'N/A')}")
        if page_scale_info.get("verified_scale"):
            scale_val = page_scale_info["verified_scale"]
            st.markdown(f"**Scale value:** 1\" = {scale_val/12:.0f}' ({scale_val} inches)")
        raw = page_scale_info.get("raw_response", "")
        if raw:
            st.markdown("**LLM Response:**")
            st.code(raw[:500], language=None)
        extracted = page_scale_info.get("extracted_text_sample", "")
        if extracted:
            st.markdown("**Extracted PDF Text (first 1500 chars):**")
            st.code(extracted, language=None)

    # ---- Auto-detected section ----
    unified_page_data = st.session_state.unified_measurements.get(page_key, {})
    auto_lines = unified_page_data.get("auto_lines", [])

    if auto_lines:
        accepted_auto = unified_page_data.get("accepted_auto", set())
        accepted_count = len(accepted_auto)
        accepted_ft = sum(auto_lines[i].get("length_feet", 0) for i in accepted_auto if i < len(auto_lines))

        auto_matched_key = f"auto_matched_indices_{page_key}"
        auto_matched_indices = st.session_state.get(auto_matched_key, set())
        currently_assigned = sum(
            1 for idx in auto_matched_indices
            if idx in st.session_state.line_assignments.get(page_key, {})
        )

        st.markdown("#### Auto-Detected Fence Lines (Pre-Selected)")
        auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
        with auto_col1:
            if currently_assigned > 0:
                st.success(f"{currently_assigned} lines matched & selected ({accepted_ft:.1f} ft)")
            elif accepted_count > 0:
                st.warning(f"{accepted_count} auto lines detected but not yet synced to selections")
            else:
                st.info("No auto-detected lines")
        with auto_col2:
            if st.button("Re-sync Auto", key=f"resync_auto_{page_num}",
                         help="Re-match auto-detected lines to selectable vector lines"):
                for k in [k for k in list(st.session_state.keys())
                          if k.startswith(f"auto_synced_{page_key}")]:
                    del st.session_state[k]
                st.session_state.pop(auto_matched_key, None)
                st.rerun()
        with auto_col3:
            if st.button("Clear Auto", key=f"clear_auto_{page_num}",
                         help="Remove all auto-detected lines from selection"):
                page_assigns = st.session_state.line_assignments.get(page_key, {})
                for idx in auto_matched_indices:
                    page_assigns.pop(idx, None)
                st.session_state.unified_measurements[page_key]["accepted_auto"] = set()
                st.session_state[auto_matched_key] = set()
                for k in [k for k in list(st.session_state.keys())
                          if k.startswith(f"auto_synced_{page_key}")]:
                    del st.session_state[k]
                st.rerun()

        auto_by_cat = {}
        for i in accepted_auto:
            if i < len(auto_lines):
                cat = auto_lines[i].get("category", "Uncategorized")
                if cat not in auto_by_cat:
                    auto_by_cat[cat] = {"count": 0, "feet": 0}
                auto_by_cat[cat]["count"] += 1
                auto_by_cat[cat]["feet"] += auto_lines[i].get("length_feet", 0)
        if auto_by_cat:
            for cat, data in auto_by_cat.items():
                cat_color = page_categories.get(cat, {}).get("color", (0, 255, 255))
                st.markdown(
                    f"<span style='color: rgb{cat_color};'>●</span> "
                    f"**{cat}**: {data['count']} lines, {data['feet']:.1f} ft",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ---- Category selector ----
    st.markdown("#### Fence Categories (This Page)")
    cat_col1, cat_col2 = st.columns([3, 1])
    with cat_col1:
        category_options = list(page_categories.keys())
        if category_options:
            current_active = st.session_state.active_category_per_page.get(page_key)
            active_cat = st.selectbox(
                "Assign lines to:",
                options=category_options,
                index=category_options.index(current_active) if current_active in category_options else 0,
                key=f"category_selector_{page_num}",
            )
            st.session_state.active_category_per_page[page_key] = active_cat
            if active_cat:
                color = page_categories[active_cat]["color"]
                st.markdown(
                    f"<span style='color: rgb{color}; font-size: 20px;'>●</span> "
                    "Click lines to assign",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No fence categories detected on this page.")

    with cat_col2:
        with st.popover("Add"):
            new_cat_name = st.text_input("Category name:", key=f"new_cat_{page_num}")
            if st.button("Add", key=f"add_cat_btn_{page_num}") and new_cat_name:
                if new_cat_name not in st.session_state.page_categories[page_key]:
                    color_idx = len(st.session_state.page_categories[page_key])
                    st.session_state.page_categories[page_key][new_cat_name] = {
                        "indicator": "",
                        "keyword": new_cat_name,
                        "color": CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)],
                    }
                    st.session_state.active_category_per_page[page_key] = new_cat_name
                    st.rerun()

    # ---- Mode toggle ----
    mode_col1, mode_col2 = st.columns([1, 1])
    with mode_col1:
        if page_key not in st.session_state.drawing_mode:
            st.session_state.drawing_mode[page_key] = "select"
        drawing_mode = st.radio(
            "Mode:",
            options=["select", "draw"],
            format_func=lambda x: "Select Lines" if x == "select" else "Draw Lines",
            horizontal=True,
            key=f"mode_{page_num}",
        )
        st.session_state.drawing_mode[page_key] = drawing_mode
    with mode_col2:
        if drawing_mode == "draw":
            st.caption("Draw lines on the image. They will be assigned to the active category.")

    if not _HAS_IMG_COORDS and drawing_mode == "draw":
        st.caption(
            "Note: streamlit-image-coordinates is not installed; draw mode "
            "is disabled. Install the package to enable interactive drawing."
        )

    # ---- Cache line stats keyed by (page, min_line_pts, scale) ----
    effective_scale = page_scale_input
    line_stats_key = f"line_stats_{page_num}_{min_line_pts}_{effective_scale}"
    if line_stats_key not in st.session_state:
        for k in [k for k in list(st.session_state.keys())
                  if k.startswith(f"line_stats_{page_num}_") and k != line_stats_key]:
            del st.session_state[k]
        stats = []
        for i, line in enumerate(lines):
            length_inches = line["length_pts"] / 72.0
            length_feet = (length_inches * effective_scale) / 12.0
            stats.append({
                "index": i,
                "length_pts": line["length_pts"],
                "length_feet": length_feet,
                "layer": line.get("layer") or "default",
                "start": line["start"],
                "end": line["end"],
            })
        st.session_state[line_stats_key] = stats
    line_stats = st.session_state[line_stats_key]

    # ---- Render base image ----
    base_img_bytes = _render_base_image(pdf_path, page_idx,
                                       dpi=int(150 if zoom_level >= 0 else 150))
    if not base_img_bytes:
        st.warning("Image not available")
        return

    # ---- Cache zoomed base image (WEBP-compressed) ----
    base_img_cache_key = f"base_img_{page_num}_{zoom_level}"
    if base_img_cache_key not in st.session_state:
        for k in [k for k in list(st.session_state.keys())
                  if (k.startswith(f"base_img_{page_num}_")
                      or k.startswith(f"base_img_size_{page_num}_")
                      or k.startswith(f"drawn_img_{page_num}_"))
                  and k != base_img_cache_key]:
            del st.session_state[k]
        try:
            base_img = Image.open(BytesIO(base_img_bytes)).convert("RGB")
        except Exception as e:
            st.warning(f"Image decode failed: {e}")
            return
        orig_width, orig_height = base_img.size
        ratio = zoom_level / orig_width
        new_width = zoom_level
        new_height = int(orig_height * ratio)
        base_img = base_img.resize((new_width, new_height), Image.LANCZOS)
        buf = BytesIO()
        base_img.save(buf, format="WEBP", quality=88, method=6)
        st.session_state[base_img_cache_key] = buf.getvalue()
        st.session_state[f"base_img_size_{page_num}_{zoom_level}"] = (new_width, new_height)
        st.session_state[f"orig_img_size_{page_num}"] = (orig_width, orig_height)
        del base_img, buf

    base_img_cached = Image.open(BytesIO(st.session_state[base_img_cache_key]))
    img_width, img_height = st.session_state[f"base_img_size_{page_num}_{zoom_level}"]

    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height

    line_assignments = st.session_state.line_assignments.get(page_key, {})

    assignment_tuple = tuple(sorted(line_assignments.items()))
    drawn_img_cache_key = f"drawn_img_{page_num}_{zoom_level}_{hash(assignment_tuple)}"

    if drawn_img_cache_key not in st.session_state:
        for k in [k for k in list(st.session_state.keys())
                  if k.startswith(f"drawn_img_{page_num}_") and k != drawn_img_cache_key]:
            del st.session_state[k]
        display_img = base_img_cached.copy()
        draw = ImageDraw.Draw(display_img)
        # Pass 1: unassigned hints
        for i, ls in enumerate(line_stats):
            if i in line_assignments:
                continue
            x0 = ls["start"][0] * scale_x
            y0 = ls["start"][1] * scale_y
            x1 = ls["end"][0] * scale_x
            y1 = ls["end"][1] * scale_y
            draw.line([(x0, y0), (x1, y1)], fill=(150, 180, 200), width=1)
        # Pass 2: assigned with category color
        for i, ls in enumerate(line_stats):
            if i in line_assignments:
                category = line_assignments[i]
                cat_info = page_categories.get(category, {})
                color = cat_info.get("color", (0, 255, 0))
                x0 = ls["start"][0] * scale_x
                y0 = ls["start"][1] * scale_y
                x1 = ls["end"][0] * scale_x
                y1 = ls["end"][1] * scale_y
                draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=6)
                draw.line([(x0, y0), (x1, y1)], fill=color, width=4)
                draw.ellipse([(x0 - 5, y0 - 5), (x0 + 5, y0 + 5)], fill=color)
                draw.ellipse([(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)], fill=color)

        drawn_buf = BytesIO()
        display_img.save(drawn_buf, format="WEBP", quality=90, method=6)
        st.session_state[drawn_img_cache_key] = drawn_buf.getvalue()
        del drawn_buf

    display_img = Image.open(BytesIO(st.session_state[drawn_img_cache_key])).convert("RGB")

    # ---- Display image + side info panel ----
    col_img, col_info = st.columns([3, 1])

    with col_img:
        if page_key not in st.session_state.user_drawn_lines:
            st.session_state.user_drawn_lines[page_key] = []

        if drawing_mode == "draw" and _HAS_IMG_COORDS:
            pending_start = st.session_state.pending_line_start.get(page_key)
            draw_img = display_img.copy()
            draw_overlay = ImageDraw.Draw(draw_img)
            user_lines = st.session_state.user_drawn_lines.get(page_key, [])
            for ul in user_lines:
                cat = ul.get("category")
                cat_info = page_categories.get(cat, {})
                color = cat_info.get("color", (0, 255, 0))
                x0 = ul["start"][0] * scale_x
                y0 = ul["start"][1] * scale_y
                x1 = ul["end"][0] * scale_x
                y1 = ul["end"][1] * scale_y
                draw_overlay.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=5)
                draw_overlay.line([(x0, y0), (x1, y1)], fill=color, width=3)
                draw_overlay.ellipse([(x0 - 4, y0 - 4), (x0 + 4, y0 + 4)], fill=color)
                draw_overlay.ellipse([(x1 - 4, y1 - 4), (x1 + 4, y1 + 4)], fill=color)
            if pending_start:
                px, py = pending_start
                img_px = px * scale_x
                img_py = py * scale_y
                draw_overlay.ellipse(
                    [(img_px - 8, img_py - 8), (img_px + 8, img_py + 8)],
                    fill=(255, 255, 0), outline=(0, 0, 0),
                )

            click_key = f"draw_click_{page_num}"
            if click_key not in st.session_state:
                st.session_state[click_key] = None

            click_result = streamlit_image_coordinates(
                draw_img, key=f"draw_img_{page_num}",
            )
            if click_result is not None:
                current_click = (click_result.get("x", 0), click_result.get("y", 0))
                if current_click != st.session_state[click_key]:
                    st.session_state[click_key] = current_click
                    click_x, click_y = current_click
                    pdf_click_x = click_x / scale_x
                    pdf_click_y = click_y / scale_y
                    if pending_start is None:
                        st.session_state.pending_line_start[page_key] = (pdf_click_x, pdf_click_y)
                        st.rerun()
                    else:
                        active_cat = st.session_state.active_category_per_page.get(page_key)
                        start_x, start_y = pending_start
                        end_x, end_y = pdf_click_x, pdf_click_y
                        length_pts = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
                        length_inches = length_pts / 72.0
                        length_feet = (length_inches * effective_scale) / 12.0
                        new_line = {
                            "start": (start_x, start_y),
                            "end": (end_x, end_y),
                            "category": active_cat,
                            "length_pts": length_pts,
                            "length_feet": length_feet,
                        }
                        if page_key not in st.session_state.user_drawn_lines:
                            st.session_state.user_drawn_lines[page_key] = []
                        st.session_state.user_drawn_lines[page_key].append(new_line)
                        st.session_state.pending_line_start[page_key] = None
                        st.rerun()

        elif drawing_mode == "select" and _HAS_IMG_COORDS:
            click_key = f"last_click_{page_num}"
            if click_key not in st.session_state:
                st.session_state[click_key] = None
            click_result = streamlit_image_coordinates(
                display_img, key=f"click_img_{page_num}",
            )
            if click_result is not None:
                current_click = (click_result.get("x", 0), click_result.get("y", 0))
                if current_click != st.session_state[click_key]:
                    st.session_state[click_key] = current_click
                    click_x, click_y = current_click
                    pdf_click_x = click_x / scale_x
                    pdf_click_y = click_y / scale_y
                    min_dist = float("inf")
                    nearest_idx = -1
                    for i, ls in enumerate(line_stats):
                        dist = _point_to_line_distance(
                            pdf_click_x, pdf_click_y,
                            ls["start"][0], ls["start"][1],
                            ls["end"][0], ls["end"][1],
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = i
                    if nearest_idx >= 0 and min_dist < 30:
                        active_cat = st.session_state.active_category_per_page.get(page_key)
                        current_assignment = st.session_state.line_assignments[page_key].get(nearest_idx)
                        if current_assignment == active_cat:
                            st.session_state.line_assignments[page_key].pop(nearest_idx, None)
                        else:
                            if active_cat:
                                st.session_state.line_assignments[page_key][nearest_idx] = active_cat
                        st.rerun()
        else:
            # Fallback: render the image without click capture so users
            # still see the preview when streamlit-image-coordinates is
            # unavailable.
            st.image(display_img, use_container_width=True)

    with col_info:
        st.markdown(f"**{len(lines)} detected lines**")
        if drawing_mode == "select":
            st.caption("Click to assign to category")
        else:
            pending = st.session_state.pending_line_start.get(page_key)
            if pending:
                st.warning("Click end point")
                if st.button("Cancel", key=f"cancel_draw_{page_num}"):
                    st.session_state.pending_line_start[page_key] = None
                    st.rerun()
            else:
                st.caption("Click start point")

        clear_col1, clear_col2 = st.columns(2)
        with clear_col1:
            if st.button("Clear Sel", key=f"clear_sel_{page_num}"):
                st.session_state.line_assignments[page_key] = {}
                st.rerun()
        with clear_col2:
            if st.button("Clear Drawn", key=f"clear_drawn_{page_num}"):
                st.session_state.user_drawn_lines[page_key] = []
                st.rerun()

        line_assignments = st.session_state.line_assignments.get(page_key, {})
        if line_assignments:
            by_category = {}
            for idx, cat in line_assignments.items():
                by_category.setdefault(cat, []).append(idx)
            st.markdown(f"**Selected: {len(line_assignments)}**")
            for cat, indices in by_category.items():
                cat_info = page_categories.get(cat, {})
                color = cat_info.get("color", (0, 255, 0))
                cat_total = sum(line_stats[i]["length_feet"] for i in indices if i < len(line_stats))
                st.markdown(
                    f"<span style='color: rgb{color};'>●</span> "
                    f"**{cat}**: {len(indices)} lines, {cat_total:.1f} ft",
                    unsafe_allow_html=True,
                )

        user_lines = st.session_state.user_drawn_lines.get(page_key, [])
        if user_lines:
            st.markdown("---")
            st.markdown(f"**Drawn: {len(user_lines)}**")
            drawn_by_cat = {}
            for ul in user_lines:
                cat = ul.get("category", "Uncategorized")
                drawn_by_cat.setdefault(cat, []).append(ul)
            for cat, cat_lines in drawn_by_cat.items():
                cat_info = page_categories.get(cat, {})
                color = cat_info.get("color", (0, 255, 0))
                cat_total = sum(ul.get("length_feet", 0) for ul in cat_lines)
                st.markdown(
                    f"<span style='color: rgb{color};'>●</span> "
                    f"**{cat}**: {len(cat_lines)} drawn, {cat_total:.1f} ft",
                    unsafe_allow_html=True,
                )


def _render_summary(fence_pages, min_line_pts):
    """Bottom-of-UMT cross-page summary table + grand total."""
    st.markdown("---")
    st.markdown("### Overall Summary")

    category_totals = {}
    grand_total_feet = 0
    grand_total_lines = 0

    for page_data in fence_pages:
        page_num = _pg_num(page_data)
        page_key = f"page_{page_num}"
        lines_cache_key = f"lines_{page_num}_{min_line_pts}"

        page_scale_info = st.session_state.per_page_scale_info.get(page_key, {}) or {}
        page_scale = (page_scale_info.get("verified_scale")
                      or page_scale_info.get("text_scale") or 360.0)

        auto_matched = st.session_state.get(f"auto_matched_indices_{page_key}", set())
        lines = st.session_state.get(lines_cache_key, [])
        line_assignments = st.session_state.line_assignments.get(page_key, {})

        for i, category in line_assignments.items():
            if i < len(lines):
                line = lines[i]
                length_inches = line["length_pts"] / 72.0
                length_feet = (length_inches * page_scale) / 12.0
                if category not in category_totals:
                    category_totals[category] = {"auto": 0, "lines": 0, "feet": 0, "drawn": 0}
                if i in auto_matched:
                    category_totals[category]["auto"] += 1
                else:
                    category_totals[category]["lines"] += 1
                category_totals[category]["feet"] += length_feet
                grand_total_feet += length_feet
                grand_total_lines += 1

        user_lines = st.session_state.user_drawn_lines.get(page_key, [])
        for ul in user_lines:
            category = ul.get("category", "Uncategorized")
            length_feet = ul.get("length_feet", 0)
            if category not in category_totals:
                category_totals[category] = {"auto": 0, "lines": 0, "feet": 0, "drawn": 0}
            category_totals[category]["drawn"] += 1
            category_totals[category]["feet"] += length_feet
            grand_total_feet += length_feet
            grand_total_lines += 1

    if grand_total_lines > 0:
        st.markdown("#### By Category")
        for cat, totals in category_totals.items():
            color = (0, 255, 0)
            for _pk, pc in st.session_state.page_categories.items():
                if cat in pc:
                    color = pc[cat].get("color", (0, 255, 0))
                    break
            col_cat, col_lines, col_feet = st.columns([3, 1, 1])
            with col_cat:
                st.markdown(
                    f"<span style='color: rgb{color}; font-size: 18px;'>●</span> **{cat}**",
                    unsafe_allow_html=True,
                )
            with col_lines:
                auto = totals.get("auto", 0)
                selected = totals["lines"]
                drawn = totals.get("drawn", 0)
                parts = []
                if auto:
                    parts.append(f"auto:{auto}")
                if selected:
                    parts.append(f"sel:{selected}")
                if drawn:
                    parts.append(f"drawn:{drawn}")
                st.markdown(", ".join(parts) if parts else "0")
            with col_feet:
                st.metric("Length", f"{totals['feet']:.1f} ft", label_visibility="collapsed")

        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Lines", grand_total_lines)
        with col_s2:
            st.metric("**Grand Total**", f"{grand_total_feet:.1f} ft")
        with col_s3:
            pages_with_assign = sum(
                1 for p in fence_pages
                if st.session_state.line_assignments.get(f"page_{_pg_num(p)}", {})
            )
            st.metric("Pages", pages_with_assign)

        if st.button("Clear All Assignments", key="clear_all_selections"):
            st.session_state.line_assignments = {}
            st.rerun()
    else:
        st.info("Click lines in the page tabs above and assign them to categories to calculate totals.")


def _render_element_specs(element_details):
    if not element_details:
        return
    st.markdown("---")
    st.markdown("#### Element Specifications (Cross-Page Details)")
    spec_rows = []
    for elem_name, details in element_details.items():
        if any(v for v in details.values() if v):
            spec_rows.append({
                "Element": elem_name,
                "Height": details.get("height", ""),
                "Post Type": details.get("post_type", ""),
                "Post Spacing": details.get("post_spacing", ""),
                "Material": details.get("material", ""),
                "Gauge": details.get("gauge", ""),
                "Mesh Size": details.get("mesh_size", ""),
                "Foundation": details.get("foundation", ""),
                "Gate Info": details.get("gate_info", ""),
                "Detail Page": details.get("detail_page", ""),
            })
    if spec_rows:
        st.dataframe(pd.DataFrame(spec_rows), hide_index=True, use_container_width=True)
        with st.expander("Full Detail Text per Element", expanded=False):
            for elem_name, details in element_details.items():
                full = details.get("full_details", "")
                notes = details.get("notes", "")
                if full or notes:
                    st.markdown(f"**{elem_name}:**")
                    if full:
                        st.markdown(f"  {full}")
                    if notes:
                        st.markdown(f"  *Notes: {notes}*")


def _render_downloads(pdf_path, fence_pages, min_line_pts):
    st.markdown("---")
    st.markdown("#### Downloads")
    dl_col1, dl_col2 = st.columns(2)

    uploaded_pdf_name = (
        st.session_state.get("uploaded_pdf_name")
        or (os.path.basename(pdf_path) if pdf_path else "document.pdf")
    )

    line_assignments = st.session_state.get("line_assignments", {})
    user_drawn_lines = st.session_state.get("user_drawn_lines", {})
    page_categories = st.session_state.get("page_categories", {})
    per_page_scale_info = st.session_state.get("per_page_scale_info", {})
    element_details = st.session_state.get("element_details", {}) or {}

    # Build lines_by_page from cached vector lines so spreadsheet
    # lookups can resolve assigned line indices.
    lines_by_page = {}
    for page_data in fence_pages:
        pn = _pg_num(page_data)
        page_key = f"page_{pn}"
        cached = st.session_state.get(f"lines_{pn}_{min_line_pts}")
        if cached:
            lines_by_page[page_key] = cached

    with dl_col1:
        try:
            pdf_bytes, pdf_name = generate_measurement_pdf(
                pdf_path or "",
                fence_pages,
                line_assignments,
                user_drawn_lines,
                page_categories,
                uploaded_pdf_name=uploaded_pdf_name,
            )
            if pdf_bytes:
                st.download_button(
                    "Download PDF with Measurements",
                    pdf_bytes,
                    pdf_name,
                    "application/pdf",
                    key="dl_measurement_pdf_umt",
                )
            else:
                st.error("Error generating PDF")
        except Exception as e:
            st.error(f"PDF export error: {e}")

    with dl_col2:
        try:
            xlsx_data = generate_measurement_spreadsheet(
                fence_pages,
                line_assignments,
                user_drawn_lines,
                page_categories,
                per_page_scale_info,
                element_details,
                lines_by_page=lines_by_page,
                min_line_pts=float(min_line_pts),
            )
            if xlsx_data:
                base_name = os.path.splitext(uploaded_pdf_name)[0]
                st.download_button(
                    "Download Measurements Excel",
                    xlsx_data,
                    f"{base_name}_measurements.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_measurement_xlsx_umt",
                )
            else:
                st.caption("No measurements to export yet — assign or draw lines first.")
        except Exception as e:
            st.error(f"Excel export error: {e}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_umt(
    pdf_path,
    fence_pages,
    per_page_scale_info,
    unified_measurements,
    element_details,
    enable_nonlayer=False,
    low_dpi_mode=False,
    job_id="",
):
    """Render the entire Unified Measurement Tool UI for a completed
    analysis job. See module docstring for argument semantics.
    """
    if not fence_pages:
        return

    st.markdown("---")
    st.markdown("<h2>Unified Measurement Tool</h2>", unsafe_allow_html=True)
    st.caption(
        "Auto-detected lines pre-selected | Click to select manual lines | "
        "Switch to Draw mode for custom lines"
    )

    # Lazy gate — the canvas builds heavy per-page caches; only load on demand.
    UMT_LOADED_KEY = f"_umt_tool_loaded_{job_id or 'default'}"
    if not st.session_state.get(UMT_LOADED_KEY):
        col1, col2 = st.columns([1, 3])
        with col1:
            def _on_open_umt():
                st.session_state[UMT_LOADED_KEY] = True
            st.button(
                "Open Measurement Tool",
                key=f"_umt_open_btn_{job_id or 'default'}",
                type="primary",
                use_container_width=True,
                on_click=_on_open_umt,
            )
        with col2:
            st.caption(
                "Interactive canvas for auto-detected + manually drawn fence lines. "
                "Vector line detection runs per loaded page. Only loads when "
                "requested. Your assignments are saved."
            )
        return

    # Initialise per-job session state buckets.
    if "per_page_scale_info" not in st.session_state:
        st.session_state.per_page_scale_info = {}
    # Seed from the API-provided per_page_scale_info on first run so we
    # don't re-detect the scale for pages that already have it.
    for p in fence_pages:
        pn = _pg_num(p)
        pk = f"page_{pn}"
        if pk in st.session_state.per_page_scale_info:
            continue
        api_si = _scale_lookup(per_page_scale_info, pn)
        if api_si:
            st.session_state.per_page_scale_info[pk] = api_si

    if "user_drawn_lines" not in st.session_state:
        st.session_state.user_drawn_lines = {}
    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = {}
    if "pending_line_start" not in st.session_state:
        st.session_state.pending_line_start = {}
    if "line_assignments" not in st.session_state:
        st.session_state.line_assignments = {}
    if "page_categories" not in st.session_state:
        st.session_state.page_categories = {}
    if "active_category_per_page" not in st.session_state:
        st.session_state.active_category_per_page = {}
    if "unified_measurements" not in st.session_state:
        st.session_state.unified_measurements = {}
    # Pre-seed unified_measurements from the API-provided dict so that
    # _render_page_panel's auto-line sync logic can find the analysis
    # output. Only seed when the user has not already populated it.
    for p in fence_pages:
        pn = _pg_num(p)
        pk = f"page_{pn}"
        if pk in st.session_state.unified_measurements:
            continue
        api_meas = _meas_lookup(unified_measurements, pn)
        if api_meas:
            # Stash the raw API blob so the panel can sync auto_lines from
            # it later. We don't pre-build auto_lines here because the
            # sync needs the enable_nonlayer toggle which is per-call.
            st.session_state.unified_measurements[pk] = {
                "auto_lines": [], "manual_lines": [],
                "drawn_lines": [], "accepted_auto": set(),
            }
    if "element_details" not in st.session_state:
        st.session_state.element_details = element_details or {}

    # Global controls
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        min_line_pts = st.number_input(
            "Min line length (pts)",
            min_value=5, max_value=200, value=30, step=5,
            help="Filter out short lines (hatching, text)",
            key=f"min_line_pts_input_{job_id or 'default'}",
        )
    with col_g2:
        st.info("Scale detected per page (see each tab)")

    zoom_level = st.slider(
        "Zoom",
        min_value=600, max_value=2000, value=1200, step=100,
        help="Adjust image display width",
        key=f"umt_zoom_{job_id or 'default'}",
    )

    # display dpi is captured in the rendered base image; keep it bounded
    # by the low_dpi sidebar toggle to save memory for big PDFs.
    _ = 110 if low_dpi_mode else 150  # acknowledged via _render_base_image dpi default

    page_tabs = st.tabs([f"Page {_pg_num(p)}" for p in fence_pages])

    for tab, page_data in zip(page_tabs, fence_pages):
        with tab:
            pg_num = _pg_num(page_data)
            pg_flag = f"_umt_pg_loaded_{job_id or 'default'}_{pg_num}"
            if not st.session_state.get(pg_flag):
                btn_col, cap_col = st.columns([1, 3])
                with btn_col:
                    def _on_load_page(active=pg_num, flag=pg_flag, prefix=f"_umt_pg_loaded_{job_id or 'default'}_"):
                        _evict_other_umt_pages(active)
                        for k in list(st.session_state.keys()):
                            if k.startswith(prefix) and k != flag:
                                del st.session_state[k]
                        st.session_state[flag] = True

                    st.button(
                        f"Load page {pg_num}",
                        key=f"_umt_pg_btn_{job_id or 'default'}_{pg_num}",
                        type="primary",
                        use_container_width=True,
                        on_click=_on_load_page,
                    )
                with cap_col:
                    st.caption(
                        "Click to detect vector lines + render the measurement "
                        "canvas for this page. Opens this page and offloads "
                        "any other currently-loaded page's images from memory. "
                        "Your line assignments persist across page switches."
                    )
            else:
                try:
                    _render_page_panel(
                        page_data, zoom_level, min_line_pts, pdf_path,
                        per_page_scale_info, unified_measurements,
                        enable_nonlayer, job_id,
                    )
                except Exception as e:
                    st.error(f"Page render error: {e}")
                    print(f"[UMT] page {pg_num} render error: {e}")

    _render_summary(fence_pages, min_line_pts)
    _render_element_specs(st.session_state.get("element_details") or element_details or {})
    _render_downloads(pdf_path, fence_pages, min_line_pts)
