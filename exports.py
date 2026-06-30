"""PDF and spreadsheet export generation.

Works on results data from the API — no Streamlit dependency.
Used by the Streamlit frontend for client-side export generation.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO

import fitz
import pandas as pd

log = logging.getLogger("exports")


def _scale_inches_to_points_per_foot(scale_inches) -> float:
    """Convert architectural scale inches to PDF points per real foot."""
    try:
        scale = float(scale_inches)
    except (TypeError, ValueError):
        scale = 360.0
    if scale <= 0:
        scale = 360.0
    return 864.0 / scale


def generate_measurement_pdf(
    pdf_path: str,
    fence_pages: list[dict],
    line_assignments: dict,
    user_drawn_lines: dict,
    page_categories: dict,
    uploaded_pdf_name: str = "document.pdf",
) -> tuple[bytes | None, str]:
    """Generate a PDF with measurement lines overlaid by category color.

    Returns (pdf_bytes, filename) or (None, filename) on failure.
    """
    input_doc = None
    output_doc = None
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return None, "measurements.pdf"

        input_doc = fitz.open(pdf_path)
        output_doc = fitz.open()

        sorted_pages = sorted(fence_pages, key=lambda x: x.get("page_idx", x.get("page_index_in_original_doc", 0)))

        for res_data in sorted_pages:
            page_idx = res_data.get("page_idx", res_data.get("page_index_in_original_doc"))
            page_num = res_data.get("page_num", res_data.get("page_number"))
            if page_idx is None:
                continue

            page_key = f"page_{page_num}"
            try:
                output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
                page_out = output_doc.load_page(len(output_doc) - 1)

                rotation = page_out.rotation
                mw = page_out.mediabox.width
                mh = page_out.mediabox.height

                def _rot(x0, y0, x1, y1):
                    if rotation == 90:
                        return y0, mh - x1, y1, mh - x0
                    elif rotation == 180:
                        return mw - x1, mh - y1, mw - x0, mh - y0
                    elif rotation == 270:
                        return mw - y1, x0, mw - y0, x1
                    return x0, y0, x1, y1

                # Draw everything into ONE shape committed once per page.
                # page.draw_line/draw_rect/draw_circle each create *and commit*
                # their own shape, and every commit re-parses the whole page
                # content stream (wrap_contents → _count_q_balance). That's
                # O(n²): a dense page can carry tens of thousands of auto lines
                # (one report hit 24k on a single page), which blew past the
                # 600s worker timeout and surfaced as "Download failed: Failed
                # to fetch". Batching the draws and committing once is O(n).
                shape = page_out.new_shape()

                def_rects = []
                for d in res_data.get("definitions", []):
                    if all(k in d for k in ("x0", "y0", "x1", "y1")):
                        r = fitz.Rect(*_rot(d["x0"], d["y0"], d["x1"], d["y1"]))
                        r.normalize()
                        if not r.is_empty and r.is_valid:
                            def_rects.append(r)
                if def_rects:
                    for r in def_rects:
                        shape.draw_rect(r)
                    shape.finish(color=(0, 0.9, 0), width=2.0)

                inst_rects = []
                for inst in res_data.get("instances", []):
                    if all(k in inst for k in ("x0", "y0", "x1", "y1")):
                        r = fitz.Rect(*_rot(inst["x0"], inst["y0"], inst["x1"], inst["y1"]))
                        r.normalize()
                        if not r.is_empty and r.is_valid:
                            inst_rects.append(r)
                if inst_rects:
                    for r in inst_rects:
                        shape.draw_rect(r)
                    shape.finish(color=(0.9, 0, 0.9), width=2.0)

                categories = page_categories.get(page_key, {})
                pa = line_assignments.get(page_key, {})
                lines = res_data.get("auto_lines", [])

                # Group auto lines by stroke color so each color batch needs a
                # single finish() — a finish per line would emit one graphics
                # group per segment (24k of them on the worst page).
                segs_by_color: dict[tuple, list] = {}
                for line_idx_str, category in pa.items():
                    idx = int(line_idx_str) if isinstance(line_idx_str, str) else line_idx_str
                    if idx < len(lines):
                        line = lines[idx]
                        cat_info = categories.get(category, {})
                        c_rgb = cat_info.get("color", (0, 255, 0))
                        color = (c_rgb[0] / 255, c_rgb[1] / 255, c_rgb[2] / 255)
                        if isinstance(line, dict):
                            sx, sy = line["start"]
                            ex, ey = line["end"]
                        else:
                            sx, sy = line.start
                            ex, ey = line.end
                        ax0, ay0, ax1, ay1 = _rot(sx, sy, ex, ey)
                        segs_by_color.setdefault(color, []).append(((ax0, ay0), (ax1, ay1)))
                for color, segs in segs_by_color.items():
                    for p1, p2 in segs:
                        shape.draw_line(p1, p2)
                    shape.finish(color=color, width=3.0)

                for ul in user_drawn_lines.get(page_key, []):
                    cat = ul.get("category")
                    cat_info = categories.get(cat, {})
                    c_rgb = cat_info.get("color", (0, 255, 0))
                    color = (c_rgb[0] / 255, c_rgb[1] / 255, c_rgb[2] / 255)
                    sx, sy = ul["start"]
                    ex, ey = ul["end"]
                    ax0, ay0, ax1, ay1 = _rot(sx, sy, ex, ey)
                    shape.draw_line((ax0, ay0), (ax1, ay1))
                    shape.finish(color=color, width=3.0)
                    shape.draw_circle((ax0, ay0), 3)
                    shape.draw_circle((ax1, ay1), 3)
                    shape.finish(color=color, fill=color)

                shape.commit(overlay=True)

            except Exception as e:
                log.warning(f"Measurement PDF page {page_idx}: {e}")

        base, ext = os.path.splitext(uploaded_pdf_name or "document.pdf")
        fname = f"{base}_measurements{ext}"

        try:
            pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
        except Exception:
            pdf_bytes = None

        return pdf_bytes, fname

    except Exception as e:
        log.warning(f"Measurement PDF generation failed: {e}")
        return None, "measurements.pdf"
    finally:
        if input_doc:
            input_doc.close()
        if output_doc:
            output_doc.close()


def _lookup_element_details(category: str, element_details: dict) -> dict:
    if not element_details:
        return {}
    if category in element_details:
        return element_details[category]
    cat_lower = category.lower()
    for name, details in element_details.items():
        if name.lower() == cat_lower:
            return details
    for name, details in element_details.items():
        if name.lower() in cat_lower or cat_lower in name.lower():
            return details
    return {}


def generate_measurement_spreadsheet(
    fence_pages: list[dict],
    line_assignments: dict,
    user_drawn_lines: dict,
    page_categories: dict,
    per_page_scale_info: dict,
    element_details: dict,
    lines_by_page: dict | None = None,
    min_line_pts: float = 10.0,
) -> bytes | None:
    """Generate an Excel workbook with measurement data.

    Returns the workbook bytes or None on failure.
    """
    DETAIL_COLS = ["Height", "Post Type", "Post Spacing", "Material",
                   "Gauge", "Mesh Size", "Detail Page", "Full Details"]

    rows = []
    for page_data in fence_pages:
        page_num = page_data.get("page_num", page_data.get("page_number"))
        page_key = f"page_{page_num}"

        scale_info = per_page_scale_info.get(page_key, {})
        scale_inches = scale_info.get("verified_scale") or scale_info.get("text_scale") or 360.0
        page_scale = _scale_inches_to_points_per_foot(scale_inches)

        lines = (lines_by_page or {}).get(page_key, [])
        pa = line_assignments.get(page_key, {})

        for line_idx_str, category in pa.items():
            idx = int(line_idx_str) if isinstance(line_idx_str, str) else line_idx_str
            if idx < len(lines):
                line = lines[idx]
                if isinstance(line, dict):
                    sx, sy = line["start"]
                    ex, ey = line["end"]
                else:
                    sx, sy = line.start
                    ex, ey = line.end
                length_pts = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
                if length_pts < min_line_pts:
                    continue
                length_ft = length_pts / page_scale
                row = _build_row(page_num, category, "auto", length_ft, length_pts,
                                page_scale, element_details)
                rows.append(row)

        for ul in user_drawn_lines.get(page_key, []):
            sx, sy = ul["start"]
            ex, ey = ul["end"]
            length_pts = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
            if length_pts < min_line_pts:
                continue
            length_ft = length_pts / page_scale
            category = ul.get("category", "Unknown")
            row = _build_row(page_num, category, "user", length_ft, length_pts,
                            page_scale, element_details)
            rows.append(row)

    if not rows:
        return None

    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="Measurements", index=False)

            summary_rows = []
            for cat in df["Category"].unique():
                cat_df = df[df["Category"] == cat]
                total_ft = cat_df["Length (ft)"].sum()
                count = len(cat_df)
                summary_rows.append({
                    "Category": cat,
                    "Total Length (ft)": round(total_ft, 2),
                    "Line Count": count,
                })
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

            if element_details:
                detail_rows = []
                for name, details in element_details.items():
                    row = {"Element": name}
                    row.update(details)
                    detail_rows.append(row)
                if detail_rows:
                    pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Element Specs", index=False)

        return buf.getvalue()
    except Exception as e:
        log.warning(f"Spreadsheet generation failed: {e}")
        return None


def _build_row(page_num, category, row_type, length_ft, length_pts,
               page_scale, element_details):
    row = {
        "Page": page_num,
        "Category": category,
        "Type": row_type,
        "Length (ft)": round(length_ft, 2),
        "Length (pts)": round(length_pts, 2),
        "Scale": page_scale,
    }
    details = _lookup_element_details(category, element_details)
    row["Height"] = details.get("height", "")
    row["Post Type"] = details.get("post_type", "")
    row["Post Spacing"] = details.get("post_spacing", "")
    row["Material"] = details.get("material", "")
    row["Gauge"] = details.get("gauge", "")
    row["Mesh Size"] = details.get("mesh_size", "")
    row["Detail Page"] = details.get("detail_page", "")
    row["Full Details"] = details.get("full_details", "")
    return row
