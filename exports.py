"""PDF and spreadsheet export generation.

Works on results data from the API — no Streamlit dependency.
Used by the Streamlit frontend for client-side export generation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from io import BytesIO

import fitz
import pandas as pd

log = logging.getLogger("exports")

#: Lines shorter than this many PDF points are excluded from exports —
#: they are almost always vector noise (hatch ticks, dimension arrows).
MIN_LINE_PTS = 10.0


@dataclass
class NumberedLine:
    """One exported line with its per-page display number.

    The number is the cross-reference key between the Excel "Line #"
    column and the label drawn on the measurement PDF, so both exports
    MUST obtain it from enumerate_page_lines — never number lines
    independently.
    """
    number: int                      # 1-based, per page
    source: str                      # "auto" | "user"
    category: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_pts: float


def enumerate_page_lines(
    auto_lines: list,
    page_assignments: dict,
    user_lines: list[dict],
    min_line_pts: float = MIN_LINE_PTS,
) -> list[NumberedLine]:
    """Enumerate one page's exportable lines in a deterministic order.

    Single source of truth for both the Excel and the measurement PDF:
    assigned auto lines sorted by their integer index, then user-drawn
    lines in saved order, with the same min-length filter applied to
    both. Survivors are numbered 1..N.
    """
    numbered: list[NumberedLine] = []

    entries = []
    for idx_str, category in (page_assignments or {}).items():
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            continue
        entries.append((idx, category))
    entries.sort(key=lambda t: t[0])

    for idx, category in entries:
        if idx < 0 or idx >= len(auto_lines):
            continue
        line = auto_lines[idx]
        try:
            if isinstance(line, dict):
                sx, sy = line["start"]
                ex, ey = line["end"]
            else:
                sx, sy = line.start
                ex, ey = line.end
        except (KeyError, AttributeError, TypeError, ValueError, IndexError):
            continue
        length_pts = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
        if length_pts < min_line_pts:
            continue
        numbered.append(NumberedLine(
            number=len(numbered) + 1,
            source="auto",
            category=category,
            start=(sx, sy),
            end=(ex, ey),
            length_pts=length_pts,
        ))

    for ul in user_lines or []:
        try:
            sx, sy = float(ul["start"][0]), float(ul["start"][1])
            ex, ey = float(ul["end"][0]), float(ul["end"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        length_pts = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
        if length_pts < min_line_pts:
            continue
        numbered.append(NumberedLine(
            number=len(numbered) + 1,
            source="user",
            category=ul.get("category") or "Unknown",
            start=(sx, sy),
            end=(ex, ey),
            length_pts=length_pts,
        ))

    return numbered


def insert_legend_page(
    doc: fitz.Document,
    title: str,
    source_name: str,
    line_swatches: list[tuple[str, tuple[float, float, float]]],
    box_swatches: list[tuple[str, tuple[float, float, float]]],
    notes: list[str],
    pno: int = 0,
) -> None:
    """Prepend a letter-size legend page explaining the markup colors.

    line_swatches/box_swatches are (label, rgb 0-1) pairs, rendered as a
    stroke sample and an outlined box sample respectively. Also used by
    ops/highlight_pdf_worker.py for the fence/detection-overview PDF.
    """
    page = doc.new_page(pno=pno, width=612, height=792)
    x, y = 54, 72
    page.insert_text((x, y), title, fontname="hebo", fontsize=18, color=(0, 0, 0))
    y += 22
    page.insert_text((x, y), f"Source: {source_name}", fontname="helv",
                     fontsize=10, color=(0.35, 0.35, 0.35))
    y += 14
    page.insert_text((x, y), f"Generated: {date.today().isoformat()}",
                     fontname="helv", fontsize=10, color=(0.35, 0.35, 0.35))
    y += 30

    def _section(header: str) -> None:
        nonlocal y
        page.insert_text((x, y), header, fontname="hebo", fontsize=12,
                         color=(0, 0, 0))
        y += 18

    MAX_ROWS = 30
    if line_swatches:
        _section("Line categories (measured fence lines)")
        for label, rgb in line_swatches[:MAX_ROWS]:
            page.draw_line((x, y - 3), (x + 40, y - 3), color=rgb, width=3.0)
            page.insert_text((x + 50, y), str(label), fontname="helv",
                             fontsize=10, color=(0, 0, 0))
            y += 16
        if len(line_swatches) > MAX_ROWS:
            page.insert_text((x + 50, y),
                             f"… and {len(line_swatches) - MAX_ROWS} more categories",
                             fontname="helv", fontsize=10, color=(0.35, 0.35, 0.35))
            y += 16
        y += 14

    if box_swatches:
        _section("Detection boxes")
        for label, rgb in box_swatches:
            page.draw_rect(fitz.Rect(x, y - 9, x + 14, y + 1), color=rgb, width=1.5)
            page.insert_text((x + 24, y), str(label), fontname="helv",
                             fontsize=10, color=(0, 0, 0))
            y += 16
        y += 14

    if notes:
        _section("Notes")
        for note in notes:
            rect = fitz.Rect(x, y - 10, 612 - 54, y + 60)
            # "-" not "•": U+2022 is outside the base-14 helv charset and
            # renders as "?".
            spent = page.insert_textbox(rect, f"-  {note}", fontname="helv",
                                        fontsize=10, color=(0, 0, 0))
            # insert_textbox returns unused height; advance by what was used.
            y += max(14, 70 - spent if spent >= 0 else 14)


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
    min_line_pts: float = MIN_LINE_PTS,
    max_labels_per_page: int = 150,
) -> tuple[bytes | None, str]:
    """Generate a PDF with measurement lines overlaid by category color.

    Lines are numbered per page with enumerate_page_lines — the same
    enumeration (and min-length filter) the Excel export uses — and each
    line gets a small numbered label matching the Excel "Line #" column.
    Pages with more than max_labels_per_page lines get a corner note
    instead of per-line labels.

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

        rendered_page_keys: list[str] = []
        for res_data in sorted_pages:
            page_idx = res_data.get("page_idx", res_data.get("page_index_in_original_doc"))
            page_num = res_data.get("page_num", res_data.get("page_number"))
            if page_idx is None:
                continue

            page_key = f"page_{page_num}"
            try:
                output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
                rendered_page_keys.append(page_key)
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

                # Same enumeration the Excel export uses — the label drawn
                # next to each line must match its "Line #" row.
                numbered = enumerate_page_lines(
                    lines, pa, user_drawn_lines.get(page_key, []), min_line_pts)

                def _cat_color(category):
                    c_rgb = categories.get(category, {}).get("color", (0, 255, 0))
                    return (c_rgb[0] / 255, c_rgb[1] / 255, c_rgb[2] / 255)

                # Group lines by stroke color so each color batch needs a
                # single finish() — a finish per line would emit one graphics
                # group per segment (24k of them on the worst page).
                segs_by_color: dict[tuple, list] = {}
                user_pts_by_color: dict[tuple, list] = {}
                for nl in numbered:
                    color = _cat_color(nl.category)
                    ax0, ay0, ax1, ay1 = _rot(*nl.start, *nl.end)
                    segs_by_color.setdefault(color, []).append(((ax0, ay0), (ax1, ay1)))
                    if nl.source == "user":
                        user_pts_by_color.setdefault(color, []).extend(
                            [(ax0, ay0), (ax1, ay1)])
                for color, segs in segs_by_color.items():
                    for p1, p2 in segs:
                        shape.draw_line(p1, p2)
                    shape.finish(color=color, width=3.0)
                for color, pts in user_pts_by_color.items():
                    for pt in pts:
                        shape.draw_circle(pt, 3)
                    shape.finish(color=color, fill=color)

                # Numbered labels. Font size tracks sheet width: ARCH D/E
                # sheets are ~2600-3456 pt wide, where a fixed 8 pt label
                # would be unreadable.
                fs = min(18, max(7, page_out.rect.width / 200))
                if numbered and len(numbered) <= max_labels_per_page:
                    halos = []
                    texts = []
                    for nl in numbered:
                        ax0, ay0, ax1, ay1 = _rot(*nl.start, *nl.end)
                        mx, my = (ax0 + ax1) / 2, (ay0 + ay1) / 2
                        dx, dy = ax1 - ax0, ay1 - ay0
                        seg_len = (dx * dx + dy * dy) ** 0.5 or 1.0
                        # Nudge the label off the stroke, perpendicular.
                        off = fs * 0.7
                        cx, cy = mx - dy / seg_len * off, my + dx / seg_len * off
                        label = str(nl.number)
                        tw = fitz.get_text_length(label, fontname="helv", fontsize=fs)
                        pad = fs * 0.2
                        half = max(tw, fs) / 2 + pad
                        halos.append(fitz.Rect(cx - half, cy - half, cx + half, cy + half))
                        texts.append(((cx - tw / 2, cy + fs * 0.35), label,
                                      _cat_color(nl.category)))
                    for r in halos:
                        shape.draw_rect(r)
                    shape.finish(color=None, fill=(1, 1, 1), fill_opacity=0.75)
                    for pt, label, color in texts:
                        shape.insert_text(pt, label, fontsize=fs, color=color,
                                          rotate=rotation)
                elif numbered:
                    note = (f"{len(numbered)} measured lines on this page — "
                            f"labels omitted for readability. Lines appear in "
                            f"the same order as the Excel 'Line #' column.")
                    tw = fitz.get_text_length(note, fontname="helv", fontsize=fs)
                    nx, ny = 12, fs * 1.6
                    shape.draw_rect(fitz.Rect(nx - 4, ny - fs * 1.1,
                                              nx + tw + 4, ny + fs * 0.45))
                    shape.finish(color=None, fill=(1, 1, 1), fill_opacity=0.85)
                    shape.insert_text((nx, ny), note, fontsize=fs,
                                      color=(0.8, 0, 0), rotate=rotation)

                shape.commit(overlay=True)

            except Exception as e:
                log.warning(f"Measurement PDF page {page_idx}: {e}")

        # Legend page up front so the color coding and Line # cross-reference
        # are explained inside the document itself.
        try:
            seen: set[tuple] = set()
            line_swatches: list[tuple[str, tuple[float, float, float]]] = []
            for page_key in rendered_page_keys:
                for cat_name, cat_info in (page_categories.get(page_key) or {}).items():
                    c = cat_info.get("color", (0, 255, 0))
                    rgb = (c[0] / 255, c[1] / 255, c[2] / 255)
                    if (cat_name, rgb) in seen:
                        continue
                    seen.add((cat_name, rgb))
                    line_swatches.append((cat_name, rgb))
            insert_legend_page(
                output_doc,
                title="Fence Measurements — Legend",
                source_name=uploaded_pdf_name or "document.pdf",
                line_swatches=line_swatches,
                box_swatches=[
                    ("Fence legend / definition callout", (0, 0.9, 0)),
                    ("Fence instance", (0.9, 0, 0.9)),
                ],
                notes=[
                    "Each measured line carries a numbered label matching the "
                    "'Line #' column in the measurements Excel export.",
                    f"Lines shorter than {min_line_pts:g} pts are excluded from "
                    "both the Excel and this PDF.",
                    f"Pages with more than {max_labels_per_page} measured lines "
                    "omit per-line labels for readability; their Excel rows "
                    "follow the same order.",
                    "User-drawn lines are marked with endpoint dots.",
                ],
            )
        except Exception as e:
            log.warning(f"Measurement PDF legend page failed: {e}")

        base, ext = os.path.splitext(uploaded_pdf_name or "document.pdf")
        fname = f"{base}_measurements{ext}"

        try:
            # garbage=4 (full cross-object compaction), not garbage=2. The
            # page loop calls insert_pdf once per page, and each insert has its
            # own dedup scope — so a raster/font/XObject shared across N source
            # pages is copied N times. garbage=2 only cleans within a stream and
            # leaves those duplicates, ballooning image-heavy CAD decks to
            # multiple GB (one 50-page report serialized to 6.8 GB, undownloadable
            # in the browser). garbage=4 collapses the duplicate objects back to
            # one, cutting size by up to Nx.
            pdf_bytes = output_doc.tobytes(garbage=4, deflate=True)
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
    min_line_pts: float = MIN_LINE_PTS,
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

        for nl in enumerate_page_lines(lines, pa, user_drawn_lines.get(page_key, []),
                                       min_line_pts):
            length_ft = nl.length_pts / page_scale
            row = _build_row(page_num, nl.number, nl.category, nl.source,
                             length_ft, nl.length_pts, page_scale,
                             nl.start, nl.end, element_details)
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


def _build_row(page_num, line_no, category, row_type, length_ft, length_pts,
               page_scale, start, end, element_details):
    # "Line #" matches the numbered label drawn next to the line on the
    # measurement PDF; the midpoint locates the line even without labels
    # (dense pages omit them for readability).
    row = {
        "Page": page_num,
        "Line #": line_no,
        "Category": category,
        "Type": row_type,
        "Length (ft)": round(length_ft, 2),
        "Length (pts)": round(length_pts, 2),
        "Scale": page_scale,
        "Midpoint X (pts)": round((start[0] + end[0]) / 2, 1),
        "Midpoint Y (pts)": round((start[1] + end[1]) / 2, 1),
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
