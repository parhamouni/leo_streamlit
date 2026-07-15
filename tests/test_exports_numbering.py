"""Excel ↔ measurement-PDF line-number cross-referencing.

The customer-facing contract: every row in the Excel "Line #" column
corresponds to a numbered label drawn next to that line on the measurement
PDF. Both sides MUST derive their numbering from
exports.enumerate_page_lines — these tests guard that they stay in sync,
including the shared min-length filter.
"""
import io

import fitz
import pandas as pd
import pytest

from exports import (
    MIN_LINE_PTS,
    enumerate_page_lines,
    generate_measurement_pdf,
    generate_measurement_spreadsheet,
)


class _ObjLine:
    def __init__(self, start, end):
        self.start = start
        self.end = end


# ---------------------------------------------------------------- unit tests

def test_enumeration_sorted_by_int_index():
    lines = [
        {"start": [0, 0], "end": [100, 0]},
        {"start": [0, 10], "end": [100, 10]},
        {"start": [0, 20], "end": [100, 20]},
    ]
    # Dict order deliberately scrambled; "10" must not sort before "2".
    pa = {"2": "B", "0": "A", "1": "C"}
    numbered = enumerate_page_lines(lines, pa, [])
    assert [(nl.number, nl.category) for nl in numbered] == \
        [(1, "A"), (2, "C"), (3, "B")]


def test_min_length_filter_and_contiguous_numbering():
    lines = [
        {"start": [0, 0], "end": [100, 0]},   # long
        {"start": [0, 0], "end": [5, 0]},     # short -> dropped
        {"start": [0, 0], "end": [0, 50]},    # long
    ]
    pa = {"0": "A", "1": "A", "2": "A"}
    user = [
        {"start": [0, 0], "end": [3, 0], "category": "U"},   # short -> dropped
        {"start": [0, 0], "end": [0, 40], "category": "U"},  # long
    ]
    numbered = enumerate_page_lines(lines, pa, user)
    assert [nl.number for nl in numbered] == [1, 2, 3]
    assert [nl.source for nl in numbered] == ["auto", "auto", "user"]
    assert all(nl.length_pts >= MIN_LINE_PTS for nl in numbered)


def test_enumeration_handles_object_lines_and_bad_input():
    lines = [_ObjLine((0, 0), (100, 0))]
    pa = {"0": "A", "not-an-int": "A", "99": "A"}
    numbered = enumerate_page_lines(lines, pa, [{"bogus": True}])
    assert len(numbered) == 1
    assert numbered[0].start == (0, 0) and numbered[0].end == (100, 0)


# ------------------------------------------------------------- shared fixture

@pytest.fixture
def export_inputs(tmp_path):
    """One-page source PDF plus matching export-call inputs: three assigned
    auto lines (one below the length floor) and one user-drawn line."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    pdf_path = tmp_path / "src.pdf"
    doc.save(str(pdf_path))
    doc.close()

    auto_lines = [
        {"start": [100, 100], "end": [300, 100]},
        {"start": [100, 200], "end": [100, 205]},   # 5pt -> dropped everywhere
        {"start": [100, 300], "end": [300, 300]},
    ]
    fence_pages = [{
        "page_idx": 0,
        "page_num": 1,
        "auto_lines": auto_lines,
        "definitions": [],
        "instances": [],
    }]
    line_assignments = {"page_1": {"0": "Chain Link", "1": "Chain Link",
                                   "2": "Wood Fence"}}
    user_drawn_lines = {"page_1": [
        {"start": [400, 400], "end": [400, 500], "category": "Chain Link"},
    ]}
    page_categories = {"page_1": {
        "Chain Link": {"color": (0, 255, 0)},
        "Wood Fence": {"color": (255, 165, 0)},
    }}
    return {
        "pdf_path": str(pdf_path),
        "fence_pages": fence_pages,
        "line_assignments": line_assignments,
        "user_drawn_lines": user_drawn_lines,
        "page_categories": page_categories,
        "lines_by_page": {"page_1": auto_lines},
    }


def _read_excel(inputs) -> pd.DataFrame:
    data = generate_measurement_spreadsheet(
        inputs["fence_pages"],
        inputs["line_assignments"],
        inputs["user_drawn_lines"],
        inputs["page_categories"],
        per_page_scale_info={},
        element_details={},
        lines_by_page=inputs["lines_by_page"],
    )
    assert data
    return pd.read_excel(io.BytesIO(data), sheet_name="Measurements")


def _render_pdf(inputs, **kwargs) -> fitz.Document:
    pdf_bytes, _ = generate_measurement_pdf(
        inputs["pdf_path"],
        inputs["fence_pages"],
        inputs["line_assignments"],
        inputs["user_drawn_lines"],
        inputs["page_categories"],
        **kwargs,
    )
    assert pdf_bytes
    return fitz.open(stream=pdf_bytes, filetype="pdf")


# ------------------------------------------------------------------ Excel

def test_excel_line_numbers(export_inputs):
    df = _read_excel(export_inputs)
    assert "Line #" in df.columns
    # 2 long auto lines + 1 user line; the 5pt line has no row.
    assert list(df["Line #"]) == [1, 2, 3]
    assert list(df["Type"]) == ["auto", "auto", "user"]
    assert "Midpoint X (pts)" in df.columns and "Midpoint Y (pts)" in df.columns


# ------------------------------------------------------------------- PDF

def test_pdf_labels_match_excel(export_inputs):
    df = _read_excel(export_inputs)
    doc = _render_pdf(export_inputs)
    # Page 0 = legend, page 1 = the measured page.
    assert "Legend" in doc.load_page(0).get_text()
    page_words = {w[4] for w in doc.load_page(1).get_text("words")}
    excel_labels = {str(n) for n in df[df["Page"] == 1]["Line #"]}
    assert excel_labels <= page_words, (
        f"Excel Line # {excel_labels} not all labeled on PDF: {page_words}")
    # 3 distinct drawn segments (the 5pt one excluded, matching the Excel).
    # Count undirected unique segments: finish() closes subpaths, so
    # get_drawings can report a closing line item per subpath as well.
    segs = {
        frozenset({(round(item[1].x), round(item[1].y)),
                   (round(item[2].x), round(item[2].y))})
        for d in doc.load_page(1).get_drawings()
        for item in d["items"] if item[0] == "l"
    }
    assert segs == {
        frozenset({(100, 100), (300, 100)}),
        frozenset({(100, 300), (300, 300)}),
        frozenset({(400, 400), (400, 500)}),
    }
    doc.close()


def test_pdf_density_cap_replaces_labels_with_note(export_inputs):
    doc = _render_pdf(export_inputs, max_labels_per_page=2)
    text = doc.load_page(1).get_text()
    assert "labels omitted" in text
    words = {w[4] for w in doc.load_page(1).get_text("words")}
    assert "1" not in words  # per-line labels suppressed
    doc.close()


def test_pdf_labels_on_rotated_page(tmp_path, export_inputs):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.set_rotation(90)
    pdf_path = tmp_path / "rot.pdf"
    doc.save(str(pdf_path))
    doc.close()
    export_inputs["pdf_path"] = str(pdf_path)

    out = _render_pdf(export_inputs)
    words = {w[4] for w in out.load_page(1).get_text("words")}
    assert {"1", "2", "3"} <= words
    out.close()


def test_legend_page_lists_categories(export_inputs):
    doc = _render_pdf(export_inputs)
    text = doc.load_page(0).get_text()
    assert "Chain Link" in text and "Wood Fence" in text
    assert "Line #" in text
    doc.close()
