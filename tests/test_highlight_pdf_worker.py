"""End-to-end tests for ops/highlight_pdf_worker.py (subprocess IPC).

Guards the 2026-07-07 regression: the worker used to insert_pdf page-by-page
into a fresh doc and serialize with tobytes(garbage=4), which duplicated
shared resources (~18 GB RSS on a 300-page scanned deck) and blew the
subprocess timeout, leaving jobs without a highlighted.pdf. The select()-based
rewrite must keep only drawable fence pages and write the file atomically.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import fitz
import pytest

WORKER = Path(__file__).resolve().parent.parent / "ops" / "highlight_pdf_worker.py"


def _run_worker(task: dict) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(WORKER)],
        input=json.dumps(task).encode(),
        capture_output=True,
        timeout=60,
    )
    out = json.loads(proc.stdout.decode().strip() or "{}")
    return proc.returncode, out


@pytest.fixture
def src_pdf(tmp_path):
    """Three-page PDF; page indexes 0/1/2."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"page {i}")
    path = tmp_path / "src.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_keeps_only_drawable_fence_pages(src_pdf, tmp_path):
    out_path = tmp_path / "highlighted.pdf"
    task = {
        "pdf_path": str(src_pdf),
        "out_path": str(out_path),
        "fence_pages": [
            # drawable: has a definition bbox
            {"page_index_in_original_doc": 2,
             "definitions": [{"x0": 10, "y0": 10, "x1": 100, "y1": 50}],
             "instances": [], "keyword_matches": []},
            # NOT drawable: no boxes at all -> must be dropped
            {"page_index_in_original_doc": 1,
             "definitions": [], "instances": [], "keyword_matches": []},
        ],
    }
    rc, out = _run_worker(task)
    assert rc == 0 and out["ok"] is True

    doc = fitz.open(str(out_path))
    assert doc.page_count == 1
    # The kept page is original page 2, and the definition rect was drawn.
    assert "page 2" in doc.load_page(0).get_text()
    drawings = doc.load_page(0).get_drawings()
    greens = [
        d for d in drawings
        if d.get("color") and tuple(round(c, 1) for c in d["color"]) == (0.0, 0.9, 0.0)
    ]
    assert greens, "expected the green definition rectangle to be drawn"
    doc.close()


def test_out_of_range_and_duplicate_indices_ignored(src_pdf, tmp_path):
    out_path = tmp_path / "highlighted.pdf"
    box = {"x0": 10, "y0": 10, "x1": 100, "y1": 50}
    task = {
        "pdf_path": str(src_pdf),
        "out_path": str(out_path),
        "fence_pages": [
            {"page_index_in_original_doc": 0, "definitions": [box],
             "instances": [], "keyword_matches": []},
            {"page_index_in_original_doc": 0, "definitions": [box],
             "instances": [], "keyword_matches": []},  # duplicate
            {"page_index_in_original_doc": 99, "definitions": [box],
             "instances": [], "keyword_matches": []},  # out of range
        ],
    }
    rc, out = _run_worker(task)
    assert rc == 0 and out["ok"] is True
    doc = fitz.open(str(out_path))
    assert doc.page_count == 1
    doc.close()


def test_zero_drawable_pages_fails_cleanly(src_pdf, tmp_path):
    out_path = tmp_path / "highlighted.pdf"
    task = {
        "pdf_path": str(src_pdf),
        "out_path": str(out_path),
        "fence_pages": [
            {"page_index_in_original_doc": 0,
             "definitions": [], "instances": [], "keyword_matches": []},
        ],
    }
    rc, out = _run_worker(task)
    assert rc == 1 and out["ok"] is False
    assert not out_path.exists()
