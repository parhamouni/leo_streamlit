"""Tests for Phase 1a's per-page subprocess (`ops/page_extractor.py` +
`pipeline._phase1a_extract_one` / `_phase1a_extract_batch`).

Phase 1a's subprocess captures `get_text("text")` + dims AND probes the
broader MuPDF call set later phases rely on (`get_text("words")`,
`get_text("dict")`, `get_drawings()`). A page that survives the
subprocess (with SIGKILL on timeout) is guaranteed safe for every
later in-process fitz call. Pages that throw or hang go into
`broken_pages` and are filtered out of every subsequent phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import (  # noqa: E402
    _PHASE1A_PAGE_TIMEOUT,
    _phase1a_extract_batch,
    _phase1a_extract_one,
)


@pytest.fixture
def healthy_pdf(tmp_path: Path) -> str:
    """A 3-page PDF with text, written to disk."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 72), f"Hello page {i}")
    out = tmp_path / "healthy.pdf"
    doc.save(str(out))
    doc.close()
    return str(out)


@pytest.fixture
def truncated_pdf(tmp_path: Path) -> str:
    """A non-PDF file that fitz.open will reject — stand-in for a
    damaged PDF. Phase 1a should report ok=False without hanging."""
    out = tmp_path / "broken.pdf"
    out.write_bytes(b"%PDF-1.4\n%damaged\n")
    return str(out)


def test_phase1a_one_healthy_returns_text_and_dims(healthy_pdf: str):
    res = _phase1a_extract_one(healthy_pdf, 0, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is True
    assert "Hello page 0" in res["text"]
    assert res["dims"]["width"] > 0
    assert res["dims"]["height"] > 0
    assert "rotation" in res["dims"]


def test_phase1a_one_out_of_range_returns_damaged(healthy_pdf: str):
    res = _phase1a_extract_one(healthy_pdf, 99, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is False
    assert "out of range" in res["error"]


def test_phase1a_one_missing_pdf_returns_error(tmp_path: Path):
    res = _phase1a_extract_one(
        str(tmp_path / "nope.pdf"), 0, timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    assert res == {"ok": False, "error": "PDF disk file missing"}


def test_phase1a_one_truncated_pdf_returns_damaged(truncated_pdf: str):
    res = _phase1a_extract_one(truncated_pdf, 0, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is False
    # Either fitz.open fails (no page_result emitted) or the probe itself
    # reports the page as damaged. Both are acceptable — the contract
    # is "ok=False, doesn't hang".
    assert res.get("error")


def test_phase1a_batch_all_healthy(healthy_pdf: str):
    res = _phase1a_extract_batch(
        healthy_pdf, [0, 1, 2], per_page_timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    assert set(res.keys()) == {0, 1, 2}
    for pi, r in res.items():
        assert r["ok"] is True
        assert f"Hello page {pi}" in r["text"]
        assert r["dims"]["width"] > 0


def test_phase1a_batch_empty_indices_returns_empty(healthy_pdf: str):
    assert _phase1a_extract_batch(healthy_pdf, [], per_page_timeout_s=5) == {}


def test_phase1a_batch_truncated_pdf_no_hang(truncated_pdf: str):
    """fitz.open fails inside the worker → no page_results emitted →
    batch returns empty dict. Caller falls back to per-page (also fails
    fast) and marks all pages damaged. Important contract: never hang,
    always return."""
    res = _phase1a_extract_batch(
        truncated_pdf, [0], per_page_timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    # Either empty (open failed; no page_result) or a single ok=False
    # entry — both are non-hang outcomes.
    if res:
        assert res[0]["ok"] is False


def test_phase1a_one_healthy_page_runs_full_probe(healthy_pdf: str):
    """A healthy real PDF page returning ok=True implicitly proves the
    full broader-probe ran end-to-end (get_text "text"/"words"/"dict"
    + get_drawings + dims) — any of those throwing in-subprocess would
    flip the result to ok=False with an `<op> failed: ...` error.
    `_phase1a_extract_one` reads the JSON line emitted by
    `_extract_one_phase1a` after every probe op completes, so this
    test exercises the production code path without faking."""
    res = _phase1a_extract_one(healthy_pdf, 0, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is True
    assert isinstance(res["text"], str)
    # The probe-only ops (words/dict/drawings) succeeded too — otherwise
    # the script would have returned ok=False with the failing op name
    # before the dims block, and we'd never see text/dims in the result.
    assert "text" in res
    assert "dims" in res
