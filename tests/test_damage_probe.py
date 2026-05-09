"""Tests for the comprehensive damage probe (ops/page_extractor.py +
pipeline.py helpers).

The probe runs every MuPDF call later phases need (get_text "text"/
"words"/"dict", get_drawings, mediabox/rotation) in a subprocess, with
SIGKILL on timeout. Pages that fail are filtered out of the pipeline
before Phase 1a, so Phase 3 can stay fully concurrent without per-call
subprocess overhead.

These tests verify the contract end-to-end, hitting the real subprocess
via `pipeline._damage_probe_one` / `_damage_probe_batch`.
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
    _damage_probe_batch,
    _damage_probe_one,
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
    """A non-PDF file that fitz.open will reject — stand-in for a damaged
    PDF. The probe should report ok=False without hanging."""
    out = tmp_path / "broken.pdf"
    out.write_bytes(b"%PDF-1.4\n%damaged\n")
    return str(out)


def test_probe_one_healthy_page_returns_ok(healthy_pdf: str):
    res = _damage_probe_one(healthy_pdf, 0, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res == {"ok": True}


def test_probe_one_out_of_range_returns_damaged(healthy_pdf: str):
    res = _damage_probe_one(healthy_pdf, 99, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is False
    assert "out of range" in res["error"]


def test_probe_one_missing_pdf_returns_error(tmp_path: Path):
    res = _damage_probe_one(
        str(tmp_path / "nope.pdf"), 0, timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    assert res == {"ok": False, "error": "PDF disk file missing"}


def test_probe_one_truncated_pdf_returns_damaged(truncated_pdf: str):
    res = _damage_probe_one(truncated_pdf, 0, timeout_s=_PHASE1A_PAGE_TIMEOUT)
    assert res["ok"] is False
    # Either the open fails (no page_result emitted) or the probe itself
    # reports the page as damaged. Both are acceptable — the contract
    # is "ok=False, doesn't hang".
    assert res.get("error")


def test_probe_batch_all_healthy(healthy_pdf: str):
    res = _damage_probe_batch(
        healthy_pdf, [0, 1, 2], per_page_timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    assert set(res.keys()) == {0, 1, 2}
    assert all(r == {"ok": True} for r in res.values())


def test_probe_batch_empty_indices_returns_empty(healthy_pdf: str):
    assert _damage_probe_batch(healthy_pdf, [], per_page_timeout_s=5) == {}


def test_probe_batch_truncated_pdf_no_results(truncated_pdf: str):
    """fitz.open fails inside the worker → no page_results emitted →
    batch returns empty dict. Caller falls back to per-page probe (also
    fails fast) and marks all pages damaged. Important contract: never
    hang, always return."""
    res = _damage_probe_batch(
        truncated_pdf, [0], per_page_timeout_s=_PHASE1A_PAGE_TIMEOUT
    )
    # Either empty (open failed; no page_result) or a single ok=False
    # entry — both are non-hang outcomes.
    if res:
        assert res[0]["ok"] is False
