"""DB ↔ disk alignment: stale job rows are swept, and re-uploading a
document whose source PDF was reaped from /tmp heals it instead of
bouncing off the dedup into an unrecoverable loop."""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pytest


os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"

# Dedup (and the heal) only run for Supabase-style UUID users.
UUID_USER = "22222222-2222-2222-2222-222222222222"


def _make_completed_job(job_registry, *, completed_at, with_results=True):
    job_id = job_registry.create_job(
        user_id=UUID_USER, filename="doc.pdf", pdf_path="/nonexistent/doc.pdf"
    )
    if with_results:
        job_registry.save_results(job_id, {"fence_pages": [], "non_fence_pages": []})
    job_registry.update_job(job_id, status="completed", completed_at=completed_at)
    return job_id


def test_orphan_sweep_removes_completed_job_without_results(job_registry_temp):
    old = int(time.time()) - 7200

    orphan = _make_completed_job(job_registry_temp, completed_at=old)
    healthy = _make_completed_job(job_registry_temp, completed_at=old)

    # Simulate incident cleanup / lost volume: results.json vanishes.
    results_dir = Path(job_registry_temp.get_job(orphan)["results_dir"])
    (results_dir / "results.json").unlink()

    removed = job_registry_temp.cleanup_orphaned_jobs()

    assert removed == 1
    assert job_registry_temp.get_job(orphan) is None
    assert not results_dir.exists()  # remnants cleared too
    assert job_registry_temp.get_job(healthy) is not None


def test_orphan_sweep_spares_recently_completed_jobs(job_registry_temp):
    # A job that completed seconds ago may still be mid-write — never sweep.
    fresh = _make_completed_job(
        job_registry_temp, completed_at=int(time.time()), with_results=False
    )
    results_dir = Path(job_registry_temp.get_job(fresh)["results_dir"])
    assert not (results_dir / "results.json").exists()

    assert job_registry_temp.cleanup_orphaned_jobs() == 0
    assert job_registry_temp.get_job(fresh) is not None


def test_orphan_sweep_ignores_non_completed_jobs(job_registry_temp):
    queued = job_registry_temp.create_job(
        user_id=UUID_USER, filename="doc.pdf", pdf_path="/nonexistent/doc.pdf"
    )
    assert job_registry_temp.cleanup_orphaned_jobs() == 0
    assert job_registry_temp.get_job(queued) is not None


@pytest.fixture
def api_server_module(job_registry_temp, fence_cache_root, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "API_AUTH_MODE", "legacy_header", raising=False)
    sys.modules.pop("api_server", None)
    import api_server

    return api_server


def test_reupload_of_deduped_document_restores_missing_source_pdf(
    api_server_module, tmp_path, monkeypatch
):
    """The stuck loop: source PDF reaped from /tmp → exports say "re-upload"
    → dedup returns the same document. The heal must write the re-uploaded
    bytes back to the hash-derived path every reference points at."""
    from fastapi.testclient import TestClient
    import hashlib

    from config import cfg

    # cfg is a frozen dataclass — monkeypatch.setattr would raise.
    orig_tmp_dir = cfg.PDF_TMP_DIR
    object.__setattr__(cfg, "PDF_TMP_DIR", str(tmp_path / "fence_pdfs"))
    try:
        pdf_bytes = b"%PDF-1.4\n% dedup heal test\n"
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        fake_doc = {
            "id": "33333333-3333-3333-3333-333333333333",
            "latest_job_id": "44444444-4444-4444-4444-444444444444",
            "job_status": "completed",
            "original_filename": "doc.pdf",
        }
        monkeypatch.setattr(
            api_server_module.db, "find_document_by_hash", lambda *_a: fake_doc
        )

        client = TestClient(api_server_module.app)
        resp = client.post(
            "/api/jobs",
            files={"pdf": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers={"X-User-Id": UUID_USER},
        )

        assert resp.status_code == 202
        assert resp.json()["status"] == "deduped"
        restored = Path(cfg.PDF_TMP_DIR) / UUID_USER / f"job_{pdf_hash[:16]}.pdf"
        assert restored.read_bytes() == pdf_bytes
    finally:
        object.__setattr__(cfg, "PDF_TMP_DIR", orig_tmp_dir)
