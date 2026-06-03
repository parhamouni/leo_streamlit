from __future__ import annotations

import os
import sys

import pytest


USER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def api_server_module(job_registry_temp, fence_cache_root, monkeypatch):
    os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"
    import config as cfg

    monkeypatch.setattr(cfg, "API_AUTH_MODE", "legacy_header", raising=False)
    sys.modules.pop("api_server", None)
    import api_server

    return api_server


def test_sqlite_job_is_exposed_as_document_fallback(api_server_module, job_registry_temp):
    job_id = job_registry_temp.create_job(
        user_id=USER_ID,
        filename="prior-run.pdf",
        pdf_path="/tmp/prior-run.pdf",
    )
    job_registry_temp.update_job(
        job_id,
        status="completed",
        completed_at=1_700_000_100,
        total_pages=7,
        fence_count=3,
    )

    docs = api_server_module._merge_sqlite_fallback_documents(USER_ID, [])

    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == job_id
    assert doc["latest_job_id"] == job_id
    assert doc["original_filename"] == "prior-run.pdf"
    assert doc["job_status"] == "completed"
    assert doc["progress_percent"] == 100
    assert doc["source"] == "sqlite"


def test_sqlite_fallback_does_not_duplicate_postgres_document(
    api_server_module, job_registry_temp
):
    job_id = job_registry_temp.create_job(
        user_id=USER_ID,
        filename="mirrored.pdf",
        pdf_path="/tmp/mirrored.pdf",
    )
    pg_doc = {
        "id": "22222222-2222-2222-2222-222222222222",
        "latest_job_id": job_id,
        "created_at": "2026-06-03T00:00:00+00:00",
    }

    docs = api_server_module._merge_sqlite_fallback_documents(USER_ID, [pg_doc])

    assert docs == [pg_doc]


def test_documents_endpoint_includes_sqlite_fallback(
    api_server_module, job_registry_temp, monkeypatch
):
    from fastapi.testclient import TestClient

    job_id = job_registry_temp.create_job(
        user_id=USER_ID,
        filename="sqlite-only.pdf",
        pdf_path="/tmp/sqlite-only.pdf",
    )
    job_registry_temp.update_job(job_id, status="completed", completed_at=1)
    monkeypatch.setattr(api_server_module.db, "list_documents", lambda user_id: [])
    api_server_module.app.dependency_overrides[
        api_server_module.require_supabase_jwt
    ] = lambda: USER_ID
    try:
        with TestClient(api_server_module.app) as client:
            resp = client.get("/api/documents")
    finally:
        api_server_module.app.dependency_overrides.clear()

    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert [d["id"] for d in docs] == [job_id]
    assert docs[0]["latest_job_id"] == job_id


def test_delete_job_removes_postgres_only_old_job(
    api_server_module, job_registry_temp, monkeypatch
):
    import asyncio

    job_id = "33333333-3333-3333-3333-333333333333"
    deleted = []

    def fake_delete(pg_job_id: str, user_id: str) -> bool:
        deleted.append((pg_job_id, user_id))
        return True

    monkeypatch.setattr(
        api_server_module,
        "_delete_postgres_document_for_job",
        fake_delete,
    )

    resp = asyncio.run(api_server_module.delete_job(job_id, user_id=USER_ID))

    assert resp == {"status": "deleted"}
    assert deleted == [(job_id, USER_ID)]
