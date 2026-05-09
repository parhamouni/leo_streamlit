"""Phase 10.1 — Cross-user authorization tests.

Verifies every job-scoped endpoint with an ownership check rejects
requests from a different user (403/404, never 200/302). Uses
`X-User-Id` auth with `FENCE_API_AUTH_MODE=legacy_header` so the
tests don't need real Supabase JWTs.

Postgres-backed routes (`/api/me`, `/api/documents*`) are JWT-only —
they enforce ownership through `db.get_document(document_id, user_id)`
(every read is filtered by `user_id`) and aren't reachable in
legacy_header mode at all. Covered separately when we wire JWT
mocking.
"""
from __future__ import annotations

import os
import sys

# Force legacy_header BEFORE config / api_server import. Both modules
# read this once at import time.
os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"

import pytest


USER_A = "11111111-1111-1111-1111-111111111111"
USER_B = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def app_client(job_registry_temp, fence_cache_root, monkeypatch):
    """FastAPI TestClient with the temp job_registry wired in."""
    # Belt-and-suspenders: in case config was imported before this fixture
    # ran, force the runtime value too.
    import config as cfg
    monkeypatch.setattr(cfg, "API_AUTH_MODE", "legacy_header", raising=False)

    # Drop any prior api_server import so Depends() resolves with the
    # patched config / job_registry refs.
    sys.modules.pop("api_server", None)
    import api_server  # noqa: F401

    from fastapi.testclient import TestClient
    return TestClient(api_server.app)


@pytest.fixture
def job_a(job_registry_temp, tmp_path):
    """Completed job owned by USER_A, with a 1-page PDF + minimal
    results.json on disk. Enough for every read endpoint to get past
    the ownership check before hitting whatever it returns."""
    import fitz

    pdf_path = tmp_path / "owned_by_a.pdf"
    doc = fitz.open()
    doc.new_page()  # blank A4 page
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_A,
        filename="owned_by_a.pdf",
        pdf_path=str(pdf_path),
    )
    job_registry_temp.update_job(
        job_id,
        status="completed",
        completed_at=1700000000,
        total_pages=1,
        fence_count=1,
        non_fence_count=0,
    )
    job_registry_temp.save_results(
        job_id,
        {
            "fence_pages": [
                {
                    "page_idx": 0,
                    "page_num": 1,
                    "scale_info": {"verified_scale": 360},
                    "measurements": {},
                    "legend_entries": [],
                    "instances": [],
                }
            ],
            "non_fence_pages": [],
        },
    )

    return job_id


# (method, path_template, body_factory). body_factory is None for
# methods without a body. Path templates use {job_id}.
ENDPOINTS = [
    ("GET", "/api/jobs/{job_id}", None),
    ("GET", "/api/jobs/{job_id}/results", None),
    ("GET", "/api/jobs/{job_id}/highlighted-pdf", None),
    ("GET", "/api/jobs/{job_id}/page-image/1", None),
    ("GET", "/api/jobs/{job_id}/page-vector-lines/1", None),
    ("GET", "/api/jobs/{job_id}/page-vector-lines/1/smart-assign", None),
    ("GET", "/api/jobs/{job_id}/measurement-pdf", None),
    ("GET", "/api/jobs/{job_id}/measurement-excel", None),
    ("GET", "/api/jobs/{job_id}/measurement-summary", None),
    ("GET", "/api/jobs/{job_id}/umt-state", None),
    (
        "PUT",
        "/api/jobs/{job_id}/umt-state/1",
        lambda: {"categories": {}, "line_assignments": {}, "user_drawn_lines": []},
    ),
    ("DELETE", "/api/jobs/{job_id}/umt-state/1", None),
    # `/progress` is an SSE long-poll — TestClient.stream would be needed.
    # Same ownership check fires synchronously before the generator runs;
    # covered by the GET /api/jobs/{id} path above.
    ("DELETE", "/api/jobs/{job_id}", None),
]


@pytest.mark.parametrize(
    "method,path,body_factory",
    ENDPOINTS,
    ids=[f"{m} {p}" for m, p, _ in ENDPOINTS],
)
def test_user_b_cannot_access_user_a_resources(
    app_client, job_a, method, path, body_factory
):
    """Cross-user attempt: USER_B requests USER_A's resources → 403/404."""
    url = path.format(job_id=job_a)
    headers = {"X-User-Id": USER_B}
    body = body_factory() if body_factory else None

    resp = app_client.request(method, url, headers=headers, json=body)

    assert resp.status_code in (403, 404), (
        f"{method} {url} as USER_B returned {resp.status_code}; "
        f"expected 403/404. body: {resp.text[:300]}"
    )


def test_unknown_job_returns_404(app_client):
    """A job_id that doesn't exist returns 404 even for a valid user."""
    resp = app_client.get(
        "/api/jobs/99999999-9999-9999-9999-999999999999",
        headers={"X-User-Id": USER_A},
    )
    assert resp.status_code == 404


def test_anonymous_cannot_access_owned_resources(app_client, job_a):
    """No X-User-Id → user_id defaults to 'anonymous' — must not match USER_A."""
    resp = app_client.get(f"/api/jobs/{job_a}")
    assert resp.status_code in (403, 404)


def test_user_a_can_access_own_job(app_client, job_a):
    """Sanity: ownership check lets the real owner through."""
    resp = app_client.get(f"/api/jobs/{job_a}", headers={"X-User-Id": USER_A})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == USER_A


def test_list_jobs_filters_by_user(app_client, job_a):
    """USER_A's job list contains the job; USER_B's does not."""
    resp_a = app_client.get("/api/jobs", headers={"X-User-Id": USER_A})
    assert resp_a.status_code == 200
    a_ids = {j["job_id"] for j in resp_a.json().get("jobs", [])}
    assert job_a in a_ids

    resp_b = app_client.get("/api/jobs", headers={"X-User-Id": USER_B})
    assert resp_b.status_code == 200
    b_ids = {j["job_id"] for j in resp_b.json().get("jobs", [])}
    assert job_a not in b_ids


def test_healthz_is_public(app_client):
    """The health-check endpoint should not require auth."""
    resp = app_client.get("/api/healthz")
    assert resp.status_code == 200


def test_jwt_only_endpoints_reject_legacy_header(app_client):
    """Endpoints behind `require_supabase_jwt` must NOT accept X-User-Id —
    they only honor a valid Supabase Bearer token."""
    for path in ("/api/me", "/api/documents"):
        resp = app_client.get(path, headers={"X-User-Id": USER_A})
        # 401 (no Bearer) is the canonical answer; 403 also acceptable if
        # the JWT verifier classifies missing-creds as forbidden.
        assert resp.status_code in (401, 403), (
            f"GET {path} with X-User-Id returned {resp.status_code}; "
            "JWT-only endpoints must reject legacy header auth."
        )
