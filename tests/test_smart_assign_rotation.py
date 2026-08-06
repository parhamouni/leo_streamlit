"""Smart Assign must work on rotated pages.

Instance bboxes are persisted in DISPLAY space (ops/analysis_worker._xform
transforms words through the rotation matrix before instance-finding). Until
2026-08 the smart-assign endpoint multiplied them by page.rotation_matrix
AGAIN, pushing every indicator box off-page on rotated sheets (254/801 pages
on the live box) — proximity voting found zero candidates and the feature
silently degraded to the strict layer-token fallback.

The rotated test here fails on the pre-fix code (decided_by == "none") and
passes post-fix (decided_by == "indicator-vote"); the rotation-0 companion
pins the identity behaviour that was never broken.
"""
from __future__ import annotations

import os
import sys

import fitz
import pytest

os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"

USER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def api_server_module(job_registry_temp, fence_cache_root, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "API_AUTH_MODE", "legacy_header", raising=False)
    sys.modules.pop("api_server", None)
    import api_server

    return api_server


@pytest.fixture
def app_client(api_server_module):
    from fastapi.testclient import TestClient

    return TestClient(api_server_module.app)


def _make_job(job_registry_temp, tmp_path, rotation: int) -> str:
    """One-page job: 4 clustered lines on OCG layer V-FENC (drawn in native
    space, rotation applied after), one F1 indicator instance persisted in
    display space exactly as ops/analysis_worker does (native * rotation
    matrix)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    ocg = doc.add_ocg("V-FENC")
    for i in range(4):
        y = 300.0 + i * 10
        page.draw_line((200, y), (400, y), color=(0, 0, 0), width=1.0, oc=ocg)
    if rotation:
        page.set_rotation(rotation)
    # Indicator bbox around the cluster centre (native ~(300, 315)),
    # transformed native → display like analysis_worker._xform.
    disp = fitz.Rect(290, 305, 310, 325) * page.rotation_matrix
    disp.normalize()

    pdf_path = tmp_path / f"rot{rotation}.pdf"
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_ID, filename=pdf_path.name, pdf_path=str(pdf_path))
    job_registry_temp.update_job(
        job_id, status="completed", completed_at=1700000000,
        total_pages=1, fence_count=1, non_fence_count=0)
    job_registry_temp.save_results(job_id, {
        "fence_pages": [{
            "page_idx": 0, "page_num": 1,
            "scale_info": {"verified_scale": 360},
            "measurements": {},
            "legend_entries": [{
                "indicator": "F1",
                "keyword": "CHAINLINK BARRIER",
                "description": "Chain link fence",
            }],
            "instances": [{
                "indicator": "F1",
                "x0": disp.x0, "y0": disp.y0, "x1": disp.x1, "y1": disp.y1,
            }],
        }],
        "non_fence_pages": [], "element_details": {},
    })
    return job_id


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_smart_assign_votes_by_indicator(
    api_server_module, app_client, job_registry_temp, tmp_path, rotation
):
    job_id = _make_job(job_registry_temp, tmp_path, rotation)
    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1/smart-assign",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    la = body["layer_assignments"].get("V-FENC")
    assert la is not None, body["layer_assignments"]
    assert la["decided_by"] == "indicator-vote", la
    assert la["category"] == "F1: CHAINLINK BARRIER"
    assert body["stats"]["by_indicator"] == 4
    assert set(body["assignments"].values()) == {"F1: CHAINLINK BARRIER"}
