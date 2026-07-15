"""Guards against the UMT path resurrecting the dense-page skip.

The pipeline marks pages with >MAX_FENCE_LINES as measurement_method=
"skipped" and the auto export honours it (commit 3cca2bb). But the UMT
canvas seeds line_assignments from /page-vector-lines and autosaves them
into umt_state — so if that endpoint ignores the skip, one user edit on a
skipped page persists 18k+ noise assignments and the UMT export re-creates
the dense-page 504. Two layers of defense are tested here:

1. /page-vector-lines returns no auto_assignments for skipped pages (and
   flags measurement_skipped so the UI can explain why).
2. _umt_export_for_page caps how many saved assignments a page can export.
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


def _make_job(job_registry_temp, tmp_path, measurements: dict) -> str:
    """Completed one-page job whose PDF really contains the fence line, so
    coord-matching in /page-vector-lines can seed assignments if allowed."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line((100, 100), (460, 100), color=(0, 0, 0), width=1.0)
    pdf_path = tmp_path / "source.pdf"
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_ID,
        filename="source.pdf",
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
                    "measurements": measurements,
                    "legend_entries": [],
                    "instances": [],
                }
            ],
            "non_fence_pages": [],
            "per_page_scale_info": {"page_1": {"verified_scale": 360}},
            "element_details": {},
        },
    )
    return job_id


FENCE_LINE = {"start": [100, 100], "end": [460, 100], "length_pts": 360, "layer": ""}


def test_page_vector_lines_seeds_assignments_when_not_skipped(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    job_id = _make_job(job_registry_temp, tmp_path, {
        "all_fence_lines": [FENCE_LINE],
        "layer_to_category": {},
        "measurement_method": "llm_guided",
    })
    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["measurement_skipped"] is False
    assert body["auto_assignments"], "expected the fence line to be seeded"


def test_page_vector_lines_honours_skipped_pages(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    job_id = _make_job(job_registry_temp, tmp_path, {
        "all_fence_lines": [FENCE_LINE],
        "layer_to_category": {},
        "measurement_method": "skipped",
        "skip_reason": "18000 lines exceeds limit of 5000",
    })
    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_assignments"] == {}
    assert body["measurement_skipped"] is True
    assert "exceeds limit" in body["skip_reason"]
    # Manual drawing must still work: the page's vector lines are returned.
    assert body["lines"]


def test_umt_export_caps_pathological_assignment_counts(api_server_module):
    n = api_server_module.MAX_EXPORT_LINES_PER_PAGE + 1000
    vlines = [
        {"start": [0, float(i)], "end": [500, float(i)], "length_pts": 500,
         "layer": ""}
        for i in range(n)
    ]
    page_state = {
        "categories": {"Fence": {"indicator": "", "keyword": "Fence",
                                 "color": [0, 255, 0]}},
        "line_assignments": {str(i): "Fence" for i in range(n)},
        "user_drawn_lines": [],
    }
    cats, pa, auto_lines, user_drawn = api_server_module._umt_export_for_page(
        page_state, vlines)
    assert len(auto_lines) == api_server_module.MAX_EXPORT_LINES_PER_PAGE
    assert len(pa) == api_server_module.MAX_EXPORT_LINES_PER_PAGE
