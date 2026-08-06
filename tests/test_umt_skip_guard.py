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


def test_sub_10pt_lines_get_no_auto_assignment(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    """Canvas auto-assignments share the exports' 10pt floor: sub-10pt
    segments are hatch/pattern fill (2026-08-05 'random lines' report),
    and the Excel/PDF exports never list them either."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line((100, 100), (460, 100), color=(0, 0, 0), width=1.0)
    page.draw_line((100, 200), (105, 200), color=(0, 0, 0), width=1.0)  # 5pt
    pdf_path = tmp_path / "source2.pdf"
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_ID, filename="source2.pdf", pdf_path=str(pdf_path))
    job_registry_temp.update_job(
        job_id, status="completed", completed_at=1700000000,
        total_pages=1, fence_count=1, non_fence_count=0)
    job_registry_temp.save_results(job_id, {
        "fence_pages": [{
            "page_idx": 0, "page_num": 1,
            "scale_info": {"verified_scale": 360},
            "measurements": {
                "all_fence_lines": [
                    FENCE_LINE,
                    {"start": [100, 200], "end": [105, 200],
                     "length_pts": 5, "layer": ""},
                ],
                "layer_to_category": {},
                "measurement_method": "llm_guided",
            },
            "legend_entries": [], "instances": [],
        }],
        "non_fence_pages": [], "element_details": {},
    })

    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_assignments_suppressed"] is False
    # Only the 360pt line is seeded; the 5pt fragment is skipped.
    assigned = set(body["auto_assignments"].values())
    assert assigned == {"Auto-detected"}
    long_idx = [str(ln["idx"]) for ln in body["lines"]
                if ln["length_pts"] > 100]
    short_idx = [str(ln["idx"]) for ln in body["lines"]
                 if ln["length_pts"] < 10]
    assert all(i in body["auto_assignments"] for i in long_idx)
    assert all(i not in body["auto_assignments"] for i in short_idx)


def test_unmapped_fallback_bucket_suppressed_above_cap(
    api_server_module, app_client, job_registry_temp, tmp_path, monkeypatch
):
    """With layer_to_category empty, an Auto-detected bucket bigger than
    AUTO_ASSIGN_FALLBACK_CAP is suppressed (returned with the flag) instead
    of painting the canvas with non-fence linework."""
    monkeypatch.setattr(api_server_module, "AUTO_ASSIGN_FALLBACK_CAP", 3)

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    lines = []
    for i in range(6):
        y = 100.0 + i * 20
        page.draw_line((100, y), (200, y), color=(0, 0, 0), width=1.0)
        lines.append({"start": [100, y], "end": [200, y],
                      "length_pts": 100, "layer": ""})
    pdf_path = tmp_path / "source3.pdf"
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_ID, filename="source3.pdf", pdf_path=str(pdf_path))
    job_registry_temp.update_job(
        job_id, status="completed", completed_at=1700000000,
        total_pages=1, fence_count=1, non_fence_count=0)
    job_registry_temp.save_results(job_id, {
        "fence_pages": [{
            "page_idx": 0, "page_num": 1,
            "scale_info": {"verified_scale": 360},
            "measurements": {
                "all_fence_lines": lines,
                "layer_to_category": {},
                "measurement_method": "llm_guided",
            },
            "legend_entries": [], "instances": [],
        }],
        "non_fence_pages": [], "element_details": {},
    })

    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_assignments_suppressed"] is True
    assert body["auto_assignments"] == {}
    assert "Auto-detected" not in body["auto_categories"]
    assert "cap" in (body["suppressed_reason"] or "")
    # Manual measurement still possible: lines are served in full.
    assert len(body["lines"]) == 6

    # A mapped layer_to_category is never suppressed — the mapping means
    # the pipeline actually identified fence layers.
    job2 = _make_job(job_registry_temp, tmp_path, {
        "all_fence_lines": [dict(FENCE_LINE, layer="V-FENC")],
        "layer_to_category": {"V-FENC": "Chain Link"},
        "measurement_method": "llm_guided",
    })
    monkeypatch.setattr(api_server_module, "AUTO_ASSIGN_FALLBACK_CAP", 0)
    resp = app_client.get(
        f"/api/jobs/{job2}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID},
    )
    body = resp.json()
    assert body["auto_assignments_suppressed"] is False
    assert set(body["auto_assignments"].values()) == {"Chain Link"}


def test_summary_floors_and_caps_auto_detected(api_server_module, monkeypatch):
    """measurement-summary's auto branch mirrors the canvas guard: sub-10pt
    fragments don't count, and an over-cap unmapped bucket reports no
    phantom Auto-detected footage."""
    fp = {
        "scale_info": {"verified_scale": 360},
        "measurements": {
            "all_fence_lines": [
                {"start": [0, 0], "end": [50, 0], "length_pts": 50, "layer": ""},
                {"start": [0, 10], "end": [60, 10], "length_pts": 60, "layer": ""},
                {"start": [0, 20], "end": [5, 20], "length_pts": 5, "layer": ""},
            ],
            "layer_to_category": {},
        },
    }
    s = api_server_module._summary_for_page(0, fp, None)
    assert s["per_category"]["Auto-detected"]["auto"] == 2  # 5pt dropped
    assert s["per_category"]["Auto-detected"]["pts"] == 110.0

    monkeypatch.setattr(api_server_module, "AUTO_ASSIGN_FALLBACK_CAP", 1)
    s = api_server_module._summary_for_page(0, fp, None)
    assert "Auto-detected" not in s["per_category"]

    # Mapped categories are unaffected by the cap.
    fp_mapped = {
        "scale_info": {"verified_scale": 360},
        "measurements": {
            "all_fence_lines": [
                {"start": [0, 0], "end": [50, 0], "length_pts": 50,
                 "layer": "V-FENC"},
                {"start": [0, 10], "end": [60, 10], "length_pts": 60,
                 "layer": "V-FENC"},
            ],
            "layer_to_category": {"V-FENC": "Chain Link"},
        },
    }
    s = api_server_module._summary_for_page(0, fp_mapped, None)
    assert s["per_category"]["Chain Link"]["auto"] == 2


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
