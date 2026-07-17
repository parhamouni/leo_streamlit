from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def completed_export_job(job_registry_temp, tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% test placeholder\n")

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
                    "measurements": {
                        "all_fence_lines": [
                            {
                                "start": [0, 0],
                                "end": [360, 0],
                                "length_pts": 360,
                                "layer": "FENCE",
                            }
                        ],
                        "layer_to_category": {"FENCE": "Fence"},
                    },
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


@pytest.fixture
def skipped_export_job(job_registry_temp, tmp_path):
    """A completed job whose fence pages have no measurable lines.

    Mirrors a report whose 'fence' pages carry no CAD fence layers and
    had measurement skipped — the export should report 'nothing to
    export' (422), not a server fault (500)."""
    pdf_path = tmp_path / "drainage.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% test placeholder\n")

    job_id = job_registry_temp.create_job(
        user_id=USER_ID,
        filename="drainage.pdf",
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
                    "measurements": {
                        "all_fence_lines": [],
                        "layer_to_category": {},
                        "measurement_method": "skipped",
                        "skip_reason": "non-layer suggestions disabled and no fence layers found",
                    },
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


def _poll_measurement_pdf(client, job_id, timeout_s=10.0):
    """GET measurement-pdf until the background build resolves (non-202)."""
    import time

    deadline = time.time() + timeout_s
    while True:
        resp = client.get(
            f"/api/jobs/{job_id}/measurement-pdf",
            headers={"X-User-Id": USER_ID},
        )
        if resp.status_code != 202 or time.time() > deadline:
            return resp
        time.sleep(0.02)


def test_measurement_excel_no_lines_returns_422(
    api_server_module, app_client, skipped_export_job
):
    resp = app_client.get(
        f"/api/jobs/{skipped_export_job}/measurement-excel",
        headers={"X-User-Id": USER_ID},
    )
    assert resp.status_code == 422
    assert "No measurements to export" in resp.text


def test_measurement_pdf_no_lines_returns_422_without_subprocess(
    api_server_module, app_client, skipped_export_job, monkeypatch
):
    calls = []

    def fake_helper(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        api_server_module,
        "_generate_measurement_pdf_in_subprocess",
        fake_helper,
    )

    resp = _poll_measurement_pdf(app_client, skipped_export_job)
    assert resp.status_code == 422
    assert "No measurements to export" in resp.text
    # The expensive overlay subprocess must be short-circuited.
    assert calls == []


def test_measurement_pdf_builds_artifact_via_subprocess_helper(
    api_server_module, app_client, completed_export_job, monkeypatch
):
    calls = []

    def fake_helper(**kwargs):
        calls.append(kwargs)
        Path(kwargs["out_path"]).write_bytes(b"%PDF isolated")

    monkeypatch.setattr(
        api_server_module,
        "_generate_measurement_pdf_in_subprocess",
        fake_helper,
    )

    first = app_client.get(
        f"/api/jobs/{completed_export_job}/measurement-pdf",
        headers={"X-User-Id": USER_ID},
    )
    assert first.status_code == 202  # build started in the background

    resp = _poll_measurement_pdf(app_client, completed_export_job)
    assert resp.status_code == 200
    assert resp.content == b"%PDF isolated"
    assert "source_measurement.pdf" in resp.headers.get("content-disposition", "")
    assert len(calls) == 1

    # Cached artifact: later downloads (and their ranged chunks) must not
    # trigger a rebuild.
    again = app_client.get(
        f"/api/jobs/{completed_export_job}/measurement-pdf",
        headers={"X-User-Id": USER_ID},
    )
    assert again.status_code == 200
    assert len(calls) == 1


def test_measurement_pdf_umt_edit_invalidates_cached_artifact(
    api_server_module, app_client, completed_export_job, monkeypatch
):
    builds = []

    def fake_helper(**kwargs):
        builds.append(kwargs)
        Path(kwargs["out_path"]).write_bytes(f"%PDF build {len(builds)}".encode())

    monkeypatch.setattr(
        api_server_module,
        "_generate_measurement_pdf_in_subprocess",
        fake_helper,
    )
    # With saved UMT edits the export state re-extracts vector lines via the
    # vector worker; stub it — the fixture's source.pdf is a placeholder.
    monkeypatch.setattr(
        api_server_module,
        "_extract_export_vector_lines",
        lambda **_kw: {
            0: [{"start": [0, 0], "end": [720, 0], "length_pts": 720, "layer": "FENCE"}]
        },
    )

    resp = _poll_measurement_pdf(app_client, completed_export_job)
    assert resp.status_code == 200
    assert len(builds) == 1

    put = app_client.put(
        f"/api/jobs/{completed_export_job}/umt-state/1",
        json={
            "categories": {"Fence": {"keyword": "Fence", "color": [0, 255, 0]}},
            "line_assignments": {"0": "Fence"},
            "user_drawn_lines": [],
        },
        headers={"X-User-Id": USER_ID},
    )
    assert put.status_code == 200

    resp = _poll_measurement_pdf(app_client, completed_export_job)
    assert resp.status_code == 200
    assert len(builds) == 2


def test_measurement_pdf_timeout_returns_504_and_health_stays_available(
    api_server_module, app_client, completed_export_job, monkeypatch
):
    def fake_helper(**_kwargs):
        raise TimeoutError("measurement worker timeout")

    monkeypatch.setattr(
        api_server_module,
        "_generate_measurement_pdf_in_subprocess",
        fake_helper,
    )

    resp = _poll_measurement_pdf(app_client, completed_export_job)
    health = app_client.get("/api/healthz")

    assert resp.status_code == 504
    assert "measurement worker timeout" in resp.text
    assert health.status_code == 200

    # The failure is surfaced once, then cleared — the next request must
    # start a fresh build rather than replaying the stale error.
    retry_first = app_client.get(
        f"/api/jobs/{completed_export_job}/measurement-pdf",
        headers={"X-User-Id": USER_ID},
    )
    assert retry_first.status_code == 202


def test_measurement_pdf_worker_failure_returns_500(
    api_server_module, app_client, completed_export_job, monkeypatch
):
    def fake_helper(**_kwargs):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(
        api_server_module,
        "_generate_measurement_pdf_in_subprocess",
        fake_helper,
    )

    resp = _poll_measurement_pdf(app_client, completed_export_job)

    assert resp.status_code == 500
    assert "worker exploded" in resp.text


def test_measurement_pdf_helper_leaves_artifact_at_out_path(
    api_server_module, tmp_path, monkeypatch
):
    worker = tmp_path / "measurement_pdf_worker.py"
    worker.write_text("# placeholder\n")
    monkeypatch.setattr(api_server_module, "_MEASUREMENT_PDF_WORKER", worker)

    def fake_run(_cmd, *, input, **_kwargs):
        task = json.loads(input)
        out_path = Path(task["out_path"])
        # Mirror the real worker: atomic .tmp + rename.
        tmp = Path(str(out_path) + ".tmp")
        tmp.write_bytes(b"%PDF worker output")
        tmp.rename(out_path)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "out_path": str(out_path), "wall_s": 0.01}),
            stderr="",
        )

    monkeypatch.setattr(api_server_module.subprocess, "run", fake_run)

    artifact = tmp_path / "measurement.pdf"
    api_server_module._generate_measurement_pdf_in_subprocess(
        pdf_path=str(tmp_path / "source.pdf"),
        fence_pages=[],
        line_assignments={},
        user_drawn_lines={},
        page_categories={},
        uploaded_pdf_name="source.pdf",
        out_path=str(artifact),
    )

    assert artifact.read_bytes() == b"%PDF worker output"
    assert not Path(str(artifact) + ".tmp").exists()


def test_export_state_with_umt_edits_uses_vector_worker(
    api_server_module, completed_export_job, monkeypatch
):
    from backend.app import umt_state
    import job_registry

    umt_state.save(
        completed_export_job,
        {
            "version": 1,
            "pages": {
                "page_1": {
                    "categories": {
                        "Fence": {
                            "indicator": "",
                            "keyword": "Fence",
                            "color": [0, 255, 0],
                        }
                    },
                    "line_assignments": {"0": "Fence"},
                    "user_drawn_lines": [],
                }
            },
        },
    )

    calls = []

    def fake_extract(**kwargs):
        calls.append(kwargs)
        return {
            0: [
                {
                    "start": [0, 0],
                    "end": [720, 0],
                    "length_pts": 720,
                    "layer": "FENCE",
                }
            ]
        }

    monkeypatch.setattr(api_server_module, "_extract_export_vector_lines", fake_extract)

    job = job_registry.get_job(completed_export_job)
    results = job_registry.load_results(completed_export_job)
    state = api_server_module._build_export_state(
        completed_export_job,
        results,
        job["pdf_path"],
    )

    assert calls
    assert calls[0]["page_indices"] == [0]
    assert state[5]["page_1"][0]["length_pts"] == 720


def test_vector_worker_timeout_maps_to_excel_504(
    api_server_module, app_client, completed_export_job, monkeypatch
):
    from backend.app import umt_state

    umt_state.save(
        completed_export_job,
        {
            "version": 1,
            "pages": {
                "page_1": {
                    "categories": {"Fence": {"keyword": "Fence", "color": [0, 255, 0]}},
                    "line_assignments": {"0": "Fence"},
                    "user_drawn_lines": [],
                }
            },
        },
    )

    def fake_extract(**_kwargs):
        raise api_server_module.ExportWorkerTimeout("vector timeout")

    monkeypatch.setattr(api_server_module, "_extract_export_vector_lines", fake_extract)

    resp = app_client.get(
        f"/api/jobs/{completed_export_job}/measurement-excel",
        headers={"X-User-Id": USER_ID},
    )

    assert resp.status_code == 504
    assert "vector timeout" in resp.text


def test_architectural_scale_converts_to_points_per_foot(api_server_module):
    import exports

    # 1" = 30' means 360 real inches per drawing inch. Since a PDF inch
    # is 72 points, one real foot is 72 / 30 = 2.4 PDF points.
    assert api_server_module._scale_inches_to_points_per_foot(360) == pytest.approx(2.4)
    assert exports._scale_inches_to_points_per_foot(360) == pytest.approx(2.4)
