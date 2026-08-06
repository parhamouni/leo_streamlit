"""Stage-3 guards: vision verification of layer candidates.

Verdicts are advisory and per-(page, layer): "not_fence" hides a layer from
auto-colouring and summary totals; absent/unsure verdicts never reject; a
failed vision call yields None (no verdicts), never a rejection.
"""
from __future__ import annotations

import os
import sys

import fitz
import pytest

os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"

from utils_ade.vision_verify import (  # noqa: E402
    layer_rejected,
    verify_page_layers,
)

USER_ID = "11111111-1111-1111-1111-111111111111"


class FakeVisionClient:
    """Mimics the OpenAI responses API surface verify_page_layers uses."""

    def __init__(self, output_text):
        self._out = output_text
        self.calls = []

        outer = self

        class _Responses:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if isinstance(outer._out, Exception):
                    raise outer._out

                class R:
                    output_text = outer._out
                return R()

        self.responses = _Responses()


def _page_with_layers():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    return doc, page


MEAS = {
    "fence_layers": ["V-FENC", "A-WALL-PATT"],
    "all_fence_lines": [
        {"start": [10, 10], "end": [200, 10], "length_pts": 190, "layer": "V-FENC"},
        {"start": [10, 50], "end": [12, 50], "length_pts": 2, "layer": "A-WALL-PATT"},
    ],
}


def test_verify_parses_numbered_verdicts():
    doc, page = _page_with_layers()
    client = FakeVisionClient(
        '{"verdicts": {"1": {"verdict": "fence", "reason": "long runs"},'
        ' "2": {"verdict": "not_fence", "reason": "hatch"}}}')
    out = verify_page_layers(page, MEAS, client, model="test-model")
    assert out["verdicts"]["V-FENC"]["verdict"] == "fence"
    assert out["verdicts"]["A-WALL-PATT"]["verdict"] == "not_fence"
    assert out["model"] == "test-model"
    # one image + prompt went out
    content = client.calls[0]["input"][0]["content"]
    assert any(c.get("type") == "input_image" for c in content)
    doc.close()


def test_verify_failure_returns_none_not_rejection():
    doc, page = _page_with_layers()
    out = verify_page_layers(page, MEAS, FakeVisionClient(RuntimeError("down")))
    assert out is None
    doc.close()


def test_verify_invalid_verdict_values_dropped():
    doc, page = _page_with_layers()
    client = FakeVisionClient(
        '{"verdicts": {"1": {"verdict": "maybe"}, "2": {"verdict": "unsure"}}}')
    out = verify_page_layers(page, MEAS, client)
    assert "V-FENC" not in out["verdicts"]          # invalid value dropped
    assert out["verdicts"]["A-WALL-PATT"]["verdict"] == "unsure"
    doc.close()


def test_layer_rejected_semantics():
    m = {"vision_verdicts": {"verdicts": {
        "JUNK": {"verdict": "not_fence"},
        "OK": {"verdict": "fence"},
        "MEH": {"verdict": "unsure"},
    }}}
    assert layer_rejected(m, "JUNK") is True
    assert layer_rejected(m, "OK") is False
    assert layer_rejected(m, "MEH") is False
    assert layer_rejected(m, "ABSENT") is False
    assert layer_rejected({}, "ANY") is False


# --- endpoint honours verdicts ---------------------------------------------

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


def test_rejected_layer_gets_no_auto_assignment(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line((100, 100), (460, 100), color=(0, 0, 0), width=1.0)
    page.draw_line((100, 200), (460, 200), color=(0, 0, 0), width=1.0)
    pdf_path = tmp_path / "src.pdf"
    doc.save(str(pdf_path))
    doc.close()

    job_id = job_registry_temp.create_job(
        user_id=USER_ID, filename="src.pdf", pdf_path=str(pdf_path))
    job_registry_temp.update_job(
        job_id, status="completed", completed_at=1700000000,
        total_pages=1, fence_count=1, non_fence_count=0)
    job_registry_temp.save_results(job_id, {
        "fence_pages": [{
            "page_idx": 0, "page_num": 1,
            "scale_info": {"verified_scale": 360},
            "measurements": {
                "all_fence_lines": [
                    {"start": [100, 100], "end": [460, 100],
                     "length_pts": 360, "layer": "GOOD"},
                    {"start": [100, 200], "end": [460, 200],
                     "length_pts": 360, "layer": "JUNK"},
                ],
                "layer_to_category": {},
                "measurement_method": "llm_guided",
                "vision_verdicts": {"verdicts": {
                    "JUNK": {"verdict": "not_fence", "reason": "hatch"},
                }},
            },
            "legend_entries": [], "instances": [],
        }],
        "non_fence_pages": [], "element_details": {},
    })

    resp = app_client.get(
        f"/api/jobs/{job_id}/page-vector-lines/1",
        headers={"X-User-Id": USER_ID})
    assert resp.status_code == 200
    body = resp.json()
    # Exactly one line auto-assigned (the GOOD one); JUNK suppressed.
    assert len(body["auto_assignments"]) == 1
    assigned_idx = next(iter(body["auto_assignments"]))
    ln = next(l for l in body["lines"] if str(l["idx"]) == assigned_idx)
    assert ln["start"][1] == 100.0  # the y=100 line is GOOD's

    # Summary skips the rejected layer too.
    s = api_server_module._summary_for_page(
        0, job_registry_temp.load_results(job_id)["fence_pages"][0], None)
    assert s["per_category"]["Auto-detected"]["auto"] == 1
