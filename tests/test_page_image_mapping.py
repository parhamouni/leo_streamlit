"""page-image must render the RIGHT page of the highlighted PDF.

Guards the 2026-08-05 "measurement canvas is drawing lines randomly"
regression: the endpoint mapped original page → highlighted-PDF index by
naive position in fence_pages, but the highlighted PDF gained a prepended
legend page (9a43ea6) and blank-page skipping (0ecd24c) — so the canvas
background was the legend sheet (first fence page) or the *previous* fence
sheet (every later page), with correct vector lines drawn over the wrong
drawing.

Covers `_compute_highlighted_page_map` (persisted kept-list, derivation via
the worker's `_has_drawable`, legend text sniff, both builder shapes,
consistency-check bail-out) and the endpoint end-to-end with distinct
solid-colour pages so a wrong-sheet regression fails on pixel colour, not
just on metadata.
"""
from __future__ import annotations

import os
import sys

import fitz
import pytest

os.environ["FENCE_API_AUTH_MODE"] = "legacy_header"

USER_ID = "11111111-1111-1111-1111-111111111111"

LEGEND_TITLE = "Fence Detection Overview — Legend"

# Distinct fill colours per original page (1-based) so the rendered PNG
# identifies which sheet was actually served.
PAGE_COLORS = {
    1: (1.0, 0.0, 0.0),
    2: (0.0, 0.8, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (1.0, 1.0, 0.0),
}


@pytest.fixture
def api_server_module(job_registry_temp, fence_cache_root, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "API_AUTH_MODE", "legacy_header", raising=False)
    sys.modules.pop("api_server", None)
    import api_server

    # The page-map cache is module-level and keyed by job_id + mtime; tests
    # in one session can collide on recycled tmp paths, so start clean.
    api_server._hl_page_map_cache.clear()
    return api_server


@pytest.fixture
def app_client(api_server_module):
    from fastapi.testclient import TestClient

    return TestClient(api_server_module.app)


def _blank_doc(n_pages: int, legend: bool) -> fitz.Document:
    doc = fitz.open()
    if legend:
        page = doc.new_page(width=612, height=792)
        page.insert_text((54, 72), LEGEND_TITLE, fontsize=18)
    for _ in range(n_pages):
        doc.new_page(width=612, height=792)
    return doc


def _fence_page(page_idx: int, drawable: bool) -> dict:
    fp = {
        "page_idx": page_idx,
        "page_num": page_idx + 1,
        "definitions": (
            [{"x0": 10, "y0": 10, "x1": 100, "y1": 50}] if drawable else []
        ),
        "instances": [],
        "keyword_matches": [],
        "legend_entries": [],
        "measurements": {},
        "scale_info": {},
    }
    return fp


def _save_hl(tmp_path, doc: fitz.Document, name: str = "highlighted.pdf"):
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# _compute_highlighted_page_map unit tests
# ---------------------------------------------------------------------------

def test_map_legend_all_drawable(api_server_module, tmp_path):
    """Worker shape: legend page shifts every fence page by +1."""
    hl = _save_hl(tmp_path, _blank_doc(3, legend=True))
    results = {"fence_pages": [_fence_page(i, True) for i in (2, 5, 7)]}
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m == {3: 2, 6: 3, 8: 4}


def test_map_blank_skip_drift(api_server_module, tmp_path):
    """A non-drawable page is dropped by the worker; later pages shift."""
    hl = _save_hl(tmp_path, _blank_doc(2, legend=True))
    results = {"fence_pages": [
        _fence_page(2, True), _fence_page(5, False), _fence_page(7, True),
    ]}
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m == {3: 2, 8: 3}
    assert 6 not in m


def test_map_no_legend_fallback_shape(api_server_module, tmp_path):
    """Inline-fallback / pre-legend PDFs: no legend page, no blank-skip —
    the unfiltered hypothesis must win when the drawable one mismatches."""
    hl = _save_hl(tmp_path, _blank_doc(3, legend=False))
    results = {"fence_pages": [
        _fence_page(2, True), _fence_page(5, False), _fence_page(7, True),
    ]}
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m == {3: 1, 6: 2, 8: 3}


def test_map_count_mismatch_returns_none(api_server_module, tmp_path):
    """Neither hypothesis reconciles with the actual page count → None,
    so the endpoint degrades to serving the original PDF."""
    hl = _save_hl(tmp_path, _blank_doc(5, legend=True))
    results = {"fence_pages": [_fence_page(2, True), _fence_page(7, True)]}
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m is None


def test_map_persisted_info_used_verbatim(api_server_module, tmp_path):
    """highlight_pdf_info from the pipeline wins over derivation — proven
    by a kept order derivation would never produce."""
    hl = _save_hl(tmp_path, _blank_doc(2, legend=True))
    results = {
        "fence_pages": [_fence_page(2, True), _fence_page(7, True)],
        "highlight_pdf_info": {
            "kept_page_indices": [7, 2],
            "legend_inserted": True,
            "builder": "worker",
        },
    }
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m == {8: 2, 3: 3}


def test_map_persisted_info_inconsistent_returns_none(api_server_module, tmp_path):
    """Persisted list that doesn't reconcile with the PDF is a red flag —
    don't fall back to guessing; serve originals."""
    hl = _save_hl(tmp_path, _blank_doc(4, legend=True))
    results = {
        "fence_pages": [_fence_page(2, True), _fence_page(7, True)],
        "highlight_pdf_info": {
            "kept_page_indices": [2, 7],
            "legend_inserted": True,
        },
    }
    m = api_server_module._compute_highlighted_page_map(results, hl)
    assert m is None


# ---------------------------------------------------------------------------
# Endpoint tests — colour-coded pages catch wrong-sheet serving
# ---------------------------------------------------------------------------

def _make_colour_job(job_registry_temp, tmp_path, *, fence_idx: list[int],
                     hl_kept_idx: list[int], legend: bool = True,
                     highlight_pdf_info: dict | None = None) -> str:
    """Completed 4-page job. Source pages are solid-filled with
    PAGE_COLORS; highlighted.pdf = optional legend + copies of the
    hl_kept_idx source pages (0-based)."""
    src = fitz.open()
    for pn in (1, 2, 3, 4):
        page = src.new_page(width=612, height=792)
        page.draw_rect(page.rect, color=None, fill=PAGE_COLORS[pn])
    src_path = tmp_path / "source.pdf"
    src.save(str(src_path))

    job_id = job_registry_temp.create_job(
        user_id=USER_ID, filename="source.pdf", pdf_path=str(src_path),
    )
    job_registry_temp.update_job(
        job_id, status="completed", completed_at=1700000000,
        total_pages=4, fence_count=len(fence_idx),
        non_fence_count=4 - len(fence_idx),
    )
    results = {
        "fence_pages": [_fence_page(i, True) for i in fence_idx],
        "non_fence_pages": [],
        "element_details": {},
    }
    if highlight_pdf_info is not None:
        results["highlight_pdf_info"] = highlight_pdf_info
    job_registry_temp.save_results(job_id, results)

    hl = fitz.open()
    if legend:
        page = hl.new_page(width=612, height=792)
        page.insert_text((54, 72), LEGEND_TITLE, fontsize=18)
    for i in hl_kept_idx:
        hl.insert_pdf(src, from_page=i, to_page=i)
    results_dir = job_registry_temp.get_job(job_id)["results_dir"]
    hl.save(os.path.join(results_dir, "highlighted.pdf"))
    hl.close()
    src.close()
    return job_id


def _center_color(png_bytes: bytes) -> tuple[int, int, int]:
    pix = fitz.Pixmap(png_bytes)
    r, g, b = pix.pixel(pix.width // 2, pix.height // 2)[:3]
    return (round(r / 255), round(g / 255), round(b / 255))


def _get_image(app_client, job_id: str, page_num: int, source: str = "auto"):
    return app_client.get(
        f"/api/jobs/{job_id}/page-image/{page_num}?dpi=72&source={source}",
        headers={"X-User-Id": USER_ID},
    )


def test_page_image_serves_the_right_highlighted_page(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    job_id = _make_colour_job(
        job_registry_temp, tmp_path, fence_idx=[1, 3], hl_kept_idx=[1, 3],
    )

    # Fence page 2 → highlighted PDF page 2 (after the legend), NOT the
    # legend sheet the buggy positional mapping used to serve.
    resp = _get_image(app_client, job_id, 2)
    assert resp.status_code == 200
    assert resp.headers["X-Image-Source"] == "highlighted"
    assert _center_color(resp.content) == (0, 1, 0)  # page 2 green

    # Fence page 4 → highlighted PDF page 3 (used to get page 2's sheet).
    resp = _get_image(app_client, job_id, 4)
    assert resp.headers["X-Image-Source"] == "highlighted"
    assert _center_color(resp.content) == (1, 1, 0)  # page 4 yellow

    # Non-fence page 3 → original PDF.
    resp = _get_image(app_client, job_id, 3)
    assert resp.headers["X-Image-Source"] == "original"
    assert _center_color(resp.content) == (0, 0, 1)  # page 3 blue

    # source=highlighted on a non-fence page stays a 404.
    resp = _get_image(app_client, job_id, 3, source="highlighted")
    assert resp.status_code == 404


def test_page_image_uses_persisted_kept_list(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    job_id = _make_colour_job(
        job_registry_temp, tmp_path, fence_idx=[1, 3], hl_kept_idx=[1, 3],
        highlight_pdf_info={
            "kept_page_indices": [1, 3],
            "legend_inserted": True,
            "builder": "worker",
        },
    )
    resp = _get_image(app_client, job_id, 4)
    assert resp.headers["X-Image-Source"] == "highlighted"
    assert _center_color(resp.content) == (1, 1, 0)


def test_page_image_degrades_to_original_when_map_unresolvable(
    api_server_module, app_client, job_registry_temp, tmp_path
):
    """Highlighted PDF whose page count matches no hypothesis: auto must
    fall back to the original sheet (right page, no overlays) rather than
    guess and serve a wrong sheet."""
    job_id = _make_colour_job(
        job_registry_temp, tmp_path, fence_idx=[1, 3],
        hl_kept_idx=[0, 1, 2, 3],  # 5 pages incl. legend; no hypothesis fits
    )
    resp = _get_image(app_client, job_id, 2)
    assert resp.status_code == 200
    assert resp.headers["X-Image-Source"] == "original"
    assert _center_color(resp.content) == (0, 1, 0)  # still page 2's sheet

    resp = _get_image(app_client, job_id, 2, source="highlighted")
    assert resp.status_code == 404
