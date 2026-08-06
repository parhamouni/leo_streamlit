"""Stage-4 guards: bounded per-document agentic resolver (no network).

A scripted FakeClient mimics the openai==2.4.0 Responses API shapes the
resolver is written against: `responses.create(model, input, tools,
tool_choice)` returning objects whose `.output` is a list of items with
`.type` in {"function_call", "message"}; function_call items carry
`.name` / `.arguments` (JSON string) / `.call_id`.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentic.resolver import apply_resolutions, find_disputes, run_resolver

CHAIN_LINK = "3: 6' CHAIN LINK FENCE"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_results() -> dict:
    """Two fence pages; page 1 has one unmatched layer + one pre-mapped."""
    return {
        "legend_registry": {"entries": [
            {"indicator": "3", "keyword": "6' CHAIN LINK FENCE",
             "description": "galvanized", "source_page": 1},
        ]},
        "fence_pages": [
            {"page_num": 1, "page_idx": 0,
             "legend_entries": [{"indicator": "3",
                                 "keyword": "6' CHAIN LINK FENCE",
                                 "description": "galvanized"}],
             "instances": [], "scale_info": {"scale": "1\"=20'"},
             "measurements": {
                 "fence_layers": ["V-FENC-A", "V-FENC-B"],
                 "layer_to_category": {"V-FENC-B": CHAIN_LINK},
                 "unmatched_layers": ["V-FENC-A"],
                 "all_fence_lines": [
                     {"start": [10, 10], "end": [110, 10],
                      "length_pts": 100.0, "layer": "V-FENC-A"},
                     {"start": [10, 40], "end": [60, 40],
                      "length_pts": 50.0, "layer": "V-FENC-B"},
                 ],
                 "measurement_method": "vector",
                 "category_source": {"V-FENC-B": "page-legend"},
             }},
            {"page_num": 2, "page_idx": 1, "legend_entries": [],
             "instances": [], "scale_info": {},
             "measurements": {
                 "fence_layers": ["V-FENC-B"],
                 "layer_to_category": {"V-FENC-B": CHAIN_LINK},
                 "unmatched_layers": [],
                 "all_fence_lines": [
                     {"start": [0, 0], "end": [10, 0],
                      "length_pts": 10.0, "layer": "V-FENC-B"},
                 ],
                 "measurement_method": "vector",
             }},
        ],
    }


@pytest.fixture
def tiny_pdf(tmp_path) -> str:
    import fitz

    path = tmp_path / "plans.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"SITE PLAN SHEET {i + 1}")
        page.insert_text((72, 110), "6' CHAIN LINK FENCE")
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# FakeClient mimicking the Responses API
# ---------------------------------------------------------------------------

def _call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name,
                           arguments=json.dumps(args), call_id=call_id)


def _msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message", role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)])


def _resp(*items) -> SimpleNamespace:
    return SimpleNamespace(output=list(items))


class FakeResponses:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.scripted:
            return _resp(_msg("script exhausted"))
        return self.scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.responses = FakeResponses(scripted)


def scripted_client() -> FakeClient:
    """The canonical six-step session from the spec."""
    return FakeClient([
        _resp(_call("list_pages", {}, "c1")),
        _resp(_call("list_layers", {"page_num": 1}, "c2")),
        _resp(_msg("Let me look at the linework."),
              _call("render_page_crop",
                    {"page_num": 1, "x0": 0.05, "y0": 0.05,
                     "x1": 0.6, "y1": 0.5, "dpi": 110}, "c3")),
        _resp(_call("record_assignment",
                    {"layer": "V-FENC-A",
                     "category": "3: 6' chain link fence",  # case-insensitive
                     "pages": [1],
                     "evidence": "page 1: crop shows the dashed run labeled "
                                 "6' CHAIN LINK FENCE on V-FENC-A",
                     "confidence": "high"}, "c4")),
        _resp(_call("record_assignment",
                    {"layer": "V-FENC-A", "category": "MADE UP",
                     "pages": [1], "evidence": "none really",
                     "confidence": "low"}, "c5")),
        _resp(_call("finish", {"summary": "resolved V-FENC-A"}, "c6")),
    ])


# ---------------------------------------------------------------------------
# find_disputes
# ---------------------------------------------------------------------------

def test_find_disputes_unmatched_shape():
    disputes = find_disputes(make_results())
    assert disputes == [{"page_num": 1, "kind": "unmatched_layers",
                         "layers": ["V-FENC-A"]}]


def test_find_disputes_no_mapping_shape():
    results = make_results()
    m = results["fence_pages"][1]["measurements"]
    m["layer_to_category"] = {}
    m["unmatched_layers"] = []
    disputes = find_disputes(results)
    assert {"page_num": 2, "kind": "no_mapping",
            "layers": ["V-FENC-B"]} in disputes


def test_find_disputes_empty_when_all_mapped():
    results = make_results()
    m = results["fence_pages"][0]["measurements"]
    m["layer_to_category"]["V-FENC-A"] = CHAIN_LINK
    m["unmatched_layers"] = []
    assert find_disputes(results) == []


# ---------------------------------------------------------------------------
# run_resolver: happy path
# ---------------------------------------------------------------------------

def test_run_resolver_happy_path(tiny_pdf):
    client = scripted_client()
    res = run_resolver(make_results(), tiny_pdf, client=client)

    # Exactly one VALID assignment survives, normalized to canonical casing.
    assert len(res["assignments"]) == 1
    a = res["assignments"][0]
    assert a["layer"] == "V-FENC-A"
    assert a["category"] == CHAIN_LINK
    assert a["pages"] == [1]
    assert a["confidence"] == "high"
    assert a["decided_by"] == "agentic-resolver"
    assert "page 1" in a["evidence"]

    # Invalid category dropped + noted.
    assert "MADE UP" in res["notes"]

    # Budgets accounted: all six scripted calls executed, one image rendered.
    assert res["tool_calls_used"] == 6
    assert res["images_used"] == 1
    assert res["budget_exhausted"] is False
    assert res["disputes"] == [{"page_num": 1, "kind": "unmatched_layers",
                                "layers": ["V-FENC-A"]}]

    # Request shape: function tools + auto tool choice on every call.
    first = client.responses.requests[0]
    assert first["tool_choice"] == "auto"
    tool_names = {t["name"] for t in first["tools"]}
    assert tool_names == {"list_pages", "get_legend_registry", "list_layers",
                          "search_text", "render_page_crop",
                          "record_assignment", "finish"}
    assert all(t["type"] == "function" for t in first["tools"])

    # The loop fed the rendered crop back as an input_image content part in
    # the function_call_output for call c3 (visible in the NEXT request).
    after_render = client.responses.requests[3]["input"]
    fco = [it for it in after_render
           if isinstance(it, dict) and it.get("type") == "function_call_output"
           and it.get("call_id") == "c3"]
    assert len(fco) == 1
    parts = fco[0]["output"]
    assert isinstance(parts, list)
    kinds = [p["type"] for p in parts]
    assert "input_text" in kinds and "input_image" in kinds
    image = next(p for p in parts if p["type"] == "input_image")
    assert image["image_url"].startswith("data:image/png;base64,")
    meta = json.loads(next(p for p in parts if p["type"] == "input_text")["text"])
    assert meta["image"] is True and meta["page_num"] == 1
    assert "_png_base64" not in meta  # raw bytes never echoed as text

    # History pairing: each function_call echoed before its output.
    types_seq = [it.get("type") for it in after_render
                 if isinstance(it, dict) and it.get("type")]
    assert types_seq.count("function_call") == types_seq.count(
        "function_call_output")


# ---------------------------------------------------------------------------
# run_resolver: budgets + short-circuits
# ---------------------------------------------------------------------------

def test_budget_exhaustion_tool_calls(tiny_pdf):
    res = run_resolver(make_results(), tiny_pdf, client=scripted_client(),
                       max_tool_calls=2)
    assert res["budget_exhausted"] is True
    assert res["tool_calls_used"] == 2  # third call never executed
    assert res["assignments"] == []  # nothing recorded before cutoff
    assert "budget" in res["notes"]


def test_client_none_short_circuits(tiny_pdf):
    res = run_resolver(make_results(), tiny_pdf, client=None)
    assert res["assignments"] == []
    assert res["tool_calls_used"] == 0
    assert res["images_used"] == 0
    assert res["budget_exhausted"] is False
    assert "no LLM client" in res["notes"]
    assert res["disputes"]  # disputes still reported


def test_no_disputes_short_circuits(tiny_pdf):
    results = make_results()
    m = results["fence_pages"][0]["measurements"]
    m["layer_to_category"]["V-FENC-A"] = CHAIN_LINK
    m["unmatched_layers"] = []

    class ExplodingClient:
        def __getattr__(self, name):  # any use would fail the test
            raise AssertionError("client must not be touched")

    res = run_resolver(results, tiny_pdf, client=ExplodingClient())
    assert res["disputes"] == []
    assert res["assignments"] == []
    assert "no disputes" in res["notes"]


def test_api_error_retries_then_finishes_gracefully(tiny_pdf):
    class FlakyResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("boom")

    client = SimpleNamespace(responses=FlakyResponses())
    res = run_resolver(make_results(), tiny_pdf, client=client)
    assert client.responses.calls == 2  # original + one retry
    assert res["budget_exhausted"] is False
    assert "retry" in res["notes"]
    assert res["assignments"] == []


# ---------------------------------------------------------------------------
# search_text handler (rotation-aware, bigram join)
# ---------------------------------------------------------------------------

def test_search_text_finds_multiword_query(tiny_pdf):
    from agentic.resolver import _ResolverSession

    session = _ResolverSession(make_results(), tiny_pdf, notes=[])
    try:
        out = session.tool_search_text("chain link")
    finally:
        session.close()
    assert out["pages_searched"] == 2
    hits = {p["page_num"] for p in out["pages"]}
    assert hits == {1, 2}
    bbox = out["pages"][0]["bboxes_display"][0]
    assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]


# ---------------------------------------------------------------------------
# apply_resolutions
# ---------------------------------------------------------------------------

def test_apply_fills_only_unmatched_and_stamps_provenance():
    results = make_results()
    resolution = {"assignments": [
        {"layer": "V-FENC-A", "category": CHAIN_LINK, "pages": [1],
         "evidence": "e", "confidence": "high",
         "decided_by": "agentic-resolver"},
        # Attempted override of an existing mapping — must be ignored.
        {"layer": "V-FENC-B", "category": "9: WOOD FENCE", "pages": [1, 2],
         "evidence": "e", "confidence": "low",
         "decided_by": "agentic-resolver"},
    ]}
    applied = apply_resolutions(results["fence_pages"], resolution)
    assert applied == 1

    m1 = results["fence_pages"][0]["measurements"]
    assert m1["layer_to_category"]["V-FENC-A"] == CHAIN_LINK
    assert m1["unmatched_layers"] == []
    assert m1["category_source"]["V-FENC-A"] == "agentic-resolver"
    # Pre-existing mapping and its provenance untouched.
    assert m1["layer_to_category"]["V-FENC-B"] == CHAIN_LINK
    assert m1["category_source"]["V-FENC-B"] == "page-legend"
    m2 = results["fence_pages"][1]["measurements"]
    assert m2["layer_to_category"] == {"V-FENC-B": CHAIN_LINK}
    assert "category_source" not in m2


def test_apply_pages_all_touches_every_page_with_the_layer():
    results = make_results()
    # Make V-FENC-A unmatched on page 2 as well.
    m2 = results["fence_pages"][1]["measurements"]
    m2["fence_layers"].append("V-FENC-A")
    m2["unmatched_layers"] = ["V-FENC-A"]
    resolution = {"assignments": [
        {"layer": "V-FENC-A", "category": CHAIN_LINK, "pages": "all",
         "evidence": "e", "confidence": "medium",
         "decided_by": "agentic-resolver"},
    ]}
    assert apply_resolutions(results["fence_pages"], resolution) == 2
    for fp in results["fence_pages"]:
        m = fp["measurements"]
        assert m["layer_to_category"]["V-FENC-A"] == CHAIN_LINK
        assert m["category_source"]["V-FENC-A"] == "agentic-resolver"
        assert "V-FENC-A" not in m["unmatched_layers"]


def test_apply_skips_pages_where_layer_absent():
    results = make_results()
    resolution = {"assignments": [
        {"layer": "V-FENC-A", "category": CHAIN_LINK, "pages": [2],
         "evidence": "e", "confidence": "high",
         "decided_by": "agentic-resolver"},
    ]}
    # Page 2 has no V-FENC-A at all -> nothing to apply.
    assert apply_resolutions(results["fence_pages"], resolution) == 0
    assert "V-FENC-A" not in results["fence_pages"][1][
        "measurements"]["layer_to_category"]
