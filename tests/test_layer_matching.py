"""Stage-1 guards for the layer→category path.

Locks in the 2026-08-06 fixes:
- pipeline passes PROCESSED legend_entries/figure_instances to
  measure_fence_elements (until then it passed raw ADE chunks positionally,
  so llm_match_layers_to_definitions never had indicator/keyword keys and
  layer_to_category was {} on every page of every live job);
- llm_match_layers_to_definitions no longer force-maps unmatched layers to
  the first category (nor all layers on exception);
- LLM JSON parsing survives fences/nesting (the old non-greedy regex
  truncated at the first closing brace);
- unmatched layers are surfaced in measurements["unmatched_layers"].
"""
from __future__ import annotations

import ast
from pathlib import Path

import fitz

from utils_ade.llm import (
    _parse_json_payload,
    llm_identify_fence_layers,
    llm_match_layers_to_definitions,
)
from utils_ade.measure import measure_fence_elements

REPO = Path(__file__).resolve().parent.parent

DEFS = [
    {"indicator": "3", "keyword": "6' CHAIN LINK FENCE", "description": "galv"},
    {"indicator": "5", "keyword": "WOOD FENCE", "description": "cedar"},
]


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response

        class R:
            content = self.response
        return R()


# --- _parse_json_payload ---------------------------------------------------

def test_parse_handles_fenced_and_nested():
    txt = 'Sure! Here you go:\n```json\n{"a": {"nested": 1}, "b": [1, 2]}\n```'
    assert _parse_json_payload(txt) == {"a": {"nested": 1}, "b": [1, 2]}


def test_parse_nested_object_not_truncated():
    # The old regex r'{.*?}' returned '{"x": {"y": 1}' fragments → json error.
    txt = 'prefix {"x": {"y": 1}, "z": "s}with brace"} suffix'
    assert _parse_json_payload(txt) == {"x": {"y": 1}, "z": "s}with brace"}


def test_parse_array():
    assert _parse_json_payload('noise ["A", "B"] more', opener="[") == ["A", "B"]


def test_parse_garbage_returns_none():
    assert _parse_json_payload("no json here") is None


# --- llm_match_layers_to_definitions --------------------------------------

def test_match_keeps_only_genuine_mappings():
    llm = FakeLLM('{"V-FENC-CL": "3: 6\' CHAIN LINK FENCE", "A-WALL-PATT": null}')
    out = llm_match_layers_to_definitions(llm, ["V-FENC-CL", "A-WALL-PATT"], DEFS)
    assert out == {"V-FENC-CL": "3: 6' CHAIN LINK FENCE"}


def test_match_no_force_map_on_unmatched():
    llm = FakeLLM('{}')
    out = llm_match_layers_to_definitions(llm, ["MYSTERY-LAYER"], DEFS)
    assert out == {}          # old code: {"MYSTERY-LAYER": "3: 6' CHAIN LINK FENCE"}


def test_match_exception_returns_empty_not_first_category():
    llm = FakeLLM(RuntimeError("api down"))
    out = llm_match_layers_to_definitions(llm, ["V-FENC-CL"], DEFS)
    assert out == {}


def test_match_raw_chunks_yield_empty():
    """Raw ADE chunks (no indicator/keyword keys) — the exact shape the
    pre-fix pipeline passed — must short-circuit to {} without an LLM call."""
    llm = FakeLLM('{"V-FENC-CL": "anything"}')
    out = llm_match_layers_to_definitions(
        llm, ["V-FENC-CL"], [{"chunk_type": "legend", "text": "..."}])
    assert out == {}
    assert llm.prompts == []


# --- llm_identify_fence_layers ---------------------------------------------

def test_identify_short_fragment_cannot_explode():
    layers = ["A-ONE", "B-TWO", "C-THREE"]
    llm = FakeLLM('["E"]')     # 1-char fragment: old code expanded to all matches
    assert llm_identify_fence_layers(llm, layers, DEFS) == []


def test_identify_valid_and_partial():
    layers = ["SITE BASE|V-SITE-FENC", "A-BLDG-WALL-TILE-PATT"]
    llm = FakeLLM('```json\n["V-SITE-FENC"]\n```')
    assert llm_identify_fence_layers(llm, layers, DEFS) == ["SITE BASE|V-SITE-FENC"]


# --- measure_fence_elements surfaces unmatched layers -----------------------

class TwoCallLLM:
    """identify → picks the layer; match → maps nothing."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1

        class R:
            content = '["FENCE-L"]' if "identify" not in prompt else "[]"
        # First call is identify (array), second is match (object).
        if "Match each PDF layer" in prompt:
            R.content = "{}"
        return R()


def test_measure_reports_unmatched_layers(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    ocg = doc.add_ocg("FENCE-L")
    page.draw_line((100, 100), (400, 100), color=(0, 0, 0), width=1.0, oc=ocg)
    m = measure_fence_elements(
        page, DEFS, [{"indicator": "3", "x0": 1, "y0": 1, "x1": 2, "y1": 2}],
        llm=None, light_llm=TwoCallLLM(), scale_factor=360.0,
    )
    assert m["layer_to_category"] == {}
    assert m["unmatched_layers"] == ["FENCE-L"]
    assert len(m["all_fence_lines"]) == 1
    doc.close()


# --- pipeline call-site regression (AST) -----------------------------------

def test_pipeline_passes_processed_objects_to_measure():
    """Guards the one-line bug that silently disabled categorization in the
    live pipeline: measure_fence_elements must receive legend_entries and
    figure_instances, never the raw legend_chunks/figure_chunks."""
    tree = ast.parse((REPO / "pipeline.py").read_text())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "measure_fence_elements"
    ]
    assert calls, "measure_fence_elements call not found in pipeline.py"
    for call in calls:
        args = [a.id for a in call.args[1:3] if isinstance(a, ast.Name)]
        assert args == ["legend_entries", "figure_instances"], (
            f"measure_fence_elements called with {args} — raw chunks again?")
