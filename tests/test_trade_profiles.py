"""Tests for the multi-trade (fence / electrical) analysis mode.

Covers the trade-profile registry, the backward-compatible fence default,
and that the LLM prompt builders actually inject the selected trade's wording
+ detail-field schema (so electrical mode doesn't ask the model about fence
heights and posts).
"""
from __future__ import annotations

import config
import utils_ade as ade
from pipeline import PipelineConfig


class CaptureLLM:
    """Fake LLM that records the prompt text it was handed and returns a
    canned response. Accepts either a string prompt or a list of LangChain
    messages (the batch classifier path)."""

    def __init__(self, response: str):
        self.calls: list[str] = []
        self._response = response

    def invoke(self, arg):
        if isinstance(arg, list):
            text = "\n".join(getattr(m, "content", str(m)) for m in arg)
        else:
            text = str(arg)
        self.calls.append(text)

        class _R:
            content = self._response

        return _R()


# --- registry ---------------------------------------------------------------

def test_registry_has_fence_and_electrical():
    assert set(config.TRADE_PROFILES) >= {"fence", "electrical"}
    assert config.DEFAULT_TRADE == "fence"
    elec = config.trade_profile("electrical")
    assert elec["subject"] == "electrical"
    assert elec["supports_measurement"] is False
    # The product-owner keywords are present.
    kws = " ".join(elec["keywords"]).lower()
    for term in ["one line", "panel schedule", "conduit schedule", "loads"]:
        assert term in kws


def test_unknown_trade_falls_back_to_fence():
    assert config.trade_profile("plumbing")["subject"] == "fence"
    assert config.trade_profile(None)["subject"] == "fence"


def test_fence_keywords_single_source_of_truth():
    # cfg.DEFAULT_FENCE_KEYWORDS mirrors the fence profile.
    assert list(config.cfg.DEFAULT_FENCE_KEYWORDS) == config.trade_profile("fence")["keywords"]


# --- prompt wording ---------------------------------------------------------

def test_classify_prompt_is_electrical_for_electrical_profile():
    llm = CaptureLLM('{"results": [{"id": 0, "is_relevant": true, "confidence": 0.9, "signals": [], "reason": "x"}]}')
    prof = config.trade_profile("electrical")
    out = ade.llm_classify_pages_batch(
        llm, [(0, "PANEL SCHEDULE 208V", ["panel schedule"])],
        prof["keywords"], profile=prof,
    )
    assert out[0]["is_fence_related"] is True  # internal key preserved
    prompt = llm.calls[0].lower()
    assert "electrical-related content" in prompt
    assert "fence" not in prompt


def test_classify_prompt_defaults_to_fence():
    llm = CaptureLLM('{"results": [{"id": 0, "is_relevant": false, "confidence": 0.1, "signals": [], "reason": "x"}]}')
    ade.llm_classify_pages_batch(llm, [(0, "some text", [])], ["fence"])  # no profile
    assert "fence-related content" in llm.calls[0].lower()


def test_detail_extraction_uses_trade_fields():
    llm = CaptureLLM("[]")
    prof = config.trade_profile("electrical")
    ade.extract_element_details(llm, ["PANEL A"], {1: "PANEL A 208V 3Ø"}, profile=prof)
    prompt = llm.calls[0]
    assert "electrical work" in prompt.lower()
    # Electrical detail fields present; fence-specific ones absent.
    assert '"rating"' in prompt and '"wire_size"' in prompt
    assert '"post_type"' not in prompt and '"mesh_size"' not in prompt


def test_detail_extraction_defaults_to_fence_fields():
    llm = CaptureLLM("[]")
    ade.extract_element_details(llm, ["9 GAUGE FABRIC"], {1: "fence text"})  # no profile
    prompt = llm.calls[0]
    assert '"post_type"' in prompt and '"mesh_size"' in prompt


# --- pipeline config --------------------------------------------------------

def test_pipeline_config_threads_trade():
    assert PipelineConfig().trade == "fence"
    assert PipelineConfig(trade="electrical").trade == "electrical"
