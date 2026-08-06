"""Stage-2 guards: document-level legend registry."""
from __future__ import annotations

from utils_ade.legend_registry import build_registry, registry_match_pages


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


PAGES = [
    {"page_num": 1, "legend_entries": [
        {"indicator": "3", "keyword": "6' CHAIN LINK FENCE", "description": "galv"},
        {"indicator": "3", "keyword": "6' chain link fence", "description": "galvanized, see 2/A-500"},
    ]},
    {"page_num": 4, "legend_entries": [
        {"indicator": "45", "keyword": "CHAIN LINK FENCE", "description": ""},
    ]},
    {"page_num": 5, "legend_entries": []},
]


def test_build_registry_dedupes_without_llm():
    reg = build_registry(PAGES, llm=None)
    assert reg["consolidated"] is False
    kws = sorted(e["keyword"].upper() for e in reg["entries"])
    assert kws == ["6' CHAIN LINK FENCE", "CHAIN LINK FENCE"]
    # richer description won for the duplicate row
    e3 = next(e for e in reg["entries"] if e["indicator"] == "3")
    assert "A-500" in e3["description"]


def test_build_registry_llm_failure_degrades_to_union():
    reg = build_registry(PAGES, llm=FakeLLM(RuntimeError("down")))
    assert reg["consolidated"] is False
    assert len(reg["entries"]) == 2


def test_build_registry_rejects_hallucinated_entries():
    resp = ('{"entries": [{"indicator": "3", "keyword": "6\' CHAIN LINK FENCE",'
            ' "description": "merged", "source_page": 1},'
            ' {"indicator": "99", "keyword": "INVENTED ITEM", "description": "x"}]}')
    reg = build_registry(PAGES, llm=FakeLLM(resp))
    assert reg["consolidated"] is True
    inds = {e["indicator"] for e in reg["entries"]}
    assert "3" in inds and "99" not in inds


def test_registry_match_adds_only_unmatched_and_stamps_provenance():
    fence_results = [{
        "page_num": 7,
        "measurements": {
            "fence_layers": ["V-FENC-CHLK", "MYSTERY"],
            "layer_to_category": {"V-FENC-CHLK": "existing: same-page"},
            "unmatched_layers": ["MYSTERY"],
        },
    }]
    reg = {"entries": [{"indicator": "3", "keyword": "6' CHAIN LINK FENCE",
                        "description": ""}]}
    llm = FakeLLM('{"MYSTERY": "3: 6\' CHAIN LINK FENCE"}')
    changed = registry_match_pages(fence_results, reg, light_llm=llm)
    m = fence_results[0]["measurements"]
    assert changed == 1
    assert m["layer_to_category"]["V-FENC-CHLK"] == "existing: same-page"  # untouched
    assert m["layer_to_category"]["MYSTERY"] == "3: 6' CHAIN LINK FENCE"
    assert m["unmatched_layers"] == []
    assert m["category_source"] == {"MYSTERY": "document-registry"}
    # only the unmatched layer was sent to the LLM
    assert "V-FENC-CHLK" not in llm.prompts[0].split("FENCE LAYERS FOUND")[1].split("FENCE CATEGORIES")[0]


def test_registry_match_noop_when_nothing_unmatched_or_no_registry():
    fr = [{"page_num": 1, "measurements": {
        "fence_layers": ["A"], "layer_to_category": {"A": "cat"}}}]
    assert registry_match_pages(fr, {"entries": [{"indicator": "1", "keyword": "F"}]},
                                light_llm=FakeLLM("{}")) == 0
    assert registry_match_pages(fr, {"entries": []}, light_llm=FakeLLM("{}")) == 0
