"""Round-trip tests for fence_cache: put/get/probe and pickle/json paths."""

from __future__ import annotations

import fence_cache


def test_cache_root_uses_env_override(fence_cache_root):
    root = fence_cache.cache_root()
    assert root == fence_cache_root
    assert root.exists()


def test_json_phase_round_trip(fence_cache_root):
    """A non-pickle phase round-trips a plain dict via JSON."""
    sha = "a" * 64
    params = fence_cache.params_hash(model="gpt-5.1", keywords=["fence"])
    payload = {"page_idx": 3, "fence_found": True, "score": 0.87}

    fence_cache.put("phase1c", sha, params, payload, page_idx=3)
    roundtripped = fence_cache.get("phase1c", sha, params, page_idx=3)

    assert roundtripped == payload


class _TypedPayload:
    """Module-level so pickle can resolve it (local classes can't be pickled)."""
    def __init__(self, x):
        self.x = x


def test_pickle_phase_round_trip(fence_cache_root):
    """A pickle phase round-trips a typed object that JSON would mangle."""
    sha = "b" * 64
    params = fence_cache.params_hash(version=1)
    obj = _TypedPayload(x=42)

    fence_cache.put("phase1a", sha, params, obj)
    loaded = fence_cache.get("phase1a", sha, params)

    assert loaded is not None
    assert loaded.x == 42


def test_get_miss_returns_none(fence_cache_root):
    sha = "c" * 64
    params = fence_cache.params_hash()
    assert fence_cache.get("phase1c", sha, params, page_idx=0) is None


def test_probe_per_page_phase(fence_cache_root):
    """probe() reports which pages are cached for a per-page phase."""
    sha = "d" * 64
    params = fence_cache.params_hash()

    fence_cache.put("phase1c", sha, params, {"ok": True}, page_idx=0)
    fence_cache.put("phase1c", sha, params, {"ok": True}, page_idx=2)

    res = fence_cache.probe(sha, "phase1c", params, page_indices=[0, 1, 2])
    assert res["covered"] == {0, 2}
    assert res["missing"] == {1}
    assert res["complete"] is False

    fence_cache.put("phase1c", sha, params, {"ok": True}, page_idx=1)
    res2 = fence_cache.probe(sha, "phase1c", params, page_indices=[0, 1, 2])
    assert res2["complete"] is True
    assert res2["missing"] == set()


def test_probe_whole_doc_phase(fence_cache_root):
    """phase1a is whole-document, not per-page."""
    sha = "e" * 64
    params = fence_cache.params_hash()

    miss = fence_cache.probe(sha, "phase1a", params)
    assert miss["complete"] is False

    fence_cache.put("phase1a", sha, params, {"text": "x"})
    hit = fence_cache.probe(sha, "phase1a", params)
    assert hit["complete"] is True


def test_params_hash_is_deterministic_and_order_independent(fence_cache_root):
    h1 = fence_cache.params_hash(b=2, a=1)
    h2 = fence_cache.params_hash(a=1, b=2)
    assert h1 == h2

    h3 = fence_cache.params_hash(a=1, b=3)
    assert h1 != h3


def test_purge_pdf(fence_cache_root):
    """purge_pdf removes all phases for one PDF, leaves others intact."""
    sha_a = "1" * 64
    sha_b = "2" * 64
    params = fence_cache.params_hash()

    fence_cache.put("phase1c", sha_a, params, {"a": 1}, page_idx=0)
    fence_cache.put("phase1c", sha_b, params, {"b": 1}, page_idx=0)

    fence_cache.purge_pdf(sha_a)

    assert fence_cache.get("phase1c", sha_a, params, page_idx=0) is None
    assert fence_cache.get("phase1c", sha_b, params, page_idx=0) == {"b": 1}
