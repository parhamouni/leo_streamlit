"""Document-level legend registry (Stage 2 of the agentic takeoff plan).

Fence legends usually live on a cover/legend sheet, not on the sheets that
carry the fence linework — so per-page layer→category matching is starved of
definitions on exactly the pages that need them (production: layer_to_category
was empty on effectively every page even before the pipeline call-site bug).

This module runs AFTER phase 3, deliberately leaving the per-page flow
untouched:
  1. `build_registry` unions every page's extracted `legend_entries` into one
     document-wide registry (deterministic dedupe, then one LLM consolidation
     call to merge near-duplicates and drop non-definition rows; LLM failure
     degrades to the deterministic union).
  2. `registry_match_pages` gives pages a second chance: any page whose
     measurements have fence layers still unmatched gets one light-model
     `llm_match_layers_to_definitions` call against the WHOLE registry.

The registry only ever ADDS mappings for otherwise-unmatched layers; it never
overrides a same-page match. Geometry is untouched — category bucketing is
recomputed downstream from all_fence_lines + layer_to_category.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .llm import _parse_json_payload, llm_match_layers_to_definitions

# Consolidation is a document-level semantic cleanup — worth the analysis
# model; the per-page rematch stays on the light model like same-page
# matching does.
MAX_REGISTRY_ENTRIES = 120


def _dedupe(entries: List[Dict]) -> List[Dict]:
    seen = {}
    for e in entries:
        ind = (e.get("indicator") or "").strip()
        kw = (e.get("keyword") or "").strip()
        if not kw and not ind:
            continue
        key = (ind.lower(), kw.lower())
        if key in seen:
            # Keep the richer description.
            if len(e.get("description") or "") > len(seen[key].get("description") or ""):
                seen[key]["description"] = e.get("description")
            continue
        seen[key] = {
            "indicator": ind,
            "keyword": kw,
            "description": (e.get("description") or "").strip(),
            "source_page": e.get("source_page"),
        }
    return list(seen.values())


def build_registry(fence_results: List[Dict], llm=None) -> Dict:
    """Union all pages' legend_entries into one document registry.

    Returns {"entries": [...], "pages_contributing": [...], "consolidated":
    bool}. Entries keep the legend_entries dict shape (indicator / keyword /
    description) so every existing consumer of definitions works unchanged.
    """
    raw: List[Dict] = []
    pages = []
    for fp in fence_results:
        entries = fp.get("legend_entries") or []
        if not entries:
            continue
        pages.append(fp.get("page_num"))
        for e in entries:
            if isinstance(e, dict):
                raw.append({**e, "source_page": fp.get("page_num")})

    entries = _dedupe(raw)[:MAX_REGISTRY_ENTRIES]
    registry = {"entries": entries, "pages_contributing": pages,
                "consolidated": False}
    if not entries or llm is None:
        return registry

    rows = "\n".join(
        f'  - indicator="{e["indicator"]}" keyword="{e["keyword"]}" '
        f'(page {e.get("source_page")}; {e.get("description", "")[:100]})'
        for e in entries
    )
    prompt = f"""These legend/keynote rows were extracted from multiple sheets of ONE
construction drawing set. Consolidate them into a clean document-wide legend for the
fencing trade.

ROWS:
{rows}

Rules:
- Merge rows that describe the SAME physical item (same indicator on different
  sheets, or trivially different wording). Keep the clearest keyword and the
  richest description.
- Drop rows that are not definitions of physical fence-scope items (sheet notes,
  scale bars, title-block text, revision notes).
- Keep indicators EXACTLY as printed — they must match flags on the drawings.
- Do not invent items that are not in the rows.

Return STRICT JSON: {{"entries": [{{"indicator": "...", "keyword": "...",
"description": "...", "source_page": <page number of the best source row>}}]}}"""
    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        text = getattr(raw_response, "content", str(raw_response))
        parsed = _parse_json_payload(text, opener="{") or {}
        out = parsed.get("entries")
        if isinstance(out, list) and out:
            cleaned = _dedupe([e for e in out if isinstance(e, dict)])
            if cleaned:
                # Consolidation must never hallucinate: every surviving
                # (indicator, keyword) must trace back to an input row's
                # indicator or keyword.
                known_inds = {e["indicator"].lower() for e in entries if e["indicator"]}
                known_kws = {e["keyword"].lower() for e in entries if e["keyword"]}
                cleaned = [
                    e for e in cleaned
                    if (e["indicator"].lower() in known_inds
                        or e["keyword"].lower() in known_kws)
                ]
            if cleaned:
                registry["entries"] = cleaned[:MAX_REGISTRY_ENTRIES]
                registry["consolidated"] = True
    except Exception as e:
        print(f"[DEBUG] legend registry consolidation failed: {e}")
    return registry


def registry_match_pages(
    fence_results: List[Dict],
    registry: Dict,
    light_llm=None,
) -> int:
    """Second-chance layer→category matching against the document registry.

    Mutates each page's measurements in place (layer_to_category additions +
    unmatched_layers shrink) and stamps `category_source` per layer so the UI
    and the agentic resolver can show provenance. Returns the number of pages
    that gained at least one mapping.
    """
    entries = (registry or {}).get("entries") or []
    if not entries or light_llm is None:
        return 0

    changed = 0
    for fp in fence_results:
        m = fp.get("measurements") or {}
        fence_layers = m.get("fence_layers") or []
        if not fence_layers:
            continue
        current = m.get("layer_to_category") or {}
        unmatched = [l for l in fence_layers if l not in current]
        if not unmatched:
            continue
        mapping = llm_match_layers_to_definitions(light_llm, unmatched, entries)
        if not mapping:
            continue
        current.update(mapping)
        m["layer_to_category"] = current
        m["unmatched_layers"] = [l for l in fence_layers if l not in current]
        src = m.setdefault("category_source", {})
        for layer in mapping:
            src[layer] = "document-registry"
        fp["measurements"] = m
        changed += 1
    return changed
