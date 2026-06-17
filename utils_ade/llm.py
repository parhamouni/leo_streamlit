"""Submodule of utils_ade — split from the original 3,153-line monolith.

The public surface is preserved via utils_ade/__init__.py (re-export shim).
External callers should keep importing as `import utils_ade as ade` and
calling `ade.<function>` — the shim makes that work unchanged.
"""

from __future__ import annotations

import re
import json
import os
import time
import functools
import requests
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

try:
    from google.cloud import documentai_v1 as documentai
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None

try:
    from utils_vector import (
        extract_vector_lines,
        extract_layer_names,
        extract_lines_by_layers,
        group_lines_by_layer,
        group_connected_lines,
        calculate_total_length,
        find_lines_near_bbox,
        find_fence_run_from_indicator,
        infer_scale_from_page,
        detect_scale_with_vision,
        VectorLine,
    )
    VECTOR_UTILS_AVAILABLE = True
except ImportError:
    VECTOR_UTILS_AVAILABLE = False

# ADE API endpoint constant (used by ade_api.py; harmless elsewhere).
ADE_PARSE_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"

# Module-level cache for Google Doc AI client (used by ocr.py).
_DOCAI_CLIENT_CACHE = None

# Cross-submodule imports.
from .instances import find_best_bbox


# --- Trade-aware prompt helpers --------------------------------------------
# The classification / legend / detail prompts used to hardcode "fence". They
# now take an optional `profile` dict (config.TRADE_PROFILES[trade]) so the
# same pipeline can analyse other trades (electrical, …). profile=None keeps
# the original fence behavior — these inline fallbacks make that work even if
# config can't be imported (e.g. an isolated unit test).
_FENCE_LOOK_FOR = [
    "Fence specifications, dimensions, or materials",
    "Gate details or schedules",
    "Barrier or guardrail references",
    "Fence post details",
    "Chain link, mesh, panel references",
    "Any fence-related construction details",
]
_FENCE_DETAIL_FIELDS = [
    {"key": "height", "desc": "fence height if found (e.g. \"6'-0\\\"\", \"8 FT\")"},
    {"key": "post_type", "desc": "post type/size (e.g. \"2-1/2\\\" SS40 ROUND\", \"W6x9\")"},
    {"key": "post_spacing", "desc": "post spacing (e.g. \"10'-0\\\" O.C.\", \"10 FT MAX\")"},
    {"key": "top_rail", "desc": "top rail details"},
    {"key": "bottom_rail", "desc": "bottom rail or tension wire details"},
    {"key": "material", "desc": "material/coating (e.g. \"Galvanized\", \"Vinyl Coated\")"},
    {"key": "gauge", "desc": "wire/mesh gauge (e.g. \"9 gauge\", \"11 gauge\")"},
    {"key": "mesh_size", "desc": "mesh/opening size (e.g. \"2 inch\")"},
    {"key": "foundation", "desc": "footing/foundation details"},
    {"key": "gate_info", "desc": "gate details if applicable"},
]


def _trade_bits(profile):
    """Resolve (subject, look_for, detail_fields) from a trade profile dict,
    defaulting to the fence profile so profile=None preserves prior behavior."""
    profile = profile or {}
    subject = (profile.get("subject") or "fence").strip()
    look_for = profile.get("look_for") or _FENCE_LOOK_FOR
    detail_fields = profile.get("detail_fields") or _FENCE_DETAIL_FIELDS
    return subject, look_for, detail_fields


def llm_extract_fence_elements_batch(llm, texts_by_id, keywords: List[str], batch_size: int = 6, profile=None):
    """Extract fence-related legend entries from multiple chunks in one
    LLM round-trip. Output is attributed back to input chunk ids.

    Args:
      texts_by_id: dict {chunk_id (hashable): chunk_text}
      keywords: fence keywords hint

    Returns:
      dict {chunk_id: [items]} where items match llm_extract_fence_elements output.

    On parse failure or missing ids, falls back to per-chunk
    llm_extract_fence_elements so robustness matches the unbatched path.
    """
    out = {}
    if not llm or not texts_by_id:
        return {k: [] for k in (texts_by_id or {})}

    ids = list(texts_by_id.keys())
    hint_keywords = ", ".join(sorted(set(keywords)))
    subject, _look_for, _fields = _trade_bits(profile)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        _use_messages = True
    except Exception:
        _use_messages = False

    system_prompt = (
        "You are an assistant reviewing engineering drawing documentation. "
        f"For EACH chunk provided, extract {subject}-related legend entries, callouts or "
        "tags, and return them as paired indicator + text elements.\n\n"
        f"Only return items that clearly map to: {hint_keywords}.\n\n"
        "Respond with JSON ONLY, exactly this shape:\n"
        '{"results": [{"chunk_id": "<string>", "items": [{"indicator": "...", '
        '"text_element": "...", "description": "..."}]}, ...]}\n'
        "The chunk_id in each result must match the <chunk id=\"...\"> from the "
        "user message. Include one result per chunk — never skip a chunk. "
        f"If a chunk has no {subject}-related items, return items=[]."
    )

    # Process in batches so we don't blow past the model's context.
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        chunk_blocks = []
        for cid in batch_ids:
            txt = (texts_by_id[cid] or "").strip()[:4000]
            chunk_blocks.append(f'<chunk id="{cid}">\n{txt}\n</chunk>')
        user_content = "Chunks:\n" + "\n\n".join(chunk_blocks)

        parsed_by_id = {}
        try:
            if _use_messages:
                raw = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
            else:
                raw = llm.invoke(system_prompt + "\n\n" + user_content) if hasattr(llm, "invoke") \
                      else llm(system_prompt + "\n\n" + user_content)
            text = getattr(raw, "content", str(raw))
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                for entry in data.get("results", []):
                    if not isinstance(entry, dict):
                        continue
                    cid = str(entry.get("chunk_id", ""))
                    if not cid:
                        continue
                    clean = []
                    for it in entry.get("items", []) or []:
                        if not isinstance(it, dict):
                            continue
                        ind = str(it.get("indicator") or "").strip()
                        txt_e = str(it.get("text_element") or "").strip()
                        desc = str(it.get("description") or "").strip()
                        if ind or txt_e:
                            clean.append({"indicator": ind, "text_element": txt_e, "description": desc})
                    parsed_by_id[cid] = clean
        except Exception as e:
            print(f"[llm_extract_fence_elements_batch] batch parse failed: {e}")

        # Fill in each chunk — if the model skipped one, fall back to a
        # per-chunk call so the caller always gets a value for every id.
        for cid in batch_ids:
            key = str(cid)
            if key in parsed_by_id:
                out[cid] = parsed_by_id[key]
            else:
                print(f"[llm_extract_fence_elements_batch] chunk {cid} missing from batch response; falling back")
                try:
                    out[cid] = llm_extract_fence_elements(llm, texts_by_id[cid], keywords, profile=profile)
                except Exception as _e2:
                    out[cid] = []
    return out


def llm_extract_fence_elements(llm, text: str, keywords: List[str], max_items: int = 100, profile=None) -> List[Dict]:
    if not llm or not text:
        return []
    hint_keywords = ", ".join(sorted(set(keywords)))
    subject, _look_for, _fields = _trade_bits(profile)
    print(f"[DEBUG] Asking LLM to extract items from text length {len(text)}...")

    analysis_prompt = f"""
You are an assistant reviewing engineering drawing documentation. Extract {subject}-related
legend entries, callouts or tags and provide paired indicator + text elements.
Only return items that clearly map to: {hint_keywords}.

Text to analyse:
<TEXT>
{text.strip()[:4000]}
</TEXT>

Respond with a JSON array where each element has:
- "indicator": the numeric or symbolic tag (e.g., "1", "F-3", "A", "3301")
- "text_element": the textual description (e.g., "existing fence", "chain link")
- "description": concise sentence on how the element relates to {subject} work
"""
    try:
        raw_response = llm.invoke(analysis_prompt) if hasattr(llm, "invoke") else llm(analysis_prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
    except Exception as exc:
        print(f"[DEBUG] ⚠️ LLM call failed: {exc}")
        return []

    parsed = []
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
            if isinstance(parsed_json, list):
                for item in parsed_json:
                    if not isinstance(item, dict):
                        continue
                    ind = str(item.get("indicator") or "").strip()
                    txt = str(item.get("text_element") or "").strip()
                    desc = str(item.get("description") or "").strip()
                    if ind or txt:
                        parsed.append({"indicator": ind, "text_element": txt, "description": desc})
    except Exception:
        pass
    print(f"[DEBUG] LLM found {len(parsed)} candidates.")
    return parsed


def extract_legend_entries(
    legend_chunks: List[Dict],
    pdf_lines: List[Dict],
    ocr_lines: List[Dict],
    fence_keywords: List[str],
    llm,
    figure_chunks: List[Dict] = None,
    prefilled_legend_items=None,  # optional {chunk_idx: [items]} — skip LLM when hit
    prefilled_figure_items=None,  # optional {chunk_idx: [items]} — skip LLM when hit
    profile=None,  # trade profile for prompt wording (None => fence)
) -> List[Dict]:
    """extract_legend_entries with optional prefilled-items kwargs.

    When a caller (the fast build) has already called a batched LLM
    extractor for multiple pages' chunks at once, it can pass the
    per-chunk items in via these dicts to skip the per-chunk LLM call
    here. Behavior is 100% backward-compatible when kwargs are None.
    """
    print("[DEBUG] Extracting Legend Entries and Matching BBoxes...")
    results = []

    # Helper: Make a shortened version of the text
    def get_substring(text):
        words = text.split()
        if len(words) > 4:
            return " ".join(words[:4])
        return text

    def _process_chunk(chunk, items, results, extraction_pass="legend"):
        """Process LLM-extracted items from a chunk into results."""
        chunk_bbox = (chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"])
        for item in items:
            desc = item["text_element"]
            ind = item["indicator"]
            bbox_desc = None
            bbox_ind = None

            # A. Description
            if desc:
                bbox_desc = find_best_bbox(desc, pdf_lines, ocr_lines, chunk_bbox)
                if not bbox_desc:
                    bbox_desc = find_best_bbox(get_substring(desc), pdf_lines, ocr_lines, chunk_bbox)
                if not bbox_desc:
                    bbox_desc = find_best_bbox(desc, pdf_lines, ocr_lines, (0, 0, 10000, 10000))

            # B. Indicator
            if ind:
                bbox_ind = find_best_bbox(ind, pdf_lines, ocr_lines, chunk_bbox)
                if not bbox_ind:
                    bbox_ind = find_best_bbox(ind, pdf_lines, ocr_lines, (0, 0, 10000, 10000))

            if bbox_desc:
                results.append({
                    "indicator": ind,
                    "keyword": desc,
                    "description": item["description"],
                    "x0": bbox_desc["x0"], "y0": bbox_desc["y0"],
                    "x1": bbox_desc["x1"], "y1": bbox_desc["y1"],
                    "source": bbox_desc.get("source", "unknown") + "_desc",
                    "extraction_pass": extraction_pass
                })

            if bbox_ind:
                is_duplicate = False
                if bbox_desc:
                    if abs(bbox_ind["x0"] - bbox_desc["x0"]) < 5.0 and abs(bbox_ind["y0"] - bbox_desc["y0"]) < 5.0:
                        is_duplicate = True
                if not is_duplicate:
                    results.append({
                        "indicator": ind,
                        "keyword": ind,
                        "description": "Indicator Code",
                        "x0": bbox_ind["x0"], "y0": bbox_ind["y0"],
                        "x1": bbox_ind["x1"], "y1": bbox_ind["y1"],
                        "source": bbox_ind.get("source", "unknown") + "_ind",
                        "extraction_pass": extraction_pass
                    })

    # Pass 1: Process legend chunks (primary source)
    for _ci, chunk in enumerate(legend_chunks):
        text = chunk.get("text", "")
        if not text:
            continue
        if prefilled_legend_items is not None and _ci in prefilled_legend_items:
            items = prefilled_legend_items[_ci]
        else:
            items = llm_extract_fence_elements(llm, text, fence_keywords, profile=profile)
        _process_chunk(chunk, items, results, extraction_pass="legend")

    # Check quality of legend results — filter out bad indicators
    good_results = [r for r in results if r.get("indicator", "").strip() not in ("", ".", "(bullet point)", "bullet point", "-", "*")]
    
    # Pass 2: If legend chunks yielded few good results, also extract from figure chunks
    if len(good_results) < 3 and figure_chunks:
        print(f"[DEBUG] Legend extraction yielded only {len(good_results)} good results, trying figure chunks...")
        for _ci, chunk in enumerate(figure_chunks):
            text = chunk.get("text", "")
            if not text:
                continue
            # Only process figure chunks whose text contains fence keywords
            text_lower = text.lower()
            if not any(kw.lower() in text_lower for kw in fence_keywords):
                continue
            if prefilled_figure_items is not None and _ci in prefilled_figure_items:
                items = prefilled_figure_items[_ci]
            else:
                items = llm_extract_fence_elements(llm, text, fence_keywords, profile=profile)
            _process_chunk(chunk, items, results, extraction_pass="figure")
    
    # Pass 3: If still few results, try extracting from OCR lines with fence keywords
    good_results = [r for r in results if r.get("indicator", "").strip() not in ("", ".", "(bullet point)", "bullet point", "-", "*")]
    if len(good_results) < 3 and ocr_lines:
        print(f"[DEBUG] Still only {len(good_results)} good results, trying OCR fence lines...")
        fence_ocr_text = []
        for line in ocr_lines:
            t = line.get('text', '')
            if any(kw.lower() in t.lower() for kw in fence_keywords):
                fence_ocr_text.append(t)
        if fence_ocr_text:
            combined_text = "\n".join(fence_ocr_text[:50])
            items = llm_extract_fence_elements(llm, combined_text, fence_keywords, profile=profile)
            if items:
                # Use a page-wide bbox for OCR-sourced items
                page_bbox = {"x0": 0, "y0": 0, "x1": 10000, "y1": 10000}
                for item in items:
                    desc = item["text_element"]
                    ind = item["indicator"]
                    bbox_desc = None
                    if desc:
                        bbox_desc = find_best_bbox(desc, pdf_lines, ocr_lines, (0, 0, 10000, 10000))
                    if bbox_desc:
                        results.append({
                            "indicator": ind,
                            "keyword": desc,
                            "description": item["description"],
                            "x0": bbox_desc["x0"], "y0": bbox_desc["y0"],
                            "x1": bbox_desc["x1"], "y1": bbox_desc["y1"],
                            "source": bbox_desc.get("source", "unknown") + "_ocr_fallback",
                            "extraction_pass": "ocr"
                        })

    print(f"[DEBUG] Finished Legend Extraction. Total mapped items: {len(results)}")
    return results


def extract_element_details(
    llm,
    element_names: List[str],
    page_texts: Dict[int, str],
    profile=None,
) -> Dict[str, Dict]:
    """
    Cross-reference fence element names with detailed specifications found across pages.
    
    For each element (e.g. "9 GAUGE FABRIC"), searches all page texts for detail info
    and uses LLM to extract structured specs (height, post spacing, material, etc.).
    
    Args:
        llm: Language model instance
        element_names: List of element keywords/categories to look up
        page_texts: {page_number: full_text_of_page} for all fence-related pages
        
    Returns:
        Dict mapping element name -> {height, post_spacing, material, gauge, 
        top_rail, bottom_rail, detail_page, full_details, ...}
    """
    if not llm or not element_names or not page_texts:
        return {}
    
    print(f"[DETAILS] Extracting details for {len(element_names)} elements across {len(page_texts)} pages")
    
    # Combine all page texts with page markers (truncate each to keep within token limits)
    combined_parts = []
    for page_num in sorted(page_texts.keys()):
        text = page_texts[page_num].strip()
        if text:
            # Truncate very long pages but keep enough for detail extraction
            truncated = text[:6000]
            combined_parts.append(f"--- PAGE {page_num} ---\n{truncated}")
    
    combined_text = "\n\n".join(combined_parts)
    # Cap total to avoid exceeding LLM context
    combined_text = combined_text[:20000]
    
    elements_list = "\n".join(f"- {name}" for name in element_names)

    subject, _look_for, detail_fields = _trade_bits(profile)
    field_lines = "\n".join(f'- "{f["key"]}": {f["desc"]}' for f in detail_fields)

    prompt = f"""You are reviewing engineering drawing documentation for {subject} work.

The following {subject} elements/categories were identified in the drawings:
{elements_list}

Below is the text extracted from multiple pages of the drawing set. Some pages contain
plan / layout views, while others contain DETAIL pages or schedules carrying the
specifications for these elements.

Your task: For EACH element listed above, find any detailed specifications mentioned
anywhere in the text below. Cross-reference by indicator numbers, element names, or
any matching descriptions.

Text from drawing pages:
<PAGES>
{combined_text}
</PAGES>

Respond with a JSON array where each element has:
- "element_name": the element name (must match one from the list above)
{field_lines}
- "detail_page": page number(s) where detail was found
- "full_details": a concise text summary of ALL specifications found for this element
- "notes": any other relevant notes or specs

If no details are found for an element, still include it with empty strings.
Only return the JSON array, no other text."""

    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
    except Exception as exc:
        print(f"[DETAILS] ⚠️ LLM call failed: {exc}")
        return {}
    
    result = {}
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("element_name", "")).strip()
                    if not name:
                        continue
                    # Match to closest element_name from our list (case-insensitive)
                    matched_name = None
                    for en in element_names:
                        if en.lower() == name.lower() or en.lower() in name.lower() or name.lower() in en.lower():
                            matched_name = en
                            break
                    if not matched_name:
                        matched_name = name

                    # Pull the trade's spec fields dynamically, then the
                    # three fields every trade shares.
                    normalized = {
                        f["key"]: str(item.get(f["key"], "")).strip()
                        for f in detail_fields
                    }
                    normalized["detail_page"] = str(item.get("detail_page", "")).strip()
                    normalized["full_details"] = str(item.get("full_details", "")).strip()
                    normalized["notes"] = str(item.get("notes", "")).strip()
                    result[matched_name] = normalized
    except Exception as e:
        print(f"[DETAILS] JSON parse error: {e}")
    
    print(f"[DETAILS] Extracted details for {len(result)} elements")
    for name, details in result.items():
        summary = details.get('full_details', '')[:80]
        print(f"[DETAILS]   {name}: {summary}...")
    
    return result


def llm_classify_page(llm, page_text: str, fence_keywords: List[str], profile=None) -> Dict:
    """
    Use LLM to classify whether a page is relevant to the active trade.
    Returns an ``is_fence_related`` key regardless of trade (kept for
    backward compatibility — the pipeline reads that field).
    """
    if not llm or not page_text:
        return {"is_fence_related": False, "confidence": 0.0, "reason": "No LLM or text"}

    subject, look_for, _fields = _trade_bits(profile)
    look_for_block = "\n".join(f"- {x}" for x in look_for)
    # FIX 1: Increase text limit from 8000 to 16000 to capture content
    # that often appears at the end of pages (legends, notes, schedules)
    text_for_llm = page_text[:16000] if len(page_text) > 16000 else page_text
    keywords_hint = ", ".join(fence_keywords[:15])

    prompt = f"""You are analyzing an engineering drawing page to determine if it contains {subject}-related content.

Keywords to look for: {keywords_hint}

Page text:
<TEXT>
{text_for_llm}
</TEXT>

Analyze the text and determine whether this page is about {subject} work or related elements.
Look for:
{look_for_block}

Respond with JSON only:
{{"is_relevant": true/false, "confidence": 0.0-1.0, "signals": ["keyword1", "keyword2"], "reason": "brief explanation"}}
"""

    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))

        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "is_fence_related": result.get("is_relevant", result.get("is_fence_related", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "signals": result.get("signals", []),
                "reason": result.get("reason", "")
            }
    except Exception as e:
        print(f"[DEBUG] LLM page classification failed: {e}")
    
    return {"is_fence_related": False, "confidence": 0.0, "reason": "LLM parsing failed"}


def llm_classify_pages_batch(llm, pages, fence_keywords, batch_size=10, profile=None):
    """Classify multiple pages in a single LLM round-trip.

    Prompt layout is stable-prefix-first (system message with rubric +
    keyword context) followed by a user message containing the dynamic
    page blocks. OpenAI auto-caches identical prompt prefixes ≥1024
    tokens across calls in a short window — so keeping the rubric in
    the system message and pages separate lets the provider skip
    re-processing the prefix on subsequent batches in the same run.

    Args:
      llm: a ChatOpenAI-like instance with .invoke(prompt) → obj with .content
      pages: list of (page_idx, page_text, matched_keywords) tuples
      fence_keywords: full keyword list (for prompt context)
      batch_size: how many pages per LLM call

    Returns:
      dict {page_idx: {"is_fence_related": bool, "confidence": float,
                       "signals": list, "reason": str}}

    Missing pages (parse failure or model drift) fall back to
    per-page llm_classify_page so robustness matches the unbatched path.
    """
    out = {}
    if not pages:
        return out
    if not llm:
        for idx, _text, _kws in pages:
            out[idx] = {"is_fence_related": True, "confidence": 0.4,
                        "signals": [], "reason": "LLM unavailable, keyword-only pass"}
        return out

    kw_hint = ", ".join(fence_keywords[:15])
    subject, look_for, _fields = _trade_bits(profile)
    look_for_inline = "; ".join(look_for)

    # Static system prompt — identical across every batch in a run (for a
    # given trade), which is what OpenAI's prompt cache keys on. If this ever
    # grows past 1024 tokens, cache hits light up automatically.
    system_prompt = (
        "You are analyzing engineering drawing pages to determine which "
        f"contain {subject}-related content.\n\n"
        f"{subject.capitalize()}-related terms to watch for: {kw_hint}\n\n"
        "For EACH page provided by the user, decide independently whether "
        f"it's {subject}-related.\n"
        f"Look for: {look_for_inline}.\n"
        "Do NOT flag a page as relevant just because it mentions a generic "
        "construction or engineering term that could appear in any drawing — "
        f"the context must clearly point to {subject} work.\n\n"
        "Respond with JSON ONLY, exactly this shape:\n"
        '{"results": [{"id": <int>, "is_relevant": true|false, '
        '"confidence": 0.0-1.0, "signals": [...], "reason": "<brief>"}, ...]}\n'
        "The id field must match the <page id=\"N\"> from the user message. "
        "Include one result per page — never skip a page."
    )

    # LangChain's SystemMessage / HumanMessage give OpenAI a clean
    # system+user split, which is what the cache groups on.
    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        _use_messages = True
    except Exception:
        _use_messages = False

    for batch_start in range(0, len(pages), batch_size):
        batch = pages[batch_start:batch_start + batch_size]

        page_blocks = []
        for idx, text, matched in batch:
            excerpt = (text or "")[:1200]
            matched_hint = ", ".join(matched[:8]) if matched else ""
            page_blocks.append(
                f'<page id="{idx}" matched="{matched_hint}">\n{excerpt}\n</page>'
            )
        user_content = "Pages:\n" + "\n\n".join(page_blocks)

        parsed_by_id = {}
        try:
            if _use_messages:
                raw = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
            else:
                # Fallback for bare-callable LLMs: concatenate in
                # prefix-first order so caching still has a chance.
                prompt = system_prompt + "\n\n" + user_content
                raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
            text = getattr(raw, "content", str(raw))
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                for entry in data.get("results", []):
                    try:
                        pid = int(entry.get("id"))
                    except (TypeError, ValueError):
                        continue
                    parsed_by_id[pid] = {
                        "is_fence_related": bool(entry.get("is_relevant", entry.get("is_fence_related", False))),
                        "confidence": float(entry.get("confidence", 0.0)),
                        "signals": entry.get("signals", []) or [],
                        "reason": entry.get("reason", "") or "",
                    }
        except Exception as e:
            print(f"[llm_classify_pages_batch] batch parse failed: {e}")

        # Fill in each page — if the model skipped one, fall back to a
        # per-page call so robustness matches the unbatched path.
        for idx, text, _kws in batch:
            if idx in parsed_by_id:
                out[idx] = parsed_by_id[idx]
            else:
                print(f"[llm_classify_pages_batch] page {idx} missing from batch response; falling back")
                try:
                    out[idx] = llm_classify_page(llm, text, fence_keywords, profile=profile)
                except Exception as _e2:
                    out[idx] = {"is_fence_related": False, "confidence": 0.0,
                                "reason": f"batch+fallback failed: {_e2}"}
    return out


def llm_identify_fence_layers(llm, layer_names: List[str], fence_definitions: List[Dict]) -> List[str]:
    """
    Use LLM to intelligently identify which layers contain fence-related elements.
    
    Args:
        llm: Language model instance
        layer_names: List of all layer names in the PDF
        fence_definitions: Already-detected fence indicators from ADE
    
    Returns:
        List of layer names that likely contain fence elements
    """
    if not llm or not layer_names:
        return []
    
    # Format fence definitions for the prompt
    indicators_text = ""
    if fence_definitions:
        for defn in fence_definitions[:10]:  # Limit to first 10
            ind = defn.get("indicator", "")
            kw = defn.get("keyword", "")
            desc = defn.get("description", "")
            if ind or kw:
                indicators_text += f'  - "{ind}" → "{kw}" ({desc})\n'
    
    if not indicators_text:
        indicators_text = "  (No fence indicators detected yet)"
    
    layers_list = "\n".join(f"  - {layer}" for layer in layer_names[:50])  # Limit to 50 layers
    
    prompt = f"""You are analyzing an engineering/architectural PDF drawing. 
Given the layer names from this drawing and the detected fence-related indicators, 
identify which layers likely contain fence-related geometric elements (lines, polylines).

LAYER NAMES FROM PDF:
{layers_list}

DETECTED FENCE INDICATORS:
{indicators_text}

Based on typical CAD/PDF layer naming conventions, identify layers that might contain:
- Fence lines (chain link, wood, metal, etc.)
- Gate elements
- Barriers, guardrails, handrails
- Property boundaries
- Perimeter/enclosure elements

Return ONLY a JSON array of matching layer names. If unsure, include layers with related terms.
Example response: ["V-SITE-FENCE", "A-BARRIER", "SITE-PERIMETER"]

If no layers seem fence-related, return: []
"""

    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
        
        # Parse JSON array from response
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            if isinstance(result, list):
                # Validate that returned layers exist in the actual layer names
                valid_layers = []
                for res_layer in result:
                    if res_layer in layer_names:
                        valid_layers.append(res_layer)
                    else:
                        # Try to find partial match (e.g. LLM returns "V-SITE-FENCE" for "SITE BASE|V-SITE-FENC")
                        matches = [l for l in layer_names if res_layer in l]
                        if matches:
                            valid_layers.extend(matches)
                
                # Deduplicate
                valid_layers = list(set(valid_layers))
                print(f"[DEBUG] LLM identified fence layers: {valid_layers}")
                return valid_layers
    except Exception as e:
        print(f"[DEBUG] LLM layer identification failed: {e}")
    
    return []


def llm_match_layers_to_definitions(llm, fence_layers: List[str], fence_definitions: List[Dict]) -> Dict[str, str]:
    """
    Use LLM to match fence layer names to fence definition categories.
    
    Args:
        llm: Language model instance
        fence_layers: List of detected fence layer names (e.g. ["V-SITE-FENC-CL", "V-SITE-FENC-WOOD"])
        fence_definitions: Detected fence definitions with indicator/keyword/description
    
    Returns:
        Dict mapping layer_name -> category_name (e.g. {"V-SITE-FENC-CL": "3: 6' CHAIN LINK FENCE"})
    """
    if not llm or not fence_layers or not fence_definitions:
        return {}
    
    # Build definition list for prompt
    defs_text = ""
    cat_names = []
    for defn in fence_definitions:
        ind = defn.get("indicator", "")
        kw = defn.get("keyword", "")
        desc = defn.get("description", "")
        cat_name = f"{ind}: {kw}" if ind else kw
        if cat_name:
            cat_names.append(cat_name)
            defs_text += f'  - Category: "{cat_name}" (description: {desc})\n'
    
    if not cat_names:
        return {}
    
    layers_text = "\n".join(f'  - "{layer}"' for layer in fence_layers)
    
    prompt = f"""You are analyzing a construction/architectural PDF drawing.
Match each PDF layer name to the most appropriate fence category based on naming conventions.

FENCE LAYERS FOUND IN PDF:
{layers_text}

FENCE CATEGORIES FROM LEGEND:
{defs_text}

Common CAD layer naming patterns:
- "FENC-CL" or "FENC-CHNLK" → chain link fence
- "FENC-WOOD" or "FENC-WD" → wood fence  
- "FENC-VINYL" or "FENC-VNL" → vinyl fence
- "FENC-IRON" or "FENC-WI" → wrought iron fence
- "FENC-METAL" → metal fence
- "WALL" → wall/barrier
- Generic "FENC" → match to the most common/default fence type

Return a JSON object mapping each layer name to the best matching category name.
Use EXACT category names from the list above.
If a layer doesn't clearly match any category, map it to the most likely one.

Example: {{"V-SITE-FENC-CL": "3: 6' CHAIN LINK FENCE", "V-SITE-FENC-WOOD": "5: 6' WOOD FENCE"}}
"""

    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
        
        # Parse JSON object from response
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            if isinstance(result, dict):
                # Validate: only keep mappings where layer exists and category exists
                valid_mapping = {}
                for layer, cat in result.items():
                    if layer in fence_layers and cat in cat_names:
                        valid_mapping[layer] = cat
                    elif layer in fence_layers:
                        # LLM returned a category name that doesn't exactly match - find closest
                        for real_cat in cat_names:
                            if cat.lower() in real_cat.lower() or real_cat.lower() in cat.lower():
                                valid_mapping[layer] = real_cat
                                break
                
                # For any unmatched layers, assign to first category as fallback
                for layer in fence_layers:
                    if layer not in valid_mapping and cat_names:
                        valid_mapping[layer] = cat_names[0]
                
                print(f"[DEBUG] LLM layer→category mapping: {valid_mapping}")
                return valid_mapping
    except Exception as e:
        print(f"[DEBUG] LLM layer→category matching failed: {e}")
    
    # Fallback: map all layers to first category
    return {layer: cat_names[0] for layer in fence_layers} if cat_names else {}


def llm_suggest_filter_params(llm, line_stats: dict, page_context: str = "") -> dict:
    """
    Ask LLM to suggest filtering parameters based on line statistics.
    
    Args:
        llm: LangChain LLM instance
        line_stats: Dictionary with line statistics
        page_context: Additional context about the page
    
    Returns:
        Dictionary with suggested parameters
    """
    if not llm:
        return {'min_length': 80.0, 'proximity_margin': 50.0, 'reasoning': 'No LLM - using defaults'}
    
    prompt = f"""You are analyzing vector line statistics from an architectural site plan PDF.

PAGE CONTEXT: {page_context if page_context else 'Site plan drawing'}

LINE STATISTICS:
- Total lines: {line_stats.get('total', 0):,}
- Length distribution:
  - Under 10 pts: {line_stats.get('under_10', 0):,} ({line_stats.get('pct_under_10', 0):.1f}%)
  - 10-50 pts: {line_stats.get('range_10_50', 0):,} ({line_stats.get('pct_10_50', 0):.1f}%)
  - 50-100 pts: {line_stats.get('range_50_100', 0):,} ({line_stats.get('pct_50_100', 0):.1f}%)
  - Over 100 pts: {line_stats.get('over_100', 0):,} ({line_stats.get('pct_over_100', 0):.1f}%)
- Layers found: {line_stats.get('layers', 'None')}
- Detected fence indicators: {line_stats.get('indicators', 'Unknown')}

TASK: Suggest filtering parameters to identify FENCE/WALL/BARRIER lines while EXCLUDING:
- Parking stripes (repeating short parallel lines in parking areas)
- Hatching patterns (tiny connected segments for area fills)
- Text and annotation lines

Based on the statistics, suggest:
1. min_length_pts: Minimum line length to consider (fence lines are typically continuous)
2. proximity_margin_pts: How close a line must be to a fence indicator to be included

IMPORTANT: 
- If most lines are tiny (<10 pts), the drawing likely has lots of hatching - use higher min_length
- If there are layer names with FENC/WALL, those should be prioritized (return 0 for min_length)
- Fence lines are typically 50-500+ pts, not tiny segments

Respond with ONLY valid JSON (no markdown):
{{"min_length_pts": <number>, "proximity_margin_pts": <number>, "reasoning": "<brief explanation>"}}"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON from response
        import json
        import re
        
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            min_len = float(result.get('min_length_pts', 80))
            margin = float(result.get('proximity_margin_pts', 50))
            reasoning = result.get('reasoning', 'LLM suggested')
            
            # Sanity bounds
            min_len = max(10, min(200, min_len))
            margin = max(20, min(150, margin))
            
            print(f"[DEBUG] LLM suggested: min_length={min_len}, margin={margin}")
            print(f"[DEBUG] Reasoning: {reasoning}")
            
            return {
                'min_length': min_len,
                'proximity_margin': margin,
                'reasoning': reasoning
            }
    except Exception as e:
        print(f"[DEBUG] LLM filter suggestion failed: {e}")
    
    return {'min_length': 80.0, 'proximity_margin': 50.0, 'reasoning': 'Fallback defaults'}
