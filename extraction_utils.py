# extraction_utils.py
from typing import Dict, Any, List
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage

from llm_utils import (
    json_invoke,
    make_legend_discovery_prompt,
    make_legend_merge_prompt,
    make_page_ref_prompt,
)


def _page_text_excerpt(doc: fitz.Document, i: int, get_text_fn) -> str:
    """
    Helper to obtain comprehensive text for a single page via provided callable.
    get_text_fn(page_idx:int) -> str (already includes OCR/annotations as configured upstream)
    """
    return get_text_fn(i)

def build_document_legend_index(
    llm,
    doc: fitz.Document,
    get_text_fn,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Run legend discovery per page (concurrently), then merge to a normalized index via LLM.

    Returns:
        {
          "index": [ { "identifier": str, "synonyms": [...], "definition_pages": [...],
                       "title": str|None, "short_description": str|None,
                       "overall_confidence": float } ],
          "raw_items": [ {page-scoped candidates...} ],
          "id_list": [ "F1", "F2A", ... ]
        }
    """
    candidates: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for i in range(len(doc)):
            page_text = _page_text_excerpt(doc, i, get_text_fn)
            prompt = make_legend_discovery_prompt(i + 1, page_text)
            futures[ex.submit(json_invoke, llm, [HumanMessage(content=prompt)])] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                resp = fut.result()
                content = resp.content if hasattr(resp, "content") else str(resp)
                data = json.loads(content)
                items = data.get("legend_items", [])
                for it in items:
                    it.setdefault("page", i + 1)
                candidates.extend(items)
            except Exception as e:
                print(f"[legend] page {i+1} failed: {e}")

    # Merge/normalize across document
    try:
        candidates_json = json.dumps(candidates, separators=(",", ":"))
        merge_prompt = LEGEND_MERGE_PROMPT.format(candidates_json=candidates_json)
        merge_resp = json_invoke(llm, [HumanMessage(content=merge_prompt)])
        merge_content = merge_resp.content if hasattr(merge_resp, "content") else str(merge_resp)
        merged = json.loads(merge_content)
        index = merged.get("index", [])
    except Exception as e:
        print(f"[legend-merge] failed: {e}")
        index = []

    id_list = [x.get("identifier") for x in index if x.get("identifier")]
    return {"index": index, "raw_items": candidates, "id_list": id_list}

def detect_cross_references(
    llm,
    doc: fitz.Document,
    get_text_fn,
    legend_index: Dict[str, Any],
    max_workers: int = 4,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Per-page reference detection (concurrent), using the normalized legend index.

    Returns:
        { page_number (1-based): [ {"identifier": str, "evidence": str, "confidence": float}, ... ] }
    """
    id_list = legend_index.get("id_list", [])
    if not id_list:
        return {}

    # Keep prompt size reasonable
    id_list_preview = ", ".join(id_list[:60]) + (" …" if len(id_list) > 60 else "")
    results: Dict[int, List[Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for i in range(len(doc)):
            page_text = _page_text_excerpt(doc, i, get_text_fn)
            prompt = make_page_ref_prompt(id_list_preview, page_text)
            futures[ex.submit(json_invoke, llm, [HumanMessage(content=prompt)])] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                resp = fut.result()
                content = resp.content if hasattr(resp, "content") else str(resp)
                data = json.loads(content)
                refs = data.get("references", [])
                if refs:
                    results[i + 1] = refs
            except Exception as e:
                print(f"[xref] page {i+1} failed: {e}")

    return results
