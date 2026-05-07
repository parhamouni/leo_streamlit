"""
utils_ade_official.py - Standalone ADE utility functions for hybrid_page_extractor
"""

import json
import time
import requests
from typing import Dict, List

ADE_PARSE_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"


def ade_parse_document_official(pdf_bytes: bytes, api_key: str, zdr: bool = False) -> Dict:
    """Parse a PDF document using LandingAI's ADE API."""
    if not api_key:
        return {"success": False, "error": "Missing ADE API Key"}

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {"options": json.dumps({"zdr": zdr})} if zdr else {}

    for attempt in range(3):
        try:
            response = requests.post(
                ADE_PARSE_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=180
            )
            response.raise_for_status()
            result = response.json()
            
            chunks = result.get("chunks", [])
            pages = {c.get("grounding", {}).get("page", 0) for c in chunks}
            total_pages = max(pages) + 1 if pages else 0
            
            return {
                "success": True,
                "data": {
                    "chunks": chunks,
                    "total_pages": total_pages,
                    "raw": result
                }
            }
        except Exception as e:
            if attempt == 2:
                return {"success": False, "error": str(e)}
            time.sleep(2 * (attempt + 1))
    
    return {"success": False, "error": "Unknown error after retries"}


def align_ade_chunks_to_page(ade_result: Dict, page_idx: int, page_width: float, page_height: float) -> List[Dict]:
    """Convert ADE normalized coordinates to absolute PDF points."""
    if not ade_result.get("success"):
        return []

    chunks = ade_result.get("data", {}).get("chunks", [])
    page_chunks = []

    for chunk in chunks:
        grounding = chunk.get("grounding", {})
        if grounding.get("page") != page_idx:
            continue
        
        box = grounding.get("box", {})
        x0 = float(box.get("left", 0.0)) * page_width
        y0 = float(box.get("top", 0.0)) * page_height
        x1 = float(box.get("right", 1.0)) * page_width
        y1 = float(box.get("bottom", 1.0)) * page_height

        page_chunks.append({
            "id": chunk.get("id", ""),
            "type": chunk.get("type", "unknown"),
            "text": chunk.get("text") or chunk.get("markdown") or "",
            "markdown": chunk.get("markdown", ""),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "bbox": (x0, y0, x1, y1)
        })
    
    return page_chunks
