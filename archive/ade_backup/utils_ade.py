"""
utils_ade.py - Unified ADE Utility Module (Refactored for Line-Based Highlighting)
"""

import os
import re
import json
import time
import random
import io
import base64
from typing import List, Dict, Optional, Tuple, Union, Set
from io import BytesIO
from statistics import median

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageFont

# Optional Google Cloud imports
try:
    from google.cloud import documentai_v1 as documentai
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None

# Optional LangChain imports
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    HumanMessage = None

# ==============================================================================
# 1. ADE (LandingAI) Integration
# ==============================================================================

ADE_PARSE_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"

def ade_parse_document(pdf_bytes: bytes, api_key: str, zdr: bool = False) -> Dict:
    """Parse a full PDF document using LandingAI's ADE API."""
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

def segment_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Segment chunks into 'legend_like' and 'figure_like'."""
    legend_like = []
    figure_like = []
    
    for chunk in chunks:
        raw_type = (chunk.get("type") or "").lower()
        text_lower = (chunk.get("text") or "").lower()
        
        is_figure = raw_type in {"figure", "architectural_drawing"}
        has_legend_hint = any(token in text_lower for token in {"legend", "keynote", "abbreviation", "symbol"})

        if not is_figure or has_legend_hint:
            legend_like.append(chunk)
        elif is_figure:
            figure_like.append(chunk)
            
    return legend_like, figure_like

# ==============================================================================
# 2. Google Cloud Document AI (OCR)
# ==============================================================================

_DOCAI_CLIENT_CACHE = None

def get_docai_client(google_cloud_config: Dict):
    global _DOCAI_CLIENT_CACHE
    if _DOCAI_CLIENT_CACHE: return _DOCAI_CLIENT_CACHE
    if not GOOGLE_CLOUD_AVAILABLE: return None

    try:
        service_info = google_cloud_config.get("service_account_info")
        if service_info:
            creds = service_account.Credentials.from_service_account_info(service_info)
            _DOCAI_CLIENT_CACHE = documentai.DocumentProcessorServiceClient(credentials=creds)
            return _DOCAI_CLIENT_CACHE
    except Exception as e:
        print(f"❌ Error creating DocAI client: {e}")
    return None

def run_google_ocr(page_bytes: bytes, google_cloud_config: Dict, page_width_pts: float, page_height_pts: float) -> List[Dict]:
    """Run Google OCR on a single page image."""
    client = get_docai_client(google_cloud_config)
    if not client: return []

    project_id = google_cloud_config.get("project_number")
    location = google_cloud_config.get("location")
    processor_id = google_cloud_config.get("processor_id")
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    raw_document = documentai.RawDocument(content=page_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    try:
        result = client.process_document(request=request)
        document = result.document
    except Exception as e:
        print(f"❌ DocAI Processing Failed: {e}")
        return []

    ocr_tokens = []
    if not document.pages: return []

    page = document.pages[0]
    for token in page.tokens:
        text_content = ""
        layout = token.layout
        if layout.text_anchor.text_segments:
            seg = layout.text_anchor.text_segments[0]
            text_content = document.text[seg.start_index:seg.end_index]
        
        text_content = text_content.strip()
        if not text_content: continue

        vertices = layout.bounding_poly.normalized_vertices
        if not vertices: continue
            
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        x0, x1 = min(xs) * page_width_pts, max(xs) * page_width_pts
        y0, y1 = min(ys) * page_height_pts, max(ys) * page_height_pts

        ocr_tokens.append({
            "text": text_content,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "confidence": layout.confidence,
            "source": "ocr"
        })

    return ocr_tokens

# ==============================================================================
# 3. PDF Native Text Extraction
# ==============================================================================

def get_pdf_text_elements(page: fitz.Page) -> Tuple[List[Dict], List[Dict]]:
    """
    Get native text words AND blocks using PyMuPDF.
    Returns (words, blocks).
    """
    # Get Words
    words_raw = page.get_text("words") 
    words = []
    for w in words_raw:
        words.append({
            "text": w[4],
            "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
            "source": "native_word"
        })
        
    # Get Blocks (for fallback)
    blocks_raw = page.get_text("blocks")
    blocks = []
    for b in blocks_raw:
        blocks.append({
            "text": b[4],
            "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
            "source": "native_block"
        })
        
    return words, blocks

# ==============================================================================
# 4. Line-Based Matching Logic (RESTORED FROM BACKUP)
# ==============================================================================

def group_items_by_line(items: List[Dict], y_tolerance: float) -> List[List[Dict]]:
    """
    Group items (OCR tokens or PDF words) by line based on Y coordinates.
    This is crucial for detecting the full "Code + Description" line.
    """
    if not items:
        return []
    
    # Filter out items with no text
    valid_items = [item for item in items if item.get("text")]
    
    # Sort by Y position primarily, then X
    sorted_items = sorted(
        valid_items,
        key=lambda it: (it.get("y0", 0.0), it.get("x0", 0.0))
    )
    
    lines: List[List[Dict]] = []
    current: List[Dict] = []
    last_y: Optional[float] = None
    
    for item in sorted_items:
        y0 = float(item.get("y0", 0.0))
        # Check if this item belongs on the same line as the previous one
        if last_y is None or abs(y0 - last_y) <= y_tolerance:
            current.append(item)
        else:
            if current:
                # Sort items within the line by X coordinate (left to right)
                lines.append(sorted(current, key=lambda it: it.get("x0", 0.0)))
            current = [item]
        last_y = y0
        
    if current:
        lines.append(sorted(current, key=lambda it: it.get("x0", 0.0)))
        
    return lines

def combine_bbox(items: List[Dict]) -> Tuple[float, float, float, float]:
    """Calculate the union bounding box of multiple items."""
    if not items:
        return (0,0,0,0)
    x0 = min(float(item.get("x0", 0.0)) for item in items)
    y0 = min(float(item.get("y0", 0.0)) for item in items)
    x1 = max(float(item.get("x1", 0.0)) for item in items)
    y1 = max(float(item.get("y1", 0.0)) for item in items)
    return x0, y0, x1, y1

def make_highlight(items: List[Dict], fallback_text: str) -> Optional[Dict]:
    """Create a highlight result covering all items in the list."""
    if not items:
        return None
    x0, y0, x1, y1 = combine_bbox(items)
    text_join = " ".join(item.get("text", "") for item in items).strip()
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "text": text_join or fallback_text,
        "source": items[0].get("source", "unknown")
    }

def match_text_in_lines(
    search_text: str,
    lines: List[List[Dict]],
    chunk_bbox: Optional[Tuple[float, float, float, float]] = None,
    tolerance: float = 50.0
) -> Optional[Dict]:
    """
    Match text in lines. 
    KEY FEATURE: If a match is found, returns the bbox for the ENTIRE LINE.
    """
    if not search_text or not lines:
        return None
    
    # Normalize search text
    target = re.sub(r'[^0-9A-Za-z]', '', search_text).lower()
    if not target:
        return None
    original_text = search_text
    
    # Prepare region filter
    chunk_x0, chunk_y0, chunk_x1, chunk_y1 = (0,0,0,0)
    if chunk_bbox:
        chunk_x0, chunk_y0, chunk_x1, chunk_y1 = chunk_bbox
    
    def is_in_region(x0, y0, x1, y1):
        if not chunk_bbox: return True
        # Allow tolerance around the chunk
        return not (x1 < chunk_x0 - tolerance or x0 > chunk_x1 + tolerance or
                   y1 < chunk_y0 - tolerance or y0 > chunk_y1 + tolerance)
    
    for line in lines:
        if not line: continue
        
        # Filter line items by region (only keep items near/in the chunk)
        line_items = [item for item in line if is_in_region(
            float(item.get("x0", 0)), float(item.get("y0", 0)),
            float(item.get("x1", 0)), float(item.get("y1", 0))
        )]
        if not line_items: continue
        
        # Strategy 1: Direct token match
        for item in line_items:
            raw_text = item.get("text", "") or ""
            clean_text = re.sub(r'[^0-9A-Za-z]', '', raw_text).lower()
            
            # Check if token matches or contains target
            if clean_text and (clean_text == target or clean_text.startswith(target) or target in clean_text):
                # Found it! Return the WHOLE LINE context
                return make_highlight(line_items, original_text)
        
        # Strategy 2: Combined sequential tokens (for phrases split across tokens)
        tokens = []
        for item in line_items:
            c_txt = re.sub(r'[^0-9A-Za-z]', '', item.get("text", "") or "").lower()
            if c_txt: tokens.append((item, c_txt))
            
        n = len(tokens)
        for i in range(n):
            combined = ""
            for j in range(i, n):
                combined += tokens[j][1]
                if combined == target:
                    return make_highlight(line_items, original_text)
                if not target.startswith(combined):
                    break
    
    return None

def find_text_in_ocr_and_pdf(
    search_text: str,
    ocr_tokens: List[Dict],
    pdf_blocks: List[Dict],
    pdf_words: List[Dict],
    chunk_bbox: Optional[Tuple[float, float, float, float]] = None,
    tolerance: float = 50.0
) -> Optional[Dict]:
    """
    Orchestrator to find text in OCR or PDF sources using line-based matching.
    """
    # 1. Search OCR Tokens (Grouped by line)
    if ocr_tokens:
        # y_tolerance 10.0 works well for standard document text size
        ocr_lines = group_items_by_line(ocr_tokens, y_tolerance=10.0)
        result = match_text_in_lines(search_text, ocr_lines, chunk_bbox, tolerance)
        if result:
            result["source"] = "ocr"
            return result
            
    # 2. Search PDF Words (Grouped by line)
    if pdf_words:
        # y_tolerance 5.0 is usually enough for digital PDF text
        pdf_lines = group_items_by_line(pdf_words, y_tolerance=5.0)
        result = match_text_in_lines(search_text, pdf_lines, chunk_bbox, tolerance)
        if result:
            result["source"] = "pdf_words"
            return result

    # 3. Fallback: PDF Blocks (Search text within the block)
    # This doesn't give tight bounding boxes, but it's a safety net
    if pdf_blocks:
        target_clean = re.sub(r'[^0-9A-Za-z]', '', search_text.lower())
        if not target_clean: return None
        
        # Prepare region bounds
        cx0, cy0, cx1, cy1 = (0,0,0,0)
        if chunk_bbox: cx0, cy0, cx1, cy1 = chunk_bbox
        
        for block in pdf_blocks:
            bx0, by0, bx1, by1 = block["x0"], block["y0"], block["x1"], block["y1"]
            
            # Check region
            if chunk_bbox:
                if (bx1 < cx0 - tolerance or bx0 > cx1 + tolerance or 
                    by1 < cy0 - tolerance or by0 > cy1 + tolerance):
                    continue
            
            block_text_clean = re.sub(r'[^0-9A-Za-z]', '', (block.get("text") or "").lower())
            
            if target_clean in block_text_clean:
                return {
                    "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
                    "text": block.get("text", ""),
                    "source": "pdf_block"
                }
                
    return None

# ==============================================================================
# 5. Keyword & Indicator Extraction
# ==============================================================================

def llm_extract_fence_elements(llm, text: str, keywords: List[str], max_items: int = 100) -> List[Dict]:
    """Robust LLM extraction of fence-related legend entries."""
    if not llm or not text: return []

    hint_keywords = ", ".join(sorted(set(keywords)))
    analysis_prompt = f"""
    You are an assistant reviewing engineering drawing documentation. Extract fence-related
    legend entries, callouts or tags and provide paired indicator + text elements. 
    Only return items that clearly map to: {hint_keywords}.

    Text to analyse:
    <TEXT>
    {text.strip()[:4000]}
    </TEXT>

    Respond with a JSON array where each element has:
    - "indicator": the numeric or symbolic tag (e.g., "1", "F-3", "A", "3301")
    - "text_element": the textual description (e.g., "existing fence", "chain link")
    - "description": concise sentence on how the element relates to fencing
    """
    try:
        raw_response = llm.invoke(analysis_prompt) if hasattr(llm, "invoke") else llm(analysis_prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
    except Exception as exc:
        print(f"⚠️ LLM call failed: {exc}")
        return []

    parsed = []
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(0))
            if isinstance(parsed_json, list):
                for item in parsed_json:
                    if not isinstance(item, dict): continue
                    ind = str(item.get("indicator") or "").strip()
                    txt = str(item.get("text_element") or "").strip()
                    desc = str(item.get("description") or "").strip()
                    if ind or txt:
                        parsed.append({"indicator": ind, "text_element": txt, "description": desc})
    except Exception:
        pass
    return parsed

def extract_legend_entries(
    legend_chunks: List[Dict],
    ocr_tokens: List[Dict],
    pdf_words: List[Dict],
    pdf_blocks: List[Dict],
    fence_keywords: List[str],
    llm
) -> List[Dict]:
    """
    Process legend chunks using LINE-BASED matching.
    Returns full-line bounding boxes for matched items.
    """
    results = []
    
    for chunk in legend_chunks:
        text = chunk.get("text", "")
        if not text: continue
        
        # 1. LLM Extraction
        items = llm_extract_fence_elements(llm, text, fence_keywords)
        
        chunk_bbox = (chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"])
        
        # 2. Match back using Line-Based Logic
        for item in items:
            found_bbox = None
            
            # A. Try matching the 'text_element' (Description)
            # This typically finds the whole line "3301 CHAIN LINK FENCE" if on same line
            if item["text_element"]:
                found_bbox = find_text_in_ocr_and_pdf(
                    search_text=item["text_element"],
                    ocr_tokens=ocr_tokens,
                    pdf_blocks=pdf_blocks,
                    pdf_words=pdf_words,
                    chunk_bbox=chunk_bbox,
                    tolerance=50.0
                )
            
            # B. If not found, try the indicator
            if not found_bbox and item["indicator"]:
                found_bbox = find_text_in_ocr_and_pdf(
                    search_text=item["indicator"],
                    ocr_tokens=ocr_tokens,
                    pdf_blocks=pdf_blocks,
                    pdf_words=pdf_words,
                    chunk_bbox=chunk_bbox,
                    tolerance=50.0
                )

            if found_bbox:
                results.append({
                    "indicator": item["indicator"],
                    "keyword": item["text_element"],
                    "description": item["description"],
                    "x0": found_bbox["x0"], 
                    "y0": found_bbox["y0"], 
                    "x1": found_bbox["x1"], 
                    "y1": found_bbox["y1"],
                    "source": found_bbox.get("source", "legend")
                })
                
    return results

def find_instances_in_figures(
    legend_entries: List[Dict],
    figure_chunks: List[Dict],
    all_tokens: List[Dict]
) -> List[Dict]:
    """
    Find indicator codes from legend appearing in figure regions.
    (This keeps the strict token matching for drawings as desired)
    """
    instances = []
    indicators_to_find = {item["indicator"] for item in legend_entries if item["indicator"]}
    
    if not indicators_to_find: return []
    
    # Filter tokens to only those inside figure chunks
    figure_tokens = []
    for chunk in figure_chunks:
        cx0, cy0, cx1, cy1 = chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]
        ft = [t for t in all_tokens if t["x0"] >= cx0 and t["y0"] >= cy0 and t["x1"] <= cx1 and t["y1"] <= cy1]
        figure_tokens.extend(ft)
        
    for token in figure_tokens:
        token_text = token["text"].strip()
        clean_text = re.sub(r'[^\w]', '', token_text)
        
        if clean_text in indicators_to_find:
            instances.append({
                "indicator": clean_text,
                "x0": token["x0"], "y0": token["y0"], "x1": token["x1"], "y1": token["y1"],
                "source": "figure_instance"
            })
            
    return instances

# ==============================================================================
# 6. Visualization & Utils
# ==============================================================================

# ==============================================================================
# 6. Visualization & Utils
# ==============================================================================

def highlight_page_image(
    page_image_bytes: bytes, 
    definitions: List[Dict], 
    instances: List[Dict],
    pdf_width: float,   # <--- NEW ARGUMENT
    pdf_height: float   # <--- NEW ARGUMENT
) -> bytes:
    """Draw boxes with scaling correction (PDF Points -> Image Pixels)."""
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Calculate Scale Factor (Image Pixels / PDF Points)
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0

        # Helper to scale a box
        def scale_box(box_dict):
            return [
                box_dict.get("x0", 0) * scale_x,
                box_dict.get("y0", 0) * scale_y,
                box_dict.get("x1", 0) * scale_x,
                box_dict.get("y1", 0) * scale_y
            ]

        # Definitions (Green)
        for d in definitions:
            box = scale_box(d)
            draw.rectangle(box, outline=(0, 255, 0, 255), width=3)
            draw.rectangle(box, fill=(0, 255, 0, 40))

        # Instances (Magenta)
        for i in instances:
            box = scale_box(i)
            draw.rectangle(box, outline=(255, 0, 255, 255), width=3)
        
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"Visualization Error: {e}")
        return page_image_bytes
    
    
def create_single_page_pdf(pdf_bytes: bytes, page_index: int) -> bytes:
    """Extract a single page as a new PDF file."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out

# ==============================================================================
# 3. PDF Native Text Extraction
# ==============================================================================

def get_pdf_text_elements(page: fitz.Page) -> Tuple[List[Dict], List[Dict]]:
    """
    Get native text words AND blocks using PyMuPDF.
    Returns (words, blocks).
    """
    # Get Words
    words_raw = page.get_text("words") 
    words = []
    for w in words_raw:
        words.append({
            "text": w[4],
            "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
            "source": "native_word"
        })
        
    # Get Blocks (for fallback)
    blocks_raw = page.get_text("blocks")
    blocks = []
    for b in blocks_raw:
        blocks.append({
            "text": b[4],
            "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
            "source": "native_block"
        })
        
    return words, blocks