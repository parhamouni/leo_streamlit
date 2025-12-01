"""
utils_ade.py - Unified ADE Utility (Native Lines + Robust Overlap + DEBUG LOGGING)
"""

import re
import json
import time
import requests
from typing import List, Dict, Optional, Tuple, Set
from io import BytesIO
from difflib import SequenceMatcher
from collections import defaultdict

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

# Optional Google Cloud imports
try:
    from google.cloud import documentai_v1 as documentai
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    documentai = None
    print("[DEBUG] Google Cloud SDK not found.")

# ==============================================================================
# 1. ADE (LandingAI) Integration
# ==============================================================================

ADE_PARSE_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"


def ade_parse_document(pdf_bytes: bytes, api_key: str, zdr: bool = False) -> Dict:
    print(f"[DEBUG] Starting ADE Parsing for document ({len(pdf_bytes)} bytes)...")
    if not api_key:
        return {"success": False, "error": "Missing ADE API Key"}

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {"options": json.dumps({"zdr": zdr})} if zdr else {}

    for attempt in range(3):
        try:
            print(f"[DEBUG] ADE API Request - Attempt {attempt+1}")
            response = requests.post(
                ADE_PARSE_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=600  # 10 minutes for large architectural PDFs
            )
            response.raise_for_status()
            result = response.json()

            chunks = result.get("chunks", [])
            pages = {c.get("grounding", {}).get("page", 0) for c in chunks}
            total_pages = max(pages) + 1 if pages else 0
            print(f"[DEBUG] ADE Success! Found {len(chunks)} chunks across {total_pages} pages.")

            return {
                "success": True,
                "data": {
                    "chunks": chunks,
                    "total_pages": total_pages,
                    "raw": result
                }
            }
        except Exception as e:
            print(f"[DEBUG] ADE Attempt {attempt+1} Failed: {e}")
            if attempt == 2:
                return {"success": False, "error": str(e)}
            time.sleep(2 * (attempt + 1))

    return {"success": False, "error": "Unknown error after retries"}


def align_ade_chunks_to_page(ade_result: Dict, page_idx: int, page_width: float, page_height: float) -> List[Dict]:
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
    """
    Segment ADE chunks into legend-like (tables, text with keywords) and figure-like (drawings).
    """
    legend_like = []
    figure_like = []
    
    for chunk in chunks:
        raw_type = (chunk.get("type") or "").lower()
        text_lower = (chunk.get("text") or "").lower()
        
        # Tables are always legend-like (they contain structured data)
        if raw_type == "table":
            legend_like.append(chunk)
            continue
        
        # Check for legend/keynote keywords
        has_legend_hint = any(token in text_lower for token in {"legend", "keynote", "abbreviation", "symbol", "notes"})
        
        # Figures without legend hints go to figure_like
        is_figure = raw_type in {"figure", "architectural_drawing", "image"}
        
        if is_figure and not has_legend_hint:
            figure_like.append(chunk)
        else:
            # Text chunks, or figures with legend hints
            legend_like.append(chunk)
    
    print(f"[DEBUG] Segmented: {len(legend_like)} Legend-like, {len(figure_like)} Figure-like")
    return legend_like, figure_like


# ==============================================================================
# 2. Google Cloud Document AI (OCR) - SAFE HIGH DPI (DEBUGGED)
# ==============================================================================

_DOCAI_CLIENT_CACHE = None


def get_docai_client(google_cloud_config: Dict):
    global _DOCAI_CLIENT_CACHE
    if _DOCAI_CLIENT_CACHE:
        return _DOCAI_CLIENT_CACHE
    if not GOOGLE_CLOUD_AVAILABLE:
        print("[DEBUG] Google Cloud libraries missing.")
        return None

    try:
        service_info = google_cloud_config.get("service_account_info")
        if service_info:
            creds = service_account.Credentials.from_service_account_info(service_info)
            _DOCAI_CLIENT_CACHE = documentai.DocumentProcessorServiceClient(credentials=creds)
            return _DOCAI_CLIENT_CACHE
    except Exception as e:
        print(f"[DEBUG] ❌ Error creating DocAI client: {e}")
        return None


def run_google_ocr_blocks(page_bytes: bytes, google_cloud_config: Dict, w: float, h: float) -> List[Dict]:
    print("[DEBUG] Starting Google OCR...")
    client = get_docai_client(google_cloud_config)
    if not client:
        return []

    project_id = google_cloud_config.get("project_number")
    location = google_cloud_config.get("location")
    processor_id = google_cloud_config.get("processor_id")
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    doc = fitz.open(stream=page_bytes, filetype="pdf")
    page = doc[0]

    # --- DYNAMIC ZOOM ---
    max_dimension = 4000.0
    current_max = max(page.rect.width, page.rect.height)
    zoom = min(3.0, max_dimension / current_max)
    print(f"[DEBUG] Rendering page image with Zoom: {zoom:.2f}...")

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # --- JPEG COMPRESSION ---
    image_content = pix.tobytes("jpeg", jpg_quality=85)
    img_size_mb = len(image_content) / (1024 * 1024)
    print(f"[DEBUG] Image rendered. Size: {img_size_mb:.2f} MB. Sending to Google API...")

    raw_document = documentai.RawDocument(content=image_content, mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    try:
        result = client.process_document(request=request)
        doc_result = result.document
        print(f"[DEBUG] Google API returned successfully. Text length: {len(doc_result.text)}")
    except Exception as e:
        print(f"[DEBUG] ❌ DocAI Processing Failed: {e}")
        return []

    if not doc_result.pages:
        return []
    ocr_page = doc_result.pages[0]

    ocr_lines = []

    for paragraph in ocr_page.paragraphs:
        text_content = ""
        layout = paragraph.layout
        for segment in layout.text_anchor.text_segments:
            text_content += doc_result.text[segment.start_index:segment.end_index]

        text_content = text_content.strip()
        if not text_content:
            continue

        vertices = layout.bounding_poly.normalized_vertices
        if not vertices:
            continue

        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]

        ocr_lines.append({
            "text": text_content,
            "x0": min(xs) * w, "y0": min(ys) * h,
            "x1": max(xs) * w, "y1": max(ys) * h,
            "source": "ocr_paragraph"
        })

    print(f"[DEBUG] OCR extraction complete. Found {len(ocr_lines)} paragraphs.")
    return ocr_lines


# ==============================================================================
# 3. PDF Native Text Extraction
# ==============================================================================


def get_native_pdf_lines(page: fitz.Page) -> List[Dict]:
    structure = page.get_text("dict")
    lines = []
    for block in structure.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                bbox = line["bbox"]
                if text:
                    lines.append({
                        "text": text,
                        "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3],
                        "source": "pdf_native"
                    })
    print(f"[DEBUG] PDF Native extraction: Found {len(lines)} lines.")
    return lines


# ==============================================================================
# 4. Matching Logic (Center-Point Strategy)
# ==============================================================================


def is_center_inside(item_box: Dict, chunk_box: Tuple[float, float, float, float], tolerance: float = 15.0) -> bool:
    item_cx = (item_box["x0"] + item_box["x1"]) / 2
    item_cy = (item_box["y0"] + item_box["y1"]) / 2
    cx0, cy0, cx1, cy1 = chunk_box

    inside_x = (cx0 - tolerance) <= item_cx <= (cx1 + tolerance)
    inside_y = (cy0 - tolerance) <= item_cy <= (cy1 + tolerance)
    return inside_x and inside_y


def find_best_bbox(search_text: str, pdf_lines: List[Dict], ocr_lines: List[Dict], chunk_bbox: Tuple[float, float, float, float], **kwargs) -> Optional[Dict]:
    if not search_text:
        return None
    target = re.sub(r'[^0-9a-zA-Z]', '', search_text).lower()
    if not target:
        return None

    def check_candidates(candidates):
        for line in candidates:
            line_clean = re.sub(r'[^0-9a-zA-Z]', '', line["text"]).lower()
            if target not in line_clean:
                continue
            # Global search override (if chunk is huge) or local check
            if chunk_bbox[2] > 5000 or is_center_inside(line, chunk_bbox):
                return line
        return None

    match = check_candidates(pdf_lines)
    if match:
        return match
    match = check_candidates(ocr_lines)
    if match:
        return match
    return None


# ==============================================================================
# 5. Keyword & Indicator Extraction
# ==============================================================================


def parse_table_rows(table_text: str) -> List[Dict]:
    """
    Parse HTML table rows from ADE markdown to extract indicator→description pairs.
    ADE returns tables like: <tr><td>18</td><td>CMU SCREEN WALL</td></tr>
    """
    items = []
    
    # Find all table rows
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
    
    for row_match in row_pattern.finditer(table_text):
        row_content = row_match.group(1)
        cells = cell_pattern.findall(row_content)
        
        if len(cells) >= 2:
            # First cell is usually indicator, second is description
            indicator = re.sub(r'<[^>]+>', '', cells[0]).strip()
            description = re.sub(r'<[^>]+>', '', cells[1]).strip()
            
            # Clean up HTML entities
            description = description.replace('&amp;', '&').replace('&quot;', '"').replace('&lt;', '<').replace('&gt;', '>')
            
            if indicator and description:
                items.append({
                    "indicator": indicator,
                    "text_element": description,
                    "description": f"Table entry: {description[:50]}..."
                })
    
    print(f"[DEBUG] Parsed {len(items)} items from table HTML")
    return items


def llm_extract_fence_elements(llm, text: str, keywords: List[str], max_items: int = 100) -> List[Dict]:
    if not llm or not text:
        return []
    hint_keywords = ", ".join(sorted(set(keywords)))
    print(f"[DEBUG] Asking LLM to extract items from text length {len(text)}...")

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
    llm
) -> List[Dict]:
    print("[DEBUG] Extracting Legend Entries and Matching BBoxes...")
    results = []
    fence_keywords_lower = [kw.lower() for kw in fence_keywords]

    # Helper: Make a shortened version of the text
    def get_substring(text):
        words = text.split()
        if len(words) > 4:
            return " ".join(words[:4])
        return text
    
    # Helper: Check if text is fence-related
    def is_fence_related(text):
        text_lower = text.lower()
        return any(kw in text_lower for kw in fence_keywords_lower)

    for chunk in legend_chunks:
        text = chunk.get("text", "")
        markdown = chunk.get("markdown", "")
        chunk_type = chunk.get("type", "").lower()
        
        if not text and not markdown:
            continue
        
        items = []
        
        # For TABLE chunks, parse HTML structure first
        if chunk_type == "table" and "<tr" in (markdown or text):
            table_items = parse_table_rows(markdown or text)
            # Filter to fence-related items only
            items = [item for item in table_items if is_fence_related(item.get("text_element", ""))]
            print(f"[DEBUG] Table chunk: {len(table_items)} total rows, {len(items)} fence-related")
        
        # If no table items or not a table, use LLM extraction
        if not items:
            items = llm_extract_fence_elements(llm, text, fence_keywords)
        
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
                    "source": bbox_desc.get("source", "unknown") + "_desc"
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
                        "source": bbox_ind.get("source", "unknown") + "_ind"
                    })
    print(f"[DEBUG] Finished Legend Extraction. Total mapped items: {len(results)}")
    return results


def find_instances_in_figures(legend_entries: List[Dict], figure_chunks: List[Dict], all_tokens: List[Dict]) -> List[Dict]:
    print("[DEBUG] Finding Instances in Figures...")
    instances = []
    indicators_to_find = {item["indicator"] for item in legend_entries if item["indicator"]}

    if not indicators_to_find:
        return []

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
    print(f"[DEBUG] Found {len(instances)} figure instances.")
    return instances


# ==============================================================================
# 6. Visualization & Utils
# ==============================================================================


def highlight_page_image(page_image_bytes: bytes, definitions: List[Dict], instances: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    print("[DEBUG] Generating Highlighted Image...")
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0

        def scale_box(box_dict):
            return [
                box_dict.get("x0", 0) * scale_x,
                box_dict.get("y0", 0) * scale_y,
                box_dict.get("x1", 0) * scale_x,
                box_dict.get("y1", 0) * scale_y
            ]

        for d in definitions:
            box = scale_box(d)
            draw.rectangle(box, outline=(0, 255, 0, 255), width=3)
            draw.rectangle(box, fill=(0, 255, 0, 40))
        for i in instances:
            box = scale_box(i)
            draw.rectangle(box, outline=(255, 0, 255, 255), width=3)

        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Visualization Error: {e}")
        return page_image_bytes


def create_single_page_pdf(pdf_bytes: bytes, page_index: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def debug_visualize_coordinates(page_bytes: bytes, ade_chunks: List[Dict], pdf_lines: List[Dict], ocr_lines: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    print("[DEBUG] Generating Layer Visualization...")
    try:
        img = Image.open(BytesIO(page_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0

        def get_rect(item):
            return [
                item.get("x0", 0) * scale_x,
                item.get("y0", 0) * scale_y,
                item.get("x1", 0) * scale_x,
                item.get("y1", 0) * scale_y
            ]

        for line in pdf_lines:
            draw.rectangle(get_rect(line), outline=(0, 0, 255, 128), width=1)
        for line in ocr_lines:
            draw.rectangle(get_rect(line), outline=(255, 165, 0, 128), width=1)
        for chunk in ade_chunks:
            draw.rectangle(get_rect(chunk), outline=(255, 0, 0, 255), width=3)

        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Debug Viz Error: {e}")
        return page_bytes


# ==============================================================================
# 7. NEW: Per-Page ADE Processing (Fixes Timeout)
# ==============================================================================

def ade_parse_single_page(pdf_bytes: bytes, page_idx: int, api_key: str, zdr: bool = False) -> Dict:
    """Parse a single page via ADE to avoid timeout on large documents."""
    single_page_pdf = create_single_page_pdf(pdf_bytes, page_idx)
    print(f"[DEBUG] Parsing single page {page_idx + 1} ({len(single_page_pdf)} bytes)...")
    
    if not api_key:
        return {"success": False, "error": "Missing ADE API Key", "chunks": []}

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": ("page.pdf", single_page_pdf, "application/pdf")}
    data = {"options": json.dumps({"zdr": zdr})} if zdr else {}

    for attempt in range(3):
        try:
            print(f"[DEBUG] ADE Page {page_idx + 1} - Attempt {attempt + 1}")
            response = requests.post(
                ADE_PARSE_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=180  # 3 min per page is enough
            )
            response.raise_for_status()
            result = response.json()
            chunks = result.get("chunks", [])
            
            # Normalize page index to 0 for single-page PDF
            for chunk in chunks:
                if "grounding" in chunk:
                    chunk["grounding"]["page"] = page_idx
            
            print(f"[DEBUG] Page {page_idx + 1} Success: {len(chunks)} chunks")
            return {"success": True, "chunks": chunks}
        except Exception as e:
            print(f"[DEBUG] Page {page_idx + 1} Attempt {attempt + 1} Failed: {e}")
            if attempt == 2:
                return {"success": False, "error": str(e), "chunks": []}
            time.sleep(2 * (attempt + 1))

    return {"success": False, "error": "Unknown error", "chunks": []}


def ade_parse_document_pagewise(pdf_bytes: bytes, api_key: str, zdr: bool = False) -> Dict:
    """Parse document page-by-page to avoid timeout issues."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    doc.close()
    
    print(f"[DEBUG] Starting page-wise ADE parsing for {total_pages} pages...")
    all_chunks = []
    failed_pages = []
    
    for page_idx in range(total_pages):
        result = ade_parse_single_page(pdf_bytes, page_idx, api_key, zdr)
        if result["success"]:
            all_chunks.extend(result["chunks"])
        else:
            failed_pages.append(page_idx + 1)
            print(f"[DEBUG] Warning: Page {page_idx + 1} failed: {result.get('error')}")
    
    if failed_pages:
        print(f"[DEBUG] ADE completed with {len(failed_pages)} failed pages: {failed_pages}")
    
    return {
        "success": len(all_chunks) > 0 or len(failed_pages) < total_pages,
        "data": {
            "chunks": all_chunks,
            "total_pages": total_pages,
            "failed_pages": failed_pages
        },
        "error": f"Failed pages: {failed_pages}" if failed_pages else None
    }


# ==============================================================================
# 8. NEW: Unified Text Extraction (PDF + OCR Merged) - IMPROVED
# ==============================================================================

def normalize_text_for_matching(text: Optional[str]) -> str:
    """Normalize text for matching, handling common OCR errors."""
    if not text:
        return ""
    # Common OCR substitutions
    replacements = {
        "O": "0", "o": "0",
        "I": "1", "l": "1",
        "B": "8",
        "S": "5",
    }
    cleaned = "".join(replacements.get(ch, ch) for ch in str(text).strip())
    return cleaned.replace(" ", "").upper()


def extract_pdf_text_words(page: fitz.Page) -> List[Dict]:
    """Extract word-level tokens from PDF text layer."""
    words = []
    for x0, y0, x1, y1, text, block_no, line_no, word_no in page.get_text("words"):
        words.append({
            "text": text,
            "normalized_text": normalize_text_for_matching(text),
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "block_no": block_no,
            "line_no": line_no,
            "word_no": word_no,
            "source": "pdf_native"
        })
    return words


def run_ocr_word_level(
    page: fitz.Page,
    google_cloud_config: Dict,
    pdf_bytes: bytes,
    page_idx: int
) -> List[Dict]:
    """Run OCR and return word-level tokens with coordinates."""
    if not google_cloud_config or not GOOGLE_CLOUD_AVAILABLE:
        return []
    
    client = get_docai_client(google_cloud_config)
    if not client:
        return []
    
    project_id = google_cloud_config.get("project_number")
    location = google_cloud_config.get("location")
    processor_id = google_cloud_config.get("processor_id")
    name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
    
    # Create single page PDF and render
    single_page_pdf = create_single_page_pdf(pdf_bytes, page_idx)
    doc = fitz.open(stream=single_page_pdf, filetype="pdf")
    ocr_page = doc[0]
    
    w, h = page.rect.width, page.rect.height
    
    # Dynamic zoom for quality
    max_dimension = 4000.0
    current_max = max(ocr_page.rect.width, ocr_page.rect.height)
    zoom = min(3.0, max_dimension / current_max)
    
    mat = fitz.Matrix(zoom, zoom)
    pix = ocr_page.get_pixmap(matrix=mat, alpha=False)
    image_content = pix.tobytes("jpeg", jpg_quality=85)
    
    raw_document = documentai.RawDocument(content=image_content, mime_type="image/jpeg")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    
    try:
        result = client.process_document(request=request)
        doc_result = result.document
    except Exception as e:
        print(f"[DEBUG] OCR failed: {e}")
        return []
    
    if not doc_result.pages:
        return []
    
    ocr_page_result = doc_result.pages[0]
    tokens = []
    
    # Extract word-level tokens
    for token in ocr_page_result.tokens:
        anchor = token.layout.text_anchor
        text = ""
        if anchor and anchor.text_segments:
            for segment in anchor.text_segments:
                text += doc_result.text[segment.start_index:segment.end_index]
        text = text.strip()
        if not text:
            continue
        
        layout = token.layout
        poly = getattr(layout, "bounding_poly", None)
        if not poly:
            continue
        
        vertices = list(getattr(poly, "normalized_vertices", []))
        if not vertices:
            continue
        
        xs = [v.x for v in vertices if v.x is not None]
        ys = [v.y for v in vertices if v.y is not None]
        if len(xs) < 2 or len(ys) < 2:
            continue
        
        tokens.append({
            "text": text,
            "normalized_text": normalize_text_for_matching(text),
            "x0": min(xs) * w,
            "y0": min(ys) * h,
            "x1": max(xs) * w,
            "y1": max(ys) * h,
            "confidence": getattr(layout, "confidence", 1.0),
            "source": "ocr"
        })
    
    doc.close()
    print(f"[DEBUG] OCR extracted {len(tokens)} word-level tokens")
    return tokens


def calculate_overlap_ratio(bbox1: Dict, bbox2: Dict) -> float:
    """Calculate overlap ratio relative to smaller box."""
    x1_min, y1_min = float(bbox1.get("x0", 0)), float(bbox1.get("y0", 0))
    x1_max, y1_max = float(bbox1.get("x1", 0)), float(bbox1.get("y1", 0))
    x2_min, y2_min = float(bbox2.get("x0", 0)), float(bbox2.get("y0", 0))
    x2_max, y2_max = float(bbox2.get("x1", 0)), float(bbox2.get("y1", 0))
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    smaller_area = min(area1, area2)
    if smaller_area <= 0:
        return 0.0
    
    return intersection_area / smaller_area


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using normalized text."""
    norm1 = normalize_text_for_matching(text1)
    norm2 = normalize_text_for_matching(text2)
    
    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # One contains the other
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    
    # Character overlap (Jaccard)
    set1 = set(norm1)
    set2 = set(norm2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def merge_pdf_ocr_tokens(
    pdf_words: List[Dict],
    ocr_tokens: List[Dict],
    overlap_threshold: float = 0.2,
    text_similarity_threshold: float = 0.5,
) -> List[Dict]:
    """
    Merge PDF and OCR tokens by finding overlaps and combining them.
    Based on the approach from three_level_extraction_merge.ipynb.
    
    Returns unified list with source tracking.
    """
    print(f"[DEBUG] Merging {len(pdf_words)} PDF words + {len(ocr_tokens)} OCR tokens...")
    
    merged_tokens = []
    ocr_used = set()
    
    # Start with PDF words as base
    for pdf_word in pdf_words:
        pdf_bbox = {
            "x0": float(pdf_word.get("x0", 0)),
            "y0": float(pdf_word.get("y0", 0)),
            "x1": float(pdf_word.get("x1", 0)),
            "y1": float(pdf_word.get("y1", 0)),
        }
        pdf_text = pdf_word.get("text", "")
        
        # Find matching OCR tokens
        matching_ocr = []
        for idx, ocr_token in enumerate(ocr_tokens):
            if idx in ocr_used:
                continue
            
            ocr_bbox = {
                "x0": float(ocr_token.get("x0", 0)),
                "y0": float(ocr_token.get("y0", 0)),
                "x1": float(ocr_token.get("x1", 0)),
                "y1": float(ocr_token.get("y1", 0)),
            }
            ocr_text = ocr_token.get("text", "")
            
            # Check overlap
            overlap = calculate_overlap_ratio(pdf_bbox, ocr_bbox)
            if overlap < overlap_threshold:
                continue
            
            # Check text similarity
            similarity = calculate_text_similarity(pdf_text, ocr_text)
            if similarity < text_similarity_threshold:
                continue
            
            matching_ocr.append((idx, ocr_token, overlap, similarity))
        
        # Merge if matches found
        if matching_ocr:
            # Merge bounding boxes (union)
            all_x0 = [pdf_bbox["x0"]] + [ocr_tokens[idx]["x0"] for idx, _, _, _ in matching_ocr]
            all_y0 = [pdf_bbox["y0"]] + [ocr_tokens[idx]["y0"] for idx, _, _, _ in matching_ocr]
            all_x1 = [pdf_bbox["x1"]] + [ocr_tokens[idx]["x1"] for idx, _, _, _ in matching_ocr]
            all_y1 = [pdf_bbox["y1"]] + [ocr_tokens[idx]["y1"] for idx, _, _, _ in matching_ocr]
            
            # Use best text (prefer longer, or OCR if higher confidence)
            texts = [pdf_text] + [ocr_tokens[idx]["text"] for idx, _, _, _ in matching_ocr]
            best_text = max(texts, key=len)
            
            # Mark OCR tokens as used
            for idx, _, _, _ in matching_ocr:
                ocr_used.add(idx)
            
            merged_tokens.append({
                "text": best_text,
                "normalized_text": normalize_text_for_matching(best_text),
                "x0": min(all_x0),
                "y0": min(all_y0),
                "x1": max(all_x1),
                "y1": max(all_y1),
                "source": "pdf+ocr",
                "sources_count": 1 + len(matching_ocr),
                "original_pdf_text": pdf_text,
            })
        else:
            # No match, keep PDF word as-is
            merged_tokens.append({
                "text": pdf_text,
                "normalized_text": normalize_text_for_matching(pdf_text),
                "x0": pdf_bbox["x0"],
                "y0": pdf_bbox["y0"],
                "x1": pdf_bbox["x1"],
                "y1": pdf_bbox["y1"],
                "source": "pdf_only",
                "sources_count": 1,
            })
    
    # Add remaining OCR tokens that weren't matched
    for idx, ocr_token in enumerate(ocr_tokens):
        if idx in ocr_used:
            continue
        
        merged_tokens.append({
            "text": ocr_token.get("text", ""),
            "normalized_text": normalize_text_for_matching(ocr_token.get("text", "")),
            "x0": float(ocr_token.get("x0", 0)),
            "y0": float(ocr_token.get("y0", 0)),
            "x1": float(ocr_token.get("x1", 0)),
            "y1": float(ocr_token.get("y1", 0)),
            "source": "ocr_only",
            "sources_count": 1,
        })
    
    # Count sources
    source_counts = defaultdict(int)
    for token in merged_tokens:
        source_counts[token.get("source", "unknown")] += 1
    
    print(f"[DEBUG] Merge result: {len(merged_tokens)} tokens")
    print(f"[DEBUG]   pdf+ocr: {source_counts['pdf+ocr']}, pdf_only: {source_counts['pdf_only']}, ocr_only: {source_counts['ocr_only']}")
    
    return merged_tokens


def get_unified_text_lines(
    page: fitz.Page,
    google_cloud_config: Optional[Dict] = None,
    pdf_bytes: Optional[bytes] = None,
    page_idx: int = 0
) -> List[Dict]:
    """
    Merge PDF native text and OCR text into a unified list.
    Uses word-level extraction and proper overlap-based merging.
    """
    print("[DEBUG] Building unified text (word-level PDF + OCR)...")
    
    # Get PDF word-level tokens
    pdf_words = extract_pdf_text_words(page)
    print(f"[DEBUG] PDF text layer: {len(pdf_words)} words")
    
    # Get OCR word-level tokens if configured
    ocr_tokens = []
    if google_cloud_config and pdf_bytes:
        ocr_tokens = run_ocr_word_level(page, google_cloud_config, pdf_bytes, page_idx)
    
    # Merge with proper deduplication
    if ocr_tokens:
        unified = merge_pdf_ocr_tokens(pdf_words, ocr_tokens)
    else:
        unified = pdf_words
        print(f"[DEBUG] No OCR, using {len(pdf_words)} PDF words only")
    
    return unified


def compute_iou(box1: Dict, box2: Dict) -> float:
    """Compute Intersection over Union for two boxes."""
    x1 = max(box1["x0"], box2["x0"])
    y1 = max(box1["y0"], box2["y0"])
    x2 = min(box1["x1"], box2["x1"])
    y2 = min(box1["y1"], box2["y1"])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1["x1"] - box1["x0"]) * (box1["y1"] - box1["y0"])
    area2 = (box2["x1"] - box2["x0"]) * (box2["y1"] - box2["y0"])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ==============================================================================
# 9. NEW: Page Classification (Is This Page Fence-Related?)
# ==============================================================================

FENCE_CLASSIFICATION_PROMPT = """Analyze this architectural drawing page text and determine if it contains fence, gate, barrier, or wall-related information.

Text from page:
<TEXT>
{text}
</TEXT>

Respond with JSON:
{{
    "is_fence_related": true/false,
    "confidence": 0.0-1.0,
    "fence_indicators": ["list", "of", "found", "indicators"],
    "fence_keywords": ["list", "of", "fence", "keywords", "found"],
    "reason": "brief explanation"
}}

Look for:
- Legend entries mentioning fence, gate, barrier, guardrail, wall, CMU, chain link, mesh, bollard
- Indicator codes (like 0401, 3301, F-1) associated with fence elements
- Drawing callouts referencing fence specifications
"""

def classify_page_fence_related(page_text: str, llm, fence_keywords: List[str] = None) -> Dict:
    """
    Use LLM to determine if a page contains fence-related content.
    Returns classification result with indicators and keywords found.
    
    IMPORTANT: page_text should be full page text from page.get_text(), 
    NOT fragmented word tokens!
    """
    if not llm or not page_text:
        return {"is_fence_related": False, "confidence": 0.0, "fence_indicators": [], "fence_keywords": [], "reason": "No text or LLM"}
    
    # Quick keyword pre-check - if ANY keyword found, page is fence-related
    text_lower = page_text.lower()
    quick_keywords = fence_keywords or ["fence", "gate", "barrier", "guardrail", "wall", "cmu", "chain link", "bollard", "screen"]
    found_keywords = [kw for kw in quick_keywords if kw in text_lower]
    
    if not found_keywords:
        print("[DEBUG] Page classification: No fence keywords found (quick check)")
        return {"is_fence_related": False, "confidence": 0.9, "fence_indicators": [], "fence_keywords": [], "reason": "No fence keywords in text"}
    
    # Keywords found - page IS fence-related (skip LLM to avoid false negatives)
    print(f"[DEBUG] Page classification: fence_related=True (keywords found: {found_keywords})")
    return {
        "is_fence_related": True, 
        "confidence": 0.8, 
        "fence_indicators": [], 
        "fence_keywords": found_keywords, 
        "reason": f"Keywords found: {', '.join(found_keywords)}"
    }


# ==============================================================================
# 10. NEW: Indicator Validation (Regex Patterns)
# ==============================================================================

# Common indicator patterns in architectural drawings
INDICATOR_PATTERNS = [
    re.compile(r'^[A-Z]?-?\d{1,4}[A-Z]?$'),           # 3301, A-1, F3, 0401
    re.compile(r'^\d{4}$'),                            # 3301, 0401 (4-digit codes)
    re.compile(r'^[A-Z]\d{1,3}$'),                     # A1, F12, B123
    re.compile(r'^\d{1,2}$'),                          # 1, 12, 45 (simple numbers)
    re.compile(r'^[A-Z]{1,2}-\d{1,3}$'),               # F-1, AB-12
]

# Patterns that look like indicators but are NOT (dimensions, scales, etc.)
INVALID_INDICATOR_PATTERNS = [
    re.compile(r"^\d+['\"]"),                          # 6', 12" (dimensions)
    re.compile(r"^\d+'-\d+"),                          # 6'-0" (dimensions)
    re.compile(r"^\d+/\d+"),                           # 1/4, 3/8 (fractions)
    re.compile(r"^\d+:\d+"),                           # 1:100 (scales)
    re.compile(r"^\d{5,}$"),                           # 5+ digit numbers (not indicators)
    re.compile(r"^\d+\.\d+$"),                         # 12.5 (decimals)
]


def is_valid_indicator(text: str) -> bool:
    """Check if text matches valid indicator patterns and not invalid ones."""
    text = text.strip()
    if not text:
        return False
    
    # Check invalid patterns first
    for pattern in INVALID_INDICATOR_PATTERNS:
        if pattern.match(text):
            return False
    
    # Check valid patterns
    for pattern in INDICATOR_PATTERNS:
        if pattern.match(text):
            return True
    
    return False


def validate_extracted_indicators(items: List[Dict]) -> List[Dict]:
    """Filter extracted items to only include valid indicators."""
    validated = []
    for item in items:
        indicator = item.get("indicator", "").strip()
        if indicator and is_valid_indicator(indicator):
            validated.append(item)
        elif indicator:
            print(f"[DEBUG] Filtered invalid indicator: '{indicator}'")
    
    print(f"[DEBUG] Indicator validation: {len(items)} -> {len(validated)} valid")
    return validated


# ==============================================================================
# 11. NEW: Fuzzy Matching for OCR Errors
# ==============================================================================

def fuzzy_match_indicator(text: str, indicators: Set[str], threshold: float = 0.75) -> Optional[str]:
    """
    Find best matching indicator using fuzzy string matching.
    Handles OCR errors like O/0, l/1, etc.
    """
    text_clean = re.sub(r'[^\w]', '', text).upper()
    if not text_clean:
        return None
    
    # Exact match first
    if text_clean in indicators:
        return text_clean
    
    # Common OCR substitutions
    ocr_variants = [
        text_clean,
        text_clean.replace('O', '0'),
        text_clean.replace('0', 'O'),
        text_clean.replace('l', '1'),
        text_clean.replace('1', 'l'),
        text_clean.replace('I', '1'),
        text_clean.replace('S', '5'),
        text_clean.replace('5', 'S'),
        text_clean.replace('B', '8'),
        text_clean.replace('8', 'B'),
    ]
    
    for variant in ocr_variants:
        if variant in indicators:
            return variant
    
    # Fuzzy match as last resort
    best_match = None
    best_score = threshold
    
    for indicator in indicators:
        score = SequenceMatcher(None, text_clean, indicator).ratio()
        if score > best_score:
            best_score = score
            best_match = indicator
    
    return best_match


def find_instances_in_figures_fuzzy(
    legend_entries: List[Dict],
    figure_chunks: List[Dict],
    all_tokens: List[Dict],
    fuzzy_threshold: float = 0.75
) -> List[Dict]:
    """
    Find instances with fuzzy matching to handle OCR errors.
    """
    print("[DEBUG] Finding Instances in Figures (with fuzzy matching)...")
    instances = []
    indicators_to_find = {item["indicator"].upper() for item in legend_entries if item.get("indicator")}
    
    if not indicators_to_find:
        print("[DEBUG] No indicators to search for")
        return []
    
    # Get tokens within figure regions
    figure_tokens = []
    for chunk in figure_chunks:
        cx0, cy0, cx1, cy1 = chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]
        ft = [t for t in all_tokens if t["x0"] >= cx0 - 5 and t["y0"] >= cy0 - 5 and t["x1"] <= cx1 + 5 and t["y1"] <= cy1 + 5]
        figure_tokens.extend(ft)
    
    # Also search all tokens if no figure chunks defined
    if not figure_tokens:
        figure_tokens = all_tokens
    
    seen_positions = set()  # Avoid duplicate instances at same position
    
    for token in figure_tokens:
        token_text = token.get("text", "").strip()
        if not token_text:
            continue
        
        matched_indicator = fuzzy_match_indicator(token_text, indicators_to_find)
        if matched_indicator:
            pos_key = (round(token["x0"], 1), round(token["y0"], 1))
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                instances.append({
                    "indicator": matched_indicator,
                    "original_text": token_text,
                    "x0": token["x0"], "y0": token["y0"],
                    "x1": token["x1"], "y1": token["y1"],
                    "source": "figure_instance_fuzzy"
                })
    
    print(f"[DEBUG] Found {len(instances)} figure instances (fuzzy)")
    return instances


# ==============================================================================
# 12. NEW: LLM-Based Legend Detection
# ==============================================================================

LEGEND_DETECTION_PROMPT = """Analyze these document chunks and identify which ones contain legend/keynote tables.

A legend table typically:
- Lists indicator codes (like 1, 2, 3 or A, B, C or 0401, 3301)
- Maps codes to descriptions
- Has structured rows/columns
- Contains terms like "legend", "keynote", "symbol", "abbreviation"

Chunks to analyze:
{chunks_json}

Respond with JSON:
{{
    "legend_chunk_ids": ["id1", "id2"],
    "reason": "explanation"
}}
"""

def detect_legend_chunks_llm(chunks: List[Dict], llm) -> List[str]:
    """
    Use LLM to identify which chunks contain legend tables.
    Returns list of chunk IDs that are legends.
    """
    if not llm or not chunks:
        return []
    
    # Prepare chunks summary for LLM
    chunks_summary = []
    for chunk in chunks:
        chunks_summary.append({
            "id": chunk.get("id", ""),
            "type": chunk.get("type", ""),
            "text_preview": (chunk.get("text", "") or "")[:500]
        })
    
    prompt = LEGEND_DETECTION_PROMPT.format(chunks_json=json.dumps(chunks_summary, indent=2)[:8000])
    
    try:
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(response, "content", str(response))
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            legend_ids = result.get("legend_chunk_ids", [])
            print(f"[DEBUG] LLM detected {len(legend_ids)} legend chunks: {legend_ids}")
            return legend_ids
    except Exception as e:
        print(f"[DEBUG] Legend detection failed: {e}")
    
    return []


def segment_chunks_smart(chunks: List[Dict], llm=None) -> Tuple[List[Dict], List[Dict]]:
    """
    Smart chunk segmentation using LLM when available, fallback to keywords.
    """
    # Try LLM-based detection first
    legend_ids = []
    if llm:
        legend_ids = detect_legend_chunks_llm(chunks, llm)
    
    legend_like = []
    figure_like = []
    
    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        raw_type = (chunk.get("type") or "").lower()
        text_lower = (chunk.get("text") or "").lower()
        
        # LLM identified as legend
        if chunk_id in legend_ids:
            legend_like.append(chunk)
            continue
        
        # Keyword-based fallback
        is_figure = raw_type in {"figure", "architectural_drawing", "image"}
        has_legend_hint = any(token in text_lower for token in {"legend", "keynote", "abbreviation", "symbol", "schedule"})
        
        # Check for table-like structure (numbered lists)
        has_numbered_list = bool(re.search(r'\b\d{1,4}\s+[A-Z]', chunk.get("text", "")))
        
        if has_legend_hint or has_numbered_list or raw_type == "table":
            legend_like.append(chunk)
        elif is_figure:
            figure_like.append(chunk)
        else:
            # Default: treat as potential legend source
            legend_like.append(chunk)
    
    print(f"[DEBUG] Smart segmentation: {len(legend_like)} legend-like, {len(figure_like)} figure-like")
    return legend_like, figure_like


# ==============================================================================
# 13. NEW: Improved Bbox Matching (IoU-based)
# ==============================================================================

def find_best_bbox_iou(
    search_text: str,
    all_lines: List[Dict],
    chunk_bbox: Tuple[float, float, float, float] = None,
    min_similarity: float = 0.6
) -> Optional[Dict]:
    """
    Find best matching bbox using text similarity and IoU overlap.
    More robust than center-point matching.
    """
    if not search_text:
        return None
    
    target = re.sub(r'[^0-9a-zA-Z\s]', '', search_text).lower().strip()
    if not target:
        return None
    
    best_match = None
    best_score = 0.0
    
    for line in all_lines:
        line_text = re.sub(r'[^0-9a-zA-Z\s]', '', line.get("text", "")).lower().strip()
        if not line_text:
            continue
        
        # Text similarity
        if target in line_text:
            text_score = 1.0
        else:
            text_score = SequenceMatcher(None, target, line_text).ratio()
        
        if text_score < min_similarity:
            continue
        
        # Spatial score (prefer items inside chunk)
        spatial_score = 1.0
        if chunk_bbox:
            line_box = {"x0": line["x0"], "y0": line["y0"], "x1": line["x1"], "y1": line["y1"]}
            chunk_box = {"x0": chunk_bbox[0], "y0": chunk_bbox[1], "x1": chunk_bbox[2], "y1": chunk_bbox[3]}
            iou = compute_iou(line_box, chunk_box)
            # Bonus for being inside chunk
            if is_center_inside(line, chunk_bbox, tolerance=20):
                spatial_score = 1.0 + iou
            else:
                spatial_score = 0.5 + iou
        
        combined_score = text_score * spatial_score
        if combined_score > best_score:
            best_score = combined_score
            best_match = line
    
    return best_match


# ==============================================================================
# 14. NEW: Enhanced Legend Entry Extraction
# ==============================================================================

def extract_legend_entries_enhanced(
    legend_chunks: List[Dict],
    unified_lines: List[Dict],
    fence_keywords: List[str],
    llm
) -> List[Dict]:
    """
    Enhanced legend extraction with:
    - Unified text lines (PDF + OCR)
    - IoU-based bbox matching
    - Indicator validation
    """
    print("[DEBUG] Enhanced Legend Entry Extraction...")
    results = []
    
    for chunk in legend_chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        
        # Extract items via LLM
        items = llm_extract_fence_elements(llm, text, fence_keywords)
        
        # Validate indicators
        items = validate_extracted_indicators(items)
        
        chunk_bbox = (chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"])
        
        for item in items:
            desc = item.get("text_element", "")
            ind = item.get("indicator", "")
            
            # Find bboxes using improved matching
            bbox_desc = find_best_bbox_iou(desc, unified_lines, chunk_bbox) if desc else None
            bbox_ind = find_best_bbox_iou(ind, unified_lines, chunk_bbox) if ind else None
            
            # Fallback to global search
            if not bbox_desc and desc:
                bbox_desc = find_best_bbox_iou(desc, unified_lines, None, min_similarity=0.5)
            if not bbox_ind and ind:
                bbox_ind = find_best_bbox_iou(ind, unified_lines, None, min_similarity=0.7)
            
            if bbox_desc:
                results.append({
                    "indicator": ind,
                    "keyword": desc,
                    "description": item.get("description", ""),
                    "x0": bbox_desc["x0"], "y0": bbox_desc["y0"],
                    "x1": bbox_desc["x1"], "y1": bbox_desc["y1"],
                    "source": bbox_desc.get("source", "unknown") + "_desc"
                })
            
            if bbox_ind and ind:
                # Avoid duplicate if same position as description
                is_duplicate = False
                if bbox_desc:
                    iou = compute_iou(bbox_ind, bbox_desc)
                    if iou > 0.5:
                        is_duplicate = True
                
                if not is_duplicate:
                    results.append({
                        "indicator": ind,
                        "keyword": ind,
                        "description": "Indicator Code",
                        "x0": bbox_ind["x0"], "y0": bbox_ind["y0"],
                        "x1": bbox_ind["x1"], "y1": bbox_ind["y1"],
                        "source": bbox_ind.get("source", "unknown") + "_ind"
                    })
    
    print(f"[DEBUG] Enhanced extraction: {len(results)} items")
    return results
