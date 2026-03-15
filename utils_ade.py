"""
utils_ade.py - Unified ADE Utility (Native Lines + Robust Overlap + DEBUG LOGGING)
"""

import re
import json
import time
import requests
from typing import List, Dict, Optional, Tuple
from io import BytesIO

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
    Segment chunks into legend-like (for definition extraction) and figure-like (for instance finding).
    
    A chunk is figure-like if:
    - It's typed as 'figure' or 'architectural_drawing'
    - AND it doesn't have strong legend indicators at the START of its text
    
    We check only the first 200 chars to avoid false positives from ADE's descriptive text.
    """
    legend_like = []
    figure_like = []
    for chunk in chunks:
        raw_type = (chunk.get("type") or "").lower()
        text = chunk.get("text") or ""
        # Only check the beginning of text for legend hints (avoid ADE descriptions)
        text_start = text[:200].lower()
        
        is_figure = raw_type in {"figure", "architectural_drawing"}
        # Check for explicit legend section headers
        has_legend_hint = any(token in text_start for token in {"legend", "keynote", "abbreviation", "symbols"})

        if is_figure and not has_legend_hint:
            figure_like.append(chunk)
        else:
            legend_like.append(chunk)
    print(f"[DEBUG] Segmented: {len(legend_like)} Legend-like chunks, {len(figure_like)} Figure-like chunks.")
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


def run_google_ocr_blocks(page_bytes: bytes, google_cloud_config: Dict, pdf_width: float, pdf_height: float) -> List[Dict]:
    """
    Run Google Document AI OCR on a PDF page and return text blocks with PDF coordinates.
    
    IMPORTANT: normalized_vertices from Document AI are relative to the IMAGE dimensions.
    We must convert them to PDF coordinate space using the correct scale factors.
    """
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
    
    # Store actual image dimensions for coordinate conversion
    img_width = pix.width
    img_height = pix.height
    print(f"[DEBUG] Image dimensions: {img_width} x {img_height} pixels")
    print(f"[DEBUG] PDF dimensions: {pdf_width:.1f} x {pdf_height:.1f} points")

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

    # Calculate scale factors to convert from image pixels to PDF points
    # normalized_vertices are 0-1 relative to image, so:
    # pixel_coord = normalized * img_dimension
    # pdf_coord = pixel_coord * (pdf_dimension / img_dimension)
    # Simplified: pdf_coord = normalized * pdf_dimension (since aspect ratio is preserved)
    
    # But to be safe, let's use explicit scale factors
    scale_x = pdf_width / img_width if img_width > 0 else 1.0
    scale_y = pdf_height / img_height if img_height > 0 else 1.0
    print(f"[DEBUG] Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")

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

        # Convert normalized (0-1) to pixel coordinates, then to PDF coordinates
        xs = [v.x * img_width * scale_x for v in vertices if v.x is not None]
        ys = [v.y * img_height * scale_y for v in vertices if v.y is not None]
        
        if not xs or not ys:
            continue

        ocr_lines.append({
            "text": text_content,
            "x0": min(xs), "y0": min(ys),
            "x1": max(xs), "y1": max(ys),
            "source": "ocr_paragraph"
        })

    print(f"[DEBUG] OCR extraction complete. Found {len(ocr_lines)} paragraphs.")
    doc.close()
    return ocr_lines


# ==============================================================================
# 3. PDF Native Text Extraction
# ==============================================================================


def get_native_pdf_lines(page: fitz.Page) -> List[Dict]:
    """
    Extract text lines from PDF with coordinates in display space.
    
    IMPORTANT: Transforms MediaBox coords to display coords for rotated pages.
    PDF text coordinates are in MediaBox space, but we need display space
    to match the rendered image coordinates.
    """
    structure = page.get_text("dict")
    rotation = page.rotation
    mediabox_w = page.mediabox.width
    mediabox_h = page.mediabox.height
    
    def transform_for_rotation(x0, y0, x1, y1):
        """Transform MediaBox coords to display coords based on page rotation"""
        if rotation == 0:
            return x0, y0, x1, y1
        elif rotation == 90:
            return mediabox_h - y1, x0, mediabox_h - y0, x1
        elif rotation == 180:
            return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
        elif rotation == 270:
            return y0, mediabox_w - x1, y1, mediabox_w - x0
        return x0, y0, x1, y1
    
    lines = []
    for block in structure.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                bbox = line["bbox"]
                if text:
                    # Transform from MediaBox to display coordinates
                    nx0, ny0, nx1, ny1 = transform_for_rotation(bbox[0], bbox[1], bbox[2], bbox[3])
                    lines.append({
                        "text": text,
                        "x0": nx0, "y0": ny0, 
                        "x1": nx1, "y1": ny1,
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
    llm,
    figure_chunks: List[Dict] = None
) -> List[Dict]:
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
    for chunk in legend_chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        items = llm_extract_fence_elements(llm, text, fence_keywords)
        _process_chunk(chunk, items, results, extraction_pass="legend")

    # Check quality of legend results — filter out bad indicators
    good_results = [r for r in results if r.get("indicator", "").strip() not in ("", ".", "(bullet point)", "bullet point", "-", "*")]
    
    # Pass 2: If legend chunks yielded few good results, also extract from figure chunks
    if len(good_results) < 3 and figure_chunks:
        print(f"[DEBUG] Legend extraction yielded only {len(good_results)} good results, trying figure chunks...")
        for chunk in figure_chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            # Only process figure chunks whose text contains fence keywords
            text_lower = text.lower()
            if not any(kw.lower() in text_lower for kw in fence_keywords):
                continue
            items = llm_extract_fence_elements(llm, text, fence_keywords)
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
            items = llm_extract_fence_elements(llm, combined_text, fence_keywords)
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
    
    prompt = f"""You are reviewing engineering drawing documentation for fence construction.

The following fence elements/categories were identified in the drawings:
{elements_list}

Below is the text extracted from multiple pages of the drawing set. Some pages contain 
plan views (showing where fences go), while others contain DETAIL pages with specifications 
like fence height, post type, post spacing, wire gauge, top/bottom rails, materials, 
coating, foundation details, etc.

Your task: For EACH element listed above, find any detailed specifications mentioned 
anywhere in the text below. Cross-reference by indicator numbers, element names, or 
any matching descriptions.

Text from drawing pages:
<PAGES>
{combined_text}
</PAGES>

Respond with a JSON array where each element has:
- "element_name": the element name (must match one from the list above)
- "height": fence height if found (e.g. "6'-0\"", "8 FT")
- "post_type": post type/size (e.g. "2-1/2\" SS40 ROUND", "W6x9")
- "post_spacing": post spacing (e.g. "10'-0\" O.C.", "10 FT MAX")
- "top_rail": top rail details
- "bottom_rail": bottom rail or tension wire details
- "material": material/coating (e.g. "Galvanized", "Vinyl Coated")
- "gauge": wire/mesh gauge (e.g. "9 gauge", "11 gauge")
- "mesh_size": mesh/opening size (e.g. "2 inch")
- "foundation": footing/foundation details
- "gate_info": gate details if applicable
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
                    
                    result[matched_name] = {
                        "height": str(item.get("height", "")).strip(),
                        "post_type": str(item.get("post_type", "")).strip(),
                        "post_spacing": str(item.get("post_spacing", "")).strip(),
                        "top_rail": str(item.get("top_rail", "")).strip(),
                        "bottom_rail": str(item.get("bottom_rail", "")).strip(),
                        "material": str(item.get("material", "")).strip(),
                        "gauge": str(item.get("gauge", "")).strip(),
                        "mesh_size": str(item.get("mesh_size", "")).strip(),
                        "foundation": str(item.get("foundation", "")).strip(),
                        "gate_info": str(item.get("gate_info", "")).strip(),
                        "detail_page": str(item.get("detail_page", "")).strip(),
                        "full_details": str(item.get("full_details", "")).strip(),
                        "notes": str(item.get("notes", "")).strip(),
                    }
    except Exception as e:
        print(f"[DETAILS] JSON parse error: {e}")
    
    print(f"[DETAILS] Extracted details for {len(result)} elements")
    for name, details in result.items():
        summary = details.get('full_details', '')[:80]
        print(f"[DETAILS]   {name}: {summary}...")
    
    return result


def find_instances_in_figures(legend_entries: List[Dict], figure_chunks: List[Dict], all_tokens: List[Dict], ocr_lines: List[Dict] = None) -> List[Dict]:
    """
    Find instances of legend indicators ONLY within figure/architectural_drawing chunks.
    
    Args:
        legend_entries: List of definitions with 'indicator' field
        figure_chunks: List of figure region bounding boxes (type=figure or architectural_drawing)
        all_tokens: All text tokens from the page
        ocr_lines: OCR text blocks as fallback when page has no embedded text
    """
    print("[DEBUG] Finding Instances in Figure Chunks...")
    print(f"[DEBUG] Figure chunks count: {len(figure_chunks)}")
    
    # Fallback: if no embedded text tokens, use OCR lines as token source
    if not all_tokens and ocr_lines:
        print(f"[DEBUG] No embedded text tokens, using {len(ocr_lines)} OCR lines as fallback")
        # Split OCR lines into individual word-level tokens for better matching
        for ocr_line in ocr_lines:
            text = ocr_line.get('text', '').strip()
            if not text:
                continue
            # For short text (likely a single token/number), use as-is
            words = text.split()
            if len(words) <= 1:
                all_tokens.append({
                    'text': text,
                    'x0': ocr_line.get('x0', 0),
                    'y0': ocr_line.get('y0', 0),
                    'x1': ocr_line.get('x1', 0),
                    'y1': ocr_line.get('y1', 0),
                })
            else:
                # For multi-word OCR blocks, split into words with approximate positions
                total_len = sum(len(w) for w in words)
                x0 = ocr_line.get('x0', 0)
                x1 = ocr_line.get('x1', 0)
                y0 = ocr_line.get('y0', 0)
                y1 = ocr_line.get('y1', 0)
                span = x1 - x0
                cursor = x0
                for w in words:
                    w_span = (len(w) / max(total_len, 1)) * span
                    all_tokens.append({
                        'text': w,
                        'x0': cursor,
                        'y0': y0,
                        'x1': cursor + w_span,
                        'y1': y1,
                    })
                    cursor += w_span
        print(f"[DEBUG] Created {len(all_tokens)} tokens from OCR lines")
    for i, fc in enumerate(figure_chunks):
        print(f"[DEBUG]   Figure {i}: type={fc.get('type')} bbox=({fc['x0']:.1f}, {fc['y0']:.1f}) - ({fc['x1']:.1f}, {fc['y1']:.1f})")
    
    instances = []
    
    def normalize_indicator(s):
        """Normalize dotted indicator numbers by stripping leading zeros from each segment.
        E.g. '2.09' -> '2.9', '02.03' -> '2.3', '2.9' -> '2.9'.
        Non-numeric parts are left unchanged."""
        parts = s.split('.')
        normalized_parts = []
        for p in parts:
            # Strip leading zeros from numeric segments, but keep at least one digit
            if p.lstrip('0').isdigit():
                normalized_parts.append(p.lstrip('0') or '0')
            elif p.isdigit() or (p and p[0].isdigit()):
                normalized_parts.append(p.lstrip('0') or '0')
            else:
                normalized_parts.append(p)
        return '.'.join(normalized_parts)
    
    # Collect all indicators to search for
    indicators_to_find = set()
    # Map from normalized form -> original indicator string
    norm_to_original = {}
    for item in legend_entries:
        ind = item.get("indicator", "").strip()
        if ind:
            indicators_to_find.add(ind)
            # Also add cleaned version (remove special chars)
            clean_ind = re.sub(r'[^\w]', '', ind)
            if clean_ind:
                indicators_to_find.add(clean_ind)
            # Build normalized lookup: normalized form -> original indicator
            norm_key = normalize_indicator(ind)
            norm_to_original.setdefault(norm_key, ind)
            if clean_ind:
                norm_clean = normalize_indicator(clean_ind)
                norm_to_original.setdefault(norm_clean, ind)
    
    print(f"[DEBUG] Looking for indicators: {indicators_to_find}")
    print(f"[DEBUG] Normalized indicator map: {norm_to_original}")
    
    if not indicators_to_find:
        return []
    
    if not figure_chunks:
        print("[DEBUG] No figure chunks to search in!")
        return []
    
    # Get legend bounding boxes to exclude (don't match indicators in legend area)
    # ONLY use legend-sourced definitions for exclusion — figure/OCR-sourced definitions
    # are IN the drawing area and should NOT create exclusion zones
    legend_bboxes = []
    for entry in legend_entries:
        if entry.get("extraction_pass", "legend") != "legend":
            continue
        if all(k in entry for k in ['x0', 'y0', 'x1', 'y1']):
            legend_bboxes.append((entry['x0'], entry['y0'], entry['x1'], entry['y1']))
    
    def is_in_legend_area(token):
        """Check if token is inside any legend bounding box (with margin)"""
        margin = 20  # PDF units margin
        tx, ty = (token['x0'] + token['x1']) / 2, (token['y0'] + token['y1']) / 2
        for lx0, ly0, lx1, ly1 in legend_bboxes:
            if lx0 - margin <= tx <= lx1 + margin and ly0 - margin <= ty <= ly1 + margin:
                return True
        return False
    
    # Filter tokens to only those inside figure chunks
    figure_tokens = []
    for chunk in figure_chunks:
        cx0, cy0, cx1, cy1 = chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]
        for t in all_tokens:
            # Check if token center is inside chunk
            tx, ty = (t['x0'] + t['x1']) / 2, (t['y0'] + t['y1']) / 2
            if cx0 <= tx <= cx1 and cy0 <= ty <= cy1:
                figure_tokens.append(t)
    
    print(f"[DEBUG] Tokens inside figure chunks: {len(figure_tokens)} (out of {len(all_tokens)} total)")
    
    # Search for indicator matches
    # For short numeric indicators (1-2 digits), require EXACT match to avoid false positives
    # from dimensions, grid references, etc.
    found_positions = set()
    
    for token in figure_tokens:
        token_text = token.get("text", "").strip()
        if not token_text:
            continue
        
        # Check if this token matches any indicator
        matched_indicator = None
        
        # For exact matching, the token text should match the indicator exactly
        # (not be part of a larger number like "180" matching "18")
        for ind in indicators_to_find:
            # Check for exact match
            if token_text == ind:
                matched_indicator = ind
                break
            # Check if token is the indicator with parentheses like "(20)"
            if token_text == f"({ind})" or token_text == f"[{ind}]":
                matched_indicator = ind
                break
            # For indicators with special chars like "(20)", match the token
            if ind.startswith("(") and token_text == ind:
                matched_indicator = ind
                break
        
        # Fallback: normalized numeric matching (handles "2.09" == "2.9")
        if not matched_indicator:
            # Strip parentheses/brackets from token for normalization
            stripped = token_text.strip('()[] ')
            norm_token = normalize_indicator(stripped)
            if norm_token in norm_to_original:
                matched_indicator = norm_to_original[norm_token]
                print(f"[DEBUG] Normalized match: token '{token_text}' -> '{norm_token}' matched indicator '{matched_indicator}'")
        
        if matched_indicator:
            # Skip if in legend area (we already have it as definition)
            if is_in_legend_area(token):
                print(f"[DEBUG] Skipping '{matched_indicator}' at ({token['x0']:.1f}, {token['y0']:.1f}) - in legend area")
                continue
            
            # Create position key to avoid duplicates
            pos_key = (round(token['x0']), round(token['y0']))
            if pos_key in found_positions:
                continue
            found_positions.add(pos_key)
            
            instances.append({
                "indicator": matched_indicator,
                "x0": token["x0"], "y0": token["y0"], 
                "x1": token["x1"], "y1": token["y1"],
                "source": "figure_instance"
            })
            print(f"[DEBUG] ✓ Found instance '{matched_indicator}' at ({token['x0']:.1f}, {token['y0']:.1f}) token='{token_text}'")
    
    print(f"[DEBUG] Total instances found: {len(instances)}")
    return instances


# ==============================================================================
# 6. Visualization & Utils
# ==============================================================================


def highlight_page_image(page_image_bytes: bytes, definitions: List[Dict], instances: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    print("[DEBUG] Generating Highlighted Image...")
    print(f"[DEBUG] Highlighting {len(definitions)} definitions and {len(instances)} instances")
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        print(f"[DEBUG] Image size: {img_w} x {img_h}")
        print(f"[DEBUG] PDF size: {pdf_width} x {pdf_height}")
        print(f"[DEBUG] Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
        # Sanity check: warn if scaled coordinates would be outside image
        for i in instances[:3]:
            sx = i.get('x0', 0) * scale_x
            sy = i.get('y0', 0) * scale_y
            if sx > img_w or sy > img_h:
                print(f"[DEBUG] ⚠️ Instance '{i.get('indicator')}' scaled coords ({sx:.1f}, {sy:.1f}) OUTSIDE image bounds!")

        def scale_box(box_dict):
            scaled = [
                box_dict.get("x0", 0) * scale_x,
                box_dict.get("y0", 0) * scale_y,
                box_dict.get("x1", 0) * scale_x,
                box_dict.get("y1", 0) * scale_y
            ]
            return scaled

        for d in definitions:
            box = scale_box(d)
            print(f"[DEBUG] Definition box: orig=({d.get('x0'):.1f}, {d.get('y0'):.1f}) -> scaled=({box[0]:.1f}, {box[1]:.1f})")
            draw.rectangle(box, outline=(0, 255, 0, 255), width=3)
            draw.rectangle(box, fill=(0, 255, 0, 40))
        
        for idx, i in enumerate(instances[:5]):  # Log first 5 instances
            box = scale_box(i)
            print(f"[DEBUG] Instance {idx} '{i.get('indicator')}' box: orig=({i.get('x0'):.1f}, {i.get('y0'):.1f}, {i.get('x1'):.1f}, {i.get('y1'):.1f}) -> scaled=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        
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


def create_multi_page_pdf(full_pdf_bytes: bytes, page_indices: List[int]) -> bytes:
    """Create a PDF containing only the specified pages (preserving order).
    
    In the resulting PDF, pages are numbered 0..len(page_indices)-1,
    so callers must map local index back to the original page index.
    """
    doc = fitz.open(stream=full_pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    for idx in page_indices:
        new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
    out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    return out


def create_page_batches(
    full_pdf_bytes: bytes,
    page_indices: List[int],
    max_batch_bytes: int = 15 * 1024 * 1024,
    max_pages_per_batch: int = 10
) -> List[List[int]]:
    """Split page indices into batches respecting estimated size and page count limits.
    
    Uses average page size from the full PDF as a heuristic. A 1.3x safety factor
    accounts for PDF structure overhead. This avoids the expense of creating
    temporary single-page PDFs just to measure their size.
    
    Args:
        full_pdf_bytes: Complete PDF file bytes
        page_indices: 0-indexed page numbers to include
        max_batch_bytes: Max estimated PDF size per batch (default 15MB)
        max_pages_per_batch: Hard cap on pages per batch (default 10)
    
    Returns:
        List of page-index lists, one per batch
    """
    if not page_indices:
        return []
    
    total_pdf_size = len(full_pdf_bytes)
    try:
        doc = fitz.open(stream=full_pdf_bytes, filetype="pdf")
        total_pages_in_doc = len(doc)
        doc.close()
    except Exception:
        total_pages_in_doc = max(page_indices) + 1
    
    avg_page_size = (total_pdf_size / max(total_pages_in_doc, 1)) * 1.3
    max_by_size = max(1, int(max_batch_bytes / max(avg_page_size, 1)))
    effective_max = min(max_by_size, max_pages_per_batch)
    
    print(f"[BATCH] PDF: {total_pdf_size/1024/1024:.1f}MB, {total_pages_in_doc} pages, "
          f"~{avg_page_size/1024:.0f}KB/page -> max {effective_max} pages/batch")
    
    batches = []
    for i in range(0, len(page_indices), effective_max):
        batches.append(page_indices[i:i + effective_max])
    
    print(f"[BATCH] {len(batches)} batch(es) for {len(page_indices)} fence pages: "
          + ", ".join(f"[{','.join(str(p+1) for p in b)}]" for b in batches))
    return batches


# ==============================================================================
# 7. Fallback Page Classification (Keyword + LLM)
# ==============================================================================


def scan_page_for_keywords(pdf_lines: List[Dict], ocr_lines: List[Dict], fence_keywords: List[str]) -> Dict:
    """
    Scan page text for fence-related keywords using word boundary matching.
    Returns dict with matched keywords and their locations.
    
    Uses regex word boundaries to avoid false positives like "gate" in "aggregate".
    Also handles common plural forms (gate -> gates, fence -> fences, etc.)
    """
    all_lines = pdf_lines + ocr_lines
    combined_text = " ".join(line.get("text", "") for line in all_lines).lower()
    
    matches = []
    matched_lines = []
    
    for keyword in fence_keywords:
        kw_lower = keyword.lower()
        # Use word boundary matching with optional plural suffix (s, es, ing)
        # This matches: gate, gates, gating; fence, fences, fencing; etc.
        pattern = r'\b' + re.escape(kw_lower) + r'(?:s|es|ing)?\b'
        if re.search(pattern, combined_text):
            matches.append(keyword)
            # Find lines containing this keyword (with word boundary)
            for line in all_lines:
                line_text = line.get("text", "").lower()
                if re.search(pattern, line_text):
                    matched_lines.append({
                        "keyword": keyword,
                        "text": line.get("text", ""),
                        "x0": line.get("x0", 0),
                        "y0": line.get("y0", 0),
                        "x1": line.get("x1", 0),
                        "y1": line.get("y1", 0),
                        "source": line.get("source", "unknown")
                    })
    
    return {
        "has_keywords": len(matches) > 0,
        "matched_keywords": list(set(matches)),
        "matched_lines": matched_lines,
        "total_text_length": len(combined_text)
    }


def llm_classify_page(llm, page_text: str, fence_keywords: List[str]) -> Dict:
    """
    Use LLM to classify if a page is fence-related.
    Similar to app.py's analyze_page but simpler.
    """
    if not llm or not page_text:
        return {"is_fence_related": False, "confidence": 0.0, "reason": "No LLM or text"}
    
    # FIX 1: Increase text limit from 8000 to 16000 to capture fence content
    # that often appears at the end of pages (legends, notes, schedules)
    text_for_llm = page_text[:16000] if len(page_text) > 16000 else page_text
    keywords_hint = ", ".join(fence_keywords[:15])
    
    prompt = f"""You are analyzing an engineering drawing page to determine if it contains fence-related content.

Keywords to look for: {keywords_hint}

Page text:
<TEXT>
{text_for_llm}
</TEXT>

Analyze the text and determine if this page is about fences, gates, barriers, guardrails, or related elements.
Look for:
- Fence specifications, dimensions, or materials
- Gate details or schedules
- Barrier or guardrail references
- Fence post details
- Chain link, mesh, panel references
- Any fence-related construction details

Respond with JSON only:
{{"is_fence_related": true/false, "confidence": 0.0-1.0, "signals": ["keyword1", "keyword2"], "reason": "brief explanation"}}
"""
    
    try:
        raw_response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_text = getattr(raw_response, "content", str(raw_response))
        
        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "is_fence_related": result.get("is_fence_related", False),
                "confidence": float(result.get("confidence", 0.0)),
                "signals": result.get("signals", []),
                "reason": result.get("reason", "")
            }
    except Exception as e:
        print(f"[DEBUG] LLM page classification failed: {e}")
    
    return {"is_fence_related": False, "confidence": 0.0, "reason": "LLM parsing failed"}


def fallback_fence_detection(
    pdf_lines: List[Dict],
    ocr_lines: List[Dict],
    fence_keywords: List[str],
    llm=None,
    use_llm_confirmation: bool = True
) -> Dict:
    """
    Fallback detection when ADE doesn't find structured definitions.
    1. Scan for keywords
    2. If keywords found and LLM available, confirm with LLM
    3. Return detection result with matched items
    """
    print("[DEBUG] Running fallback fence detection...")
    
    # Step 1: Keyword scan
    keyword_result = scan_page_for_keywords(pdf_lines, ocr_lines, fence_keywords)
    
    if not keyword_result["has_keywords"]:
        print("[DEBUG] No keywords found in fallback scan.")
        return {
            "fence_found": False,
            "method": "keyword_scan",
            "matched_keywords": [],
            "matched_lines": [],
            "llm_result": None
        }
    
    print(f"[DEBUG] Keywords found: {keyword_result['matched_keywords']}")
    
    # FIX 2: High-signal keywords that should NOT be overridden by LLM rejection
    # These are specific fence-related terms that strongly indicate fence content
    HIGH_SIGNAL_KEYWORDS = {
        'fence', 'fencing', 'gate', 'gates', 'chain link', 'guardrail', 
        'railing', 'handrail', 'bollard', 'barrier'
    }
    
    matched_lower = {kw.lower() for kw in keyword_result["matched_keywords"]}
    has_high_signal = bool(matched_lower & HIGH_SIGNAL_KEYWORDS)
    
    if has_high_signal:
        print(f"[DEBUG] High-signal keywords found: {matched_lower & HIGH_SIGNAL_KEYWORDS} - trusting keywords over LLM")
        return {
            "fence_found": True,
            "method": "keyword_high_signal",
            "matched_keywords": keyword_result["matched_keywords"],
            "matched_lines": keyword_result["matched_lines"],
            "llm_result": None
        }
    
    # Step 2: LLM confirmation (if enabled and available) for lower-signal keywords
    llm_result = None
    if use_llm_confirmation and llm:
        all_lines = pdf_lines + ocr_lines
        page_text = " ".join(line.get("text", "") for line in all_lines)
        llm_result = llm_classify_page(llm, page_text, fence_keywords)
        print(f"[DEBUG] LLM classification: {llm_result}")
        
        # Use LLM decision if confident
        if llm_result["confidence"] >= 0.5:
            return {
                "fence_found": llm_result["is_fence_related"],
                "method": "llm_confirmed",
                "matched_keywords": keyword_result["matched_keywords"],
                "matched_lines": keyword_result["matched_lines"],
                "llm_result": llm_result
            }
    
    # Step 3: Fall back to keyword-only decision
    # If we found keywords, consider it fence-related
    return {
        "fence_found": True,
        "method": "keyword_only",
        "matched_keywords": keyword_result["matched_keywords"],
        "matched_lines": keyword_result["matched_lines"],
        "llm_result": llm_result
    }


def highlight_keyword_matches(page_image_bytes: bytes, matched_lines: List[Dict], pdf_width: float, pdf_height: float) -> bytes:
    """
    Highlight keyword matches on the page image.
    Uses orange color to distinguish from definition (green) and instance (purple) highlights.
    """
    if not matched_lines:
        return page_image_bytes
    
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        for line in matched_lines:
            box = [
                line.get("x0", 0) * scale_x,
                line.get("y0", 0) * scale_y,
                line.get("x1", 0) * scale_x,
                line.get("y1", 0) * scale_y
            ]
            # Orange outline for keyword matches
            draw.rectangle(box, outline=(255, 165, 0, 255), width=2)
            draw.rectangle(box, fill=(255, 165, 0, 40))
        
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Keyword highlight error: {e}")
        return page_image_bytes


# ==============================================================================
# 8. Debug Visualization
# ==============================================================================


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
# 9. Smart Fence Measurement (LLM-based Layer Identification)
# ==============================================================================

# Import vector utilities
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
        VectorLine
    )
    VECTOR_UTILS_AVAILABLE = True
except ImportError:
    VECTOR_UTILS_AVAILABLE = False
    print("[DEBUG] utils_vector.py not found - fence measurement disabled")


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


def detect_dimension_lines(
    page: fitz.Page,
    scale_factor: float = 30.0,
    search_radius: float = 150.0,
    min_length_ft: float = 2.0,
    max_length_ft: float = 500.0
) -> Dict:
    """
    Detect fence measurements by finding numeric dimension text and matching to nearby lines.
    
    This method looks for dimension-style annotations (numbers with units like "45'" or "120 LF")
    and finds the vector line whose length best matches the annotated measurement.
    
    Args:
        page: PDF page object
        scale_factor: Drawing scale (e.g., 30 means 1" = 30')
        search_radius: Max distance (pts) to search for matching lines
        min_length_ft: Minimum fence length to consider (feet)
        max_length_ft: Maximum fence length to consider (feet)
    
    Returns:
        Dictionary with detected dimension lines and measurements
    """
    if not VECTOR_UTILS_AVAILABLE:
        return {"success": False, "error": "Vector utilities not available", "measurements": []}
    
    print("[DEBUG] Starting dimension line detection...")
    
    # Extract text and find numeric measurements
    text_dict = page.get_text('dict')
    measurements = []
    
    for block in text_dict.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip()
                bbox = span['bbox']
                
                # Look for patterns like: "45", "45'", "45'-0\"", "45 LF", "120'-6\""
                match = re.match(r'^(\d+\.?\d*)\s*[\'\"\-LF]*', text)
                if match:
                    try:
                        num = float(match.group(1))
                        # Filter to reasonable fence lengths
                        if min_length_ft <= num <= max_length_ft:
                            measurements.append({
                                'value_ft': num,
                                'text': text,
                                'bbox': bbox,
                                'center': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                            })
                    except:
                        pass
    
    print(f"[DEBUG] Found {len(measurements)} potential dimension texts")
    
    # Extract all lines from page
    all_lines = extract_vector_lines(page)
    # Filter to minimum length
    all_lines = [l for l in all_lines if l.length_pts >= 5.0]
    print(f"[DEBUG] Extracted {len(all_lines)} vector lines (min 5 pts)")
    
    matched_dimensions = []
    matched_lines = []
    
    for meas in measurements:
        px, py = meas['center']
        # Convert expected length from feet to points
        expected_pts = meas['value_ft'] * 12.0 / scale_factor * 72.0
        
        best_line = None
        best_score = float('inf')
        
        for line in all_lines:
            sx, sy = line.start
            ex, ey = line.end
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            
            # Distance from measurement text to line (any point)
            dist = min(
                ((sx-px)**2 + (sy-py)**2)**0.5,
                ((ex-px)**2 + (ey-py)**2)**0.5,
                ((mx-px)**2 + (my-py)**2)**0.5
            )
            
            if dist > search_radius:
                continue
            
            # Score: prioritize length match, penalize distance
            length_diff_pct = abs(line.length_pts - expected_pts) / expected_pts if expected_pts > 0 else 1
            score = length_diff_pct + (dist / search_radius) * 0.3
            
            if score < best_score:
                best_score = score
                best_line = line
        
        # Only include good matches (score < 1.0 means length match is within 70%)
        if best_line and best_score < 1.0:
            actual_ft = (best_line.length_pts / 72.0) * scale_factor / 12.0
            matched_dimensions.append({
                'expected_ft': meas['value_ft'],
                'actual_ft': actual_ft,
                'measurement_text': meas['text'],
                'text_bbox': meas['bbox'],
                'line_start': best_line.start,
                'line_end': best_line.end,
                'match_score': best_score,
                'error_pct': abs(actual_ft - meas['value_ft']) / meas['value_ft'] * 100 if meas['value_ft'] > 0 else 0
            })
            matched_lines.append(best_line)
    
    # Calculate totals
    total_expected_ft = sum(m['expected_ft'] for m in matched_dimensions)
    total_actual_ft = sum(m['actual_ft'] for m in matched_dimensions)
    
    print(f"[DEBUG] Matched {len(matched_dimensions)} dimension lines")
    print(f"[DEBUG] Total: expected={total_expected_ft:.0f}ft, actual={total_actual_ft:.0f}ft")
    
    return {
        'success': True,
        'method': 'dimension_line',
        'measurements': matched_dimensions,
        'matched_lines': matched_lines,
        'total_expected_ft': total_expected_ft,
        'total_actual_ft': total_actual_ft,
        'dimension_count': len(matched_dimensions)
    }


def measure_fence_elements(
    page: fitz.Page,
    fence_definitions: List[Dict],
    fence_instances: List[Dict],
    figure_chunks: List[Dict] = None,
    llm=None,
    scale_factor: Optional[float] = None,
    ocr_text: str = None
) -> Dict:
    """
    Measure fence-related vector elements based on detected indicators.
    
    This function:
    1. Extracts all layer names from the page
    2. Uses LLM to identify fence-related layers
    3. Extracts vector lines from those layers (constrained to figure areas)
    4. Matches lines to detected indicator instances
    5. Calculates measurements
    
    Args:
        page: PDF page object
        fence_definitions: Detected fence definitions from legend
        fence_instances: Detected fence instances in figures
        figure_chunks: ADE-detected figure/drawing areas for boundary constraint
        llm: Language model for layer identification (optional)
        scale_factor: Drawing scale override (if None, auto-inferred)
    
    Returns:
        Dictionary with measurement results
    """
    if not VECTOR_UTILS_AVAILABLE:
        return {"error": "Vector utilities not available", "measurements": {}}
    
    print("[DEBUG] Starting fence element measurement...")
    
    # Get page info
    page_width = page.rect.width
    page_height = page.rect.height
    rotation = page.rotation
    
    # Auto-infer scale if not provided
    if scale_factor is None:
        # 1) Try regex on embedded text / OCR text
        scale_factor = infer_scale_from_page(page, ocr_text=ocr_text)
        if scale_factor:
            print(f"[DEBUG] Auto-detected scale factor (regex): {scale_factor}")
        # 2) Fall back to vision model (GPT-4V analyzes page image)
        if scale_factor is None and llm is not None:
            print("[DEBUG] Regex scale detection failed, trying vision model...")
            try:
                vision_result = detect_scale_with_vision(page, llm)
                if vision_result.get('success') and vision_result.get('verified_scale'):
                    scale_factor = vision_result['verified_scale']
                    print(f"[DEBUG] Auto-detected scale factor (vision): {scale_factor}")
                else:
                    print(f"[DEBUG] Vision scale detection failed: {vision_result.get('message', 'unknown')}")
            except Exception as e:
                print(f"[DEBUG] Vision scale detection error: {e}")
        if scale_factor is None:
            scale_factor = 1.0
            print("[DEBUG] Could not detect scale, using 1.0")
    layer_names = extract_layer_names(page)
    print(f"[DEBUG] Found {len(layer_names)} layers on page")
    
    # Use LLM to identify fence-related layers
    fence_layers = []
    if llm:
        fence_layers = llm_identify_fence_layers(llm, layer_names, fence_definitions)
    
    # If LLM didn't find layers, fall back to keyword matching
    if not fence_layers:
        print("[DEBUG] Falling back to keyword-based layer detection")
        fence_keywords = ['FENC', 'WALL', 'BARRIER', 'GUARD', 'RAIL', 'GATE', 'PERIM', 'BNDY', 'BOUND']
        for layer in layer_names:
            layer_upper = layer.upper()
            if any(kw in layer_upper for kw in fence_keywords):
                fence_layers.append(layer)
        print(f"[DEBUG] Keyword-matched fence layers: {fence_layers}")
    
    # Match fence layers to definition categories using LLM
    layer_to_category = {}
    if fence_layers and llm and fence_definitions:
        layer_to_category = llm_match_layers_to_definitions(llm, fence_layers, fence_definitions)
    
    # Extract lines from fence-related layers
    if fence_layers:
        fence_lines = extract_lines_by_layers(page, fence_layers)
    else:
        fence_lines = []
    print(f"[DEBUG] Extracted {len(fence_lines)} lines from fence layers")
    
    # Group lines by layer for per-layer measurements
    lines_by_layer = group_lines_by_layer(fence_lines)
    
    # Calculate measurements per layer
    layer_measurements = {}
    for layer, lines in lines_by_layer.items():
        # Group connected lines
        connected_groups = group_connected_lines(lines)
        
        # Calculate total measurement
        total = calculate_total_length(lines, scale_factor)
        
        # Calculate per-group measurements
        group_measurements = []
        for group in connected_groups:
            group_total = calculate_total_length(group, scale_factor)
            group_measurements.append({
                'segment_count': group_total['segment_count'],
                'length_feet': round(group_total['total_feet'], 2)
            })
        
        # Sort groups by length (largest first)
        group_measurements.sort(key=lambda x: x['length_feet'], reverse=True)
        
        layer_measurements[layer] = {
            'total_segments': total['segment_count'],
            'total_length_feet': round(total['total_feet'], 2),
            'connected_runs': len(connected_groups),
            'runs': group_measurements[:10]  # Top 10 runs
        }
    
    # =========================================================================
    # FIGURE-CONSTRAINED MEASUREMENT:
    # Only measure lines that are INSIDE ADE-detected figure/drawing areas
    # =========================================================================
    
    # Get figure bounding boxes from ADE figure_chunks (the actual drawing areas)
    figure_bboxes = []
    if figure_chunks:
        for chunk in figure_chunks:
            x0 = chunk.get("x0", 0)
            y0 = chunk.get("y0", 0)
            x1 = chunk.get("x1", 0)
            y1 = chunk.get("y1", 0)
            # Only use chunks that are large enough to be actual drawing areas
            if x1 - x0 > 100 and y1 - y0 > 100:
                figure_bboxes.append((x0, y0, x1, y1))
    
    print(f"[DEBUG] Found {len(figure_bboxes)} figure/drawing area bounding boxes")
    
    # Filter fence_lines to only those inside figure bboxes
    def line_in_any_bbox(line, bboxes, margin=50.0):
        """Check if a line is inside or near any bounding box."""
        for (x0, y0, x1, y1) in bboxes:
            # Expand bbox by margin
            x0m, y0m = x0 - margin, y0 - margin
            x1m, y1m = x1 + margin, y1 + margin
            # Check if either endpoint is inside
            sx, sy = line.start
            ex, ey = line.end
            if (x0m <= sx <= x1m and y0m <= sy <= y1m) or \
               (x0m <= ex <= x1m and y0m <= ey <= y1m):
                return True
        return False
    
    # If we have figure bboxes, filter fence_lines
    if figure_bboxes and fence_lines:
        filtered_fence_lines = [l for l in fence_lines if line_in_any_bbox(l, figure_bboxes)]
        print(f"[DEBUG] Filtered to {len(filtered_fence_lines)} lines inside figures (from {len(fence_lines)})")
    else:
        filtered_fence_lines = fence_lines
    
    # =========================================================================
    # LAYER-FIRST APPROACH:
    # Use layer-based lines if we found any, otherwise fall back to proximity
    # =========================================================================
    
    indicator_measurements = {}
    final_fence_lines = []
    measurement_method = "none"
    
    if filtered_fence_lines:
        # PRIMARY: Use layer-based lines (already filtered to figures)
        measurement_method = "layer"
        final_fence_lines = filtered_fence_lines
        
        # Calculate per-indicator measurements by proximity within filtered lines
        for item in list(fence_instances) + list(fence_definitions):
            ind = item.get("indicator", "") or item.get("keyword", "")
            if not ind:
                continue
            
            bbox = (item.get("x0", 0), item.get("y0", 0), 
                    item.get("x1", 0), item.get("y1", 0))
            
            if bbox[2] - bbox[0] < 1:
                continue
            
            # Find lines near this indicator (within filtered lines)
            nearby = find_lines_near_bbox(filtered_fence_lines, bbox, margin=80.0)
            
            if nearby:
                total = calculate_total_length(nearby, scale_factor)
                if ind not in indicator_measurements:
                    indicator_measurements[ind] = {
                        'instance_count': 0,
                        'run_segment_count': 0,
                        'run_length_feet': 0.0,
                        'run_length_pts': 0.0
                    }
                indicator_measurements[ind]['instance_count'] += 1
                indicator_measurements[ind]['run_segment_count'] += total['segment_count']
                indicator_measurements[ind]['run_length_feet'] += total['total_feet']
                indicator_measurements[ind]['run_length_pts'] += total['total_pts']
        
        print(f"[DEBUG] Layer-based measurement: {len(final_fence_lines)} lines")
    
    else:
        # ALTERNATIVE: For layerless PDFs, use LLM-guided filtering
        measurement_method = "llm_guided"
        print(f"[DEBUG] No fence layers found - using LLM-guided filtering")
        
        all_page_lines = extract_vector_lines(page)
        
        # Compute line statistics for LLM
        total_lines = len(all_page_lines)
        under_10 = sum(1 for l in all_page_lines if l.length_pts < 10)
        range_10_50 = sum(1 for l in all_page_lines if 10 <= l.length_pts < 50)
        range_50_100 = sum(1 for l in all_page_lines if 50 <= l.length_pts < 100)
        over_100 = sum(1 for l in all_page_lines if l.length_pts >= 100)
        
        line_stats = {
            'total': total_lines,
            'under_10': under_10,
            'range_10_50': range_10_50,
            'range_50_100': range_50_100,
            'over_100': over_100,
            'pct_under_10': (under_10 / total_lines * 100) if total_lines > 0 else 0,
            'pct_10_50': (range_10_50 / total_lines * 100) if total_lines > 0 else 0,
            'pct_50_100': (range_50_100 / total_lines * 100) if total_lines > 0 else 0,
            'pct_over_100': (over_100 / total_lines * 100) if total_lines > 0 else 0,
            'layers': 'None detected',
            'indicators': [d.get('indicator', '') for d in fence_definitions][:5]
        }
        
        # Get LLM-suggested parameters
        filter_params = llm_suggest_filter_params(llm, line_stats)
        MIN_LINE_LENGTH = filter_params['min_length']
        PROXIMITY_MARGIN = filter_params['proximity_margin']
        
        print(f"[DEBUG] LLM params: min_length={MIN_LINE_LENGTH}, margin={PROXIMITY_MARGIN}")
        
        # Filter 1: Apply LLM-suggested length filter
        candidate_lines = [l for l in all_page_lines if l.length_pts > MIN_LINE_LENGTH]
        print(f"[DEBUG] Lines > {MIN_LINE_LENGTH} pts: {len(candidate_lines)} (from {len(all_page_lines)})")
        
        # Filter 2: Apply figure bbox constraint
        if figure_bboxes:
            candidate_lines = [l for l in candidate_lines if line_in_any_bbox(l, figure_bboxes)]
            print(f"[DEBUG] After figure constraint: {len(candidate_lines)}")
        
        # Find lines near each indicator using LLM-suggested proximity
        seen_line_ids = set()
        
        for item in list(fence_instances) + list(fence_definitions):
            ind = item.get("indicator", "") or item.get("keyword", "")
            if not ind:
                continue
            
            bbox = (item.get("x0", 0), item.get("y0", 0), 
                    item.get("x1", 0), item.get("y1", 0))
            
            if bbox[2] - bbox[0] < 1:
                continue
            
            # Find lines near this indicator using LLM-suggested margin
            nearby = find_lines_near_bbox(candidate_lines, bbox, margin=PROXIMITY_MARGIN)
            
            if nearby:
                new_lines = [l for l in nearby if id(l) not in seen_line_ids]
                final_fence_lines.extend(new_lines)
                for l in new_lines:
                    seen_line_ids.add(id(l))
                
                total = calculate_total_length(nearby, scale_factor)
                if ind not in indicator_measurements:
                    indicator_measurements[ind] = {
                        'instance_count': 0,
                        'run_segment_count': 0,
                        'run_length_feet': 0.0,
                        'run_length_pts': 0.0
                    }
                indicator_measurements[ind]['instance_count'] += 1
                indicator_measurements[ind]['run_segment_count'] += total['segment_count']
                indicator_measurements[ind]['run_length_feet'] += total['total_feet']
                indicator_measurements[ind]['run_length_pts'] += total['total_pts']
        
        print(f"[DEBUG] Length-filtered measurement: {len(final_fence_lines)} lines")
    
    # Round indicator measurements
    for ind in indicator_measurements:
        indicator_measurements[ind]['run_length_feet'] = round(
            indicator_measurements[ind]['run_length_feet'], 2
        )
        indicator_measurements[ind]['run_length_pts'] = round(
            indicator_measurements[ind]['run_length_pts'], 1
        )
    
    # Calculate totals
    grand_total_segments = sum(m['total_segments'] for m in layer_measurements.values())
    grand_total_feet = sum(m['total_length_feet'] for m in layer_measurements.values())
    
    final_total = calculate_total_length(final_fence_lines, scale_factor) if final_fence_lines else {}
    
    # Run dimension line detection (find measurement text and match to lines)
    dimension_result = detect_dimension_lines(page, scale_factor)
    
    # Add dimension lines to final fence lines if found
    if dimension_result.get('success') and dimension_result.get('matched_lines'):
        dim_lines = dimension_result['matched_lines']
        existing_endpoints = set()
        for line in final_fence_lines:
            existing_endpoints.add((round(line.start[0], 1), round(line.start[1], 1), 
                                   round(line.end[0], 1), round(line.end[1], 1)))
        
        new_dim_lines = []
        for line in dim_lines:
            key = (round(line.start[0], 1), round(line.start[1], 1),
                   round(line.end[0], 1), round(line.end[1], 1))
            if key not in existing_endpoints:
                new_dim_lines.append(line)
                existing_endpoints.add(key)
        
        if new_dim_lines:
            final_fence_lines.extend(new_dim_lines)
            final_total = calculate_total_length(final_fence_lines, scale_factor)
    
    result = {
        'page_info': {
            'width': page_width,
            'height': page_height,
            'rotation': rotation,
            'scale_factor': scale_factor,
            'scale_detected': scale_factor != 1.0
        },
        'measurement_method': measurement_method,
        'fence_layers': fence_layers,
        'layer_to_category': layer_to_category,
        'all_fence_lines': final_fence_lines,
        'layer_measurements': layer_measurements,
        'indicator_measurements': indicator_measurements,
        'dimension_measurements': dimension_result.get('measurements', []),
        'proximity_totals': {
            'total_segments': final_total.get('segment_count', 0),
            'total_length_feet': round(final_total.get('total_feet', 0), 2),
            'total_length_pts': round(final_total.get('total_pts', 0), 1)
        },
        'totals': {
            'total_layers': len(layer_measurements),
            'total_segments': grand_total_segments,
            'total_length_feet': round(grand_total_feet, 2)
        }
    }
    
    print(f"[DEBUG] Measurement complete ({measurement_method}): {final_total.get('total_feet', 0):.1f} ft from {len(final_fence_lines)} lines")
    return result


def highlight_fence_lines(
    page_image_bytes: bytes,
    fence_lines: List,  # List of VectorLine
    pdf_width: float,
    pdf_height: float
) -> bytes:
    """
    Highlight measured fence lines on the page image.
    
    Args:
        page_image_bytes: Original page image
        fence_lines: List of VectorLine objects to highlight
        pdf_width, pdf_height: PDF page dimensions
    
    Returns:
        Image bytes with highlighted lines
    """
    if not fence_lines:
        return page_image_bytes
    
    try:
        img = Image.open(BytesIO(page_image_bytes))
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        scale_x = img_w / pdf_width if pdf_width > 0 else 1.0
        scale_y = img_h / pdf_height if pdf_height > 0 else 1.0
        
        for line in fence_lines:
            x1 = line.start[0] * scale_x
            y1 = line.start[1] * scale_y
            x2 = line.end[0] * scale_x
            y2 = line.end[1] * scale_y
            
            # Draw line in cyan color
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 255, 200), width=3)
        
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as e:
        print(f"[DEBUG] Fence line highlight error: {e}")
        return page_image_bytes

