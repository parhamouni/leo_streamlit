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
    llm
) -> List[Dict]:
    print("[DEBUG] Extracting Legend Entries and Matching BBoxes...")
    results = []

    # Helper: Make a shortened version of the text
    def get_substring(text):
        words = text.split()
        if len(words) > 4:
            return " ".join(words[:4])
        return text

    for chunk in legend_chunks:
        text = chunk.get("text", "")
        if not text:
            continue

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
    """
    Find instances of legend indicators ONLY within figure/architectural_drawing chunks.
    
    Args:
        legend_entries: List of definitions with 'indicator' field
        figure_chunks: List of figure region bounding boxes (type=figure or architectural_drawing)
        all_tokens: All text tokens from the page
    """
    print("[DEBUG] Finding Instances in Figure Chunks...")
    print(f"[DEBUG] Figure chunks count: {len(figure_chunks)}")
    for i, fc in enumerate(figure_chunks):
        print(f"[DEBUG]   Figure {i}: type={fc.get('type')} bbox=({fc['x0']:.1f}, {fc['y0']:.1f}) - ({fc['x1']:.1f}, {fc['y1']:.1f})")
    
    instances = []
    
    # Collect all indicators to search for
    indicators_to_find = set()
    for item in legend_entries:
        ind = item.get("indicator", "").strip()
        if ind:
            indicators_to_find.add(ind)
            # Also add cleaned version (remove special chars)
            clean_ind = re.sub(r'[^\w]', '', ind)
            if clean_ind:
                indicators_to_find.add(clean_ind)
    
    print(f"[DEBUG] Looking for indicators: {indicators_to_find}")
    
    if not indicators_to_find:
        return []
    
    if not figure_chunks:
        print("[DEBUG] No figure chunks to search in!")
        return []
    
    # Get legend bounding boxes to exclude (don't match indicators in legend area)
    legend_bboxes = []
    for entry in legend_entries:
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


def measure_fence_elements(
    page: fitz.Page,
    fence_definitions: List[Dict],
    fence_instances: List[Dict],
    figure_chunks: List[Dict] = None,
    llm=None,
    scale_factor: Optional[float] = None
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
        scale_factor = infer_scale_from_page(page)
        if scale_factor:
            print(f"[DEBUG] Auto-detected scale factor: {scale_factor}")
        else:
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
        # NO FENCE LAYERS FOUND - Skip proximity fallback as it's unreliable
        # (Parking stripes and other repeating patterns get incorrectly detected)
        measurement_method = "no_layers"
        print(f"[DEBUG] No fence layers found - measurement skipped (proximity fallback disabled)")
        print(f"[DEBUG] Hint: PDFs without layer information cannot be reliably measured")
    
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
        'all_fence_lines': final_fence_lines,
        'layer_measurements': layer_measurements,
        'indicator_measurements': indicator_measurements,
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

