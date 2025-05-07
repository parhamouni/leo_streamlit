"""
Full OCR + Fence Detector utils using OCR.Space + optional LLM vision.
"""

import io
import requests
from PIL import Image, ImageDraw, ImageFont
import base64

def call_ocr_space(image_bytes: bytes, api_key: str) -> dict:
    files = {'file': ('page.png', image_bytes)}
    data = {'apikey': api_key, 'language': 'eng', 'isOverlayRequired': True}
    resp = requests.post('https://api.ocr.space/parse/image', files=files, data=data)
    resp.raise_for_status()
    return resp.json()

def detect_text_with_boxes(image_bytes: bytes, ocr_api_key: str) -> list[dict]:
    """
    Returns list of words with bounding boxes:
    { WordText, Confidence, Left, Top, Width, Height }
    """
    result = call_ocr_space(image_bytes, ocr_api_key)
    if result.get("IsErroredOnProcessing", True):
        msg = result.get("ErrorMessage", ["Unknown error"])[0]
        raise RuntimeError(f"OCR.Space error: {msg}")
    words = []
    for pr in result.get("ParsedResults", []):
        overlay = pr.get("TextOverlay", {})
        for line in overlay.get("Lines", []):
            for w in line.get("Words", []):
                words.append({
                    "text": w["WordText"],
                    "conf": float(w.get("Confidence", 0)),
                    "bbox": (int(w["Left"]), int(w["Top"]),
                             int(w["Width"]), int(w["Height"]))
                })
    return words

def visualize_with_bounding_boxes(image_bytes: bytes, words: list[dict]) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    # Draw
    for w in words:
        x, y, w_, h_ = w["bbox"]
        draw.rectangle([x, y, x+w_, y+h_], outline=(255,0,0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_page(page: dict, llm_vision, FENCE_KEYWORDS: list[str]) -> dict:
    """
    page: { page_number, image_bytes }
    llm_vision: ChatOpenAI vision-capable—or None
    """
    image_bytes = page["image_bytes"]
    ocr_key = page.get("ocr_api_key")
    # 1. OCR + boxes
    words = detect_text_with_boxes(image_bytes, ocr_key)
    highlighted = visualize_with_bounding_boxes(image_bytes, words)
    # 2. Keyword filter
    matches = [
        {"text": w["text"], "bbox": w["bbox"]}
        for w in words
        if any(k.lower() in w["text"].lower() for k in FENCE_KEYWORDS)
    ]
    fence_found = bool(matches)
    # 3. Optional LLM vision analysis
    llm_analysis = ""
    if llm_vision and fence_found:
        try:
            prompt = (
                "This is an engineering drawing page. Extract any references to fences, barriers, gates, "
                "or enclosures, and describe their context."
            )
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            resp = llm_vision.invoke(prompt=prompt, image=img)
            llm_analysis = getattr(resp, "content", str(resp))
        except Exception as e:
            llm_analysis = f"LLM vision error: {e}"
    else:
        llm_analysis = "No fence references found or vision disabled."
    # 4. Legend/items/indicators placeholders (no PDF legend parsing here; dummy empty)
    legend_items = []
    fence_indicators = []
    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": True,
        "vision_found": bool(llm_vision),
        "text_references": matches,
        "ocr_text_elements": words,
        "image": image_bytes,
        "highlighted_image": highlighted,
        "llm_analysis": llm_analysis,
        "legend_items": legend_items,
        "fence_indicators": fence_indicators
    }
