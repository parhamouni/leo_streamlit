"""
OCR and Fence Detection Utility using EasyOCR (Streamlit Cloud Compatible)
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr

# Load EasyOCR model globally
ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)


def detect_text_with_easyocr(image_bytes):
    """
    Perform OCR on an image using EasyOCR.

    Args:
        image_bytes: Raw image bytes

    Returns:
        List of OCR text elements with 'text', 'conf', and 'bbox'
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    results = ocr_engine.readtext(np_image)
    text_elements = []

    for bbox, text, conf in results:
        if conf > 0.3 and text.strip():
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            x = int(min(x_coords))
            y = int(min(y_coords))
            w = int(max(x_coords) - x)
            h = int(max(y_coords) - y)
            text_elements.append({
                "text": text.strip(),
                "conf": conf,
                "bbox": (x, y, w, h)
            })

    return text_elements

def visualize_with_bounding_boxes(image_bytes, text_elements):
    """
    Draw bounding boxes around OCR-detected text elements.

    Args:
        image_bytes: Original image as bytes
        text_elements: List of text elements with bbox

    Returns:
        Annotated image as bytes
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Group text by Y proximity to identify lines
    y_tolerance = 10
    lines = {}
    for i, element in enumerate(text_elements):
        _, y, _, _ = element["bbox"]
        line_key = None
        for key in lines:
            if abs(key - y) < y_tolerance:
                line_key = key
                break
        if line_key is None:
            line_key = y
            lines[line_key] = []
        lines[line_key].append(i)

    # Assign colors per line
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128)
    ]
    line_colors = {k: colors[i % len(colors)] for i, k in enumerate(sorted(lines))}

    for i, element in enumerate(text_elements):
        x, y, w, h = element["bbox"]
        line_key = next(k for k in lines if i in lines[k])
        color = line_colors[line_key]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def analyze_page(page, llm_vision, FENCE_KEYWORDS):
    """
    Analyze a page for fence-related content using OCR and optionally LLM.

    Args:
        page: Dictionary with image_bytes and page_number
        llm_vision: Optional LLM for image captioning/vision analysis
        FENCE_KEYWORDS: List of fence-related keywords to look for in text

    Returns:
        Dictionary with detection and analysis results
    """
    image_bytes = page["image_bytes"]
    text_elements = detect_text_with_easyocr(image_bytes)

    # Create image visualization
    highlighted_image = visualize_with_bounding_boxes(image_bytes, text_elements)

    # Keyword filtering from OCR text
    matched_texts = [
        {"text": el["text"]}
        for el in text_elements
        if any(k.lower() in el["text"].lower() for k in FENCE_KEYWORDS)
    ]

    fence_found = bool(matched_texts)

    # Optional: LLM-based image captioning
    llm_analysis = ""
    if llm_vision and fence_found:
        try:
            prompt = "Describe any signs of fences, barriers, or enclosures in this engineering drawing."
            image = Image.open(io.BytesIO(image_bytes))
            response = llm_vision.invoke(prompt=prompt, image=image)
            llm_analysis = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            llm_analysis = f"LLM analysis failed: {e}"

    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": True,
        "vision_found": bool(llm_vision),
        "text_references": matched_texts,
        "ocr_text_elements": text_elements,
        "image": image_bytes,
        "highlighted_image": highlighted_image,
        "llm_analysis": llm_analysis or "OCR + keyword detection used for fence-related text."
    }
