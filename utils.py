"""
OCR and Fence Detection Utility using PaddleOCR (Streamlit Cloud Compatible)
"""

import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# Load PaddleOCR once
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def detect_text_with_paddleocr(image_bytes):
    """
    Perform OCR on an image using PaddleOCR.

    Args:
        image_bytes: Raw image bytes

    Returns:
        List of OCR text elements with 'text', 'conf', and 'bbox'
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    results = ocr_engine.ocr(np_image, cls=True)
    text_elements = []

    for line in results:
        for word_info in line:
            bbox = word_info[0]
            text = word_info[1][0]
            conf = word_info[1][1]
            if conf > 0.3 and text.strip():
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x, y, w, h = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
                text_elements.append({
                    "text": text.strip(),
                    "conf": conf,
                    "bbox": (int(x), int(y), int(w), int(h))
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
    text_elements = detect_text_with_paddleocr(image_bytes)

    # Create image visualization
    highlighted_image = visualize_with_bounding_boxes(image_bytes, text_elements)

    # Keyword filtering from OCR text
    matched_texts = [
        {"text": el["text"]}
        for el in text_elements
        if any(k.lower() in el["text"].lower() for k in FENCE_KEYWORDS)
    ]

    fence_found = bool(matched_texts)

    # Optional: LLM-based analysis
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
