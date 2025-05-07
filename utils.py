"""
Full “LLM first, then Azure OCR” Fence Detector utils.
"""

import os, io, time, base64
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Init Azure CV client from env vars
_endpoint = os.getenv("AZURE_CV_ENDPOINT")
_key      = os.getenv("AZURE_CV_KEY")
_cv_client = None
if _endpoint and _key:
    _cv_client = ComputerVisionClient(_endpoint, CognitiveServicesCredentials(_key))

def ocr_with_azure(image_bytes: bytes) -> list[dict]:
    """
    Returns list of words with bounding boxes via Azure Read API:
    Each dict: { text, conf, bbox:(x,y,w,h) }
    """
    if not _cv_client:
        raise RuntimeError("Azure CV client not configured")
    # submit for OCR
    op = _cv_client.read_in_stream(io.BytesIO(image_bytes), raw=True)
    op_loc = op.headers["Operation-Location"]
    op_id = op_loc.split("/")[-1]
    # poll
    while True:
        result = _cv_client.get_read_result(op_id)
        if result.status not in ("notStarted", "running"):
            break
        time.sleep(0.5)
    words = []
    if result.status == "succeeded":
        for page in result.analyze_result.read_results:
            for line in page.lines:
                for word in line.words:
                    bb = word.bounding_box  # [x1,y1,x2,y2...]
                    xs, ys = bb[0::2], bb[1::2]
                    x, y = min(xs), min(ys)
                    w, h = max(xs)-x, max(ys)-y
                    words.append({
                        "text": word.text,
                        "conf": word.confidence,
                        "bbox": (int(x), int(y), int(w), int(h))
                    })
    return words

def visualize_with_boxes(image_bytes: bytes, words: list[dict]) -> bytes:
    """
    Draw red rectangles around each word. Return image PNG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    for w in words:
        x,y,w_,h_ = w["bbox"]
        draw.rectangle([x,y,x+w_,y+h_], outline=(255,0,0), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_page(page: dict, llm_vision, FENCE_KEYWORDS: list[str]) -> dict:
    """
    1) LLM vision YES/NO for fence.
    2) If YES → Azure OCR + keyword filter + visualize.
    Returns full dict for Streamlit.
    """
    image_bytes = page["image_bytes"]
    # 1) LLM vision check
    fence_found = False
    llm_analysis = ""
    if llm_vision:
        try:
            prompt = (
                "You are given an engineering drawing page image. "
                "Answer simply 'YES' if there are any fences, gates, barriers, or enclosures; otherwise 'NO'."
            )
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            resp = llm_vision.invoke(prompt=prompt, image=img)
            content = getattr(resp, "content", str(resp)).strip().upper()
            fence_found = content.startswith("YES")
            llm_analysis = content
        except Exception as e:
            llm_analysis = f"LLM vision error: {e}"
    else:
        llm_analysis = "No vision model provided."

    # defaults
    words, matches, highlighted = [], [], image_bytes

    # 2) If fence, run OCR+filter+viz
    if fence_found:
        words = ocr_with_azure(image_bytes)
        matches = [
            {"text": w["text"], "bbox": w["bbox"]}
            for w in words
            if any(k.lower() in w["text"].lower() for k in FENCE_KEYWORDS)
        ]
        highlighted = visualize_with_boxes(image_bytes, words)

    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": bool(words),
        "vision_found": bool(llm_vision),
        "text_references": matches,
        "ocr_text_elements": words,
        "image": image_bytes,
        "highlighted_image": highlighted,
        "llm_analysis": llm_analysis,
        "legend_items": [],
        "fence_indicators": []
    }
