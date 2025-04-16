import fitz  # PyMuPDF
import base64
import re
import time
import random
import asyncio
from langchain_core.messages import HumanMessage
from openai import RateLimitError

FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail']

def extract_pdf_data(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text()
        pix = page.get_pixmap(dpi=100)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        pages.append({
            "page_number": i + 1,
            "text": text,
            "image_bytes": img_bytes,
            "image_b64": img_b64
        })
    return pages

def extract_snippet(text):
    for kw in FENCE_KEYWORDS:
        match = re.search(rf".{{0,30}}{kw}.{{0,30}}", text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def is_positive_response(response: str) -> bool:
    response = response.strip().lower()
    return response.startswith("yes") or response.startswith("yes,") or "includes a fence" in response or "shows a fence" in response

async def retry_with_backoff_async(func, *args, retries=5, base_delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except RateLimitError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    raise RuntimeError("Max retries exceeded due to rate limit errors.")

async def analyze_page_async(page, llm_text, llm_vision, FENCE_KEYWORDS):
    text = page["text"]
    text_response = None
    text_snippet = None
    text_found = False
    vision_found = False
    vision_result = None

    # Early skip using fast keyword filtering
    skip_text_analysis = not any(kw in text.lower() for kw in FENCE_KEYWORDS)

    # --- Run text analysis if not skipped ---
    if not skip_text_analysis:
        text_prompt = f"""You are an assistant analyzing engineering drawings.
        Does the following page text refer to any fences or fence-related elements (like gates, barriers, or fencing types)? 
        Start your answer with 'Yes' or 'No'. Then explain why.

        Text:
        {text}
        """
        text_response = (await retry_with_backoff_async(llm_text.ainvoke, [HumanMessage(content=text_prompt)])).content
        text_found = is_positive_response(text_response)
        text_snippet = extract_snippet(text)

    # --- If text doesn't confirm, optionally run vision ---
    if not text_found and llm_vision:
        image_url = f"data:image/png;base64,{page['image_b64']}"
        vision_prompt = [
            {"type": "text", "text": "Does this drawing include any fences or fence-related structures? Start your answer with 'Yes' or 'No'. Then explain."},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        vision_response = await retry_with_backoff_async(llm_vision.ainvoke, [HumanMessage(content=vision_prompt)])
        vision_result = vision_response.content
        vision_found = is_positive_response(vision_result)

    fence_found = text_found or vision_found

    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": text_found,
        "vision_found": vision_found,
        "text_response": text_response,
        "vision_response": vision_result,
        "text_snippet": text_snippet,
        "image": page["image_bytes"]
    }
