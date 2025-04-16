import fitz  # PyMuPDF
import base64
import re
import time
import random
from langchain_core.messages import HumanMessage
from openai import RateLimitError

FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail']

def extract_snippet(text):
    for kw in FENCE_KEYWORDS:
        match = re.search(rf".{{0,30}}{kw}.{{0,30}}", text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def is_positive_response(response: str) -> bool:
    response = response.strip().lower()

    starts_yes = response.startswith("yes")
    contradiction = "not related" in response or "not a fence" in response or "no fence" in response

    return starts_yes and not contradiction


def retry_with_backoff(func, *args, retries=5, base_delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit hit. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    raise RuntimeError("Max retries exceeded due to rate limit errors.")

def analyze_page(page, llm_text, llm_vision, FENCE_KEYWORDS):
    text = page["text"]
    text_response = None
    text_snippet = extract_snippet(text)
    text_found = False
    vision_found = False
    vision_result = None

    # Step 1: Always run LLM text analysis
    text_prompt = f"""You are an assistant analyzing engineering drawings.
    Does the following page text refer to any fences or fence-related elements (like gates, barriers, or fencing types)? 
    Start your answer with 'Yes' or 'No'. Then explain why.

    Text:
    {text}
    """
    response = retry_with_backoff(llm_text.invoke, [HumanMessage(content=text_prompt)])
    text_response = response.content
    text_found = is_positive_response(text_response)

    # Step 2: If text doesn't confirm, and vision model is available, run vision check
    if not text_found and llm_vision:
        image_url = f"data:image/png;base64,{page['image_b64']}"
        vision_prompt = [
            {"type": "text", "text": "Does this drawing include any fences or fence-related structures? Start your answer with 'Yes' or 'No'. Then explain."},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        response = retry_with_backoff(llm_vision.invoke, [HumanMessage(content=vision_prompt)])
        vision_result = response.content
        vision_found = is_positive_response(vision_result)

    # Final result
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
