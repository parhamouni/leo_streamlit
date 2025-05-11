import fitz  # PyMuPDF
import base64
import re
import time
import random
from langchain_core.messages import HumanMessage
from openai import RateLimitError
import pdfplumber
from io import BytesIO

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

def get_fence_related_text_boxes(page_bytes, llm, fence_keywords):
    """
    Extract text lines from a PDF page and identify fence-related text elements
    using LLM analysis.
    
    Args:
        page_bytes: Binary data of the PDF page
        llm: LLM model for text analysis
        fence_keywords: List of fence-related keywords
    
    Returns:
        List of text boxes (with coordinates) that are fence-related
    """
    # Open PDF from binary data
    with pdfplumber.open(BytesIO(page_bytes)) as pdf:
        # Get the first (and only) page
        pdf_page = pdf.pages[0]
        
        # Extract all text lines with bounding boxes
        text_lines = pdf_page.extract_text_lines()
        
        # Also extract all individual characters to catch standalone numbers
        chars = pdf_page.chars
        
        # First, analyze text lines
        fence_related_boxes = []
        fence_related_ids = set()  # To track which legend items are fence-related
        
        # Analyze each text line with LLM
        for i, line in enumerate(text_lines):
            # Get the text content
            line_text = line['text']
            
            # Skip very short text that's likely not meaningful
            if len(line_text) < 1:  # Changed from 3 to 1 to include single characters/numbers
                continue
                
            # Check if the line contains any fence keywords (quick pre-filter)
            contains_keyword = any(kw.lower() in line_text.lower() for kw in fence_keywords)
            
            # If this looks like a legend item with an ID (e.g., "1. Fence Type A")
            legend_id_match = re.match(r'^\s*(\d+)[\.:]?\s+(.+)', line_text)
            
            # If basic keyword match, or might be a specification, or is a legend item
            if (contains_keyword or 
                (len(line_text) >= 2 and any(c.isdigit() for c in line_text)) or
                legend_id_match):
                
                # Prepare prompt based on what we're analyzing
                if legend_id_match:
                    prompt = f"""
                    You are analyzing a legend item in an engineering drawing.
                    Determine if this legend item is related to fences, fencing, gates, barriers, or guardrails.
                    It's critical to identify even subtle references to fence components or types.
                    
                    Legend item: "{line_text}"
                    
                    Respond with ONLY "YES" or "NO".
                    """
                else:
                    prompt = f"""
                    You are analyzing engineering drawings text.
                    Determine if this text element is related to fences, fencing, gates, barriers, or guardrails.
                    It could be a dimension, specification, note, label, or legend item about fences.
                    
                    Text: "{line_text}"
                    
                    Respond with ONLY "YES" or "NO".
                    """
                
                response = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt)])
                is_fence_related = "YES" in response.content.upper()
                
                if is_fence_related:
                    # Get bounding box from the line
                    fence_related_boxes.append({
                        'x0': line['x0'],
                        'y0': line['top'],
                        'x1': line['x1'],
                        'y1': line['bottom'],
                        'text': line_text
                    })
                    
                    # If this is a legend item, extract its ID for later reference
                    if legend_id_match:
                        fence_related_ids.add(legend_id_match.group(1))
        
        # Now, look for standalone numbers that match fence-related legend IDs
        if fence_related_ids:
            # Group chars into potential standalone numbers
            number_groups = {}
            
            for char in chars:
                # Skip if not a digit
                if not char['text'].isdigit():
                    continue
                
                # Create a key based on approximate position (within a small tolerance)
                # This groups adjacent digits together
                position_key = (round(char['x0']), round(char['top']))
                
                if position_key not in number_groups:
                    number_groups[position_key] = {
                        'text': char['text'],
                        'x0': char['x0'],
                        'y0': char['top'],
                        'x1': char['x1'],
                        'y1': char['bottom']
                    }
                else:
                    # Update the existing group
                    group = number_groups[position_key]
                    group['text'] += char['text']
                    group['x1'] = max(group['x1'], char['x1'])
                    group['y1'] = max(group['y1'], char['bottom'])
            
            # Check if any of these number groups match our fence-related IDs
            for pos_key, group in number_groups.items():
                if group['text'] in fence_related_ids:
                    # This is a standalone number that matches a fence-related legend item
                    fence_related_boxes.append(group)
        
        return fence_related_boxes, pdf_page.width, pdf_page.height