import fitz  # PyMuPDF
import base64
import re
import time
import random
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from langchain_core.messages import HumanMessage
from openai import RateLimitError

FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'chain link', 'enclosure']

def extract_fence_references(text):
    """Extract all fence-related text references with their nearby context."""
    references = []
    for kw in FENCE_KEYWORDS:
        # Find all matches for this keyword
        pattern = re.compile(rf".{{0,40}}{kw}.{{0,40}}", re.IGNORECASE)
        for match in pattern.finditer(text):
            references.append({
                "text": match.group(0),
                "keyword": kw,
                "position": match.start()
            })
    return references

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

def extract_legend_fence_items(llm_vision, image_b64):
    """Extract fence-related items from the legend to know what indicators to look for."""
    image_url = f"data:image/png;base64,{image_b64}"
    
    prompt_text = """Analyze the LEGEND section of this engineering drawing and identify all items related to FENCES, GATES, or similar barriers.

For each fence-related item in the legend, provide:
1. The item number/identifier (typically a number like "1.", "2.", etc.)
2. The complete description of the fence item

Focus ONLY on items specifically mentioning:
- Fences
- Gates
- Barriers
- Guard rails
- Enclosures

Format your response as a JSON array:
```json
[
  {
    "item_number": "1",
    "description": "6'-0\" HIGH INTERIOR COURT FENCE. COLOR: BLACK VINYL."
  },
  {
    "item_number": "2",
    "description": "3'-0\" HIGH INTERIOR COURT FENCE. COLOR: BLACK VINYL."
  }
]
```

Return ONLY the JSON array with fence-related items from the legend.
"""
    
    legend_prompt = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    
    try:
        response = retry_with_backoff(llm_vision.invoke, [HumanMessage(content=legend_prompt)])
        content = response.content.strip()
        
        # Extract JSON from the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        try:
            import json
            legend_items = json.loads(content)
            return legend_items
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {content}")
            # Simple fallback for malformed JSON
            try:
                content = content.replace("'", "\"")
                content = re.sub(r',\s*]', ']', content)
                legend_items = json.loads(content)
                return legend_items
            except:
                return []
    except Exception as e:
        print(f"Error extracting legend items: {str(e)}")
        return []

def get_fence_indicators(llm_vision, image_b64, legend_items):
    """Find indicators (circled numbers) that refer to fence items in the legend."""
    image_url = f"data:image/png;base64,{image_b64}"
    
    # Build prompt with legend item information
    legend_info = ""
    item_numbers = []
    for item in legend_items:
        num = item.get("item_number", "")
        desc = item.get("description", "")
        legend_info += f"- Item {num}: {desc}\n"
        item_numbers.append(num)
    
    prompt_text = f"""Locate all indicators (circled numbers) in this engineering drawing that refer to FENCE-RELATED items according to the legend.

The following legend items are related to fences:
{legend_info}

For each indicator that matches one of these fence item numbers ({', '.join(item_numbers)}), provide:
1. The item number it refers to
2. The coordinates of the center of the indicator [x, y] as percentages of image dimensions (0-100)
3. A small bounding box around the indicator [x1, y1, x2, y2] as percentages

IMPORTANT:
- Focus ONLY on finding the INDICATORS (circled numbers) that match fence-related item numbers
- Do NOT try to trace the entire fence lines
- Look for circled numbers like ①, ②, ③ that match the fence item numbers
- Return coordinates for EACH instance of these indicators in the drawing

Format your response as a JSON array:
```json
[
  {{
    "item_number": "1",
    "center": [25, 30],
    "bounding_box": [23, 28, 27, 32]
  }},
  {{
    "item_number": "2",
    "center": [45, 50],
    "bounding_box": [43, 48, 47, 52]
  }}
]
```

Return ONLY the JSON array with indicator locations.
"""
    
    indicators_prompt = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    
    try:
        response = retry_with_backoff(llm_vision.invoke, [HumanMessage(content=indicators_prompt)])
        content = response.content.strip()
        
        # Extract JSON from the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON
        try:
            import json
            indicators = json.loads(content)
            return indicators
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {content}")
            # Simple fallback for malformed JSON
            try:
                content = content.replace("'", "\"")
                content = re.sub(r',\s*]', ']', content)
                indicators = json.loads(content)
                return indicators
            except:
                return []
    except Exception as e:
        print(f"Error getting fence indicators: {str(e)}")
        return []

def highlight_fence_indicators(image_bytes, indicators, legend_items):
    """Highlight fence indicators on the image with appropriate styles."""
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    
    # Create a drawing context
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Try to load a font for labels
    try:
        font = ImageFont.truetype("Arial", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Create a lookup for item descriptions
    item_descriptions = {item.get("item_number", ""): item.get("description", "") for item in legend_items}
    
    # Define different colors for different fence types
    colors = {
        "1": (0, 200, 0, 255),  # Green for item 1
        "2": (255, 0, 0, 255),  # Red for item 2
        "3": (0, 120, 255, 255),  # Blue for item 3
        "default": (255, 165, 0, 255)  # Orange default
    }
    
    # Draw each fence indicator
    for idx, indicator in enumerate(indicators):
        # Get indicator information
        item_number = indicator.get("item_number", "")
        if not item_number:
            continue
            
        # Get coordinates
        center = indicator.get("center", [0, 0])
        bbox = indicator.get("bounding_box", [0, 0, 0, 0])
        
        if len(center) != 2 or len(bbox) != 4:
            continue
            
        # Convert from percentages to pixels
        center_x = int((center[0] / 100) * width)
        center_y = int((center[1] / 100) * height)
        
        x1 = int((bbox[0] / 100) * width)
        y1 = int((bbox[1] / 100) * height)
        x2 = int((bbox[2] / 100) * width)
        y2 = int((bbox[3] / 100) * height)
        
        # Make the highlight area slightly larger for visibility
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Skip if any coordinate is outside image bounds
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 > width or y1 > height or x2 > width or y2 > height:
            continue
        
        # Get color based on item number
        color = colors.get(item_number, colors["default"])
        
        # Draw a circle around the indicator with a glow effect
        for radius in range(padding + 10, padding, -2):
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                outline=(*color[:3], max(0, color[3] - (20 * (padding + 10 - radius)))),
                width=2
            )
        
        # Draw filled highlight circle/oval around the indicator
        draw.ellipse(
            [x1, y1, x2, y2],
            fill=(*color[:3], 80),  # Semi-transparent fill
            outline=color,
            width=3
        )
        
        # Get description for label
        description = item_descriptions.get(item_number, "Fence Item")
        short_desc = description.split('.')[0]
        if len(short_desc) > 30:
            short_desc = short_desc[:27] + "..."
        
        label = f"Item {item_number}: {short_desc}"
        
        # Position label near the indicator
        label_x = x2 + 10
        label_y = center_y - 10
        
        # Ensure label is within image bounds
        if label_x + 200 > width:
            label_x = x1 - 210  # Place on left side if near right edge
            if label_x < 5:  # If still out of bounds, place below
                label_x = center_x - 100
                label_y = y2 + 10
        
        label_x = max(5, min(width - 210, label_x))
        label_y = max(5, min(height - 30, label_y))
        
        # Draw text background for better readability
        text_bb = draw.textbbox((label_x, label_y), label, font=font)
        text_width = text_bb[2] - text_bb[0]
        text_height = text_bb[3] - text_bb[1]
        
        draw.rectangle(
            [label_x - 2, label_y - 2, label_x + text_width + 2, label_y + text_height + 2],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
    
    # Convert back to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def analyze_page(page, llm_text, llm_vision, FENCE_KEYWORDS):
    """Analyze a page for fence indicators using the legend-based approach."""
    text = page["text"]
    image_bytes = page["image_bytes"]
    image_b64 = page["image_b64"]
    
    # Step 1: Extract fence-related text references
    fence_references = extract_fence_references(text)
    text_found = len(fence_references) > 0
    
    # Step 2: Extract fence items from the legend
    legend_items = []
    indicators = []
    highlighted_image = None
    vision_found = False
    
    if llm_vision:
        # Extract fence-related items from the legend
        legend_items = extract_legend_fence_items(llm_vision, image_b64)
        
        if legend_items:
            # Find indicators that refer to these items
            indicators = get_fence_indicators(llm_vision, image_b64, legend_items)
            
            # Create highlighted image if indicators were found
            if indicators:
                highlighted_image = highlight_fence_indicators(image_bytes, indicators, legend_items)
                vision_found = True
    
    # Final determination - fence found if either text or vision detected fence elements
    fence_found = text_found or vision_found
    
    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": text_found,
        "vision_found": vision_found,
        "text_references": fence_references,
        "image": image_bytes,
        "highlighted_image": highlighted_image,
        "legend_items": legend_items,
        "fence_indicators": indicators
    }