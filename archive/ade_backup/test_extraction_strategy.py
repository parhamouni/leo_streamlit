#!/usr/bin/env python3
"""
Test script to verify the extraction strategy is working.
"""
import sys
from pathlib import Path
import fitz
from io import BytesIO
import json

# Load config
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    openai_api_key = secrets.get("OPENAI_API_KEY") or None
    if not openai_api_key:
        print("❌ OpenAI API key not found")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from utils_ade_official import extract_legend_keywords_and_indicators, extract_indicators_from_text_llm

# Test on page 2 (which has codes 3301, 0113, 0402)
pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
doc = fitz.open(str(pdf_path))
page = doc[1]  # Page 2

# Get page dimensions
page_width, page_height = page.rect.width, page.rect.height

# Extract PDF text layer
pdf_text_layer_words = []
words = page.get_text("words")
for word_tuple in words:
    x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
    pdf_text_layer_words.append({
        "text": text,
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "block_no": block_no,
        "line_no": line_no
    })

print(f"📄 Page 2: {len(pdf_text_layer_words)} text layer words")
print(f"   Page size: {page_width:.1f} x {page_height:.1f}")

# Group text layer by lines
text_layer_lines = []
current_line = []
prev_y = None

for word in sorted(pdf_text_layer_words, key=lambda x: (x.get("y0", 0), x.get("x0", 0))):
    text = word.get("text", "").strip()
    if not text:
        continue
    
    y = word.get("y0", 0)
    if prev_y is None or abs(y - prev_y) < 10:
        current_line.append(text)
    else:
        if current_line:
            text_layer_lines.append(" ".join(current_line))
        current_line = [text]
    prev_y = y

if current_line:
    text_layer_lines.append(" ".join(current_line))

print(f"\n📝 Grouped into {len(text_layer_lines)} lines")

# Filter lines that might contain indicators
fence_keywords = ["fence", "gate", "barrier", "guardrail", "cmu", "screen wall"]
filtered_lines = []
for line in text_layer_lines:
    line_lower = line.lower()
    if (len(line) <= 15 or 
        any(kw in line_lower for kw in fence_keywords) or
        __import__('re').search(r'\b(0\d{3}|[3-9]\d{3})\b', line)):
        filtered_lines.append(line)

print(f"🔍 Filtered to {len(filtered_lines)} potential indicator lines")
print(f"\nSample filtered lines:")
for i, line in enumerate(filtered_lines[:10]):
    print(f"  {i+1}. {line[:80]}")

# Test LLM extraction
if filtered_lines:
    text_layer_text = "\n".join(filtered_lines[:100])
    print(f"\n🤖 Sending to LLM ({len(text_layer_text)} chars)...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    
    indicators = extract_indicators_from_text_llm(text_layer_text, fence_keywords, llm)
    print(f"\n✅ LLM extracted {len(indicators)} indicators:")
    for ind in indicators:
        print(f"   - Code: {ind.get('indicator')}, Desc: {ind.get('description')[:50]}")

# Now test full extraction function
print(f"\n\n🧪 Testing full extract_legend_keywords_and_indicators function...")
print(f"   (No ADE chunks, only PDF text layer + mock OCR)")

# Create mock OCR (just use text layer words as OCR items)
mock_ocr = []
for word in pdf_text_layer_words:
    mock_ocr.append({
        "text": word["text"],
        "x0": word["x0"],
        "y0": word["y0"],
        "x1": word["x1"],
        "y1": word["y1"]
    })

highlights = extract_legend_keywords_and_indicators(
    page_chunks=[],  # No ADE chunks
    google_ocr_results=mock_ocr,
    fence_keywords=fence_keywords,
    page_width=page_width,
    page_height=page_height,
    llm=llm,
    pdf_text_layer_words=pdf_text_layer_words
)

print(f"\n✅ Function returned {len(highlights)} highlights")
print(f"\nSample highlights:")
for i, h in enumerate(highlights[:10]):
    print(f"  {i+1}. Text: '{h.get('text')}' at ({h.get('x0'):.1f}, {h.get('y0'):.1f})")
    print(f"      Tag: {h.get('tag_from_llm')}")

# Check if we found the known codes
known_codes = ['3301', '0113', '0402', '0401']
found_codes = set()
for h in highlights:
    text = h.get('text', '').strip()
    for code in known_codes:
        if code in text:
            found_codes.add(code)

print(f"\n📊 Found known codes: {found_codes}")
print(f"   Expected: {set(known_codes)}")
print(f"   Match: {'✅' if found_codes == set(known_codes) else '❌'}")

doc.close()



