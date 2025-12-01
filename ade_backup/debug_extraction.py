#!/usr/bin/env python3
"""Debug script to see what's being extracted."""
import os
import sys
from pathlib import Path
from io import BytesIO
import fitz

from utils_ade_official import (
    ade_parse_document_official,
    align_ade_chunks_to_page,
    extract_indicators_from_legend_text,
    get_page_dimensions,
    create_single_page_pdf
)

FENCE_KEYWORDS = [
    'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 
    'mesh', 'panel', 'chain link', 'masonry', 'fence details', 
    'canopy shading', 'adot specifications', 'mag specifications', 
    'rail', 'railing', 'bollards', 'handrails', 'wall', 'cmu', 'keynote'
]

pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
ade_api_key = os.getenv("LANDINGAI_API_KEY") or "cXp5ZDZiZzRjbXc0OWdxZTNtM3N5OkVPclVRdG11T2hLSXBtbTV1cE9SdFFvWUhndm5YekdM"

with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

# Process with ADE
print("Processing with ADE...")
full_doc_ade_result = ade_parse_document_official(pdf_bytes, ade_api_key, zdr=False)

if not full_doc_ade_result.get("success"):
    print(f"Error: {full_doc_ade_result.get('error')}")
    sys.exit(1)

# Check multiple pages for legend tables
for page_idx in [1, 2]:  # Pages 2 and 3
    page_num = page_idx + 1
    print(f"\n{'='*60}")
    print(f"PAGE {page_num}")
    print(f"{'='*60}")
    single_page_bytes = create_single_page_pdf(pdf_bytes, page_idx)
    page_width, page_height = get_page_dimensions(single_page_bytes)
    
    page_chunks = align_ade_chunks_to_page(full_doc_ade_result, page_idx, page_width, page_height)
    
    print(f"\nPage {page_num}: Found {len(page_chunks)} chunks")
    print(f"Chunk types: {[c.get('type') for c in page_chunks[:10]]}")
    
    # Check all text chunks for indicator codes
    print(f"\n=== PAGE {page_num} TEXT CHUNKS (checking for indicator codes) ===")
    import re
    for i, chunk in enumerate(page_chunks):
        chunk_type = chunk.get("type", "").lower()
        if chunk_type == "text":
            chunk_text = chunk.get("text", "")
            # Look for 4-digit codes
            codes = re.findall(r'\b(0\d{3}|[3-9]\d{3})\b', chunk_text)
            if codes:
                print(f"  Chunk {i} (text): Found codes {codes}")
                print(f"    Text preview: {chunk_text[:200]}")
    
    # Extract legend text - focus on finding legend tables
    legend_text_parts = []
    legend_tables = []
    
    for chunk in page_chunks:
        element_type = chunk.get("type", "").lower()
        if element_type == "table":
            table_text = chunk.get("text", "")
            # Check if this looks like a legend/keynote table
            table_lower = table_text.lower()
            # Look for keywords that suggest it's a legend/keynote table
            legend_keywords = ["keynote", "number", "description", "note", "symbol", "legend"]
            if any(kw in table_lower for kw in legend_keywords):
                legend_tables.append(table_text)
                print(f"\n  Found potential legend table:")
                print(f"    Preview: {table_text[:300]}")
        elif element_type == "text":
            element_text = chunk.get("text", "")
            if element_text:
                legend_text_parts.append(element_text)
    
    legend_text = "\n".join(legend_text_parts)
    print(f"\nLegend text length: {len(legend_text)}")
    
    # Check ALL tables for 4-digit indicator codes
    print(f"\n=== PAGE {page_num} ALL TABLES (searching for 4-digit codes) ===")
    import re
    all_tables = [chunk.get("text", "") for chunk in page_chunks if chunk.get("type", "").lower() == "table"]
    
    for table_idx, table in enumerate(all_tables):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', table, re.DOTALL)
        found_codes = []
        for cell in cells:
            cell_text = re.sub(r'<[^>]+>', '', cell).strip()
            # Look for 4-digit indicator codes
            if re.match(r'^(0\d{3}|[3-9]\d{3})$', cell_text):
                found_codes.append(cell_text)
        
        if found_codes:
            print(f"\nTable {table_idx + 1} contains indicator codes: {found_codes}")
            # Show context around these codes
            for code in found_codes[:5]:
                # Find the row containing this code
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
                for row in rows:
                    if code in row:
                        row_cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                        row_text = ' | '.join([re.sub(r'<[^>]+>', '', c).strip()[:50] for c in row_cells[:3]])
                        print(f"    Code {code} in row: {row_text}")
                        break
    
    # Extract indicators using LLM
    from utils_ade_official import extract_indicators_from_legend_text_llm
    from langchain_openai import ChatOpenAI
    import os
    
    openai_key = os.getenv("OPENAI_API_KEY") or "test"
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key) if openai_key != "test" else None
    
    if llm and legend_text:
        indicators = extract_indicators_from_legend_text_llm(legend_text, FENCE_KEYWORDS, llm)
        print(f"\nPage {page_num} - LLM extracted {len(indicators)} indicators:")
        for ind in indicators[:10]:
            print(f"  - {ind['indicator']} (description: '{ind.get('text', '')[:60]}...')")
    else:
        indicators = extract_indicators_from_legend_text(legend_text, FENCE_KEYWORDS)
        print(f"\nPage {page_num} - Regex extracted {len(indicators)} indicators:")
        for ind in indicators[:10]:
            print(f"  - {ind['indicator']} (from: '{ind['text'][:60]}...')")

