#!/usr/bin/env python3
"""
Test if OCR can detect actual indicator codes that should be highlighted.
Checks if indicator codes from legends are present in OCR results.
"""

import fitz
import json
import re
from pathlib import Path
from utils_ade_official import (
    ade_parse_document_official,
    align_ade_chunks_to_page,
    filter_ocr_by_ade_regions,
    get_google_ocr_results_with_boxes,
    extract_indicators_from_table_llm,
    extract_indicators_from_text_llm
)
from utils import create_document_ai_client
from langchain_openai import ChatOpenAI
import sys

def analyze_indicator_detection(pdf_path, page_idx=0):
    """Test if indicator codes can be detected in OCR."""
    print(f"\n{'='*80}")
    print(f"Testing Indicator Code Detection for Page {page_idx}")
    print(f"{'='*80}\n")
    
    # Load API keys
    try:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        api_key = secrets.get("LANDINGAI_API_KEY", "").strip().strip('"').strip("'")
        openai_key = secrets.get("OPENAI_API_KEY", "")
        if not openai_key:
            openai_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return
    
    # Load PDF
    doc = fitz.open(str(pdf_path))
    if page_idx >= len(doc):
        print(f"Error: Page {page_idx} not found")
        return
    
    page = doc[page_idx]
    page_width = page.rect.width
    page_height = page.rect.height
    
    # Create single-page PDF
    temp_doc = fitz.open()
    temp_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
    single_page_bytes = temp_doc.tobytes()
    temp_doc.close()
    
    # Get ADE chunks
    print("1. Parsing with ADE...")
    ade_result = ade_parse_document_official(single_page_bytes, api_key, zdr=False)
    if not ade_result.get("success"):
        print(f"   ❌ ADE parsing failed: {ade_result.get('error')}")
        return
    
    page_chunks = align_ade_chunks_to_page(ade_result, page_idx, page_width, page_height)
    print(f"   ✅ Found {len(page_chunks)} ADE chunks")
    
    # Get Google OCR
    print("\n2. Getting Google OCR results...")
    try:
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        secrets = toml.load(secrets_path)
        google_cloud_raw = {}
        if "google_cloud" in secrets:
            google_cloud_raw["project_number"] = secrets["google_cloud"].get("project_number")
            google_cloud_raw["location"] = secrets["google_cloud"].get("location")
            google_cloud_raw["processor_id"] = secrets["google_cloud"].get("processor_id")
        if "gcp_service_account" in secrets:
            google_cloud_raw["service_account_info"] = dict(secrets["gcp_service_account"])
        
        if google_cloud_raw.get("service_account_info"):
            client = create_document_ai_client(google_cloud_raw)
            processor_name = f"projects/{google_cloud_raw.get('project_number')}/locations/{google_cloud_raw.get('location')}/processors/{google_cloud_raw.get('processor_id')}"
            google_cloud_config = {
                "client": client,
                "processor_name": processor_name
            }
        else:
            print("   ❌ Google Cloud config incomplete")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    google_ocr_results = get_google_ocr_results_with_boxes(
        single_page_bytes,
        google_cloud_config
    )
    print(f"   ✅ Found {len(google_ocr_results)} OCR items")
    
    # Identify legend regions
    fence_keywords = ['fence', 'fencing', 'gate', 'barrier', 'keynote', 'legend']
    legend_regions = []
    
    for chunk in page_chunks:
        chunk_type = chunk.get("type", "").lower()
        chunk_text = (chunk.get("markdown") or chunk.get("text", "")).lower()
        
        if chunk_type == "table" or any(kw in chunk_text for kw in ["keynote", "legend", "note", "symbol"]):
            legend_regions.append({
                "x0": chunk.get("x0", 0),
                "y0": chunk.get("y0", 0),
                "x1": chunk.get("x1", 0),
                "y1": chunk.get("y1", 0),
                "type": chunk_type,
                "markdown": chunk.get("markdown", chunk.get("text", ""))
            })
    
    print(f"\n3. Found {len(legend_regions)} legend/keynote regions")
    
    # Extract indicators using LLM
    print("\n4. Extracting indicators from legends using LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    
    indicators_found = []
    for region in legend_regions:
        if region["type"] == "table":
            region_ocr = filter_ocr_by_ade_regions(google_ocr_results, [region])
            indicators = extract_indicators_from_table_llm(
                region["markdown"], fence_keywords, llm, region_ocr
            )
        else:
            indicators = extract_indicators_from_text_llm(
                region["markdown"], fence_keywords, llm
            )
        
        indicators_found.extend(indicators)
        print(f"   Region {region['type']}: {len(indicators)} indicators")
    
    indicator_codes = [ind.get("indicator", "").strip() for ind in indicators_found if ind.get("indicator")]
    print(f"\n   ✅ Total indicators extracted: {len(indicator_codes)}")
    print(f"   Sample codes: {indicator_codes[:10]}")
    
    # Check if indicator codes are in OCR
    print(f"\n5. Checking if indicator codes are present in OCR...")
    
    ocr_texts = [item.get("text", "").strip() for item in google_ocr_results]
    ocr_text_combined = " ".join(ocr_texts)
    
    found_in_ocr = []
    not_found_in_ocr = []
    
    for code in indicator_codes:
        if not code:
            continue
        
        # Check various matching patterns
        code_clean = re.sub(r'[^\dA-Za-z]', '', code)
        found = False
        
        # Exact match
        if code in ocr_text_combined or code.lower() in ocr_text_combined.lower():
            found = True
        # Clean match
        elif code_clean:
            for ocr_text in ocr_texts:
                ocr_clean = re.sub(r'[^\dA-Za-z]', '', ocr_text)
                if code_clean.lower() == ocr_clean.lower() or code_clean.lower() in ocr_clean.lower():
                    found = True
                    break
        # Word boundary match
        if not found:
            pattern = rf'\b{re.escape(code)}\b'
            if re.search(pattern, ocr_text_combined, re.IGNORECASE):
                found = True
        
        if found:
            found_in_ocr.append(code)
        else:
            not_found_in_ocr.append(code)
    
    print(f"\n   ✅ Found in OCR: {len(found_in_ocr)}/{len(indicator_codes)}")
    print(f"   ❌ Not found in OCR: {len(not_found_in_ocr)}/{len(indicator_codes)}")
    
    if found_in_ocr:
        print(f"\n   Sample codes found in OCR: {found_in_ocr[:5]}")
    
    if not_found_in_ocr:
        print(f"\n   Sample codes NOT found: {not_found_in_ocr[:5]}")
    
    # Check OCR coverage in figure regions
    print(f"\n6. Checking OCR in figure regions...")
    figure_regions = []
    for chunk in page_chunks:
        chunk_type = chunk.get("type", "").lower()
        if chunk_type in ["figure", "architectural_drawing"]:
            chunk_text = (chunk.get("markdown") or chunk.get("text", "")).lower()
            if "logo" not in chunk_text or len(chunk_text) > 100:
                figure_regions.append({
                    "x0": chunk.get("x0", 0),
                    "y0": chunk.get("y0", 0),
                    "x1": chunk.get("x1", 0),
                    "y1": chunk.get("y1", 0)
                })
    
    if figure_regions:
        figure_ocr = filter_ocr_by_ade_regions(google_ocr_results, figure_regions)
        print(f"   Found {len(figure_ocr)} OCR items in {len(figure_regions)} figure regions")
        
        # Check if any indicator codes are in figure OCR
        figure_ocr_texts = [item.get("text", "").strip() for item in figure_ocr]
        figure_ocr_combined = " ".join(figure_ocr_texts)
        
        found_in_figures = []
        for code in indicator_codes:
            if not code:
                continue
            code_clean = re.sub(r'[^\dA-Za-z]', '', code)
            
            # Check if code is in figure OCR
            for ocr_text in figure_ocr_texts:
                ocr_clean = re.sub(r'[^\dA-Za-z]', '', ocr_text)
                if (code.lower() == ocr_text.lower() or 
                    code_clean.lower() == ocr_clean.lower() or
                    re.search(rf'\b{re.escape(code)}\b', ocr_text, re.IGNORECASE)):
                    found_in_figures.append(code)
                    break
        
        print(f"   ✅ Indicator codes found in figures: {len(found_in_figures)}/{len(indicator_codes)}")
        if found_in_figures:
            print(f"   Sample: {found_in_figures[:5]}")
    
    doc.close()
    
    return {
        "indicators_extracted": len(indicator_codes),
        "found_in_ocr": len(found_in_ocr),
        "not_found_in_ocr": len(not_found_in_ocr),
        "found_in_figures": len(found_in_figures) if figure_regions else 0
    }

if __name__ == "__main__":
    import os
    pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    result = analyze_indicator_detection(pdf_path, page_idx=0)
    
    if result:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Indicators extracted: {result['indicators_extracted']}")
        print(f"Found in OCR: {result['found_in_ocr']}")
        print(f"Not found in OCR: {result['not_found_in_ocr']}")
        print(f"Found in figures: {result['found_in_figures']}")
        print(f"\n✅ Detection test complete!")



