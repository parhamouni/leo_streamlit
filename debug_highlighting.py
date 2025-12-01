#!/usr/bin/env python3
"""
Debug script to diagnose highlighting issues.
"""
import os
import sys
from pathlib import Path
try:
    import toml
except ImportError:
    import tomli as toml
import fitz

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import utils_ade as ade
from langchain_openai import ChatOpenAI

def load_config():
    """Load API keys and config from secrets.toml"""
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        print("❌ secrets.toml not found.")
        return None
    
    secrets = toml.load(secrets_path)
    
    config = {
        "openai_key": secrets.get("OPENAI_API_KEY"),
        "ade_key": secrets.get("LANDINGAI_API_KEY"),
    }
    
    # Google Cloud config
    if "google_cloud" in secrets and "gcp_service_account" in secrets:
        config["google_cloud"] = {
            "project_number": secrets["google_cloud"]["project_number"],
            "location": secrets["google_cloud"]["location"],
            "processor_id": secrets["google_cloud"]["processor_id"],
            "service_account_info": dict(secrets["gcp_service_account"])
        }
    else:
        config["google_cloud"] = None
    
    return config

def main():
    print("=" * 80)
    print("DEBUGGING HIGHLIGHTING ISSUE")
    print("=" * 80)
    
    config = load_config()
    if not config:
        return
    
    # Test PDF
    pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
    if not pdf_path.exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        return
    
    print(f"\n📄 Testing with: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Initialize LLM
    llm = ChatOpenAI(api_key=config["openai_key"], model="gpt-4o-mini", temperature=0)
    
    # Parse with ADE
    print("\n1. Calling ADE API...")
    ade_result = ade.ade_parse_document(pdf_bytes, config["ade_key"])
    
    if not ade_result["success"]:
        print(f"❌ ADE failed: {ade_result['error']}")
        return
    
    print(f"✅ ADE returned {len(ade_result['data']['chunks'])} chunks")
    
    # Process first page
    page_idx = 0
    print(f"\n2. Processing page {page_idx + 1}...")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    w, h = page.rect.width, page.rect.height
    print(f"   Page dimensions: {w:.1f} x {h:.1f} points")
    
    # Align chunks
    chunks = ade.align_ade_chunks_to_page(ade_result, page_idx, w, h)
    print(f"   Found {len(chunks)} chunks on this page")
    
    # Segment
    legend_chunks, figure_chunks = ade.segment_chunks(chunks)
    print(f"   Segmented: {len(legend_chunks)} legend, {len(figure_chunks)} figure")
    
    # Get tokens
    native_tokens = ade.get_pdf_text_tokens(page)
    print(f"   Native PDF tokens: {len(native_tokens)}")
    
    ocr_tokens = []
    if config["google_cloud"]:
        single_page_pdf = ade.create_single_page_pdf(pdf_bytes, page_idx)
        ocr_tokens = ade.run_google_ocr(single_page_pdf, config["google_cloud"], w, h)
        print(f"   Google OCR tokens: {len(ocr_tokens)}")
    
    all_tokens = native_tokens + ocr_tokens
    print(f"   Total tokens: {len(all_tokens)}")
    
    # Show sample tokens
    print(f"\n3. Sample tokens (first 10):")
    for i, t in enumerate(all_tokens[:10]):
        print(f"   [{i}] '{t['text']}' @ ({t['x0']:.1f}, {t['y0']:.1f}) - ({t['x1']:.1f}, {t['y1']:.1f}) [{t['source']}]")
    
    # Show legend chunks
    print(f"\n4. Legend chunks:")
    for i, chunk in enumerate(legend_chunks[:3]):
        print(f"   [{i}] Type: {chunk['type']}")
        print(f"       BBox: ({chunk['x0']:.1f}, {chunk['y0']:.1f}) - ({chunk['x1']:.1f}, {chunk['y1']:.1f})")
        print(f"       Text preview: {chunk['text'][:100]}...")
    
    # Extract definitions
    print(f"\n5. Extracting legend entries with LLM...")
    fence_keywords = ["fence", "gate", "barrier", "guardrail", "cmu", "wall"]
    definitions = ade.extract_legend_entries(legend_chunks, all_tokens, fence_keywords, llm)
    
    print(f"   ✅ Extracted {len(definitions)} definitions")
    for i, d in enumerate(definitions):
        print(f"   [{i}] Indicator: '{d['indicator']}', Keyword: '{d['keyword']}'")
        print(f"       BBox: ({d['x0']:.1f}, {d['y0']:.1f}) - ({d['x1']:.1f}, {d['y1']:.1f})")
    
    # Find instances
    print(f"\n6. Finding instances in figures...")
    instances = ade.find_instances_in_figures(definitions, figure_chunks, all_tokens)
    
    print(f"   ✅ Found {len(instances)} instances")
    for i, inst in enumerate(instances):
        print(f"   [{i}] Indicator: '{inst['indicator']}'")
        print(f"       BBox: ({inst['x0']:.1f}, {inst['y0']:.1f}) - ({inst['x1']:.1f}, {inst['y1']:.1f})")
    
    # Test phrase matching directly
    print(f"\n7. Testing phrase matching directly...")
    test_phrases = ["fence", "keynote", "cmu"]
    for phrase in test_phrases:
        matches = ade.find_phrase_matches(all_tokens, phrase, None)
        print(f"   Phrase '{phrase}': {len(matches)} matches")
        if matches:
            print(f"      First match: ({matches[0]['x0']:.1f}, {matches[0]['y0']:.1f}) - ({matches[0]['x1']:.1f}, {matches[0]['y1']:.1f})")
    
    doc.close()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Legend chunks: {len(legend_chunks)}")
    print(f"Figure chunks: {len(figure_chunks)}")
    print(f"Definitions extracted: {len(definitions)}")
    print(f"Instances found: {len(instances)}")
    
    if len(definitions) == 0:
        print("\n⚠️ WARNING: No definitions extracted!")
        print("   Possible issues:")
        print("   1. LLM not finding fence-related items in legend chunks")
        print("   2. Phrase matching failing to find the extracted text in tokens")
        print("   3. Tokens not within chunk bounding boxes")
    
    if len(instances) == 0 and len(definitions) > 0:
        print("\n⚠️ WARNING: Definitions found but no instances!")
        print("   Possible issues:")
        print("   1. Indicator codes not appearing in figure regions")
        print("   2. Token filtering excluding figure tokens")

if __name__ == "__main__":
    main()

