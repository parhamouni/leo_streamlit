#!/usr/bin/env python3
"""
Debug script to diagnose highlighting issues - WITHOUT ADE.
Focus on OCR token extraction and matching.
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
    print("DEBUGGING HIGHLIGHTING ISSUE (WITHOUT ADE)")
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
    
    # Process first page
    page_idx = 0
    print(f"\n1. Processing page {page_idx + 1}...")
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    w, h = page.rect.width, page.rect.height
    print(f"   Page dimensions: {w:.1f} x {h:.1f} points")
    
    # Get tokens
    native_tokens = ade.get_pdf_text_tokens(page)
    print(f"   Native PDF tokens: {len(native_tokens)}")
    
    ocr_tokens = []
    if config["google_cloud"]:
        print(f"   Running Google OCR...")
        single_page_pdf = ade.create_single_page_pdf(pdf_bytes, page_idx)
        ocr_tokens = ade.run_google_ocr(single_page_pdf, config["google_cloud"], w, h)
        print(f"   Google OCR tokens: {len(ocr_tokens)}")
    
    all_tokens = native_tokens + ocr_tokens
    print(f"   Total tokens: {len(all_tokens)}")
    
    # Show sample tokens
    print(f"\n2. Sample tokens (first 20):")
    for i, t in enumerate(all_tokens[:20]):
        print(f"   [{i:2d}] '{t['text']:20s}' @ ({t['x0']:6.1f}, {t['y0']:6.1f}) - ({t['x1']:6.1f}, {t['y1']:6.1f}) [{t['source']:6s}]")
    
    # Test phrase matching directly
    print(f"\n3. Testing phrase matching with sample phrases...")
    test_phrases = [
        "fence",
        "keynote", 
        "cmu",
        "chain link",
        "existing fence",
        "3301",
        "0113"
    ]
    
    for phrase in test_phrases:
        matches = ade.find_phrase_matches(all_tokens, phrase, None)
        print(f"   Phrase '{phrase:20s}': {len(matches):2d} matches", end="")
        if matches:
            m = matches[0]
            print(f" | First: ({m['x0']:6.1f}, {m['y0']:6.1f}) - ({m['x1']:6.1f}, {m['y1']:6.1f})")
        else:
            print()
    
    # Test with a mock legend chunk
    print(f"\n4. Testing with mock legend chunk...")
    
    # Find tokens that contain "keynote" to identify legend region
    keynote_tokens = [t for t in all_tokens if "keynote" in t["text"].lower()]
    
    if keynote_tokens:
        print(f"   Found {len(keynote_tokens)} tokens with 'keynote'")
        
        # Create a mock chunk around the keynote area
        kx0 = min(t["x0"] for t in keynote_tokens)
        ky0 = min(t["y0"] for t in keynote_tokens)
        kx1 = max(t["x1"] for t in keynote_tokens)
        ky1 = max(t["y1"] for t in keynote_tokens)
        
        # Expand to capture nearby content (legend table)
        mock_chunk = {
            "x0": kx0 - 50,
            "y0": ky0 - 50,
            "x1": kx1 + 300,  # Expand right to capture table
            "y1": ky1 + 400,  # Expand down to capture rows
            "text": "KEYNOTES (mock)",
            "type": "table"
        }
        
        print(f"   Mock chunk bbox: ({mock_chunk['x0']:.1f}, {mock_chunk['y0']:.1f}) - ({mock_chunk['x1']:.1f}, {mock_chunk['y1']:.1f})")
        
        # Filter tokens in this region
        chunk_tokens = [
            t for t in all_tokens
            if (mock_chunk["x0"] <= (t["x0"] + t["x1"])/2 <= mock_chunk["x1"] and
                mock_chunk["y0"] <= (t["y0"] + t["y1"])/2 <= mock_chunk["y1"])
        ]
        
        print(f"   Tokens in mock chunk: {len(chunk_tokens)}")
        
        # Show some tokens
        print(f"   Sample tokens from chunk:")
        for i, t in enumerate(chunk_tokens[:15]):
            print(f"      [{i:2d}] '{t['text']:20s}'")
        
        # Test matching within chunk
        print(f"\n5. Testing phrase matching WITHIN mock chunk...")
        for phrase in ["fence", "cmu", "3301", "chain link"]:
            matches = ade.find_phrase_matches(all_tokens, phrase, mock_chunk)
            print(f"   Phrase '{phrase:20s}': {len(matches):2d} matches in chunk")
    else:
        print(f"   ⚠️ No 'keynote' tokens found - cannot create mock chunk")
    
    doc.close()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if len(all_tokens) == 0:
        print("❌ CRITICAL: No tokens extracted!")
        print("   Check OCR and PDF text extraction")
    elif len(all_tokens) < 100:
        print("⚠️  WARNING: Very few tokens extracted")
        print(f"   Only {len(all_tokens)} tokens found")
    else:
        print(f"✅ Token extraction OK: {len(all_tokens)} tokens")
    
    # Test canonical matching
    print(f"\nTesting canonical text matching:")
    test_canon = [
        ("fence", "FENCE"),
        ("chain link", "CHAIN-LINK"),
        ("chain link", "CHAIN_LINK"),
        ("cmu wall", "CMU WALL"),
    ]
    
    for target, test in test_canon:
        canon_target = ade._canon(target)
        canon_test = ade._canon(test)
        match = canon_target == canon_test
        print(f"   '{target}' vs '{test}': canon('{canon_target}') == canon('{canon_test}') = {match}")

if __name__ == "__main__":
    main()



