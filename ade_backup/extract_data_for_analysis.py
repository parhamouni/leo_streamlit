#!/usr/bin/env python3
"""
Data extraction script for ADE integration analysis.

Extracts and stores:
1. Full document ADE output
2. Per-page OCR results
3. PDF text layer data
4. Page dimensions

Stores all data in data_analysis/ directory for later analysis.
"""
import os
import json
import sys
from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF
from io import BytesIO

# Import utilities
from utils_ade_official import (
    ade_parse_document_official,
    get_google_ocr_results_with_boxes,
    get_page_dimensions,
    create_single_page_pdf
)
from utils import create_document_ai_client

def load_config():
    """Load configuration from secrets."""
    try:
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            
            # ADE API key
            ade_api_key = secrets.get("LANDINGAI_API_KEY") or os.getenv("LANDINGAI_API_KEY")
            
            # Google Cloud config
            google_cloud_config = None
            if "google_cloud" in secrets and "gcp_service_account" in secrets:
                google_cloud_config_raw = {
                    "project_number": secrets["google_cloud"]["project_number"],
                    "location": secrets["google_cloud"]["location"],
                    "processor_id": secrets["google_cloud"]["processor_id"],
                    "service_account_info": dict(secrets["gcp_service_account"])
                }
                
                # Build proper Google Cloud config with client and processor_name
                client = create_document_ai_client(google_cloud_config_raw)
                if client:
                    processor_name = (
                        f"projects/{google_cloud_config_raw['project_number']}/"
                        f"locations/{google_cloud_config_raw['location']}/"
                        f"processors/{google_cloud_config_raw['processor_id']}"
                    )
                    
                    google_cloud_config = {
                        **google_cloud_config_raw,
                        "client": client,
                        "processor_name": processor_name
                    }
            
            return ade_api_key, google_cloud_config
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return None, None

def extract_ade_output(pdf_path: Path, ade_api_key: str) -> Dict:
    """Extract full document ADE output."""
    print(f"📄 Extracting ADE output from {pdf_path}...")
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    print(f"   PDF size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
    
    # Parse full document with ADE
    ade_result = ade_parse_document_official(pdf_bytes, ade_api_key, zdr=False)
    
    if not ade_result.get("success"):
        print(f"   ⚠️ ADE parsing failed: {ade_result.get('error')}")
        return None
    
    data = ade_result.get("data", {})
    total_pages = data.get("total_pages", len(data.get("pages", [])))
    print(f"   ✅ ADE parsed {total_pages} pages")
    
    # Extract chunks per page for easier analysis
    chunks_per_page = {}
    pages = data.get("pages", [])
    
    for page_idx, page_data in enumerate(pages):
        page_num = page_idx + 1
        chunks = page_data.get("chunks", []) or page_data.get("elements", []) or []
        chunks_per_page[page_num] = {
            "chunk_count": len(chunks),
            "chunks": chunks,
            "page_text": page_data.get("text", ""),
            "page_metadata": {k: v for k, v in page_data.items() if k not in ["chunks", "elements", "text"]}
        }
    
    return {
        "success": True,
        "total_pages": total_pages,
        "raw_response": data,
        "chunks_per_page": chunks_per_page,
        "metadata": {
            "pdf_path": str(pdf_path),
            "pdf_size_mb": len(pdf_bytes) / 1024 / 1024
        }
    }

def extract_ocr_per_page(pdf_path: Path, google_cloud_config: Dict) -> Dict[int, List[Dict]]:
    """Extract OCR results for each page."""
    print(f"🔍 Extracting OCR results per page from {pdf_path}...")
    
    if not google_cloud_config:
        print("   ⚠️ Google Cloud config not available, skipping OCR extraction")
        return {}
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    total_pages = len(doc)
    
    ocr_per_page = {}
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        print(f"   Processing page {page_num}/{total_pages}...")
        
        # Create single-page PDF
        single_page_bytes = create_single_page_pdf(pdf_bytes, page_idx)
        
        if not single_page_bytes:
            print(f"      ⚠️ Could not create single-page PDF for page {page_num}")
            continue
        
        # Get OCR results
        ocr_results = get_google_ocr_results_with_boxes(
            single_page_bytes,
            google_cloud_config,
            page_num
        )
        
        ocr_per_page[page_num] = ocr_results
        print(f"      ✅ Found {len(ocr_results)} OCR text elements")
    
    doc.close()
    
    return ocr_per_page

def extract_pdf_text_layer(pdf_path: Path) -> Dict[int, List[Dict]]:
    """Extract PDF text layer words for each page."""
    print(f"📝 Extracting PDF text layer from {pdf_path}...")
    
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    
    text_layer_per_page = {}
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        
        print(f"   Processing page {page_num}/{total_pages}...")
        
        # Extract words with coordinates
        words = page.get_text("words")
        
        words_data = []
        for word_tuple in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
            words_data.append({
                "text": text,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "block_no": block_no,
                "line_no": line_no,
                "word_no": word_no
            })
        
        # Also get text blocks for context
        blocks = page.get_text("blocks")
        blocks_data = []
        for block in blocks:
            if block[0] == 0:  # Text block (not image)
                x0, y0, x1, y1, text, block_no, block_type = block
                blocks_data.append({
                    "text": text,
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "block_no": block_no
                })
        
        text_layer_per_page[page_num] = {
            "words": words_data,
            "blocks": blocks_data,
            "full_text": page.get_text("text"),
            "word_count": len(words_data),
            "block_count": len(blocks_data)
        }
        
        print(f"      ✅ Extracted {len(words_data)} words, {len(blocks_data)} text blocks")
    
    doc.close()
    
    return text_layer_per_page

def extract_page_dimensions(pdf_path: Path) -> Dict[int, Dict]:
    """Extract page dimensions for each page."""
    print(f"📏 Extracting page dimensions from {pdf_path}...")
    
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    
    dimensions_per_page = {}
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        
        dimensions_per_page[page_num] = {
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "rotation": page.rotation
        }
    
    doc.close()
    
    print(f"   ✅ Extracted dimensions for {total_pages} pages")
    
    return dimensions_per_page

def save_to_json(data: Dict, filepath: Path):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"💾 Saved to {filepath}")

def main():
    """Main extraction function."""
    # Setup paths
    pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
    output_dir = Path("data_analysis")
    output_dir.mkdir(exist_ok=True)
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DATA EXTRACTION FOR ADE INTEGRATION ANALYSIS")
    print("=" * 60)
    print()
    
    # Load configuration
    print("🔧 Loading configuration...")
    ade_api_key, google_cloud_config = load_config()
    
    if not ade_api_key:
        print("❌ ADE API key not found")
        sys.exit(1)
    
    if not google_cloud_config:
        print("⚠️ Google Cloud config not found - OCR extraction will be skipped")
    
    print()
    
    # Extract ADE output
    print("Phase 1: Extracting ADE output...")
    ade_output = extract_ade_output(pdf_path, ade_api_key)
    if ade_output:
        save_to_json(ade_output, output_dir / "ade_full_document.json")
    print()
    
    # Extract OCR per page
    print("Phase 2: Extracting OCR results per page...")
    ocr_per_page = extract_ocr_per_page(pdf_path, google_cloud_config)
    if ocr_per_page:
        save_to_json(ocr_per_page, output_dir / "ocr_per_page.json")
    print()
    
    # Extract PDF text layer
    print("Phase 3: Extracting PDF text layer...")
    text_layer_per_page = extract_pdf_text_layer(pdf_path)
    if text_layer_per_page:
        save_to_json(text_layer_per_page, output_dir / "pdf_text_layer.json")
    print()
    
    # Extract page dimensions
    print("Phase 4: Extracting page dimensions...")
    dimensions_per_page = extract_page_dimensions(pdf_path)
    if dimensions_per_page:
        save_to_json(dimensions_per_page, output_dir / "page_dimensions.json")
    print()
    
    # Summary
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"✅ ADE output: {'✅' if ade_output else '❌'}")
    print(f"✅ OCR per page: {len(ocr_per_page)} pages")
    print(f"✅ PDF text layer: {len(text_layer_per_page)} pages")
    print(f"✅ Page dimensions: {len(dimensions_per_page)} pages")
    print()
    print(f"📁 Data saved to: {output_dir.absolute()}")
    print()
    print("Next step: Run analyze_extracted_data.py to analyze the extracted data")

if __name__ == "__main__":
    main()

