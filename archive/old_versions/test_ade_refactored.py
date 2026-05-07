#!/usr/bin/env python3
"""
Test script for the refactored ADE utilities.
Tests core functionality without requiring Streamlit.
"""
import os
import sys
from pathlib import Path
import toml

# Load configuration
def load_config():
    """Load API keys and config from secrets.toml"""
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        print("❌ secrets.toml not found. Please create it with your API keys.")
        return None
    
    secrets = toml.load(secrets_path)
    
    config = {
        "openai_key": secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "ade_key": secrets.get("LANDINGAI_API_KEY") or os.getenv("LANDINGAI_API_KEY"),
    }
    
    # Google Cloud config (optional)
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

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    try:
        import utils_ade
        print("✅ utils_ade imported successfully")
        
        # Check key functions exist
        required_funcs = [
            "ade_parse_document",
            "align_ade_chunks_to_page",
            "segment_chunks",
            "run_google_ocr",
            "get_pdf_text_tokens",
            "extract_legend_entries",
            "find_instances_in_figures"
        ]
        
        for func_name in required_funcs:
            if hasattr(utils_ade, func_name):
                print(f"  ✅ {func_name} found")
            else:
                print(f"  ❌ {func_name} NOT found")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ade_parsing(config):
    """Test ADE document parsing"""
    print("\n" + "=" * 60)
    print("TEST 2: ADE Document Parsing")
    print("=" * 60)
    
    if not config["ade_key"]:
        print("⚠️  Skipping: No ADE API key found")
        return None
    
    # Find a test PDF
    test_pdf_paths = [
        Path("subset_gold/selected_pages_no_annotations.pdf"),
        Path("test.pdf"),
    ]
    
    pdf_path = None
    for path in test_pdf_paths:
        if path.exists():
            pdf_path = path
            break
    
    if not pdf_path:
        print("⚠️  Skipping: No test PDF found")
        return None
    
    print(f"📄 Testing with: {pdf_path}")
    
    try:
        import utils_ade as ade
        
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        print(f"   PDF size: {len(pdf_bytes) / 1024:.1f} KB")
        print("   Calling ADE API...")
        
        result = ade.ade_parse_document(pdf_bytes, config["ade_key"], zdr=False)
        
        if result["success"]:
            data = result["data"]
            total_pages = data["total_pages"]
            chunks = data["chunks"]
            print(f"   ✅ ADE parsing successful!")
            print(f"   📊 Total pages: {total_pages}")
            print(f"   📊 Total chunks: {len(chunks)}")
            
            # Show chunk type distribution
            chunk_types = {}
            for chunk in chunks[:20]:  # Sample first 20
                ctype = chunk.get("type", "unknown")
                chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
            
            print(f"   📊 Chunk types (sample): {chunk_types}")
            return result
        else:
            print(f"   ❌ ADE parsing failed: {result.get('error')}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_page_processing(config, ade_result):
    """Test processing a single page"""
    print("\n" + "=" * 60)
    print("TEST 3: Page Processing (Chunk Alignment)")
    print("=" * 60)
    
    if not ade_result:
        print("⚠️  Skipping: No ADE result available")
        return None
    
    try:
        import utils_ade as ade
        import fitz
        
        # Get first page
        pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
        if not pdf_path.exists():
            print("⚠️  Skipping: Test PDF not found")
            return None
        
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            print("⚠️  Skipping: PDF has no pages")
            return None
        
        page = doc[0]
        page_width = page.rect.width
        page_height = page.rect.height
        
        print(f"📄 Processing page 0")
        print(f"   Page dimensions: {page_width:.1f} x {page_height:.1f} points")
        
        # Align chunks
        chunks = ade.align_ade_chunks_to_page(ade_result, 0, page_width, page_height)
        print(f"   ✅ Found {len(chunks)} chunks for this page")
        
        if chunks:
            print(f"   Sample chunk types: {[c['type'] for c in chunks[:5]]}")
            print(f"   Sample chunk bbox: {chunks[0].get('bbox')}")
        
        # Test PDF text extraction
        native_tokens = ade.get_pdf_text_tokens(page)
        print(f"   ✅ Extracted {len(native_tokens)} native text tokens")
        
        doc.close()
        return chunks
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_llm_extraction(config, chunks):
    """Test LLM-based keyword extraction"""
    print("\n" + "=" * 60)
    print("TEST 4: LLM Keyword Extraction")
    print("=" * 60)
    
    if not config["openai_key"]:
        print("⚠️  Skipping: No OpenAI API key found")
        return None
    
    if not chunks:
        print("⚠️  Skipping: No chunks available")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        import utils_ade as ade
        
        llm = ChatOpenAI(api_key=config["openai_key"], model="gpt-4o-mini", temperature=0)
        print("   ✅ LLM client initialized")
        
        # Get native tokens for the page
        import fitz
        pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        native_tokens = ade.get_pdf_text_tokens(page)
        doc.close()
        
        fence_keywords = ["fence", "gate", "barrier", "guardrail", "cmu", "wall"]
        print(f"   🔍 Extracting keywords with: {fence_keywords[:3]}...")
        
        # Segment chunks first
        legend_chunks, figure_chunks = ade.segment_chunks(chunks)
        print(f"   📊 Segments: {len(legend_chunks)} legend-like, {len(figure_chunks)} figure-like")
        
        # Extract (this will call the LLM)
        definitions = ade.extract_legend_entries(
            legend_chunks, native_tokens, fence_keywords, llm
        )
        
        print(f"   ✅ Extracted {len(definitions)} definitions")
        if definitions:
            print(f"   Sample: {definitions[0]}")
        
        return definitions
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_google_ocr(config):
    """Test Google OCR (if available)"""
    print("\n" + "=" * 60)
    print("TEST 5: Google OCR (Optional)")
    print("=" * 60)
    
    if not config.get("google_cloud"):
        print("⚠️  Skipping: Google Cloud config not available")
        return None
    
    try:
        import utils_ade as ade
        import fitz
        
        pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
        if not pdf_path.exists():
            print("⚠️  Skipping: Test PDF not found")
            return None
        
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        
        # Create single page PDF
        single_page_bytes = ade.create_single_page_pdf(
            pdf_path.read_bytes(), 0
        )
        
        print("   🔍 Running Google OCR...")
        ocr_tokens = ade.run_google_ocr(
            single_page_bytes, config["google_cloud"], w, h
        )
        
        print(f"   ✅ Found {len(ocr_tokens)} OCR tokens")
        if ocr_tokens:
            print(f"   Sample: '{ocr_tokens[0]['text']}' at ({ocr_tokens[0]['x0']:.1f}, {ocr_tokens[0]['y0']:.1f})")
        
        doc.close()
        return ocr_tokens
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ADE REFACTORED MODULE - TEST SUITE")
    print("=" * 60)
    print()
    
    # Load config
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please check dependencies.")
        sys.exit(1)
    
    # Test 2: ADE Parsing
    ade_result = test_ade_parsing(config)
    
    # Test 3: Page Processing
    chunks = test_page_processing(config, ade_result)
    
    # Test 4: LLM Extraction
    definitions = test_llm_extraction(config, chunks)
    
    # Test 5: Google OCR (optional)
    ocr_tokens = test_google_ocr(config)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Imports: PASSED")
    print(f"{'✅' if ade_result else '⚠️ '} ADE Parsing: {'PASSED' if ade_result else 'SKIPPED/FAILED'}")
    print(f"{'✅' if chunks else '⚠️ '} Page Processing: {'PASSED' if chunks else 'SKIPPED/FAILED'}")
    print(f"{'✅' if definitions is not None else '⚠️ '} LLM Extraction: {'PASSED' if definitions is not None else 'SKIPPED/FAILED'}")
    print(f"{'✅' if ocr_tokens is not None else '⚠️ '} Google OCR: {'PASSED' if ocr_tokens is not None else 'SKIPPED'}")
    print()
    
    if ade_result and chunks:
        print("🎉 Core functionality is working!")
        print("\nNext steps:")
        print("  1. Run: streamlit run app_ade.py")
        print("  2. Upload a PDF and click 'Run Analysis'")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

