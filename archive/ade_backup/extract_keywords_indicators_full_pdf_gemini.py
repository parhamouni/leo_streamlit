"""
extract_keywords_indicators_full_pdf.py

High-level workflow:
1.  Setup: Loads API keys from secrets.toml and loads the PDF.
2.  Global Analysis: Calls ADE *once* for the full document.
3.  Per-Page Loop:
    a. Gathers all text sources: Native PDF text (fitz), OCR text (Google Document AI).
    b. Finds "Definitions" (Green Boxes): Uses LLM (OpenAI) on ADE chunks
       to find fence keywords and indicators.
    c. Finds "Instances" (Magenta Boxes): Locates the extracted indicators
       within the ADE 'figure' (drawing) regions.
    d. Visualize: Saves a per-page image with boxes drawn.
4.  Final Output: Saves all results to a single JSON file.
"""

import os
import json
import re
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Dict, Optional
from PIL import Image
from pathlib import Path
import toml

# Import all helper functions from our utility file
import utils_ade as utils
from hybrid_page_extractor import normalize_document_ai_config, extract_page_text_layers

# --- CONFIGURATION ---

# 1. LOAD SECRETS FROM TOML FILE
SECRETS_PATH = Path("/home/ubuntu/leo_streamlit/.streamlit/secrets.toml")
if not SECRETS_PATH.exists():
    raise FileNotFoundError(f"Expected secrets.toml with API keys at {SECRETS_PATH.absolute()}")

secrets = toml.load(SECRETS_PATH)

# Load Document AI configuration
raw_doc_ai_config: Optional[Dict] = None
if "google_cloud" in secrets and "gcp_service_account" in secrets:
    raw_doc_ai_config = {
        "project_number": secrets["google_cloud"].get("project_number"),
        "location": secrets["google_cloud"].get("location"),
        "processor_id": secrets["google_cloud"].get("processor_id"),
        "service_account_info": dict(secrets["gcp_service_account"]),
    }
    doc_ai_config = normalize_document_ai_config(raw_doc_ai_config)
else:
    doc_ai_config = None
    print("⚠️ Google Cloud credentials missing; OCR results will be empty.")

# 2. SET YOUR FENCE KEYWORDS
# This list is used to prompt the LLM.
FENCE_KEYWORDS = [
    'fence', 'screen wall', 'CMU wall', 'masonry wall', 
    'wrought iron', 'chain link', 'gate'
]

# 3. SET INPUT/OUTPUT FILES
PDF_INPUT_PATH = "/home/ubuntu/leo_streamlit/subset_gold/selected_pages_no_annotations.pdf"  # <-- PUT YOUR PDF FILENAME HERE
OUTPUT_DIR = "results"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "extraction_results.json")

# --- CLIENT INITIALIZATION ---

def initialize_clients():
    """
    Initialize and return API clients by loading from secrets.toml.
    Returns (ade_key, llm_client, doc_ai_config)
    """
    # 1. ADE Key
    ade_key = secrets.get("LANDINGAI_API_KEY") or secrets.get("ADE_API_KEY")
    if not ade_key:
        raise ValueError("Missing LANDINGAI_API_KEY or ADE_API_KEY in secrets.toml")

    # 2. OpenAI LLM Client
    try:
        from langchain_openai import ChatOpenAI
        
        openai_key = secrets.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Missing OPENAI_API_KEY in secrets.toml")

        llm_client = ChatOpenAI(
            model="gpt-4-turbo", 
            temperature=0.0,
            api_key=openai_key
        )
    except ImportError:
        raise ImportError("Please install langchain-openai: pip install langchain-openai")
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {e}.")

    # 3. Document AI config is already loaded at module level
    if doc_ai_config and doc_ai_config.get("client"):
        print("✅ Document AI client initialized")
    else:
        print("⚠️ Document AI client not available")
    
    return ade_key, llm_client, doc_ai_config


def run_extraction(pdf_path: str, ade_key: str, llm_client, doc_ai_config: Optional[Dict]):
    """
    Main orchestration function.
    """
    # --- 1. SETUP ---
    print(f"Starting extraction for: {pdf_path}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    try:
        pdf_bytes = open(pdf_path, "rb").read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    # --- 2. GLOBAL ADE ANALYSIS ---
    ade_result = utils.ade_parse_document_official(pdf_bytes, ade_key)
    if not ade_result.get("success"):
        print(f"Critical Error: ADE parsing failed: {ade_result.get('error')}")
        return

    # --- 3. PER-PAGE LOOP ---
    final_results = {}
    total_pages = doc.page_count
    print(f"\n--- Starting Per-Page Processing ({total_pages} pages) ---")

    for page_idx, page in enumerate(doc):
        print(f"\nProcessing Page {page_idx + 1}/{total_pages}...")
        
        page_width, page_height = page.rect.width, page.rect.height
        final_results[page_idx] = {
            "page": page_idx,
            "width": page_width,
            "height": page_height,
            "definitions": [],
            "instances": []
        }

        # --- 3.A. GATHER TEXT SOURCES ---
        
        # Get ADE chunks for this page, with absolute coords
        ade_chunks = utils.align_ade_chunks_to_page(
            ade_result, page_idx, page_width, page_height
        )
        if not ade_chunks:
            print("   No ADE chunks found for this page. Skipping.")
            continue
        
        # Get native PDF text (words + blocks)
        pdf_words, pdf_blocks = utils.get_pdf_native_text(page)
        
        # Get OCR text (Google Document AI)
        text_layers = extract_page_text_layers(page, doc_ai_config)
        ocr_tokens = text_layers["ocr_tokens"]
        
        # Combine all text tokens for instance searching
        all_page_text_tokens = ocr_tokens + pdf_words

        # --- 3.B. FIND DEFINITIONS (Green Boxes) ---
        definitions = utils.extract_keywords_and_indicators_from_chunks(
            ade_chunks=ade_chunks,
            ocr_tokens=ocr_tokens,
            pdf_blocks=pdf_blocks,
            pdf_words=pdf_words,
            fence_keywords=FENCE_KEYWORDS,
            llm=llm_client
        )
        final_results[page_idx]["definitions"] = definitions
        
        # Create a set of all unique indicators found
        indicator_codes = {d['indicator'] for d in definitions if d.get('indicator')}
        # Normalize for matching (e.g., "(18)" -> "18")
        indicator_codes = {re.sub(r'[^0-9A-Za-z]', '', code) for code in indicator_codes}
        indicator_codes.discard('') # Remove empty strings

        if not indicator_codes:
            print("   No indicators found in definitions. Skipping instance search.")
        else:
            # --- 3.C. FIND INSTANCES (Magenta Boxes) ---
            figure_chunks = [c for c in ade_chunks if c['type'] == 'figure']
            
            instances = utils.find_indicators_in_figures(
                indicator_codes=indicator_codes,
                figure_chunks=figure_chunks,
                all_page_text=all_page_text_tokens
            )
            final_results[page_idx]["instances"] = instances

        # --- 3.D. VISUALIZE ---
        page_image = utils.convert_pdf_page_to_image(page)
        viz_path = os.path.join(OUTPUT_DIR, f"page_{page_idx}_visualization.png")
        
        utils.visualize_page_results(
            page_image=page_image,
            definitions=definitions,
            instances=final_results[page_idx]["instances"], # Use the just-found instances
            output_path=viz_path
        )
        
    doc.close()

    # --- 4. FINAL OUTPUT ---
    print(f"\n--- Extraction Complete ---")
    
    # Save final JSON
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"✅ All results saved to: {JSON_OUTPUT_PATH}")
    
    # Summary Statistics
    total_defs = sum(len(p["definitions"]) for p in final_results.values())
    total_instances = sum(len(p["instances"]) for p in final_results.values())
    print(f"\nSummary:")
    print(f"  - Total Definitions (Green): {total_defs}")
    print(f"  - Total Instances (Magenta): {total_instances}")
    print(f"  - Visualizations saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        ade_api_key, llm, doc_ai = initialize_clients()
        print("✅ All API clients initialized successfully.")
        run_extraction(PDF_INPUT_PATH, ade_api_key, llm, doc_ai)
    except Exception as e:
        print(f"\n--- A critical error occurred ---")
        print(f"{e}")
        print("\nPlease check your 'secrets.toml' file, credentials, and file paths.")