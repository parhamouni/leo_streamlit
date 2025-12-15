"""
Debug script to investigate why detection is inconsistent across runs.
Tests page 2 of the subset_gold PDF multiple times.
"""
import sys
import os
import fitz

# Try different toml libraries
try:
    import tomllib
    def load_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:
    try:
        import toml
        def load_toml(path):
            return toml.load(path)
    except ImportError:
        import tomli
        def load_toml(path):
            with open(path, "rb") as f:
                return tomli.load(f)
from io import BytesIO

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils_ade as ade

# Load secrets for Google Cloud config
secrets = {}
if os.path.exists(".streamlit/secrets.toml"):
    secrets = load_toml(".streamlit/secrets.toml")

google_cloud_config = None
try:
    if "google_cloud" in secrets and "gcp_service_account" in secrets:
        google_cloud_config = {
            "project_number": secrets["google_cloud"]["project_number"],
            "location": secrets["google_cloud"]["location"],
            "processor_id": secrets["google_cloud"]["processor_id"],
            "service_account_info": dict(secrets["gcp_service_account"])
        }
        print("✅ Google Cloud config loaded")
except Exception as e:
    print(f"⚠️ Could not load Google Cloud config: {e}")

# Load LLM
llm = None
openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if openai_key:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)
    print("✅ LLM loaded (gpt-4o-mini, temperature=0)")
else:
    print("⚠️ No OpenAI key found")

# Keywords
FENCE_KEYWORDS = [
    'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
    'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
    'adot specifications', 'mag specifications', 'rail', 'railing',
    'bollards', 'handrails', 'wall', 'cmu', 'keynote'
]

# Load PDF
PDF_PATH = "subset_gold/selected_pages_no_annotations.pdf"
with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()

doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
print(f"\n📄 PDF loaded: {PDF_PATH}")
print(f"   Total pages: {len(doc)}")

# Test page 2 (index 1)
PAGE_IDX = 1
page = doc[PAGE_IDX]
pdf_width, pdf_height = page.rect.width, page.rect.height
print(f"\n🔍 Testing Page {PAGE_IDX + 1} (index {PAGE_IDX})")
print(f"   Dimensions: {pdf_width:.1f} x {pdf_height:.1f}")

# Create single-page PDF
single_page_pdf = ade.create_single_page_pdf(pdf_bytes, PAGE_IDX)

# ============================================================
# TEST 1: PDF Native Lines (should be deterministic)
# ============================================================
print("\n" + "="*60)
print("TEST 1: PDF Native Text Extraction")
print("="*60)

for run in range(3):
    pdf_lines = ade.get_native_pdf_lines(page)
    combined_text = " ".join(line.get("text", "") for line in pdf_lines)
    print(f"Run {run+1}: {len(pdf_lines)} lines, {len(combined_text)} chars")
    if run == 0:
        first_pdf_text = combined_text
        print(f"   Sample: {combined_text[:200]}...")
    else:
        if combined_text != first_pdf_text:
            print(f"   ⚠️ TEXT DIFFERS from run 1!")
        else:
            print(f"   ✓ Same as run 1")

# ============================================================
# TEST 2: OCR (may vary)
# ============================================================
print("\n" + "="*60)
print("TEST 2: Google OCR Text Extraction")
print("="*60)

if google_cloud_config:
    for run in range(3):
        ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, pdf_width, pdf_height)
        combined_ocr = " ".join(line.get("text", "") for line in ocr_lines)
        print(f"Run {run+1}: {len(ocr_lines)} paragraphs, {len(combined_ocr)} chars")
        if run == 0:
            first_ocr_text = combined_ocr
            print(f"   Sample: {combined_ocr[:200]}...")
        else:
            if combined_ocr != first_ocr_text:
                print(f"   ⚠️ OCR TEXT DIFFERS from run 1!")
                # Show diff
                if len(combined_ocr) != len(first_ocr_text):
                    print(f"      Length diff: {len(first_ocr_text)} vs {len(combined_ocr)}")
            else:
                print(f"   ✓ Same as run 1")
else:
    print("   Skipped (no Google Cloud config)")
    ocr_lines = []

# ============================================================
# TEST 3: Keyword Scan (should be deterministic given same input)
# ============================================================
print("\n" + "="*60)
print("TEST 3: Keyword Scan")
print("="*60)

pdf_lines = ade.get_native_pdf_lines(page)
ocr_lines = []
if google_cloud_config:
    ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, pdf_width, pdf_height)

for run in range(3):
    result = ade.scan_page_for_keywords(pdf_lines, ocr_lines, FENCE_KEYWORDS)
    print(f"Run {run+1}: has_keywords={result['has_keywords']}, matched={result['matched_keywords']}")
    if run == 0:
        first_result = result
    else:
        if result['matched_keywords'] != first_result['matched_keywords']:
            print(f"   ⚠️ KEYWORDS DIFFER from run 1!")

# ============================================================
# TEST 4: LLM Classification (may vary even with temp=0)
# ============================================================
print("\n" + "="*60)
print("TEST 4: LLM Classification")
print("="*60)

if llm:
    all_lines = pdf_lines + ocr_lines
    page_text = " ".join(line.get("text", "") for line in all_lines)
    
    results = []
    for run in range(5):
        llm_result = ade.llm_classify_page(llm, page_text, FENCE_KEYWORDS)
        results.append(llm_result)
        print(f"Run {run+1}: is_fence_related={llm_result['is_fence_related']}, "
              f"confidence={llm_result['confidence']:.2f}, reason={llm_result['reason'][:50]}...")
    
    # Check consistency
    fence_votes = [r['is_fence_related'] for r in results]
    if len(set(fence_votes)) > 1:
        print(f"\n⚠️ LLM INCONSISTENT! Votes: {fence_votes}")
    else:
        print(f"\n✓ LLM consistent across {len(results)} runs")
else:
    print("   Skipped (no LLM)")

# ============================================================
# TEST 5: Full fallback_fence_detection (end-to-end)
# ============================================================
print("\n" + "="*60)
print("TEST 5: Full fallback_fence_detection")
print("="*60)

if llm:
    results = []
    for run in range(5):
        # Re-extract text each time to simulate real conditions
        pdf_lines = ade.get_native_pdf_lines(page)
        ocr_lines = []
        if google_cloud_config:
            ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, pdf_width, pdf_height)
        
        result = ade.fallback_fence_detection(
            pdf_lines=pdf_lines,
            ocr_lines=ocr_lines,
            fence_keywords=FENCE_KEYWORDS,
            llm=llm,
            use_llm_confirmation=True
        )
        results.append(result)
        print(f"Run {run+1}: fence_found={result['fence_found']}, "
              f"method={result['method']}, keywords={result['matched_keywords']}")
    
    # Check consistency
    fence_votes = [r['fence_found'] for r in results]
    if len(set(fence_votes)) > 1:
        print(f"\n⚠️ DETECTION INCONSISTENT! Votes: {fence_votes}")
    else:
        print(f"\n✓ Detection consistent across {len(results)} runs")
else:
    print("   Skipped (no LLM)")

doc.close()
print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
