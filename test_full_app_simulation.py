"""
Full App Simulation - Test complete flow with LLM + DocAI
Simulates exact app.py logic to diagnose server memory issue
"""
import fitz
import os
import sys
import gc
import psutil
from io import BytesIO

# Setup
sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

# Mock streamlit
class MockSecrets:
    def __init__(self):
        try:
            import toml
            secrets_path = '/Users/parhamhamouni/Desktop/leo/leo_streamlit/.streamlit/secrets.toml'
            if os.path.exists(secrets_path):
                self.data = toml.load(secrets_path)
            else:
                self.data = {}
        except:
            self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __contains__(self, key):
        return key in self.data

class MockStreamlit:
    secrets = MockSecrets()

sys.modules['streamlit'] = MockStreamlit()
import streamlit as st

from langchain_openai import ChatOpenAI
from utils import extract_comprehensive_text_from_page, analyze_page

# Configuration
PDF_PATH = "/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf"
PAGES_TO_TEST = 30
FENCE_KEYWORDS = ['fence']

def get_memory_mb():
    """Get current process RSS memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Initialize
print("Loading API keys...")
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_key:
    print("ERROR: No OpenAI API key!")
    sys.exit(1)

print("Initializing LLM...")
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key, timeout=180, max_retries=2)

print("Loading Google Cloud config...")
google_cloud_config = None
if "google_cloud" in st.secrets and "gcp_service_account" in st.secrets:
    google_cloud_config = {
        "project_number": st.secrets["google_cloud"]["project_number"],
        "location": st.secrets["google_cloud"]["location"],
        "processor_id": st.secrets["google_cloud"]["processor_id"],
        "service_account_info": dict(st.secrets["gcp_service_account"])
    }
    print("✓ Google Cloud config loaded")

print("\n" + "="*80)
print("FULL APP SIMULATION WITH AGGRESSIVE GC + CLOSE/REOPEN")
print("="*80)
print(f"PDF: {PDF_PATH}")
print(f"Testing: First {PAGES_TO_TEST} pages")
print(f"With: LLM calls + Document AI + aggressive GC")
print("="*80)

# Open PDF
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
print(f"\nTotal pages in PDF: {total_pages}")

# Track memory
memory_start = get_memory_mb()
print(f"Starting memory: {memory_start:.1f} MB\n")

for i in range(min(PAGES_TO_TEST, total_pages)):
    page_num = i + 1
    
    # Memory before page
    mem_before = get_memory_mb()
    
    # === STEP 1: AGGRESSIVE GC BEFORE PAGE ===
    for _ in range(5):
        gc.collect()
    
    # Explicit deletion of previous variables
    if i > 0:
        if 'page_obj' in locals():
            del page_obj
        if 'text_content' in locals():
            del text_content
        if 'single_page_pdf_bytes' in locals():
            del single_page_pdf_bytes
        if 'page_data' in locals():
            del page_data
        if 'analysis_result' in locals():
            del analysis_result
        gc.collect()
        gc.collect()
    
    # CRITICAL: Close and reopen PDF every 3 pages (was 5)
    if i > 0 and i % 3 == 0:
        doc.close()
        doc = fitz.open(PDF_PATH)
        print(f"   🔄 Page {page_num}: REOPENED PDF to free C memory")
        for _ in range(3):
            gc.collect()
    
    mem_after_gc = get_memory_mb()
    
    # === STEP 2: LOAD & PROCESS PAGE ===
    page_obj = doc[i]
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    is_large_page = page_width > 2000 or page_height > 2000
    
    # Extract text
    text_content = page_obj.get_text("text")
    
    # Generate page_bytes
    dpi = 30 if is_large_page else 45
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    pix_width, pix_height = pix.width, pix.height
    
    img_bytes = pix.tobytes("png")
    del pix
    gc.collect()
    gc.collect()
    
    # Wrap in PDF
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page(width=pix_width, height=pix_height)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_doc.tobytes(deflate=True, garbage=4)
    temp_doc.close()
    del img_bytes, temp_page
    gc.collect()
    gc.collect()
    
    mem_after_image = get_memory_mb()
    
    # === STEP 3: ANALYZE PAGE (LLM + DocAI) ===
    page_data = {
        "page_number": page_num,
        "text": text_content,
        "page_bytes": single_page_pdf_bytes
    }
    
    try:
        analysis_result = analyze_page(
            page_data,
            llm,
            FENCE_KEYWORDS,
            google_cloud_config,
            recall_mode="strict"
        )
    except Exception as e:
        print(f"   ERROR analyzing page: {e}")
        analysis_result = {"fence_found": False}
    
    mem_after_analysis = get_memory_mb()
    
    # === STEP 4: AGGRESSIVE GC AFTER PAGE ===
    del page_obj, text_content, single_page_pdf_bytes, page_data, analysis_result
    
    for _ in range(7):
        gc.collect()
    
    mem_after_cleanup = get_memory_mb()
    
    # Calculate
    gc_before_saved = mem_before - mem_after_gc
    image_cost = mem_after_image - mem_after_gc
    analysis_cost = mem_after_analysis - mem_after_image
    gc_after_saved = mem_after_analysis - mem_after_cleanup
    net_change = mem_after_cleanup - mem_before
    
    # Log
    large_marker = "📄 LARGE" if is_large_page else "📄 small"
    print(f"{large_marker} Page {page_num:2d}: "
          f"START={mem_before:.0f}MB → "
          f"GC={mem_after_gc:.0f}MB → "
          f"IMG={mem_after_image:.0f}MB (+{image_cost:.0f}) → "
          f"ANALYSIS={mem_after_analysis:.0f}MB (+{analysis_cost:.0f}) → "
          f"CLEANUP={mem_after_cleanup:.0f}MB ({gc_after_saved:+.0f}) | "
          f"NET={net_change:+.0f}MB")

doc.close()

# Final
memory_end = get_memory_mb()
memory_total = memory_end - memory_start

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Starting: {memory_start:.1f} MB")
print(f"Ending: {memory_end:.1f} MB")
print(f"Total increase: {memory_total:+.1f} MB")
print(f"Average per page: {memory_total/PAGES_TO_TEST:+.1f} MB")

# Projection
if PAGES_TO_TEST > 0:
    # Streamlit Cloud starts at ~730 MB (after loading PDF to temp file)
    streamlit_start = 730
    avg_per_page = memory_total / PAGES_TO_TEST
    projected = streamlit_start + (avg_per_page * 84)
    
    print(f"\n📊 PROJECTION for Streamlit Cloud (84 pages):")
    print(f"   Starting (with PDF loaded): {streamlit_start} MB")
    print(f"   Average increase per page: {avg_per_page:+.1f} MB")
    print(f"   Expected at page 24: {streamlit_start + (avg_per_page * 24):.1f} MB")
    print(f"   Expected ending (page 84): {projected:.1f} MB")
    
    if projected > 1000:
        print(f"   ⚠️  CRITICAL: Over 1000 MB limit! Need more optimization.")
    elif projected > 950:
        print(f"   ⚠️  WARNING: Close to 950 MB limit.")
    else:
        print(f"   ✅ SAFE: Under 950 MB limit")

print("="*80)

