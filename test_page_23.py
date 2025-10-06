#!/usr/bin/env python3
"""
Test specifically what happens at page 23 with DPI=35
Focus on pages 20-25 to see the memory behavior
"""
import sys
import os
import gc
import tempfile
import json
from pathlib import Path

sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

try:
    import psutil
    import resource
    def _rss_mb():
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except:
    def _rss_mb():
        return 0.0

import fitz
from utils import analyze_page, get_fence_related_text_boxes
import toml

# Load secrets
secrets_path = Path('/Users/parhamhamouni/Desktop/leo/leo_streamlit/.streamlit/secrets.toml')
if secrets_path.exists():
    secrets = toml.load(secrets_path)
    google_cloud_config = {
        "project_id": secrets.get("GCP_PROJECT_ID"),
        "processor_id": secrets.get("DOCUMENT_AI_PROCESSOR_ID"),
        "location": secrets.get("DOCUMENT_AI_LOCATION", "us"),
        "service_account_info": secrets.get("gcp_service_account"),
    }
else:
    google_cloud_config = None

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'

FENCE_KEYWORDS = [
    "fence", "fencing", "chain link", "chainlink",
    "temp fence", "temporary fence", "site fence",
    "barrier", "barricade", "hoarding"
]

print("="*80)
print("PAGE 23 SPECIFIC TEST - With DPI=35")
print("="*80)

# Simulate Cloud starting state
print(f"Starting memory: {_rss_mb():.1f}MB")

# Load PDF
with open(PDF_PATH, 'rb') as f:
    pdf_bytes = f.read()
print(f"After loading PDF: {_rss_mb():.1f}MB")

# Write to temp file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
temp_file.write(pdf_bytes)
temp_file.close()
temp_pdf_path = temp_file.name

# Free PDF bytes
pdf_bytes = None
gc.collect()
gc.collect()
print(f"After freeing PDF bytes: {_rss_mb():.1f}MB")

# Open from file
doc = fitz.open(temp_pdf_path)
print(f"Opened PDF: {_rss_mb():.1f}MB")
print(f"Total pages: {len(doc)}\n")

print("="*80)
print("Processing Pages 18-25 (Focus on page 23)")
print("="*80)

START_PAGE = 18
END_PAGE = 26

for i in range(START_PAGE - 1, min(END_PAGE, len(doc))):
    page_num = i + 1
    
    print(f"\n{'='*80}")
    print(f"📄 PAGE {page_num}")
    print(f"{'='*80}")
    
    mem_start = _rss_mb()
    print(f"Start: {mem_start:.1f}MB")
    
    # GC
    gc.collect()
    mem_after_gc = _rss_mb()
    print(f"  After GC: {mem_after_gc:.1f}MB ({mem_after_gc-mem_start:+.1f}MB)")
    
    # Load page
    page_obj = doc.load_page(i)
    mem_after_load = _rss_mb()
    print(f"  After load_page: {mem_after_load:.1f}MB ({mem_after_load-mem_after_gc:+.1f}MB)")
    
    # Extract text
    text_content = page_obj.get_text("text")
    mem_after_text = _rss_mb()
    print(f"  After get_text: {mem_after_text:.1f}MB ({mem_after_text-mem_after_load:+.1f}MB) len={len(text_content)}")
    
    # Check page size
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    if page_width > 2000 or page_height > 2000:
        dpi = 35
        print(f"  → Large page: {page_width:.0f}×{page_height:.0f}, using DPI=35")
    else:
        dpi = 45
        print(f"  → Normal page: {page_width:.0f}×{page_height:.0f}, using DPI=45")
    
    # Create pixmap
    mem_before_pix = _rss_mb()
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    mem_after_pix = _rss_mb()
    print(f"  After get_pixmap: {mem_after_pix:.1f}MB ({mem_after_pix-mem_before_pix:+.1f}MB) size={pix.width}x{pix.height}")
    
    # Convert to PNG
    mem_before_tobytes = _rss_mb()
    img_bytes = pix.tobytes("png")
    img_size_mb = len(img_bytes) / (1024*1024)
    mem_after_tobytes = _rss_mb()
    print(f"  After tobytes: {mem_after_tobytes:.1f}MB ({mem_after_tobytes-mem_before_tobytes:+.1f}MB) PNG={img_size_mb:.2f}MB")
    
    # Free pixmap
    del pix
    gc.collect()
    gc.collect()
    mem_after_free_pix = _rss_mb()
    print(f"  After free pixmap (2×GC): {mem_after_free_pix:.1f}MB ({mem_after_free_pix-mem_after_tobytes:+.1f}MB)")
    
    # Create PDF wrapper
    mem_before_wrapper = _rss_mb()
    temp_img_doc = fitz.open()
    temp_page = temp_img_doc.new_page(width=pix.width if 'pix' in dir() else 1000, height=pix.height if 'pix' in dir() else 1000)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
    pdf_size_mb = len(single_page_pdf_bytes) / (1024*1024)
    temp_img_doc.close()
    mem_after_wrapper = _rss_mb()
    print(f"  After PDF wrapper: {mem_after_wrapper:.1f}MB ({mem_after_wrapper-mem_before_wrapper:+.1f}MB) PDF={pdf_size_mb:.2f}MB")
    
    # Free img_bytes
    del img_bytes
    gc.collect()
    gc.collect()
    mem_after_free_img = _rss_mb()
    print(f"  After free img_bytes (2×GC): {mem_after_free_img:.1f}MB ({mem_after_free_img-mem_after_wrapper:+.1f}MB)")
    
    # Analyze page (if Document AI available)
    if google_cloud_config:
        page_data = {
            "page_number": page_num,
            "text": text_content,
            "page_bytes": single_page_pdf_bytes
        }
        
        try:
            mem_before_analysis = _rss_mb()
            analysis_result = analyze_page(
                page_data=page_data,
                llm_text=None,
                fence_keywords=FENCE_KEYWORDS,
                google_cloud_config=google_cloud_config
            )
            mem_after_analysis = _rss_mb()
            print(f"  After analyze_page: {mem_after_analysis:.1f}MB ({mem_after_analysis-mem_before_analysis:+.1f}MB)")
            
            # Extract signals
            try:
                jr = json.loads(analysis_result.get("text_response", "{}"))
                signals = jr.get("signals", [])
            except:
                signals = []
            
            mem_after_signals = _rss_mb()
            print(f"  After extract signals: {mem_after_signals:.1f}MB ({mem_after_signals-mem_after_analysis:+.1f}MB) count={len(signals)}")
            
        except Exception as e:
            print(f"  ❌ Analysis error: {e}")
            mem_after_analysis = mem_after_free_img
            mem_after_signals = mem_after_free_img
            analysis_result = {"fence_found": False}
            signals = []
    else:
        print(f"  ⚠️ Skipping Document AI (no config)")
        mem_after_analysis = mem_after_free_img
        mem_after_signals = mem_after_free_img
    
    # Final cleanup
    del page_obj, text_content, single_page_pdf_bytes
    if 'analysis_result' in locals():
        del analysis_result
    if 'signals' in locals():
        del signals
    gc.collect()
    gc.collect()
    gc.collect()
    
    mem_end = _rss_mb()
    print(f"\n  Final: {mem_end:.1f}MB")
    print(f"  NET CHANGE: {mem_end - mem_start:+.1f}MB")
    
    # Check if would exceed Cloud limit
    if mem_end > 1050:
        print(f"\n  ⚠️ WOULD EXCEED 1050MB LIMIT ON CLOUD!")
    
    if page_num == 23:
        print(f"\n{'='*80}")
        print(f"🎯 PAGE 23 ANALYSIS")
        print(f"{'='*80}")
        print(f"  Start:   {mem_start:.1f}MB")
        print(f"  Peak:    {max(mem_after_pix, mem_after_wrapper, mem_after_analysis):.1f}MB")
        print(f"  End:     {mem_end:.1f}MB")
        print(f"  Net:     {mem_end - mem_start:+.1f}MB")
        print(f"  Status:  {'✅ Under 1050MB' if mem_end <= 1050 else '❌ EXCEEDS 1050MB'}")

doc.close()
os.unlink(temp_pdf_path)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Tested pages {START_PAGE}-{END_PAGE-1}")
print(f"Focus: Page 23 behavior with DPI=35")
print(f"{'='*80}")

