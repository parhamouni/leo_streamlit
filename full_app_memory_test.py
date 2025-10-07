#!/usr/bin/env python3
"""
Full test with actual app logic (Document AI + OCR + highlighting)
Test first 10 pages to see if we stay under 750MB limit
"""
import sys
import os
import gc
import io
import tempfile
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
secrets = toml.load(secrets_path)

google_cloud_config = {
    "project_id": secrets.get("GCP_PROJECT_ID"),
    "processor_id": secrets.get("DOCUMENT_AI_PROCESSOR_ID"),
    "location": secrets.get("DOCUMENT_AI_LOCATION", "us"),
    "service_account_info": secrets.get("gcp_service_account"),
}

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'
NUM_PAGES = 10  # Test first 10 pages

print("="*80)
print(f"FULL APP MEMORY TEST - First {NUM_PAGES} Pages")
print("="*80)
print(f"Memory limit: 750MB (Streamlit Cloud limit for pages 1-10)")
print(f"Starting memory: {_rss_mb():.1f}MB\n")

# Load PDF
with open(PDF_PATH, 'rb') as f:
    pdf_bytes = f.read()
print(f"After loading PDF: {_rss_mb():.1f}MB")

# Write to temp file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
temp_file.write(pdf_bytes)
temp_file.close()
print(f"After writing temp file: {_rss_mb():.1f}MB")

# Free PDF bytes
pdf_bytes = None
gc.collect()
gc.collect()
print(f"After freeing PDF bytes: {_rss_mb():.1f}MB\n")

# Open from file
doc = fitz.open(temp_file.name)
print(f"Opened PDF from file: {_rss_mb():.1f}MB")
print(f"Total pages: {len(doc)}\n")

print("="*80)
print("Processing Pages")
print("="*80)

page_memories = []

for i in range(min(NUM_PAGES, len(doc))):
    page_num = i + 1
    mem_start = _rss_mb()
    
    # GC before page
    gc.collect()
    mem_after_gc = _rss_mb()
    
    # Clear cache every 2 pages
    if i % 2 == 0 and i > 0:
        # Note: can't import streamlit here, so skip cache clear
        pass
    
    # Check memory limit (adaptive)
    if i < 10:
        memory_limit = 750
    elif i < 30:
        memory_limit = 800
    else:
        memory_limit = 850
    
    if mem_after_gc > memory_limit:
        print(f"\n❌ Page {page_num}: HALT at {mem_after_gc:.1f}MB (limit: {memory_limit}MB)")
        break
    
    # Load page
    page_obj = doc.load_page(i)
    
    # Extract text
    text_content = page_obj.get_text("text")
    
    # Create PNG wrapper (adaptive DPI)
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    if page_width > 2000 or page_height > 2000:
        dpi = 50
    else:
        dpi = 60
    
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    img_bytes = pix.tobytes("png")
    
    # Create PDF wrapper
    temp_img_doc = fitz.open()
    temp_page = temp_img_doc.new_page(width=pix.width, height=pix.height)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
    temp_img_doc.close()
    
    # Cleanup
    del pix, img_bytes
    gc.collect()
    
    mem_before_analysis = _rss_mb()
    
    # Analyze page (Document AI)
    page_data = {
        "page_number": page_num,
        "text": text_content,
        "page_bytes": single_page_pdf_bytes
    }
    
    try:
        analysis_result = analyze_page(
            page_data=page_data,
            llm_text=None,  # Skip LLM for speed
            fence_keywords=[
                "fence", "fencing", "chain link", "chainlink",
                "temp fence", "temporary fence", "site fence",
                "barrier", "barricade", "hoarding"
            ],
            google_cloud_config=google_cloud_config
        )
        
        mem_after_analysis = _rss_mb()
        
        # OCR highlighting (if fence detected)
        if analysis_result.get("is_fence_page"):
            boxes = get_fence_related_text_boxes(
                page_bytes_pdf=single_page_pdf_bytes,
                google_cloud_config=google_cloud_config,
                fence_kw_list=[
                    "fence", "fencing", "chain link", "chainlink",
                    "temp fence", "temporary fence", "site fence",
                    "barrier", "barricade", "hoarding"
                ],
                extra_kw=[],
                top_k=50
            )
        else:
            boxes = []
        
        mem_after_ocr = _rss_mb()
        
    except Exception as e:
        print(f"\n❌ Page {page_num}: Error during analysis: {e}")
        mem_after_analysis = _rss_mb()
        mem_after_ocr = mem_after_analysis
        boxes = []
        analysis_result = None
    
    # Cleanup
    if analysis_result:
        del analysis_result
    del page_obj, text_content, single_page_pdf_bytes
    gc.collect()
    gc.collect()
    gc.collect()
    
    mem_end = _rss_mb()
    
    # Record
    page_memories.append({
        'page': page_num,
        'start': mem_start,
        'after_gc': mem_after_gc,
        'before_analysis': mem_before_analysis,
        'after_analysis': mem_after_analysis,
        'after_ocr': mem_after_ocr,
        'end': mem_end,
        'delta': mem_end - mem_start,
        'boxes': len(boxes) if boxes else 0
    })
    
    print(f"Page {page_num:2d}: {mem_start:6.1f}→{mem_end:6.1f}MB (Δ{mem_end-mem_start:+6.1f}MB) | Analysis: {mem_after_analysis-mem_before_analysis:+5.1f}MB | OCR: {mem_after_ocr-mem_after_analysis:+5.1f}MB | Boxes: {len(boxes) if boxes else 0}")

doc.close()
os.unlink(temp_file.name)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_delta = sum(p['delta'] for p in page_memories)
avg_delta = total_delta / len(page_memories) if page_memories else 0

print(f"\nPages processed: {len(page_memories)}/{NUM_PAGES}")
print(f"Total memory growth: {total_delta:+.1f}MB")
print(f"Average per page: {avg_delta:+.1f}MB")
print(f"Final memory: {page_memories[-1]['end']:.1f}MB" if page_memories else "N/A")

if page_memories:
    max_mem = max(p['end'] for p in page_memories)
    print(f"Peak memory: {max_mem:.1f}MB")
    
    if max_mem < 750:
        print(f"\n✅ SUCCESS: Stayed under 750MB limit!")
    else:
        print(f"\n❌ FAILED: Exceeded 750MB limit")
        
print("\n" + "="*80)

