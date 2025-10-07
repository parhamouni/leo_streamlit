#!/usr/bin/env python3
"""
Run full app logic locally to replicate Streamlit Cloud behavior
Process up to page 30 to see what happens at page 23
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
if not secrets_path.exists():
    print("❌ No secrets.toml found - will skip Document AI")
    google_cloud_config = None
else:
    secrets = toml.load(secrets_path)
    google_cloud_config = {
        "project_id": secrets.get("GCP_PROJECT_ID"),
        "processor_id": secrets.get("DOCUMENT_AI_PROCESSOR_ID"),
        "location": secrets.get("DOCUMENT_AI_LOCATION", "us"),
        "service_account_info": secrets.get("gcp_service_account"),
    }

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'
NUM_PAGES = 30  # Test up to page 30

FENCE_KEYWORDS = [
    "fence", "fencing", "chain link", "chainlink",
    "temp fence", "temporary fence", "site fence",
    "barrier", "barricade", "hoarding"
]

print("="*80)
print(f"LOCAL FULL TEST - Replicating Streamlit Cloud (First {NUM_PAGES} Pages)")
print("="*80)
print(f"Simulating Cloud baseline: ~730MB")
print(f"Starting memory: {_rss_mb():.1f}MB\n")

# Simulate Streamlit Cloud file upload
print("Step 1: Load PDF into memory (simulating file upload)")
with open(PDF_PATH, 'rb') as f:
    pdf_bytes = f.read()
mem_after_upload = _rss_mb()
print(f"   After upload: {mem_after_upload:.1f}MB\n")

# Simulate temp file creation
print("Step 2: Write to temp file")
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
temp_file.write(pdf_bytes)
temp_file.close()
temp_pdf_path = temp_file.name
print(f"   After writing: {_rss_mb():.1f}MB")

# Free PDF bytes
print("\nStep 3: Free PDF from memory")
pdf_bytes = None
gc.collect()
gc.collect()
mem_after_free = _rss_mb()
print(f"   After freeing: {mem_after_free:.1f}MB")

# Open from file
print("\nStep 4: Open PDF from file")
doc = fitz.open(temp_pdf_path)
mem_after_open = _rss_mb()
print(f"   After opening: {mem_after_open:.1f}MB")
print(f"   Total pages: {len(doc)}\n")

print("="*80)
print("Processing Pages (with Document AI)")
print("="*80)

page_memories = []
halt_occurred = False

for i in range(min(NUM_PAGES, len(doc))):
    page_num = i + 1
    mem_start = _rss_mb()
    
    print(f"\n📄 Page {page_num} START: {mem_start:.1f}MB")
    
    # GC before page
    gc.collect()
    mem_after_gc = _rss_mb()
    print(f"   1. GC cleanup: {mem_after_gc:.1f}MB ({mem_after_gc-mem_start:+.1f}MB)")
    
    # Clear cache every 2 pages
    if i % 2 == 0 and i > 0:
        # Can't clear Streamlit cache here, but note it
        print(f"   2. Cache clear: (would clear here)")
    
    # Check memory limit (adaptive - matching app.py)
    current_memory = mem_after_gc
    if i < 5:
        memory_limit = 850
    elif i < 20:
        memory_limit = 900
    elif i < 50:
        memory_limit = 950
    else:
        memory_limit = 1000
    
    if current_memory > memory_limit:
        print(f"\n❌ Page {page_num}: HALT at {current_memory:.1f}MB (limit: {memory_limit}MB)")
        halt_occurred = True
        break
    
    # Load page
    page_obj = doc.load_page(i)
    mem_after_load = _rss_mb()
    print(f"   3. load_page(): {mem_after_load:.1f}MB ({mem_after_load-mem_after_gc:+.1f}MB)")
    
    # Extract text
    text_content = page_obj.get_text("text")
    text_len = len(text_content)
    mem_after_text = _rss_mb()
    print(f"   4. get_text(): {mem_after_text:.1f}MB ({mem_after_text-mem_after_load:+.1f}MB) len={text_len}")
    
    # Adaptive DPI
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    if page_width > 2000 or page_height > 2000:
        dpi = 50
        print(f"   → Large page: {page_width:.0f}×{page_height:.0f}, using DPI=50")
    else:
        dpi = 60
    
    # Create pixmap
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    pix_size = pix.width * pix.height
    mem_after_pixmap = _rss_mb()
    print(f"   5. get_pixmap(): {mem_after_pixmap:.1f}MB ({mem_after_pixmap-mem_after_text:+.1f}MB) size={pix.width}x{pix.height}")
    
    # Convert to PNG
    img_bytes = pix.tobytes("png")
    img_size_mb = len(img_bytes) / (1024*1024)
    del pix
    gc.collect()
    gc.collect()
    mem_after_tobytes = _rss_mb()
    print(f"   6. tobytes() + free pixmap: {mem_after_tobytes:.1f}MB ({mem_after_tobytes-mem_after_pixmap:+.1f}MB) {img_size_mb:.2f}MB PNG")
    
    # Create PDF wrapper
    temp_img_doc = fitz.open()
    temp_page = temp_img_doc.new_page(width=pix.width if 'pix' in dir() else 100, height=pix.height if 'pix' in dir() else 100)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
    pdf_size_mb = len(single_page_pdf_bytes) / (1024*1024)
    temp_img_doc.close()
    mem_after_pdf = _rss_mb()
    print(f"   7. PDF wrapper: {mem_after_pdf:.1f}MB ({mem_after_pdf-mem_after_tobytes:+.1f}MB) {pdf_size_mb:.2f}MB")
    
    # Cleanup img_bytes
    del img_bytes
    gc.collect()
    gc.collect()
    mem_after_cleanup = _rss_mb()
    print(f"   8. Cleanup img_bytes: {mem_after_cleanup:.1f}MB ({mem_after_cleanup-mem_after_pdf:+.1f}MB)")
    
    # Analyze page (if Document AI available)
    if google_cloud_config:
        page_data = {
            "page_number": page_num,
            "text": text_content,
            "page_bytes": single_page_pdf_bytes
        }
        
        try:
            # Note: analyze_page expects llm_text, not llm_instance
            analysis_result = analyze_page(
                page_data=page_data,
                llm_text=None,  # Skip LLM for speed
                fence_keywords=FENCE_KEYWORDS,
                google_cloud_config=google_cloud_config
            )
            
            mem_after_analysis = _rss_mb()
            print(f"   9. analyze_page(): {mem_after_analysis:.1f}MB ({mem_after_analysis-mem_after_cleanup:+.1f}MB) fence={analysis_result.get('fence_found', False)}")
            
            # Extract signals
            try:
                jr = json.loads(analysis_result.get("text_response", "{}"))
                signals = jr.get("signals", [])
            except:
                signals = []
            
            mem_after_signals = _rss_mb()
            print(f"   10. Extract signals: {mem_after_signals:.1f}MB ({mem_after_signals-mem_after_analysis:+.1f}MB) count={len(signals)}")
            
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
            mem_after_analysis = mem_after_cleanup
            mem_after_signals = mem_after_cleanup
            analysis_result = {"fence_found": False}
            signals = []
    else:
        mem_after_analysis = mem_after_cleanup
        mem_after_signals = mem_after_cleanup
        analysis_result = {"fence_found": False}
        signals = []
    
    # Cleanup
    del page_obj, text_content, single_page_pdf_bytes
    if 'analysis_result' in locals():
        del analysis_result
    if 'signals' in locals():
        del signals
    gc.collect()
    gc.collect()
    gc.collect()
    
    mem_end = _rss_mb()
    print(f"   📊 Page {page_num} NET: {mem_end-mem_start:+.1f}MB (start={mem_start:.1f}MB, end={mem_end:.1f}MB)")
    
    # Record
    page_memories.append({
        'page': page_num,
        'start': mem_start,
        'end': mem_end,
        'delta': mem_end - mem_start,
        'limit': memory_limit
    })

doc.close()
os.unlink(temp_pdf_path)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if not page_memories:
    print("\n❌ No pages processed!")
else:
    print(f"\nPages processed: {len(page_memories)}/{NUM_PAGES}")
    
    for p in page_memories:
        status = "✅" if p['end'] < p['limit'] else "❌"
        print(f"  Page {p['page']:2d}: {p['start']:6.1f}→{p['end']:6.1f}MB (Δ{p['delta']:+6.1f}MB) limit={p['limit']}MB {status}")
    
    total_delta = sum(p['delta'] for p in page_memories)
    avg_delta = total_delta / len(page_memories)
    
    print(f"\nTotal memory growth: {total_delta:+.1f}MB")
    print(f"Average per page: {avg_delta:+.1f}MB")
    print(f"Final memory: {page_memories[-1]['end']:.1f}MB")
    
    if halt_occurred:
        print(f"\n❌ HALTED: Exceeded memory limit")
    else:
        print(f"\n✅ SUCCESS: Completed {len(page_memories)} pages without halting")

print("="*80)

