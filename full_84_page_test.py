#!/usr/bin/env python3
"""
Test ALL 84 pages locally to find the real memory pattern
This will help us understand the generalizable solution
"""
import sys
import os
import gc
import tempfile
from pathlib import Path

sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

try:
    import psutil
    def _rss_mb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
except:
    def _rss_mb():
        return 0.0

import fitz

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'

print("="*80)
print("FULL 84-PAGE TEST - Find Generalizable Solution")
print("="*80)

# Simulate Cloud baseline
baseline_local = _rss_mb()
print(f"Local baseline: {baseline_local:.1f}MB")
CLOUD_BASELINE = 730  # MB
print(f"Cloud baseline (estimated): {CLOUD_BASELINE}MB")
CLOUD_OFFSET = CLOUD_BASELINE - baseline_local
print(f"Cloud offset: +{CLOUD_OFFSET:.1f}MB\n")

# Load PDF
with open(PDF_PATH, 'rb') as f:
    pdf_bytes = f.read()

# Write to temp
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
temp_file.write(pdf_bytes)
temp_file.close()

# Free and open from file
pdf_bytes = None
gc.collect()
gc.collect()

doc = fitz.open(temp_file.name)
print(f"PDF opened from file: {_rss_mb():.1f}MB")
print(f"Total pages: {len(doc)}\n")

print("Processing all 84 pages...")
print("="*80)

page_data = []
max_memory = 0
max_memory_page = 0

for i in range(len(doc)):
    page_num = i + 1
    mem_start = _rss_mb()
    
    # Triple GC (matching app.py)
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Load page
    page_obj = doc.load_page(i)
    text_content = page_obj.get_text("text")
    
    # Adaptive DPI
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    if page_width > 2000 or page_height > 2000:
        dpi = 35
    else:
        dpi = 45
    
    # Create pixmap
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    img_bytes = pix.tobytes("png")
    
    # Free pixmap
    del pix
    gc.collect()
    gc.collect()
    
    # PDF wrapper
    temp_img_doc = fitz.open()
    temp_page = temp_img_doc.new_page(width=1000, height=1000)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
    temp_img_doc.close()
    
    # Free img_bytes
    del img_bytes
    gc.collect()
    gc.collect()
    
    # Cleanup
    del page_obj, text_content, single_page_pdf_bytes
    gc.collect()
    gc.collect()
    gc.collect()
    
    mem_end = _rss_mb()
    delta = mem_end - mem_start
    
    # Estimate Cloud memory
    cloud_memory = mem_end + CLOUD_OFFSET
    
    page_data.append({
        'page': page_num,
        'local_start': mem_start,
        'local_end': mem_end,
        'delta': delta,
        'cloud_est': cloud_memory
    })
    
    if cloud_memory > max_memory:
        max_memory = cloud_memory
        max_memory_page = page_num
    
    # Print every 5 pages and problem pages
    if page_num % 5 == 0 or delta > 50 or cloud_memory > 1100:
        status = "⚠️" if cloud_memory > 1024 else "✅"
        print(f"{status} Page {page_num:2d}: {mem_end:6.1f}MB local (Cloud: {cloud_memory:7.1f}MB) Δ{delta:+6.1f}MB")

doc.close()
os.unlink(temp_file.name)

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Find problem pages (>1024MB on Cloud)
problem_pages = [p for p in page_data if p['cloud_est'] > 1024]
print(f"\nPages exceeding 1024MB on Cloud: {len(problem_pages)}")
if problem_pages:
    print("Problem pages:")
    for p in problem_pages[:10]:  # Show first 10
        print(f"  Page {p['page']:2d}: {p['cloud_est']:.1f}MB")

# Find largest spikes
sorted_by_delta = sorted(page_data, key=lambda x: x['delta'], reverse=True)
print(f"\nTop 10 pages by memory increase:")
for p in sorted_by_delta[:10]:
    print(f"  Page {p['page']:2d}: {p['delta']:+6.1f}MB (Cloud: {p['cloud_est']:.1f}MB)")

# Memory pattern
print(f"\nMemory pattern:")
print(f"  Peak Cloud memory: {max_memory:.1f}MB at page {max_memory_page}")
print(f"  Final Cloud memory: {page_data[-1]['cloud_est']:.1f}MB")
print(f"  Average growth: {sum(p['delta'] for p in page_data) / len(page_data):+.1f}MB per page")

# Check if any pages have negative growth
cleanup_pages = [p for p in page_data if p['delta'] < -10]
print(f"\nPages with significant cleanup (Δ<-10MB): {len(cleanup_pages)}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if max_memory > 1024:
    overage = max_memory - 1024
    print(f"❌ Peak exceeds 1024MB by {overage:.1f}MB")
    print(f"\nOptions:")
    print(f"1. Lower DPI further: 35→30 (would save ~20-30MB)")
    print(f"2. Skip image generation for pages {max_memory_page-5}-{max_memory_page+5} (spike zone)")
    print(f"3. Use text-only Document AI (no page_bytes)")
    print(f"4. Process in batches of 40 pages each")
else:
    print(f"✅ Would complete all pages under 1024MB!")

print("="*80)

