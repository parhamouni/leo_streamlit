"""
Local Memory Test - Verify Aggressive GC Fix
Tests memory usage through first 30 pages with new GC strategy
"""
import fitz
import os
import sys
import gc
import psutil

# Setup
PDF_PATH = "/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf"
PAGES_TO_TEST = 30

def get_memory_mb():
    """Get current process RSS memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("="*80)
print("MEMORY TEST WITH AGGRESSIVE GC")
print("="*80)
print(f"PDF: {PDF_PATH}")
print(f"Testing: First {PAGES_TO_TEST} pages")
print(f"Strategy: 5x GC before + 7x GC after each page")
print("="*80)

# Open PDF
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
print(f"\nTotal pages in PDF: {total_pages}")

# Track memory
memory_start = get_memory_mb()
print(f"Starting memory: {memory_start:.1f} MB\n")

memory_log = []

for i in range(min(PAGES_TO_TEST, total_pages)):
    page_num = i + 1
    
    # Memory before page
    mem_before = get_memory_mb()
    
    # === AGGRESSIVE GC BEFORE PAGE (simulate app.py logic) ===
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
        gc.collect()
        gc.collect()
    
    # CRITICAL FIX: Close and reopen PDF every 5 pages to free C-level memory
    if i > 0 and i % 5 == 0:
        doc.close()
        doc = fitz.open(PDF_PATH)
        print(f"   🔄 CLOSED/REOPENED PDF at page {page_num} to free C memory")
        for _ in range(3):
            gc.collect()
    
    mem_after_gc_before = get_memory_mb()
    
    # === PROCESS PAGE ===
    page_obj = doc[i]
    page_width = page_obj.rect.width
    page_height = page_obj.rect.height
    is_large_page = page_width > 2000 or page_height > 2000
    
    # Extract text
    text_content = page_obj.get_text("text")
    
    # Generate page_bytes (simulate image generation)
    dpi = 30 if is_large_page else 45
    pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
    pix_width, pix_height = pix.width, pix.height
    
    img_bytes = pix.tobytes("png")
    img_size_mb = len(img_bytes) / (1024*1024)
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
    
    page_bytes_kb = len(single_page_pdf_bytes) / 1024
    
    mem_after_processing = get_memory_mb()
    
    # === AGGRESSIVE GC AFTER PAGE ===
    # Delete all variables (simulate app.py cleanup)
    del page_obj, text_content, single_page_pdf_bytes
    
    # 7x GC for thorough cleanup
    for _ in range(7):
        gc.collect()
    
    mem_after_gc_after = get_memory_mb()
    
    # Calculate deltas
    gc_before_saved = mem_before - mem_after_gc_before
    processing_cost = mem_after_processing - mem_after_gc_before
    gc_after_saved = mem_after_processing - mem_after_gc_after
    net_change = mem_after_gc_after - mem_before
    
    # Log
    memory_log.append({
        'page': page_num,
        'is_large': is_large_page,
        'mem_before': mem_before,
        'mem_after_gc_before': mem_after_gc_before,
        'mem_after_processing': mem_after_processing,
        'mem_after_gc_after': mem_after_gc_after,
        'gc_before_saved': gc_before_saved,
        'processing_cost': processing_cost,
        'gc_after_saved': gc_after_saved,
        'net_change': net_change,
        'page_bytes_kb': page_bytes_kb,
        'img_size_mb': img_size_mb
    })
    
    # Print summary
    large_marker = "📄 LARGE" if is_large_page else "📄 small"
    print(f"{large_marker} Page {page_num:2d}: "
          f"START={mem_before:.1f}MB → "
          f"GC-5x={mem_after_gc_before:.1f}MB ({gc_before_saved:+.1f}) → "
          f"PROCESS={mem_after_processing:.1f}MB ({processing_cost:+.1f}) → "
          f"GC-7x={mem_after_gc_after:.1f}MB ({gc_after_saved:+.1f}) | "
          f"NET={net_change:+.1f}MB")

doc.close()

# Final summary
memory_end = get_memory_mb()
memory_total_increase = memory_end - memory_start

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Starting memory: {memory_start:.1f} MB")
print(f"Ending memory: {memory_end:.1f} MB")
print(f"Total increase: {memory_total_increase:+.1f} MB")
print(f"Average per page: {memory_total_increase/PAGES_TO_TEST:+.1f} MB")

# Find problem pages (large net increase)
problem_pages = [m for m in memory_log if m['net_change'] > 20]
if problem_pages:
    print(f"\n⚠️  Pages with >20MB net increase: {len(problem_pages)}")
    for m in problem_pages:
        print(f"   Page {m['page']:2d}: {m['net_change']:+.1f} MB "
              f"(GC-before saved {m['gc_before_saved']:.1f}, GC-after saved {m['gc_after_saved']:.1f})")

# Projection for full 84 pages
if PAGES_TO_TEST > 0:
    avg_per_page = memory_total_increase / PAGES_TO_TEST
    projected_84 = memory_start + (avg_per_page * 84)
    print(f"\n📊 PROJECTION for 84 pages: {projected_84:.1f} MB")
    if projected_84 > 950:
        print(f"   ⚠️  WARNING: Still over 950 MB limit!")
    else:
        print(f"   ✅ SAFE: Under 950 MB limit")

print("="*80)

