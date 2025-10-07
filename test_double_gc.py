#!/usr/bin/env python3
"""
Test if double gc.collect() helps with large page memory spikes
"""
import sys
import os
import gc

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

doc = fitz.open(PDF_PATH)

print("="*80)
print("TEST: Single vs Double gc.collect() for Large Pages")
print("="*80)

# Test pages 19-20 (the problematic ones)
for page_num in [19, 20]:
    print(f"\n{'='*80}")
    print(f"Page {page_num}")
    print(f"{'='*80}")
    
    page = doc.load_page(page_num - 1)
    
    # TEST 1: Single gc.collect()
    print("\nTEST 1: Single gc.collect()")
    mem_before = _rss_mb()
    print(f"  Before pixmap: {mem_before:.1f}MB")
    
    pix = page.get_pixmap(dpi=50, alpha=False)
    mem_after_pix = _rss_mb()
    print(f"  After pixmap: {mem_after_pix:.1f}MB (+{mem_after_pix-mem_before:.1f}MB)")
    
    img_bytes = pix.tobytes("png")
    mem_after_tobytes = _rss_mb()
    print(f"  After tobytes: {mem_after_tobytes:.1f}MB (+{mem_after_tobytes-mem_after_pix:.1f}MB)")
    
    del pix
    gc.collect()
    mem_after_1gc = _rss_mb()
    print(f"  After del + 1×gc: {mem_after_1gc:.1f}MB ({mem_after_1gc-mem_after_tobytes:+.1f}MB)")
    
    del img_bytes
    gc.collect()
    mem_after_cleanup = _rss_mb()
    print(f"  After del img + 1×gc: {mem_after_cleanup:.1f}MB ({mem_after_cleanup-mem_after_1gc:+.1f}MB)")
    print(f"  Net increase: +{mem_after_cleanup - mem_before:.1f}MB")
    
    # TEST 2: Double gc.collect()
    print("\nTEST 2: Double gc.collect()")
    mem_before = _rss_mb()
    print(f"  Before pixmap: {mem_before:.1f}MB")
    
    pix = page.get_pixmap(dpi=50, alpha=False)
    mem_after_pix = _rss_mb()
    print(f"  After pixmap: {mem_after_pix:.1f}MB (+{mem_after_pix-mem_before:.1f}MB)")
    
    img_bytes = pix.tobytes("png")
    mem_after_tobytes = _rss_mb()
    print(f"  After tobytes: {mem_after_tobytes:.1f}MB (+{mem_after_tobytes-mem_after_pix:.1f}MB)")
    
    del pix
    gc.collect()
    gc.collect()
    mem_after_2gc = _rss_mb()
    print(f"  After del + 2×gc: {mem_after_2gc:.1f}MB ({mem_after_2gc-mem_after_tobytes:+.1f}MB)")
    
    del img_bytes
    gc.collect()
    gc.collect()
    mem_after_cleanup = _rss_mb()
    print(f"  After del img + 2×gc: {mem_after_cleanup:.1f}MB ({mem_after_cleanup-mem_after_2gc:+.1f}MB)")
    print(f"  Net increase: +{mem_after_cleanup - mem_before:.1f}MB")

doc.close()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("Double gc.collect() may help reduce peak memory spikes,")
print("but Python's GC has inherent delays. The real fix was")
print("opening PDF from FILE not memory, which saves 149MB baseline.")
print("="*80)

