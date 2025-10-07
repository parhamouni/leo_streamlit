#!/usr/bin/env python3
"""
Diagnose which pages are large and causing memory spikes
"""
import fitz

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'

doc = fitz.open(PDF_PATH)

print("="*80)
print(f"PAGE SIZE ANALYSIS - {len(doc)} pages")
print("="*80)

large_pages = []

for i in range(len(doc)):
    page = doc.load_page(i)
    width = page.rect.width
    height = page.rect.height
    area = width * height
    
    # Check if would trigger low DPI
    if width > 2000 or height > 2000:
        dpi_used = 50
        marker = "🔴 LARGE"
    else:
        dpi_used = 60
        marker = "      "
    
    # Calculate pixmap size at each DPI
    pix_60 = page.get_pixmap(dpi=60, alpha=False)
    pix_size_60 = pix_60.width * pix_60.height * 3 / (1024*1024)  # RGB, no alpha
    
    pix_50 = page.get_pixmap(dpi=50, alpha=False)
    pix_size_50 = pix_50.width * pix_50.height * 3 / (1024*1024)
    
    savings = pix_size_60 - pix_size_50
    
    print(f"{marker} Page {i+1:2d}: {width:6.0f}×{height:6.0f} | "
          f"DPI={dpi_used} | "
          f"@60DPI: {pix_size_60:5.1f}MB | "
          f"@50DPI: {pix_size_50:5.1f}MB | "
          f"Savings: {savings:+5.1f}MB")
    
    if width > 2000 or height > 2000:
        large_pages.append({
            'page': i+1,
            'width': width,
            'height': height,
            'size_60': pix_size_60,
            'size_50': pix_size_50,
            'savings': savings
        })

doc.close()

if large_pages:
    print("\n" + "="*80)
    print(f"LARGE PAGES SUMMARY ({len(large_pages)} pages)")
    print("="*80)
    for p in large_pages:
        print(f"Page {p['page']:2d}: {p['width']:.0f}×{p['height']:.0f} | "
              f"@60DPI: {p['size_60']:.1f}MB | "
              f"@50DPI: {p['size_50']:.1f}MB | "
              f"Savings: {p['savings']:+.1f}MB")
    
    total_savings = sum(p['savings'] for p in large_pages)
    print(f"\nTotal savings with adaptive DPI: {total_savings:.1f}MB")
else:
    print("\n✅ No large pages found (all under 2000×2000)")

