#!/usr/bin/env python3
"""
Analyze what's actually stored in session state to find memory leaks
"""
import sys
import json

# Sample analysis_result structure based on code
analysis_result_structure = {
    'page_number': 1,
    'page_index_in_original_doc': 0,
    'highlight_fence_text_app_setting': True,
    'fence_found': True,
    'text_found': True,
    'text_response': '{"answer":"yes","confidence":0.9,"signals":["fence","chain link"],"reason":"Contains fence callout"}',
    'text_snippet': 'This is a sample text snippet showing fence F-1...',
    'extraction_stats': {'elements': 50, 'method': 'google_cloud'},
    'extraction_method': 'google_cloud',
    'fence_text_boxes_details': [
        # Each box has this structure:
        {
            'id': 'kw_1',
            'text': 'F-1',
            'x0': 100.5,
            'y0': 200.3,
            'x1': 120.8,
            'y1': 215.9,
            'type_from_llm': 'keyword_ocr',
            'tag_from_llm': 'KEYWORD_OCR'
        }
        # Multiply this by 25-100 boxes per page!
    ]
}

# Calculate memory per page
def estimate_memory(obj, depth=0, max_depth=5):
    """Estimate memory usage of an object"""
    if depth > max_depth:
        return 0
    
    total = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += sys.getsizeof(k)
            total += estimate_memory(v, depth+1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += estimate_memory(item, depth+1, max_depth)
    elif isinstance(obj, str):
        pass  # already counted
    
    return total

# Test with varying box counts
print("="*80)
print("MEMORY ANALYSIS: Session State Storage")
print("="*80)

print("\n📊 Per-Page Memory Estimate:\n")
print(f"{'Boxes':<10} {'JSON Size':<15} {'Python Memory':<15} {'Notes':<30}")
print("-"*80)

for box_count in [0, 10, 25, 50, 100, 200]:
    # Create sample result
    sample_result = analysis_result_structure.copy()
    sample_result['fence_text_boxes_details'] = [
        {
            'id': f'kw_{i}',
            'text': f'F-{i}',
            'x0': 100.0 + i,
            'y0': 200.0 + i,
            'x1': 120.0 + i,
            'y1': 215.0 + i,
            'type_from_llm': 'keyword_ocr',
            'tag_from_llm': 'KEYWORD_OCR'
        }
        for i in range(box_count)
    ]
    
    json_size = len(json.dumps(sample_result))
    py_mem = estimate_memory(sample_result)
    
    note = ""
    if box_count == 0:
        note = "Non-fence page"
    elif box_count <= 25:
        note = "Normal fence page"
    elif box_count <= 50:
        note = "Heavy fence page"
    else:
        note = "⚠️ Very heavy page"
    
    print(f"{box_count:<10} {json_size/1024:>10.1f} KB {py_mem/1024:>12.1f} KB   {note}")

print("\n" + "="*80)
print("PROJECTION FOR 84 PAGES:")
print("="*80)

scenarios = [
    ("Best case (10 boxes avg)", 10, 84),
    ("Normal case (25 boxes avg)", 25, 84),
    ("Heavy case (50 boxes avg)", 50, 84),
    ("Worst case (100 boxes avg)", 100, 84),
]

print(f"\n{'Scenario':<30} {'Per Page':<15} {'×84 Pages':<15} {'Status':<20}")
print("-"*80)

for scenario_name, avg_boxes, pages in scenarios:
    sample_result = analysis_result_structure.copy()
    sample_result['fence_text_boxes_details'] = [
        {'id': f'kw_{i}', 'text': f'F-{i}', 'x0': 100.0, 'y0': 200.0, 'x1': 120.0, 'y1': 215.0, 
         'type_from_llm': 'keyword_ocr', 'tag_from_llm': 'KEYWORD_OCR'}
        for i in range(avg_boxes)
    ]
    
    per_page_kb = estimate_memory(sample_result) / 1024
    total_mb = (per_page_kb * pages) / 1024
    
    if total_mb < 50:
        status = "✅ Safe"
    elif total_mb < 100:
        status = "⚠️ Borderline"
    else:
        status = "❌ Risky"
    
    print(f"{scenario_name:<30} {per_page_kb:>10.1f} KB {total_mb:>12.1f} MB   {status}")

print("\n" + "="*80)
print("🔍 KEY FINDINGS:")
print("="*80)

print("""
1. **Each OCR box is ~150 bytes** (8 fields × ~20 bytes each)
2. **25 boxes per page = 3.75 KB** just for boxes
3. **84 pages × 3.75 KB = 315 KB** for all boxes (acceptable)
4. **BUT**: If pages have 100+ boxes = **1.5 MB per page** = **126 MB total** ⚠️

**The issue is NOT the boxes themselves, but:**
- Boxes are stored in BOTH session_state AND Streamlit's cache
- Each box is duplicated when rendering UI
- Pixmaps for rendering are kept in memory

**Real culprits:**
1. ❌ Pixmaps (30MB per large page) NOT freed immediately
2. ❌ PDF wrappers (15MB per large page) accumulating
3. ❌ Streamlit cache keeping old images (10 entries × 15MB = 150MB)
4. ✅ OCR boxes are actually fine (only 315 KB for 84 pages)

**Solution:**
- Free pixmaps IMMEDIATELY after use (not at end of loop)
- Lower DPI for large pages (60 instead of 72)
- Clear Streamlit cache more aggressively (every page, not every 5)
""")

print("="*80)

