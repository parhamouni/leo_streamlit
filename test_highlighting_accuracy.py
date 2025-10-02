#!/usr/bin/env python3
"""
Test highlighting accuracy: Are bounding boxes correctly placed?
This tests the OCR highlighting step (get_fence_related_text_boxes).
"""
import fitz
import sys
from pathlib import Path
from utils import get_fence_related_text_boxes
from langchain_openai import ChatOpenAI

try:
    import streamlit as st
    openai_key = st.secrets["OPENAI_API_KEY"]
    # Create proper google_cloud_config dict (matching app.py format)
    google_cloud_config = {
        "project_number": st.secrets["google_cloud"]["project_number"],
        "location": st.secrets["google_cloud"]["location"],
        "processor_id": st.secrets["google_cloud"]["processor_id"],
        "service_account_info": dict(st.secrets["gcp_service_account"])
    }
except Exception as e:
    print(f"❌ Cannot load secrets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test on page 1 (has many fence highlights)
pdf_path = Path("/Users/parhamhamouni/Desktop/leo/data/gold_standard/subset_gold/selected_pages.pdf")
doc = fitz.open(pdf_path)
page = doc[0]  # Page 1

# Create single-page PDF (PNG wrapper approach - same as app.py)
pix = page.get_pixmap(dpi=72, alpha=False)
img_bytes = pix.tobytes("png")
temp_doc = fitz.open()
temp_page = temp_doc.new_page(width=pix.width, height=pix.height)
temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
page_bytes = temp_doc.tobytes(deflate=True, garbage=4)
temp_doc.close()

print(f"📄 Testing Page 1")
print(f"   Page dimensions: {page.rect.width} x {page.rect.height}")
print(f"   PNG dimensions: {pix.width} x {pix.height}")
print(f"   Page bytes: {len(page_bytes)/1024:.1f} KB")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key)

# Test highlighting with expanded keywords (matching app.py)
FENCE_KEYWORDS = [
    "fence", "fencing", "gate", "barrier", "guardrail", "post", "mesh", "panel", "chain link",
    "screen wall", "privacy screen", "CMU wall", "masonry wall", "wall", "bollard",
    "railing", "handrail", "security barrier", "perimeter"
]

print(f"\n🔍 Running OCR highlighting...")
try:
    boxes, _, _ = get_fence_related_text_boxes(
        page_bytes,
        llm,
        FENCE_KEYWORDS,
        [],  # No extra keywords
        "gpt-4o",
        google_cloud_config
    )
    
    print(f"\n✅ Found {len(boxes)} highlight boxes")
    
    # Check if boxes are valid
    for i, box in enumerate(boxes[:5], 1):  # Show first 5
        print(f"\n   Box {i}:")
        print(f"      Coords: ({box['x0']:.1f}, {box['y0']:.1f}, {box['x1']:.1f}, {box['y1']:.1f})")
        print(f"      Size: {box['x1']-box['x0']:.1f} x {box['y1']-box['y0']:.1f}")
        print(f"      Text: \"{box.get('text', 'N/A')[:50]}...\"")
        
        # Validate coordinates
        if box['x0'] < 0 or box['y0'] < 0:
            print(f"      ⚠️  WARNING: Negative coordinates!")
        if box['x1'] > page.rect.width or box['y1'] > page.rect.height:
            print(f"      ⚠️  WARNING: Coordinates outside page bounds!")
        if box['x1'] <= box['x0'] or box['y1'] <= box['y0']:
            print(f"      ⚠️  WARNING: Invalid box dimensions!")
    
    # Compare with ground truth annotations
    print(f"\n📊 Ground Truth Comparison:")
    gt_count = 0
    for annot in (page.annots() or []):
        if annot.type[1] not in {"FreeText", "Square"}:
            gt_count += 1
    
    print(f"   Ground truth highlights: {gt_count}")
    print(f"   Detected highlights: {len(boxes)}")
    print(f"   Ratio: {len(boxes)/gt_count:.2f}" if gt_count > 0 else "   Ratio: N/A")
    
    if len(boxes) < gt_count * 0.5:
        print(f"\n   ⚠️  WARNING: Detected <50% of ground truth highlights!")
        print(f"   This may indicate highlighting accuracy issues.")
    elif len(boxes) > gt_count * 2:
        print(f"\n   ⚠️  WARNING: Detected >200% of ground truth highlights!")
        print(f"   This may indicate over-highlighting (false positives).")
    else:
        print(f"\n   ✅ Highlighting ratio looks reasonable (50-200%)")
    
    # Visual validation suggestion
    print(f"\n💡 To visually validate:")
    print(f"   1. Run the Streamlit app")
    print(f"   2. Upload selected_pages.pdf")
    print(f"   3. Check page 1 highlights match the ground truth PDF")
    
except Exception as e:
    print(f"\n❌ Error during highlighting: {e}")
    import traceback
    traceback.print_exc()

doc.close()

