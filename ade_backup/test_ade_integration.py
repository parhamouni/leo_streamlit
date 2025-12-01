#!/usr/bin/env python3
"""
Test script for ADE integration
"""
import os
import sys
import base64
import fitz
from io import BytesIO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils_ade import ade_parse_page, extract_ade_text_and_elements, get_ade_fence_boxes

def test_ade_integration():
    """Test basic ADE functionality"""
    print("🧪 Testing ADE Integration...")
    
    # Check for API key
    ade_key = os.getenv("LANDINGAI_API_KEY")
    if not ade_key:
        print("❌ LANDINGAI_API_KEY not found in environment")
        return False
    
    print("✅ ADE API key found")
    
    # Create a simple test PDF
    print("📄 Creating test PDF...")
    doc = fitz.open()
    page = doc.new_page()
    
    # Add some test text
    page.insert_text((100, 100), "FENCE SPECIFICATIONS", fontsize=12)
    page.insert_text((100, 120), "Chain link fence 6 feet high", fontsize=10)
    page.insert_text((100, 140), "Gate details and materials", fontsize=10)
    page.insert_text((100, 160), "Regular text not related to fences", fontsize=10)
    
    # Get PDF bytes
    pdf_bytes = doc.tobytes()
    doc.close()
    
    print(f"📄 Test PDF created ({len(pdf_bytes)} bytes)")
    
    # Test ADE parsing
    print("🔍 Testing ADE parsing...")
    ade_result = ade_parse_page(pdf_bytes, ade_key, zdr=True)
    
    if not ade_result.get("success"):
        print(f"❌ ADE parsing failed: {ade_result.get('error')}")
        return False
    
    print("✅ ADE parsing successful")
    
    # Extract text and elements
    print("📝 Extracting text and elements...")
    page_width, page_height = 612, 792  # Standard letter size
    ade_text, ade_elements = extract_ade_text_and_elements(ade_result, page_width, page_height)
    
    print(f"📝 Extracted text length: {len(ade_text)}")
    print(f"📝 Extracted elements: {len(ade_elements)}")
    
    if ade_text:
        print(f"📝 Sample text: {ade_text[:100]}...")
    
    if ade_elements:
        print(f"📝 Sample element: {ade_elements[0]}")
    
    # Test fence box extraction
    print("🎯 Testing fence box extraction...")
    fence_keywords = ['fence', 'gate', 'chain link']
    fence_boxes = get_ade_fence_boxes(ade_elements, fence_keywords)
    
    print(f"🎯 Found {len(fence_boxes)} fence-related boxes")
    
    for i, box in enumerate(fence_boxes):
        print(f"🎯 Box {i+1}: '{box['text']}' at ({box['x0']:.1f}, {box['y0']:.1f}, {box['x1']:.1f}, {box['y1']:.1f})")
    
    print("✅ ADE integration test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_ade_integration()
    sys.exit(0 if success else 1)



























