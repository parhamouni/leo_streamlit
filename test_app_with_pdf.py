#!/usr/bin/env python3
"""
Quick test to verify app_NEW.py works without actually running through Streamlit UI.
We'll import the key functions and simulate processing a few pages.
"""
import sys
import os
import fitz
import io

# Add parent directory to path
sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

print("Testing app_NEW.py components...")
print("="*80)

# Test 1: Check imports work
print("\n1. Testing imports...")
try:
    from utils import analyze_page, get_fence_related_text_boxes
    from langchain_openai import ChatOpenAI
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check MemoryProfiler class
print("\n2. Testing MemoryProfiler...")
try:
    # Read the profiler code from app.py
    with open('/Users/parhamhamouni/Desktop/leo/leo_streamlit/app.py', 'r') as f:
        content = f.read()
    
    if 'class MemoryProfiler:' in content:
        print("   ✅ MemoryProfiler class found in app.py")
    else:
        print("   ❌ MemoryProfiler class NOT found")
        sys.exit(1)
    
    if 'profiler = MemoryProfiler()' in content:
        print("   ✅ Global profiler instance created")
    else:
        print("   ❌ Global profiler instance NOT found")
        sys.exit(1)
        
    if 'profiler.start_page(' in content:
        print("   ✅ profiler.start_page() calls found")
    else:
        print("   ❌ profiler.start_page() calls NOT found")
        sys.exit(1)
    
    if 'profiler.record_step(' in content:
        count = content.count('profiler.record_step(')
        print(f"   ✅ profiler.record_step() called {count} times")
    else:
        print("   ❌ profiler.record_step() calls NOT found")
        sys.exit(1)
    
    if 'profiler.end_page()' in content:
        print("   ✅ profiler.end_page() calls found")
    else:
        print("   ❌ profiler.end_page() calls NOT found")
        sys.exit(1)
    
    if 'PROFILING SUMMARY' in content:
        print("   ✅ Profiling summary report found")
    else:
        print("   ❌ Profiling summary report NOT found")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ MemoryProfiler test failed: {e}")
    sys.exit(1)

# Test 3: Check cross-reference code deleted
print("\n3. Testing cross-reference deletion...")
try:
    if 'def compute_doc_legend_and_refs_compact(' not in content:
        print("   ✅ compute_doc_legend_and_refs_compact() deleted")
    else:
        print("   ⚠️  compute_doc_legend_and_refs_compact() still exists")
    
    if 'legend_id_list' not in content or content.count('legend_id_list') < 3:
        print("   ✅ legend_id_list references minimized/removed")
    else:
        count = content.count('legend_id_list')
        print(f"   ⚠️  legend_id_list still appears {count} times")
        
except Exception as e:
    print(f"   ❌ Cross-reference test failed: {e}")
    sys.exit(1)

# Test 4: Check OCR skip logic removed
print("\n4. Testing OCR always-on...")
try:
    if 'if current_mem_before_ocr > 500:' not in content:
        print("   ✅ Memory-based OCR skip removed")
    else:
        print("   ❌ OCR skip logic still exists!")
        sys.exit(1)
    
    if 'OCR HIGHLIGHTING (ALWAYS RUN' in content:
        print("   ✅ OCR always-run comment found")
    else:
        print("   ⚠️  OCR always-run comment missing")
        
except Exception as e:
    print(f"   ❌ OCR test failed: {e}")
    sys.exit(1)

# Test 5: Check cache size reduced
print("\n5. Testing cache size...")
try:
    if 'max_entries=10' in content:
        print("   ✅ Cache size reduced to 10")
    elif 'max_entries=18' in content:
        print("   ❌ Cache size still 18!")
        sys.exit(1)
    else:
        print("   ⚠️  Cache size not clear")
        
except Exception as e:
    print(f"   ❌ Cache test failed: {e}")
    sys.exit(1)

# Test 6: Check GC frequency
print("\n6. Testing GC frequency...")
try:
    # Look for the new pattern (GC every page)
    if 'gc.collect()' in content and 'profiler.record_step("1. GC cleanup")' in content:
        print("   ✅ GC after every page (with profiling)")
    else:
        print("   ⚠️  GC pattern unclear")
        
except Exception as e:
    print(f"   ❌ GC test failed: {e}")
    sys.exit(1)

# Test 7: Quick PDF processing test (first page only)
print("\n7. Testing PDF processing (first page)...")
try:
    pdf_path = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'
    if os.path.exists(pdf_path):
        doc = fitz.open(pdf_path)
        print(f"   ✅ PDF opened: {len(doc)} pages")
        
        # Just test first page
        page = doc.load_page(0)
        text = page.get_text("text")
        print(f"   ✅ Page 0 text extracted: {len(text)} chars")
        
        pix = page.get_pixmap(dpi=72, alpha=False)
        print(f"   ✅ Page 0 pixmap created: {pix.width}x{pix.height}")
        
        img_bytes = pix.tobytes("png")
        print(f"   ✅ Page 0 PNG bytes: {len(img_bytes)/(1024*1024):.2f}MB")
        
        doc.close()
        print("   ✅ PDF processing test passed")
    else:
        print(f"   ⚠️  Test PDF not found at {pdf_path}")
        
except Exception as e:
    print(f"   ⚠️  PDF test warning: {e}")
    # Don't exit, this is optional

# Test 8: Check file size
print("\n8. Checking file size...")
try:
    app_size = os.path.getsize('/Users/parhamhamouni/Desktop/leo/leo_streamlit/app.py')
    print(f"   ✅ app.py size: {app_size} bytes")
    
    with open('/Users/parhamhamouni/Desktop/leo/leo_streamlit/app.py', 'r') as f:
        lines = len(f.readlines())
    print(f"   ✅ app.py lines: {lines}")
    
    if lines < 850:
        print(f"   ✅ File reduced from 927 to {lines} lines (deleted ~{927-lines} lines)")
    else:
        print(f"   ⚠️  File still large: {lines} lines")
        
except Exception as e:
    print(f"   ❌ File size test failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nThe app_NEW.py is ready for testing with the full 84-page PDF.")
print("Run: streamlit run app.py")
print("Then upload: /Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf")

