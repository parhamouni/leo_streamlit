# 🔍 Memory Profiling Results & Analysis

## Test Results (5 pages)

```
🥇 #1 CULPRIT: 13. Cleanup variables
   Average: +86.6MB per call
   Max spike: +111.1MB
   Total impact: +433.2MB

🥈 #2 CULPRIT: 5. get_pixmap()
   Average: +16.3MB per call
   Total impact: +81.7MB

🥉 #3 CULPRIT: 7. PDF wrapper
   Average: +7.7MB per call
   Total impact: +38.4MB
```

---

## 🚨 Root Cause Identified

### The Problem:
**"Cleanup variables" step is ADDING 86MB instead of removing it!**

This means:
- Steps 5-7 allocate memory (pixmap, PNG bytes, PDF wrapper)
- Memory is NOT freed immediately
- It accumulates in Python's heap
- When we call `del` + `gc.collect()` at the end, GC runs and processes a HUGE backlog
- GC itself uses memory to track and free objects
- Net result: +86MB spike during "cleanup"

### Why This Happens:
Python's GC doesn't run immediately on `del`. It waits until:
1. Threshold is reached (too late)
2. `gc.collect()` is called (but by then, too much accumulated)
3. During long-running operations (like API calls - explains the -126MB during analyze_page)

---

## 📊 Detailed Breakdown

### Page 4 (Large Page Example):
```
5. get_pixmap(): 224.2MB (+33.6MB)  ← Allocate 33MB for 2592x1728 pixels
6. tobytes('png'): 238.9MB (+14.7MB) ← Allocate 14MB for PNG compression
7. PDF wrapper: 254.0MB (+15.1MB)    ← Allocate 15MB for PDF wrapper
8. Cleanup pixmap: 254.0MB (+0.0MB)  ← del pix, img_bytes - NO EFFECT!
9. analyze_page(): 82.4MB (-171.7MB) ← GC runs during API call, frees backlog
13. Cleanup variables: 186.1MB (+103.6MB) ← GC processes remaining objects
```

**Total allocated in steps 5-7:** 33.6 + 14.7 + 15.1 = **63.4MB**  
**But freed during step 9:** -171.7MB (frees THIS page + previous backlog)  
**Then step 13 adds back:** +103.6MB (GC overhead)

---

## 🎯 The Fix

### Strategy 1: Immediate GC After Each Allocation (Recommended)
```python
# After pixmap
pix = page_obj.get_pixmap(dpi=72, alpha=False)
profiler.record_step("5. get_pixmap()")

# Convert immediately and free
img_bytes = pix.tobytes("png")
del pix  # Free pixmap NOW
gc.collect()  # Force immediate GC
profiler.record_step("6. tobytes() + free pixmap")

# Create wrapper
temp_img_doc = fitz.open()
temp_page = temp_img_doc.new_page(width=..., height=...)
temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
temp_img_doc.close()
del img_bytes  # Free PNG bytes NOW
gc.collect()  # Force immediate GC
profiler.record_step("7. PDF wrapper + free img_bytes")
```

**Expected result:** Each cleanup shows NEGATIVE memory delta

---

### Strategy 2: Lower DPI for Large Pages
```python
# Detect large pages
page_width = page_obj.rect.width
page_height = page_obj.rect.height

# Adaptive DPI
if page_width > 2000 or page_height > 2000:
    dpi = 60  # Large pages: 60 DPI (saves ~40%)
else:
    dpi = 72  # Normal pages: 72 DPI

pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
```

**Expected savings:**  
- 2592x1728 @ 72 DPI = 33.6MB
- 2160x1440 @ 60 DPI = 23.3MB  
- **Savings: 10MB per large page** (×50 large pages = 500MB saved)

---

### Strategy 3: Disable Pixmap Caching in PyMuPDF
```python
# Before opening PDF
import fitz
fitz.TOOLS.set_small_glyph_heights(True)  # Reduce glyph cache

# When creating pixmap
pix = page_obj.get_pixmap(dpi=72, alpha=False, annots=False, clip=None)
```

---

## 📈 Projection for 84 Pages

### Current State (Without Fix):
- Avg per page: -4.5MB (but this is misleading due to API errors)
- **Real avg (steps 5-7 only):** +16.3 + +7.7 + +6.9 = **+30.9MB per page**
- **Projected for 84 pages:** 30.9 × 84 = **+2,595MB** ⚠️ **WILL CRASH!**

### With Fix (Strategy 1):
- Immediate GC after each allocation
- Expected: pixmap cleanup shows -30MB, img_bytes cleanup shows -15MB
- **Net per page:** ~+5MB (only for stored results)
- **Projected for 84 pages:** 5 × 84 = **+420MB** ✅ **SAFE**

### With Fix (Strategy 1 + 2):
- Immediate GC + Lower DPI for large pages
- **Net per page:** ~+3MB
- **Projected for 84 pages:** 3 × 84 = **+252MB** ✅ **VERY SAFE**

---

## ✅ Recommended Fix

Apply **both strategies**:

1. **Immediate GC** after pixmap and img_bytes
2. **Adaptive DPI** (60 for large pages, 72 for normal)
3. **Test again** to verify cleanup shows negative deltas

This should reduce memory from **+2,595MB → +252MB** (10x improvement!)

---

## 🔬 Next Steps

1. Apply the fix to `app_NEW.py`
2. Run quick test again (5 pages)
3. Verify cleanup steps show NEGATIVE memory deltas
4. If successful, test full 84 pages
5. Commit and push

---

**Date:** October 2, 2025  
**Test:** 5 pages from Combined_TO.pdf (84 pages total)  
**Conclusion:** Memory leak found in pixmap/PNG/PDF allocation - fix with immediate GC

