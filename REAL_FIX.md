# 🎯 THE REAL FIX - Based on Profiling Data

## 🔍 What the Profiling Revealed

From `quick_memory_test.py` results:

```
🥇 #1 CULPRIT: 13. Cleanup variables: +86.6MB per call  ⚠️ MAJOR LEAK!
🥈 #2 CULPRIT: 5. get_pixmap():  +16.3MB per call
🥉 #3 CULPRIT: 7. PDF wrapper: +7.7MB per call

Page 4 (Large 2592×1728):
5. get_pixmap(): +33.6MB
6. tobytes('png'): +14.7MB  
7. PDF wrapper: +15.1MB
8. Cleanup pixmap: +0.0MB  ← NO EFFECT!
9. analyze_page(): -171.7MB  ← GC runs during API wait
13. Cleanup variables: +103.6MB  ← GC processes backlog
```

## 🚨 Root Cause

**Pixmaps and PDF wrappers accumulate in memory and are NOT freed until much later.**

The `del pix, img_bytes` and `gc.collect()` at step 8 have **zero effect** because:
1. Python's GC doesn't run immediately on `del`
2. Objects accumulate across multiple pages
3. GC finally runs during API calls (step 9) or at cleanup (step 13)
4. By then, too much memory has accumulated

**For 84 pages:**
- ~50 large pages × 63MB = **3,150MB leak** → CRASH!

---

## ✅ THE FIX (3-Part Strategy)

### Part 1: Immediate GC After Pixmap (Critical!)

**Current code (lines 512-531):**
```python
# 5. Create pixmap
pix = page_obj.get_pixmap(dpi=72, alpha=False)
profiler.record_step("5. get_pixmap()", f"size={pix.width}x{pix.height}")

# 6. Convert to PNG
img_bytes = pix.tobytes("png")
profiler.record_step("6. tobytes('png')", f"{len(img_bytes)/(1024*1024):.2f}MB")

# 7. PDF wrapper
temp_img_doc = fitz.open()
temp_page = temp_img_doc.new_page(width=pix.width, height=pix.height)
temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
temp_img_doc.close()
profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")

# 8. Cleanup pixmap
del pix, img_bytes
gc.collect()
profiler.record_step("8. Cleanup pixmap")
```

**FIX:**
```python
# 5. Create pixmap
pix = page_obj.get_pixmap(dpi=72, alpha=False)
pix_width, pix_height = pix.width, pix.height
profiler.record_step("5. get_pixmap()", f"size={pix_width}x{pix_height}")

# 6. Convert to PNG and FREE pixmap immediately
img_bytes = pix.tobytes("png")
del pix  # Free 30MB NOW
gc.collect()  # Force immediate collection
profiler.record_step("6. tobytes() + free pixmap", f"{len(img_bytes)/(1024*1024):.2f}MB")

# 7. PDF wrapper
temp_img_doc = fitz.open()
temp_page = temp_img_doc.new_page(width=pix_width, height=pix_height)
temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
temp_img_doc.close()
profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")

# 8. Cleanup img_bytes
del img_bytes  # Free 15MB NOW
gc.collect()  # Force immediate collection
profiler.record_step("8. Cleanup img_bytes")
```

**Expected result:** Step 6 and 8 should show **NEGATIVE** memory deltas

---

### Part 2: Adaptive DPI for Large Pages

**Add before step 5:**
```python
# Detect large pages and use lower DPI
page_width = page_obj.rect.width
page_height = page_obj.rect.height
if page_width > 2000 or page_height > 2000:
    dpi = 60  # Large pages: 60 DPI (saves ~40% memory)
    profiler.record_step("→ Large page detected, using DPI=60")
else:
    dpi = 72  # Normal pages: 72 DPI

# 5. Create pixmap with adaptive DPI
pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
```

**Savings:**
- 2592×1728 @ 72 DPI = 33.6MB
- 2160×1440 @ 60 DPI = 23.3MB
- **Saves 10MB per large page** × 50 large pages = **500MB saved!**

---

### Part 3: Clear Cache After EVERY Page

**Current code (line 485-487):**
```python
# Clear cache every 5 pages
if i % 5 == 0 and i > 0:
    st.cache_data.clear()
    profiler.record_step("2. Cache clear")
```

**FIX:**
```python
# Clear cache every 3 pages (more aggressive)
if i % 3 == 0 and i > 0:
    st.cache_data.clear()
    profiler.record_step("2. Cache clear")
```

**Why:** Cache holds 10 entries × 15MB = 150MB. Clear more often to prevent buildup.

---

## 📊 Expected Results After Fix

### Before Fix (Current):
```
5. get_pixmap(): +33.6MB
6. tobytes('png'): +14.7MB
7. PDF wrapper: +15.1MB
8. Cleanup pixmap: +0.0MB  ← NOT WORKING
13. Cleanup variables: +103.6MB  ← PROCESSING BACKLOG
NET: +63MB per large page × 50 = +3,150MB → CRASH!
```

### After Fix:
```
5. get_pixmap(): +23.3MB (lower DPI)
6. tobytes() + free pixmap: -10.0MB  ← FREED IMMEDIATELY
7. PDF wrapper: +10.1MB (smaller due to lower DPI)
8. Cleanup img_bytes: -10.0MB  ← FREED IMMEDIATELY
13. Cleanup variables: +2.0MB  ← NO BACKLOG
NET: +15MB per large page × 50 = +750MB ✅ SAFE!
```

**Total savings: 3,150MB → 750MB** (4x improvement!)

---

## 🔧 Implementation Steps

1. Apply Part 1 (lines 512-531): Immediate GC after pixmap & img_bytes
2. Apply Part 2 (before line 512): Adaptive DPI for large pages
3. Apply Part 3 (line 485): Clear cache every 3 pages
4. Test with quick test (5 pages) - verify step 6 & 8 show NEGATIVE deltas
5. Test with full 84 pages - verify no crash

**Ready to apply?**

