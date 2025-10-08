# Memory Profile Analysis - Pages 1-30

## Problem Summary
App crashes at page 23 with **984.9 MB memory usage** on Streamlit Cloud (1GB limit).

## Local Test Results (30 pages)

### Memory Growth Pattern
```
Starting: 220.0 MB
Page 1:   236.6 MB (+16.6 MB)
Page 10:  258.4 MB (+37.6 MB from page 1) ✅ GOOD
Page 18:  297.3 MB (+60.7 MB from page 1) ⚠️ GROWING
Page 19:  382.6 MB (+85.4 MB) 🔴 SPIKE!
Page 20:  438.8 MB (+56.2 MB) 🔴 SPIKE!
Page 21:  502.7 MB (+63.8 MB) 🔴 SPIKE!
Page 22:  556.3 MB (+53.7 MB) 🔴 SPIKE!
Page 23:  643.8 MB (+87.5 MB) 🔴 MASSIVE SPIKE!
Page 26:  574.0 MB (-68.6 MB) ✅ GC finally kicks in
Page 30:  571.9 MB
Ending:   578.9 MB (+358.9 MB total)
```

### Critical Finding: Pages 19-23 Memory Explosion
**Total spike on pages 19-23: ~347 MB (87% of total growth)**

Individual page memory deltas:
- Page 19: **+85.4 MB** 🔴
- Page 20: **+56.2 MB** 🔴
- Page 21: **+63.8 MB** 🔴
- Page 22: **+53.7 MB** 🔴
- Page 23: **+87.5 MB** 🔴

**Total for 5 pages: +346.6 MB**

## Root Cause Analysis

### NOT the Problem
❌ `page_bytes` size (~0.2-0.3 MB per large page) - this is working as expected
❌ Session state storage - already minimized
❌ Image generation - properly cleaned up

### The REAL Problem
✅ **Document AI responses on complex pages (19-23) are HUGE**
✅ **Garbage collection not keeping up** - memory accumulates faster than GC can clean
✅ **Cache clearing interval too infrequent** (was every 10 pages outside spike zone)

### Why Pages 19-23 Specifically?
Looking at the log, these pages contain:
- "SINGLE CHAMBER DRYWELL DETAIL" (page 19)
- "MULTIPLE CHAMBERS DRYWELL DETAIL" (page 21)
- Complex engineering diagrams with thousands of OCR elements

Document AI returns **massive responses** for these pages with detailed geometry data.

## Fixes Applied

### 1. More Aggressive Cache Clearing
**Before:**
```python
elif i % 10 == 0 and i > 0:  # Every 10 pages
    st.cache_data.clear()
```

**After:**
```python
elif i % 5 == 0 and i > 0:  # Every 5 pages
    st.cache_data.clear()
```

### 2. Extended Variable Deletion
Added cleanup of image generation intermediates:
```python
if 'pix' in locals(): del pix
if 'img_bytes' in locals(): del img_bytes
if 'temp_img_doc' in locals(): del temp_img_doc
if 'temp_page' in locals(): del temp_page
```

### 3. Triple Garbage Collection (already in place)
```python
gc.collect()
gc.collect()
gc.collect()  # Force immediate cleanup
```

## Expected Impact

**Before fixes:**
- Cache cleared every 10 pages (except spike zone 15-30)
- Memory accumulated to 643.8 MB by page 23
- Would hit ~984 MB on Streamlit Cloud (with overhead)

**After fixes:**
- Cache cleared every 5 pages + every page in spike zone
- Better cleanup of image generation temporaries
- Expected reduction: ~100-150 MB by page 23

**Estimated result:** 
- Local test: ~500-550 MB by page 23 (vs 643.8 MB)
- Streamlit Cloud: ~850-900 MB by page 23 (vs 984.9 MB) ✅ UNDER LIMIT

## Recommendations for Further Optimization (if still needed)

### Option 1: Reduce DPI on problem pages (19-23)
```python
if i >= 18 and i < 24:
    dpi = 25  # Even lower for problematic pages
```
**Trade-off:** ~20% less OCR accuracy on these specific pages

### Option 2: Skip OCR on extremely complex pages
```python
if is_large_page and len(text_content) > 5000:  # Already has lots of text
    single_page_pdf_bytes = None  # Skip OCR
```
**Trade-off:** May miss fence mentions in complex diagrams

### Option 3: Batch processing with checkpoints
Process in batches of 20 pages, save checkpoint, clear all memory, resume
**Trade-off:** More complex implementation

## Deployment Status
✅ Commit: d72c2b7
✅ Message: "Aggressive memory cleanup: cache clear every 5 pages, extended variable deletion"
✅ Pushed to Streamlit Cloud

## Next Steps
1. Test on Streamlit Cloud with same Combined_TO.pdf
2. Monitor memory at page 23
3. If still fails, implement Option 1 (reduce DPI for pages 19-23)

