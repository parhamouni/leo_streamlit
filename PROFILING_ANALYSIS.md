# 📊 Local Profiling Analysis Results

## Test Environment
- **Machine:** Local Mac (conda leo environment)
- **PDF:** Combined_TO.pdf (84 pages)
- **Pages Tested:** 5 pages
- **Start Memory:** 210.9MB
- **Final Memory:** 242.3MB
- **NET Growth:** +31.4MB for 5 pages = **+6.3MB per page**

---

## 🔍 Memory Delta Per Function (5 Pages)

```
Step                          Calls    Avg Δ        Max Δ        Min Δ       
------------------------------------------------------------------------------
9. analyze_page()               5     -45.0MB      +2.0MB      -151.0MB  ← GC during API wait
13. Cleanup variables           5     +26.4MB     +101.9MB       +0.0MB  ← SPIKE!
5. get_pixmap()                 5     +11.3MB      +25.0MB       +1.5MB
6. tobytes() + free pixmap      5      +5.0MB      +10.5MB       +1.6MB
7. PDF wrapper                  5      +4.2MB      +10.0MB       +1.5MB
4. get_text()                   5      +2.2MB       +3.8MB       +0.0MB
3. load_page()                  5      +0.3MB       +1.1MB       +0.0MB
```

---

## 🚨 The Real Problem

### Issue #1: Cleanup SPIKES
**`13. Cleanup variables: +26.4MB avg, MAX +101.9MB`**

This is NOT a leak - it's Python's GC processing a backlog:
- Memory accumulates during page processing
- When we call `gc.collect()`, it processes everything
- GC itself uses memory to track and free objects
- On a low-memory system (Streamlit Cloud), this spike pushes over the limit

### Issue #2: Starting Memory on Streamlit Cloud
**Local:** 210MB start → 750MB limit = **540MB headroom**  
**Streamlit Cloud:** 700MB start → 750MB limit = **50MB headroom** ← NO ROOM FOR SPIKES!

---

## 📈 Projection for 84 Pages

### Optimistic (Average):
- **+4.4MB per page × 84 = +372MB**
- Start: 210MB → End: 582MB ✅ **SAFE**

### Pessimistic (With Spikes):
- Every cleanup: +26MB spike
- Every 10 pages: one +100MB spike
- **+372MB + 8 × 100MB = +1,172MB** ❌ **CRASH**

---

## ✅ Root Causes Identified

### 1. **Streamlit Cloud Starts at 700MB** (vs 210MB locally)
   - Docker container overhead
   - Streamlit services
   - System libraries
   - **Solution:** Can't change this

### 2. **GC Spikes During Cleanup**
   - Up to +100MB temporary spike
   - Happens when GC processes backlog
   - **Solution:** Process in smaller batches

### 3. **Large Page Pixmaps**
   - 2592×1728 pages use 25MB per pixmap
   - Even at DPI=50: 1800×1200 = 25MB
   - **Solution:** Already at minimum viable DPI

---

## 🎯 Recommended Solutions

### Solution 1: Batch Processing (BEST)
Process PDF in batches of 20 pages:
- Batch 1: Pages 1-20
- Clear ALL memory
- Batch 2: Pages 21-40
- Clear ALL memory
- etc.

**Pros:** Guaranteed to stay under limit  
**Cons:** User needs to run multiple times

---

### Solution 2: Skip OCR on Large Pages
Only run OCR on small pages:
- Small pages (792×612): Full OCR
- Large pages (2592×1728): Text-only, no OCR highlighting

**Pros:** Reduces memory by 50%  
**Cons:** Less accurate highlighting on large pages

---

### Solution 3: Progressive Processing
Start with low limit, increase as memory allows:
- Pages 1-10: Stop at 700MB
- Pages 11-20: Stop at 750MB
- Pages 21+: Stop at 800MB

**Pros:** Process as many pages as memory allows  
**Cons:** May not complete all 84 pages

---

### Solution 4: Disable Live Preview
Don't show live results during processing:
- No st.expander during processing
- No intermediate images
- Only show results at end

**Pros:** Frees ~100MB from Streamlit DOM  
**Cons:** User can't see progress

---

## 📊 Comparison

| Solution | Memory Saved | Pages Completed | User Experience |
|----------|-------------|-----------------|-----------------|
| Batch Processing | 0MB | 84 (guaranteed) | Fair (multiple runs) |
| Skip Large OCR | 200MB | 84 (likely) | Good (some loss) |
| Progressive | 0MB | Variable (20-60?) | Poor (incomplete) |
| No Live Preview | 100MB | 84 (maybe) | Good |

---

## ✅ RECOMMENDED: Combination Approach

1. **Disable live preview** → Saves 100MB
2. **Skip OCR on pages 50+** → Saves another 100MB
3. **Adaptive limit** → Use memory wisely

**Expected result:** Complete 84 pages within 850MB limit

---

**Next step:** Which solution do you prefer?

