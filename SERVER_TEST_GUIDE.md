# 🚀 Server Testing Guide

## ✅ Changes Pushed to GitHub

**Commit:** `41fa167`  
**Branch:** `main`  
**Date:** October 2, 2025

---

## 📦 What Was Committed

### Files Modified:
1. ✅ `app.py` - Main fixes applied (520 insertions, 206 deletions)
   - Added MemoryProfiler class
   - Deleted cross-reference code (130 lines)
   - Immediate GC after pixmap & PNG bytes
   - Adaptive DPI for large pages
   - Cache clear every 3 pages

2. ✅ `FINAL_SUMMARY.md` - Complete explanation
3. ✅ `MEMORY_ANALYSIS_RESULTS.md` - Profiling data analysis
4. ✅ `REAL_FIX.md` - Technical implementation details

---

## 🧪 How to Test on Streamlit Cloud

### Step 1: Wait for Deployment
Streamlit Cloud should auto-deploy the new version within 1-2 minutes.

### Step 2: Upload Test PDF
Upload: `/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf`
(84 pages)

### Step 3: Monitor the Logs
Click "Manage app" → "Logs" on Streamlit Cloud to see:

```
📄 Page 1 START: 250.0MB
      1. GC cleanup: 248.0MB (-2.0MB)
      3. load_page(): 250.0MB (+2.0MB)
      4. get_text(): 252.0MB (+2.0MB) len=1234
      5. get_pixmap(): 278.0MB (+26.0MB) size=792x612
      6. tobytes() + free pixmap: 264.0MB (-14.0MB) ✅ FREED!
      7. PDF wrapper: 266.0MB (+2.0MB)
      8. Cleanup img_bytes: 252.0MB (-14.0MB) ✅ FREED!
      9. analyze_page(): 270.0MB (+18.0MB)
      10. Extract signals: 270.0MB (+0.0MB)
      11. OCR highlighting: 290.0MB (+20.0MB)
      12. Store result: 292.0MB (+2.0MB)
      13. Cleanup variables: 288.0MB (-4.0MB)
   📊 Page 1 NET: +38.0MB
```

### Step 4: Look for These Indicators

✅ **Good Signs:**
- Steps 6 and 8 show **NEGATIVE** or small positive deltas
- Large pages show "→ Large page detected, using DPI=60"
- Average NET per page < 15MB
- All 84 pages complete without crash
- Final memory < 700MB

❌ **Bad Signs:**
- Steps 6 and 8 show large **POSITIVE** deltas (30MB+)
- Average NET per page > 30MB
- "Memory usage too high" error before page 84
- Connection reset / app crashes

---

## 📊 What to Expect

### Memory Profile (Expected):

**Small Pages (792×612):**
```
5. get_pixmap(): +4MB
6. tobytes() + free pixmap: -2MB ✅
7. PDF wrapper: +2MB
8. Cleanup img_bytes: -2MB ✅
NET: ~+5MB per page
```

**Large Pages (2592×1728):**
```
→ Large page detected: using DPI=60
5. get_pixmap(): +23MB (was +33MB) ✅ 30% reduction
6. tobytes() + free pixmap: -18MB ✅ FREED!
7. PDF wrapper: +10MB (was +15MB) ✅ smaller
8. Cleanup img_bytes: -10MB ✅ FREED!
NET: ~+8MB per page (was +63MB) ✅ 87% reduction
```

**84 Pages Total:**
- Expected: ~420-600MB total
- Should complete without crash

---

## 🎯 Success Criteria

### ✅ Test PASSES if:
1. All 84 pages complete without crash
2. Final memory < 700MB
3. Steps 6 and 8 show negative or small deltas
4. No "Memory usage too high" errors
5. Profiling summary shows at end

### ❌ Test FAILS if:
1. App crashes before page 84
2. "Connection reset by peer" error
3. Steps 6 and 8 show large positive deltas
4. Memory grows linearly above 800MB

---

## 📋 Profiling Summary

At the very end of processing, you'll see:

```
================================================================================
PROFILING SUMMARY - Memory Delta Per Function
================================================================================

Step                                Calls    Avg Δ      Max Δ      Min Δ     
--------------------------------------------------------------------------------
6. tobytes() + free pixmap             84    -12.3MB     -5.0MB    -20.0MB  ✅
8. Cleanup img_bytes                   84    -10.1MB     -3.0MB    -15.0MB  ✅
5. get_pixmap()                        84    +15.2MB    +25.0MB     +3.5MB
11. OCR highlighting                   45    +22.0MB    +40.0MB    +10.0MB
...

Total pages profiled: 84
Total NET memory change: +480.0MB
Average NET per page: +5.7MB
================================================================================
```

**Key metrics:**
- Steps 6 and 8 should be **NEGATIVE** (cleanup working!)
- Average NET should be **< 10MB per page**
- Total NET should be **< 700MB**

---

## 🔍 If It Still Crashes

### Check the logs for:
1. Which page it crashed on
2. What was the last profiling step
3. Memory value at crash

### Share this info:
```
Crashed at: Page XX
Last step: YY. ZZZ: AAAA.AMB
Final memory: BBB.BMB
```

Then we can apply more targeted fixes.

---

## 📞 Next Actions Based on Results

### If SUCCESS (no crash):
1. ✅ Verify all 84 pages processed
2. ✅ Check profiling summary
3. ✅ Confirm memory < 700MB
4. ✅ Deploy to production!

### If PARTIAL SUCCESS (crashes after page 60+):
- Good progress! Need minor tweaks:
  - Lower memory limit from 900MB to 800MB
  - Clear cache every 2 pages instead of 3
  - Reduce OCR frequency

### If STILL FAILS (crashes before page 30):
- Need deeper investigation:
  - Check if cleanup is working (steps 6 & 8 negative?)
  - Check if adaptive DPI is working (large pages use 60 DPI?)
  - May need to disable OCR on some pages

---

## ✅ Ready to Test!

The changes are live on GitHub. Streamlit Cloud should auto-deploy within 1-2 minutes.

**Good luck!** 🚀

Let me know the results and I'll help with next steps if needed.

