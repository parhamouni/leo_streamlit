# ✅ Changes Applied to app_NEW.py

## 📊 Summary
**File:** `app_NEW.py` (New test version, NOT committed yet)
**Lines:** 795 (down from 927 - removed 132 lines of dead code)
**Status:** ✅ Syntax checked, ready for local testing

---

## 🔧 Changes Applied

### 1. ✅ Added Memory Profiler Class (Lines 42-80)
```python
class MemoryProfiler:
    """Track memory per function to find leaks"""
    def start_page(self, page_num)
    def record_step(self, step_name, details="")
    def end_page(self)
```
**Purpose:** Track every function call with memory deltas to identify the leak source

**Output Example:**
```
📄 Page 1 START: 250.0MB
      1. GC cleanup: 248.0MB (-2.0MB)
      3. load_page(): 250.0MB (+2.0MB)
      4. get_text(): 252.0MB (+2.0MB) len=1234
      5. get_pixmap(): 278.0MB (+26.0MB) size=2592x1728
      6. tobytes('png'): 293.0MB (+15.0MB) 1.2MB
      7. PDF wrapper: 295.0MB (+2.0MB) 1.3MB
      8. Cleanup pixmap: 268.0MB (-27.0MB)
      9. analyze_page(): 288.0MB (+20.0MB) fence=True
      10. Extract signals: 288.0MB (+0.0MB) count=3
      11. OCR highlighting: 308.0MB (+20.0MB) boxes=25
      12. Store result: 310.0MB (+2.0MB) size=3.5KB
      13. Cleanup variables: 306.0MB (-4.0MB)
   📊 Page 1 NET: +56.0MB (start=250.0MB, end=306.0MB)
```

---

### 2. ✅ Deleted Cross-Reference Code (130 lines removed)
**Deleted:**
- `compute_doc_legend_and_refs_compact()` function (lines 367-495 in old app.py)
- Simplified `merge_extra_keywords()` to just return signals as-is
- Removed unused session state variables: `legend_id_list`, `page_refs`, `legend_index_compact`

**Why:** This code was never called (commented out at line 542-546). Just dead code wasting memory and cluttering the codebase.

---

### 3. ✅ Reduced Cache Size (Line 226)
```python
# Before:
@st.cache_data(ttl=180, show_spinner=False, max_entries=18)

# After:
@st.cache_data(ttl=180, show_spinner=False, max_entries=10)
```
**Memory Savings:** ~120MB (each cached page ~15MB × 8 fewer entries)

---

### 4. ✅ GC After EVERY Page (Line 479-487)
```python
# Before:
if i > 0 and i % 3 == 0:  # Every 3 pages
    gc.collect()
    st.cache_data.clear()

# After:
gc.collect()  # EVERY page
profiler.record_step("1. GC cleanup")

if i % 5 == 0 and i > 0:  # Clear cache every 5 pages
    st.cache_data.clear()
    profiler.record_step("2. Cache clear")
```
**Purpose:** More aggressive cleanup to prevent memory buildup

---

### 5. ✅ Instrumented Processing Loop (Lines 477-663)
Added `profiler.record_step()` after each major operation:
1. GC cleanup
2. Cache clear
3. load_page()
4. get_text()
5. get_pixmap()
6. tobytes('png')
7. PDF wrapper
8. Cleanup pixmap
9. analyze_page()
10. Extract signals
11. OCR highlighting
12. Store result
13. Cleanup variables

**Purpose:** See exactly where memory is growing

---

### 6. ✅ Removed OCR Skip Logic (Lines 569-597)
```python
# Before:
if current_mem_before_ocr > 500:  # Skip OCR if over 500MB
    print(f"⚠️ Skipping OCR...")
    analysis_result['fence_text_boxes_details'] = []
else:
    # Run OCR

# After:
# OCR HIGHLIGHTING (ALWAYS RUN - no memory-based skipping per user request)
if not fatal_err_page and highlight_fence_text_app and analysis_result.get('text_found'):
    # ALWAYS run OCR, never skip based on memory
```
**Purpose:** User requested OCR to always run, no "cheating" by hardcoding skips

---

### 7. ✅ Added Profiling Summary at End (Lines 680-711)
Prints comprehensive summary after processing:
```
================================================================================
PROFILING SUMMARY - Memory Delta Per Function
================================================================================

Step                                Calls    Avg Δ      Max Δ      Min Δ     
--------------------------------------------------------------------------------
1. GC cleanup                          84     -2.3MB     +0.5MB    -12.1MB
11. OCR highlighting                   45    +24.8MB    +87.3MB     +2.1MB
3. load_page()                         84     +1.8MB     +3.2MB     +0.9MB
5. get_pixmap()                        84    +26.4MB    +45.2MB    +18.7MB
6. tobytes('png')                      84    +14.2MB    +22.1MB     +9.3MB
7. PDF wrapper                         84     +2.1MB     +3.8MB     +1.2MB
8. Cleanup pixmap                      84    -28.7MB    -15.2MB    -42.3MB
9. analyze_page()                      84    +18.5MB    +32.1MB    +12.4MB

Total pages profiled: 84
Total NET memory change: +1247.8MB
Average NET per page: +14.9MB
================================================================================
```

**Purpose:** See which function is the culprit!

---

## 🎯 What This Will Tell Us

After running with the 84-page PDF, the profiling summary will show:

1. **Which function leaks the most memory** (highest Avg Δ)
2. **Which pages are problematic** (highest Max Δ)
3. **Whether cleanup is working** (negative Δ for cleanup steps)
4. **Average memory growth per page** (should be <15MB ideally)

If we see:
- **High Δ on `11. OCR highlighting`**: Google Document AI is the problem
- **High Δ on `9. analyze_page()`**: LLM API calls are leaking
- **High Δ on `5. get_pixmap()`**: PyMuPDF pixmap not being freed
- **Low negative Δ on `8. Cleanup pixmap`**: GC not working properly
- **High Δ on `12. Store result`**: Session state growing too large

---

## 🚀 Next Steps

### Test Locally:
```bash
cd /Users/parhamhamouni/Desktop/leo/leo_streamlit
cp app.py app_BACKUP.py  # Backup current version
cp app_NEW.py app.py      # Use new version
streamlit run app.py
# Upload: /Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf
```

### Watch the Logs:
```
📄 Page 1 START: 250.0MB
      1. GC cleanup: 248.0MB (-2.0MB)
      ...
   📊 Page 1 NET: +56.0MB

📄 Page 2 START: 306.0MB
      ...
```

### Expected Behavior:
- ✅ All 84 pages should process (no crashes)
- ✅ Memory should grow linearly, not exponentially
- ✅ At end, profiling summary shows where memory went
- ✅ OCR always runs (no skipping)
- ✅ Thumbnails still work

### If It Works:
1. Review profiling output together
2. Identify the main culprit
3. Apply targeted fix
4. Test again
5. Then commit and push

### If It Crashes:
1. Look at which page crashed
2. Check the last profiling step
3. That's the function causing the issue
4. We fix that specific function
5. Test again

---

## 📝 Files Created (Not Committed):
- ✅ `app_NEW.py` - New version with all changes
- ✅ `CHANGES_TO_APPLY.md` - Original plan
- ✅ `CHANGES_APPLIED_SUMMARY.md` - This file
- ✅ `REVERT_ANALYSIS.md` - What was lost in revert
- ✅ `SAFE_MEMORY_FIX.md` - Original minimal plan
- ✅ `app_PROFILED.py` - Partial work (ignore)

---

## 🔍 Key Differences from Before:
1. **No hardcoded OCR skips** - OCR always runs
2. **Detailed profiling** - Track every single function
3. **Deleted dead code** - 132 lines removed
4. **More aggressive GC** - Every page, not every 3
5. **Smaller cache** - 10 entries, not 18
6. **Comprehensive summary** - See exactly what's leaking

---

## ✅ Ready to Test!
The file is syntax-checked and ready. Let me know when you want to:
1. Test locally with the 84-page PDF
2. Review profiling output together
3. Apply targeted fix based on data
4. Commit and push

