# 🎯 What's Next - Action Plan

## ✅ What I Did (NOT committed yet)

### 1. Created `app_NEW.py` with These Changes:
- ✅ **Deleted 132 lines** of unused cross-reference code
- ✅ **Added MemoryProfiler** class to track every function call
- ✅ **GC after EVERY page** (not every 3)
- ✅ **Reduced cache** from 18 to 10 entries
- ✅ **OCR ALWAYS runs** (no memory-based skipping)
- ✅ **Detailed profiling** for 13 steps per page
- ✅ **Profiling summary** at end showing which function leaks

### 2. Key Features Preserved:
- ✅ OCR highlighting (always enabled)
- ✅ Thumbnails (kept as-is)
- ✅ All existing functionality
- ❌ Cross-references (deleted as requested)

### 3. Files Created (Not Committed):
- `app_NEW.py` - New version with profiling
- `CHANGES_APPLIED_SUMMARY.md` - What changed
- `TEST_INSTRUCTIONS.md` - How to test
- `WHATS_NEXT.md` - This file

---

## 🚀 Next Step: Test Locally

### Run This:
```bash
cd /Users/parhamhamouni/Desktop/leo/leo_streamlit
cp app.py app_BACKUP.py
cp app_NEW.py app.py
streamlit run app.py
```

### Upload:
```
/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf
```

### Watch Terminal for:
```
📄 Page 1 START: 250.0MB
      1. GC cleanup: 248.0MB (-2.0MB)
      ...
   📊 Page 1 NET: +56.0MB
```

---

## 📊 After Testing, We'll See:

### Example Output:
```
================================================================================
PROFILING SUMMARY
================================================================================
Step                                Calls    Avg Δ      Max Δ      Min Δ     
--------------------------------------------------------------------------------
11. OCR highlighting                   45    +24.8MB    +87.3MB     +2.1MB  ⚠️ CULPRIT!
5. get_pixmap()                        84    +26.4MB    +45.2MB    +18.7MB
9. analyze_page()                      84    +18.5MB    +32.1MB    +12.4MB
...
Average NET per page: +14.9MB
```

**This tells us:** If `11. OCR highlighting` has the highest Avg Δ, then Google Document AI is the leak!

---

## 🔧 Then We Fix the Right Thing

### If OCR is the leak:
- Option 1: Reduce OCR frequency (every N pages)
- Option 2: Process OCR in chunks
- Option 3: Clear Document AI cache more aggressively

### If pixmap is the leak:
- Option 1: More aggressive pixmap cleanup
- Option 2: Smaller DPI
- Option 3: Release pixmap immediately after use

### If analyze_page() is the leak:
- Option 1: Clear LLM conversation history
- Option 2: Reduce context window
- Option 3: Process in smaller batches

**We'll know EXACTLY which fix to apply based on the data!**

---

## 📝 What You Asked For:

✅ **"OCR should be kept fully"** - Done, OCR always runs
✅ **"No cross referencing at all"** - Deleted 132 lines
✅ **"Thumbnails should be kept"** - Kept as-is
✅ **"Profile every function"** - 13 profiling steps per page
✅ **"Change code but not push"** - app_NEW.py ready, not committed
✅ **"Test locally first"** - Instructions in TEST_INSTRUCTIONS.md

---

## 🎬 Your Options Now:

### Option A: Test Right Now (Recommended)
```bash
cd /Users/parhamhamouni/Desktop/leo/leo_streamlit
cp app.py app_BACKUP.py && cp app_NEW.py app.py
streamlit run app.py
```
Then upload the 84-page PDF and watch the terminal.

### Option B: Review Changes First
Read `CHANGES_APPLIED_SUMMARY.md` to see exactly what changed.

### Option C: Quick 10-Page Test
Same as Option A, but edit line 461 to process only 10 pages for faster feedback.

---

## 🤝 Agreement Before Commit:

**I will NOT commit until you:**
1. Test locally with the 84-page PDF
2. Review the profiling output
3. Confirm the changes work
4. Explicitly approve the commit

**Then we'll:**
1. Discuss the profiling results
2. Apply a targeted fix (if needed)
3. Test again
4. Commit + push together

---

## 💡 Why This Approach is Better:

### Before (Your Concern):
- ❌ Hardcoded OCR skips (cheating)
- ❌ Guessing what's wrong
- ❌ Too many changes at once
- ❌ No data to prove what's leaking

### Now (Data-Driven):
- ✅ OCR always runs (no cheating)
- ✅ Measure every function
- ✅ Small, targeted changes
- ✅ Data shows exact culprit

---

## Ready! 🎯

The code is ready to test. Let me know:
1. If you want to test now
2. If you want me to explain anything
3. If you want to review the changes first

**No commits will happen without your explicit approval.** 🔒

