# 🧪 Testing Instructions

## Quick Test (Recommended)

```bash
cd /Users/parhamhamouni/Desktop/leo/leo_streamlit

# 1. Backup current version
cp app.py app_BACKUP_before_profiling.py

# 2. Use new profiled version
cp app_NEW.py app.py

# 3. Run Streamlit
streamlit run app.py
```

## Upload the 84-page PDF:
```
/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf
```

## What to Watch For:

### ✅ Good Signs:
- Every page prints profiling steps (1-13)
- Memory grows steadily, not exponentially
- At end, see "PROFILING SUMMARY" table
- All 84 pages complete without crash
- OCR runs on all pages (no skips)

### ⚠️ Bad Signs:
- Memory jumps suddenly on one step
- App crashes before page 84
- Error messages about memory
- OCR gets skipped (shouldn't happen now)

## Expected Terminal Output:

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

📄 Page 2 START: 306.0MB
   ...

[After all 84 pages]

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

## 📸 Copy the Profiling Summary

Once done, **copy the entire PROFILING SUMMARY table** and paste it here so we can:
1. Identify the biggest memory leak
2. Design a targeted fix
3. Apply only what's needed
4. Test again

## 🔄 To Revert:
```bash
cp app_BACKUP_before_profiling.py app.py
```

## Questions to Answer:

1. **Did all 84 pages complete?** (Yes/No)
2. **What was the final memory?** (look at last page's NET)
3. **Which step had the highest Avg Δ?** (from summary table)
4. **Did it crash?** If yes, on which page?

---

## Alternative: Test with First 10 Pages Only

If you want faster feedback, edit `app.py` line ~461:
```python
# Add this after line 461 (for i in range(st.session_state.doc_total_pages):)
for i in range(min(10, st.session_state.doc_total_pages)):  # Only 10 pages for testing
```

This will process only 10 pages so you can see profiling data quickly.

---

Ready to test! 🚀

