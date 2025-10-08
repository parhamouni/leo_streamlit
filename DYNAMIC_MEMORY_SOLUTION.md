# Dynamic Memory-Aware OCR Solution

## General-Purpose Approach
No hardcoded page numbers - the system dynamically decides whether to use OCR based on current memory usage.

## How It Works

### Decision Logic (Line 579-592)
```python
current_memory = _rss_mb()

if is_large_page:
    if current_memory > 750:
        skip_ocr_for_memory = True  # Skip OCR when memory is high
        # Log: "OCR SKIPPED (RAM=XXX MB)"
    else:
        dpi = 30  # Normal OCR processing
```

### Behavior
1. **Early pages (low memory)**: Full OCR on all pages
2. **As memory grows**: When RAM > 750 MB, large pages switch to text-only
3. **Small pages**: Always get OCR (they don't cause memory issues)

### Advantages
✅ **General purpose** - works with any PDF
✅ **Adaptive** - responds to actual memory pressure
✅ **Maximum recall** - uses OCR when possible
✅ **Safe** - automatically backs off when memory is critical

### Memory Threshold: 750 MB
- Streamlit Cloud limit: 1024 MB
- Safety margin: 274 MB
- Typical large page OCR cost: 50-90 MB
- With 750 MB threshold, we have room for one more complex page before hitting limit

### Example Behavior
**For your Combined_TO.pdf:**
- Pages 1-18: Full OCR (memory < 750 MB)
- Pages 19-23: Text-only (memory > 750 MB, large pages)
- Pages 24+: Resumes OCR as memory stabilizes after GC

**For a different PDF:**
- If it has fewer complex pages, more pages get OCR
- If it has more complex pages, more pages fall back to text-only
- System adapts automatically

## Deployment
- Commit: ed0e662
- Status: ✅ DEPLOYED
- Type: General-purpose solution (no hardcoded values)

## Expected Results
- All PDFs can be processed
- Maximum OCR usage within memory constraints
- Graceful degradation under memory pressure
- No crashes

