# Changes to Apply - Memory Fix with Profiling

## Change 1: Remove ALL Cross-Reference Code

### Delete entire function `compute_doc_legend_and_refs_compact` (lines 327-455)
This function is never called anymore, just dead code taking up space.

### Delete function `merge_extra_keywords` (lines 458-463)
Replace with simpler version:
```python
def merge_extra_keywords(signals: list) -> list:
    """Return page-local signals only."""
    return list(signals or [])
```

### Remove from initialize_session_state (lines 491-493, 544-546)
DELETE these lines:
```python
'legend_id_list': [],
'page_refs': {},
'legend_index_compact': [],
```
AND
```python
st.session_state.legend_id_list = []
st.session_state.page_refs = {}
st.session_state.legend_index_compact = []
print(f"[CROSS] Cross-reference analysis disabled for memory optimization. RAM: {_rss_mb():.1f} MB")
```

---

## Change 2: Add Memory Profiling Per Function

### Add at top (after imports):
```python
class MemoryProfiler:
    """Track memory per function to find leaks"""
    def __init__(self):
        self.page_data = []
    
    def start_page(self, page_num):
        self.current_page = {'page': page_num, 'steps': [], 'start_mem': _rss_mb()}
    
    def record_step(self, step_name, details=""):
        mem_now = _rss_mb()
        delta = mem_now - (self.current_page['steps'][-1]['mem'] if self.current_page['steps'] else self.current_page['start_mem'])
        self.current_page['steps'].append({
            'name': step_name,
            'mem': mem_now,
            'delta': delta,
            'details': details
        })
        sign = "+" if delta >= 0 else ""
        print(f"      {step_name}: {mem_now:.1f}MB ({sign}{delta:.1f}MB) {details}")
    
    def end_page(self):
        end_mem = _rss_mb()
        net = end_mem - self.current_page['start_mem']
        self.page_data.append(self.current_page)
        print(f"   📊 Page {self.current_page['page']} NET: {net:+.1f}MB (start={self.current_page['start_mem']:.1f}MB, end={end_mem:.1f}MB)")
        return net

# Global profiler instance
profiler = MemoryProfiler()
```

---

## Change 3: Instrument the Processing Loop

### In the main processing loop, wrap each operation:

```python
for i in range(st.session_state.doc_total_pages):
    curr_pg_num = i + 1
    
    # Start profiling this page
    profiler.start_page(curr_pg_num)
    print(f"\n📄 Page {curr_pg_num} START: {_rss_mb():.1f}MB")
    
    # GC AFTER EVERY PAGE (not every 3)
    import gc
    gc.collect()
    profiler.record_step("1. GC cleanup")
    
    # Clear cache every 5 pages
    if i % 5 == 0 and i > 0:
        st.cache_data.clear()
        profiler.record_step("2. Cache clear")
    
    # Check memory limit
    current_memory = _rss_mb()
    if current_memory > 900:
        error_msg = f"⚠️ Memory limit reached ({current_memory:.1f}MB)"
        st.error(error_msg)
        break
    
    # Load page
    page_obj = doc_proc_loop.load_page(i)
    profiler.record_step("3. load_page()")
    
    # Extract text
    text_content = page_obj.get_text("text")
    profiler.record_step("4. get_text()", f"len={len(text_content)}")
    
    # Create pixmap
    pix = page_obj.get_pixmap(dpi=72, alpha=False)
    profiler.record_step("5. get_pixmap()", f"size={pix.width}x{pix.height}")
    
    # Convert to PNG
    img_bytes = pix.tobytes("png")
    profiler.record_step("6. tobytes('png')", f"{len(img_bytes)/(1024*1024):.2f}MB")
    
    # Create PDF wrapper
    temp_img_doc = fitz.open()
    temp_page = temp_img_doc.new_page(width=pix.width, height=pix.height)
    temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
    single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
    temp_img_doc.close()
    profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")
    
    # Cleanup pixmap
    del pix, img_bytes
    gc.collect()
    profiler.record_step("8. Cleanup pixmap")
    
    # Analyze page
    page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
    analysis_res_core = analyze_page(
        page_data_an, llm_analysis_instance, FENCE_KEYWORDS_APP, google_cloud_config,
        recall_mode="strict"
    )
    profiler.record_step("9. analyze_page()", f"fence={analysis_res_core.get('fence_found')}")
    
    # Extract signals
    try:
        jr = json.loads(analysis_res_core["text_response"])
        signals = jr.get("signals", [])
    except:
        signals = []
    profiler.record_step("10. Extract signals", f"count={len(signals)}")
    
    # OCR highlighting (ALWAYS RUN - no skipping based on memory)
    analysis_result = {**analysis_res_core, 'page_number': curr_pg_num, 'page_index_in_original_doc': i, 'fence_text_boxes_details': []}
    if analysis_result.get('text_found'):
        try:
            boxes, _, _ = get_fence_related_text_boxes(
                single_page_pdf_bytes,
                llm_analysis_instance,
                FENCE_KEYWORDS_APP,
                merge_extra_keywords(signals),
                st.session_state.selected_model_for_analysis,
                google_cloud_config
            )
            if boxes:
                analysis_result['fence_text_boxes_details'] = boxes
            profiler.record_step("11. OCR highlighting", f"boxes={len(boxes) if boxes else 0}")
        except Exception as e:
            print(f"OCR error on page {curr_pg_num}: {e}")
            profiler.record_step("11. OCR highlighting (FAILED)", str(e))
    
    # Store in session state
    target_list = st.session_state.fence_pages if analysis_result.get('fence_found') else st.session_state.non_fence_pages
    target_list.append(analysis_result)
    profiler.record_step("12. Store result", f"size={len(str(analysis_result))/(1024):.1f}KB")
    
    # Cleanup variables
    del page_obj, text_content, single_page_pdf_bytes, page_data_an, analysis_res_core
    gc.collect()
    profiler.record_step("13. Cleanup variables")
    
    # End page profiling
    net_change = profiler.end_page()
    
    # Display result (keep thumbnails)
    # ... existing display code ...
```

---

## Change 4: Reduce Cache Size

Line 186:
```python
@st.cache_data(ttl=180, show_spinner=False, max_entries=10)  # Changed from 18
```

---

## Change 5: Print Summary at End

After processing loop:
```python
# Print profiling summary
print("\n" + "="*80)
print("PROFILING SUMMARY")
print("="*80)
summary = {}
for page_data in profiler.page_data:
    for step in page_data['steps']:
        name = step['name']
        if name not in summary:
            summary[name] = {'count': 0, 'total': 0, 'max': 0, 'min': 999}
        summary[name]['count'] += 1
        summary[name]['total'] += step['delta']
        summary[name]['max'] = max(summary[name]['max'], step['delta'])
        summary[name]['min'] = min(summary[name]['min'], step['delta'])

print(f"\n{'Step':<30} {'Calls':<8} {'Avg Δ':<10} {'Max Δ':<10} {'Min Δ':<10}")
print("-"*80)
for name in sorted(summary.keys()):
    s = summary[name]
    avg = s['total'] / s['count']
    print(f"{name:<30} {s['count']:<8} {avg:>+9.1f}MB {s['max']:>+9.1f}MB {s['min']:>+9.1f}MB")

print(f"\nTotal pages profiled: {len(profiler.page_data)}")
print(f"Average NET per page: {sum(p['steps'][-1]['mem'] - p['start_mem'] for p in profiler.page_data) / len(profiler.page_data):.1f}MB")
print("="*80)
```

---

## Summary of Changes:

1. ✅ **Deleted cross-reference code** (never used, just bloat)
2. ✅ **Added MemoryProfiler class** (tracks every function call)
3. ✅ **GC after EVERY page** (not every 3)
4. ✅ **Cache size 10** (not 18)
5. ✅ **OCR ALWAYS runs** (no skipping)
6. ✅ **Keep thumbnails** (no changes to display code)
7. ✅ **Detailed profiling** prints memory delta for each step

---

## Expected Output:

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

This will show us EXACTLY which function is leaking!

