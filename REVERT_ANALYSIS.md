# Revert Analysis - What Was Lost & Recovery Plan

**Date:** October 2, 2025  
**Status:** Repository reverted to commit `45dc985` (stable state)  
**Reason:** Too many unstable changes pushed without proper testing

---

## ❌ **WHAT WAS LOST (Reverted Changes)**

### 1. **Detailed Per-Function Memory Logging** ❌
**What it was:**
- Step-by-step logging showing memory delta for each operation
- Tracked: load_page(), get_text(), get_pixmap(), analyze_page(), OCR, cleanup
- Format: `3️⃣ get_pixmap(72dpi): 280.0MB (Δ+26.0MB, size=2592x1728)`

**Why it was useful:**
- Would pinpoint EXACT function causing memory leak
- Show which cleanup steps aren't working

**Why it was removed:**
- Added complexity without fixing the root issue
- Created NameError bugs (analysis_result scope issue)

---

### 2. **Adaptive DPI Based on Page Size** ❌
**What it was:**
```python
page_area = page_obj.rect.width * page_obj.rect.height
is_large_page = page_area > 1500000
dpi = 60 if is_large_page else 72  # 60 DPI for large pages
```

**Why it was useful:**
- Profiling showed large pages (2592×1728) use 33MB at 72 DPI
- 60 DPI reduces this by 30% (~10MB saved per large page)
- Over 84 pages with 60% large: ~500MB savings

**Why it was removed:**
- Introduced with other unstable changes
- Not tested independently

---

### 3. **Aggressive GC (Every Page)** ❌
**What it was:**
```python
# After EVERY page:
gc.collect()

# Every 5 pages:
st.cache_data.clear()

# Every 10 pages:
gc.collect(generation=2)  # Full GC
```

**Why it was useful:**
- Profiling showed +15MB/page retention without aggressive GC
- Every-page GC could save ~420MB over 84 pages

**Why it was removed:**
- Current: GC every 3 pages (less aggressive)
- Revert went back to more conservative approach

---

### 4. **Reduced Cache Size (10 entries)** ❌
**What it was:**
```python
@st.cache_data(max_entries=10)  # Was 18
```

**Why it was useful:**
- 18 cached images = ~270MB
- 10 cached images = ~150MB
- Savings: ~120MB

**Why it was removed:**
- Reverted to max_entries=18 (current state)

---

### 5. **Adaptive OCR Skip (Projection-Based)** ❌
**What it was:**
```python
projected_mem = current_mem + 50  # Worst case from profiling
if projected_mem > 750:
    skip_ocr = True
```

**Why it was useful:**
- Based on profiling: OCR variance is -15MB to +21MB
- Uses actual memory + safety margin (not arbitrary threshold)
- Would allow OCR on more pages when memory is available

**Current state (what we have now):**
```python
if current_mem_before_ocr > 500:  # Arbitrary hardcoded threshold
    skip_ocr = True
```

**Why it was removed:**
- Threshold changes (750MB vs 900MB) caused confusion
- Wasn't properly tested

---

### 6. **Free PDF Bytes After Opening** ❌
**What it was:**
```python
doc = fitz.open(stream=io.BytesIO(st.session_state.original_pdf_bytes))
pdf_size_mb = len(st.session_state.original_pdf_bytes) / (1024*1024)
st.session_state.original_pdf_bytes = None  # Free 150MB
gc.collect()
```

**Why it was useful:**
- PDF is 150MB in session state
- After fitz.open(), we don't need session state copy
- Would free 150MB immediately

**Current state:**
- PDF stays in session state entire time (uses 150MB permanently)

**Why it was removed:**
- Caused issues with rerun display (needs PDF bytes for image generation)
- Not properly tested for side effects

---

### 7. **Analysis/Diagnostic Files** ❌ DELETED
- `MEMORY_PROFILING_ANALYSIS.md` - Profiling data & insights
- `OPTIMIZATION_VERIFICATION.md` - Proof of no overfitting
- `profile_memory_per_function.py` - Profiling script
- `test_full_document_processing.py` - Comprehensive test script
- `diagnose_crash.py` - Diagnostic tool

**Why they were useful:**
- Documented the profiling methodology
- Showed evidence for optimization decisions
- Test scripts to verify fixes

---

## ✅ **WHAT WE STILL HAVE (Current State)**

### 1. **Lightweight PNG Wrapper** ✅
- Instead of `insert_pdf()` (350MB spike)
- Using `get_pixmap() → tobytes('png') → insert_image()`
- **Savings:** ~90% memory reduction on this operation

### 2. **Memory-Aware OCR Skip** ✅
- Skips OCR if memory > 500MB
- **Downside:** Arbitrary threshold, not adaptive

### 3. **GC Every 3 Pages** ✅
- Less aggressive than "every page" but still helps
- **Downside:** May allow 15MB/page accumulation

### 4. **Cross-Reference Disabled** ✅
- No legend analysis (memory-intensive)
- **Savings:** Significant

### 5. **Error Logging & Profiling Decorator** ✅
- `log_exception()` function
- `profile_memory()` decorator
- **Useful for:** Debugging when issues occur

---

## 🎯 **RECOVERY PLAN (What to Re-implement)**

### Priority 1: CRITICAL - Test First, Then Apply 🔴

#### **1.1 Free PDF Bytes After Opening** 
**Impact:** -150MB immediately  
**Risk:** MEDIUM (might break rerun display)  

**Plan:**
1. Test locally if rerun works without `st.session_state.original_pdf_bytes`
2. If it fails, find alternative (save to temp file, reload on demand, etc.)
3. Only apply if verified working

**Code:**
```python
# After: doc = fitz.open(stream=io.BytesIO(st.session_state.original_pdf_bytes))
st.session_state.original_pdf_bytes = None
gc.collect()
```

---

#### **1.2 Adaptive DPI for Large Pages**
**Impact:** -6MB/page × 50 large pages = -300MB  
**Risk:** LOW (just changes DPI, no breaking changes)

**Plan:**
1. Add page size detection
2. Test one PDF locally
3. Verify Document AI still works at 60 DPI
4. Apply if accuracy maintained

**Code:**
```python
page_area = page_obj.rect.width * page_obj.rect.height
dpi = 60 if page_area > 1500000 else 72
```

---

### Priority 2: BENEFICIAL - Apply After Testing ✅

#### **2.1 Aggressive GC (Every Page)**
**Impact:** -5MB/page × 84 = -420MB  
**Risk:** VERY LOW (just more frequent cleanup)

**Code:**
```python
# After each page:
gc.collect()

# Every 5 pages (not 3):
if i % 5 == 0:
    st.cache_data.clear()
```

---

#### **2.2 Reduce Cache Size**
**Impact:** -120MB  
**Risk:** LOW (users may need to re-expand pages more often)

**Code:**
```python
@st.cache_data(max_entries=10)  # From 18
```

---

### Priority 3: OPTIONAL - Only If Needed 🟡

#### **3.1 Adaptive OCR Skip**
**Current:** Skip if > 500MB (arbitrary)  
**Better:** Skip if (current + 50MB) > 900MB (projected)

**Impact:** More pages get OCR when safe  
**Risk:** LOW

---

#### **3.2 Detailed Memory Logging**
**Only add if:**
- Still crashing on Streamlit Cloud
- Need to debug exact leak point

**Risk:** MEDIUM (added complexity, potential bugs)

---

## 📋 **RECOMMENDED IMPLEMENTATION ORDER**

### Phase 1: Safe & High Impact (Do This First)
1. ✅ **Adaptive DPI** (test locally, then apply)
2. ✅ **GC every page** (very safe)
3. ✅ **Cache size 10** (minimal risk)

**Expected:** ~450MB savings, should allow 60-70 pages

---

### Phase 2: Test Locally First
4. 🧪 **Free PDF bytes** (test if rerun still works)
   - If YES → +150MB savings = **600MB total savings**
   - If NO → skip this, use alternatives

---

### Phase 3: Only If Still Crashing
5. 🔧 **Adaptive OCR threshold** (if still issues)
6. 🔍 **Detailed logging** (if debugging needed)

---

## ⚠️ **LESSONS LEARNED**

1. **Test each change independently** before pushing
2. **Don't combine 5+ optimizations** in one commit
3. **Verify on Streamlit Cloud** before marking as "done"
4. **Keep diagnostic files** for future reference
5. **Document why each change works** (profiling data)

---

## 🚀 **NEXT STEPS (Awaiting Your Approval)**

**I will NOT apply anything until you approve. Here's what I recommend:**

### Step 1: Apply Safe Changes (Your Approval Needed)
- [ ] Adaptive DPI (60 for large pages)
- [ ] GC every page (not every 3)
- [ ] Cache size 10 (not 18)

**Test:** Run locally, then deploy

### Step 2: Test PDF Freeing Locally (Your Approval Needed)
- [ ] Test if rerun works without st.session_state.original_pdf_bytes
- [ ] If yes, apply
- [ ] If no, find alternative

### Step 3: Only If Still Issues
- [ ] Add back adaptive OCR threshold
- [ ] Add back detailed logging

---

**What would you like me to do?**
1. Apply Phase 1 changes (adaptive DPI + aggressive GC + smaller cache)?
2. Test PDF freeing first before anything else?
3. Something different?

Please confirm before I make ANY changes.

