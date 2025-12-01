# HIGHLIGHTING ISSUE - COMPREHENSIVE SUMMARY

## 🎯 GOAL: What We Want to Achieve

We want to **highlight fence-related content** in engineering PDF drawings with **precise bounding boxes** around the actual text, not large chunk regions.

### Specific Requirements:
1. **In Legend/Keynote regions**: Highlight BOTH the indicator code (e.g., "3301") AND the description (e.g., "CHAIN LINK FENCE")
2. **In Figure/Drawing regions**: Highlight ONLY the indicator codes where they appear as callouts
3. **Bounding boxes must be TIGHT** around the actual text tokens, not the entire ADE chunk

---

## 📁 FILES INVOLVED

### Current Implementation (What we're trying to fix):
- **`utils_ade.py`** - Our refactored utility (CURRENTLY BROKEN)
- **`app_ade.py`** - Streamlit app using the refactored utils

### Reference Implementation (What worked before):
- **`ade_backup/utils_ade.py`** - The OLD version that had better highlighting logic
- **`utils.py`** - The ORIGINAL app.py highlighting logic (proven to work well)

---

## ❌ WHAT'S INCORRECT - The Core Problem

### Problem 1: Missing the Critical Function
**`ade_backup/utils_ade.py`** has a function called:
```python
def find_text_in_ocr_and_pdf(
    search_text: str,
    ocr_tokens: List[Dict],
    pdf_blocks: List[Dict],
    pdf_words: List[Dict],
    chunk_bbox: Optional[Tuple[float, float, float, float]] = None,
    tolerance: float = 50.0
) -> Optional[Dict]:
```

This function:
1. Groups OCR tokens into **LINES** (not individual tokens)
2. Uses `match_text_in_lines()` which returns **FULL LINE bounding boxes**
3. Has sophisticated logic for:
   - Normalizing text for matching
   - Handling fragmented text across tokens
   - Combining sequential tokens
   - Returning the **entire line's bbox** when a match is found

**Our current `utils_ade.py` does NOT have this function!**

### Problem 2: Wrong Approach in Current Implementation
Our current `find_phrase_matches()` in `utils_ade.py`:
- ✅ Does handle singular/plural
- ❌ But returns individual token bboxes, not line bboxes
- ❌ Doesn't group tokens into lines properly
- ❌ Doesn't use the proven `match_text_in_lines` logic

### Problem 3: Missing Helper Functions
The old `ade_backup/utils_ade.py` has these critical helpers that we're missing:
- `group_items_by_line()` - Groups tokens into lines by Y-coordinate
- `combine_bbox()` - Combines multiple bboxes
- `make_highlight()` - Creates highlight dict from items
- `match_text_in_lines()` - The CORE matching logic that returns FULL LINE bboxes

---

## 🔧 WHAT NEEDS TO BE FIXED

### Step 1: Port the Missing Functions
From `ade_backup/utils_ade.py`, we need to copy these functions to our current `utils_ade.py`:

1. **`group_items_by_line(items, y_tolerance)`** (lines 303-334)
   - Groups tokens by Y-coordinate proximity
   - Sorts by Y, then X
   - Returns list of lines

2. **`combine_bbox(items)`** (lines 337-351)
   - Calculates union bbox of multiple items

3. **`make_highlight(items, fallback_text)`** (lines 354-375)
   - Creates highlight dict with combined bbox

4. **`match_text_in_lines(search_text, lines, chunk_bbox, tolerance)`** (lines 378-458)
   - **THIS IS THE KEY FUNCTION**
   - Searches for text in lines
   - Returns FULL LINE bbox when match found
   - Handles fragmented text (e.g., "chain" + "link" across tokens)

5. **`find_text_in_ocr_and_pdf(search_text, ocr_tokens, pdf_blocks, pdf_words, chunk_bbox, tolerance)`** (lines 461-533)
   - Orchestrates the search across OCR and PDF sources
   - Calls `match_text_in_lines` for OCR tokens
   - Calls `match_text_in_lines` for PDF words
   - Falls back to PDF blocks if needed

### Step 2: Update `extract_legend_entries()`
Change from:
```python
keyword_matches = find_phrase_matches(all_tokens, item["text_element"], chunk)
```

To:
```python
keyword_bbox = find_text_in_ocr_and_pdf(
    search_text=item["text_element"],
    ocr_tokens=ocr_tokens,
    pdf_blocks=pdf_blocks,
    pdf_words=pdf_words,
    chunk_bbox=(chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"]),
    tolerance=50.0
)
```

### Step 3: Separate OCR and PDF Tokens
Currently we merge them into `all_tokens`. We need to keep them separate:
- `ocr_tokens` - from Google OCR
- `pdf_words` - from PyMuPDF get_text("words")
- `pdf_blocks` - from PyMuPDF get_text("blocks")

---

## 📊 WHY THE OLD VERSION WORKED BETTER

The key insight from `ade_backup/utils_ade.py`:

```python
def match_text_in_lines(...):
    # When a match is found in a line...
    for item in line_items:
        if target in clean_text:
            # Return FULL LINE bbox, not just the matched token
            return make_highlight(line_items, original_text)
```

**This returns the ENTIRE LINE's bounding box**, which includes:
- The indicator code (e.g., "3301")
- The description text (e.g., "CHAIN LINK FENCE 6FT HIGH")
- All the spacing and formatting in between

This is exactly what we want for legend highlighting!

---

## 🎬 ACTION PLAN

1. **Copy functions from `ade_backup/utils_ade.py` to current `utils_ade.py`:**
   - `group_items_by_line`
   - `combine_bbox`
   - `make_highlight`
   - `match_text_in_lines`
   - `find_text_in_ocr_and_pdf`

2. **Update `extract_legend_entries()` to use the new functions**

3. **Update `app_ade.py` to pass separate token lists** (ocr_tokens, pdf_words, pdf_blocks) instead of merged `all_tokens`

4. **Test with the debug script** to verify highlighting works

---

## 📝 SUMMARY

**The Problem**: We're using individual token bboxes instead of full line bboxes  
**The Solution**: Port the proven `find_text_in_ocr_and_pdf` + `match_text_in_lines` logic from the old version  
**The Key Insight**: When you find "FENCE" in a line, highlight the ENTIRE LINE (which includes the code + description), not just the word "FENCE"



