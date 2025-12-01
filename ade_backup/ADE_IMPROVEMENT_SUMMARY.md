# Summary: How to Improve app_ade.py

## Current Problem
The current `app_ade.py` tries to extract indicators from ADE chunks and match them with OCR in one complex step, which is error-prone.

## Solution: Use app.py's proven two-step approach + ADE structure

### How app.py works (that we should copy):

1. **Step 1 - LLM Analysis** (`analyze_page()`):
   - Extracts full page text
   - Sends to LLM: "Does this page have fence content? What signals/keywords did you find?"
   - LLM returns: `{answer: "yes/no", signals: ["3301", "CMU WALL", "fence", ...]}`
   - The `signals` array contains indicators/keywords the LLM intelligently extracted

2. **Step 2 - Highlighting** (`get_fence_related_text_boxes()`):
   - Receives `signals` from Step 1
   - Does simple keyword matching against page text/OCR
   - Creates bounding boxes around matches
   - Returns highlight boxes

### How to add ADE structure (the improvement):

**Step 1 Enhanced**: Use ADE to provide context to LLM
- Identify legend regions (where codes are defined)
- Identify figure regions (where codes are referenced)
- Tell LLM: "This page has X legend regions and Y figure regions. Legend regions contain indicator definitions. Figure regions contain indicator references."
- LLM can now better understand the page structure and extract more relevant signals

**Step 2 Enhanced**: Use ADE structure to filter highlights
- In legend regions: highlight everything (keywords, indicators, descriptions)
- In figure regions: only highlight indicator codes (numeric codes like "3301"), not full descriptions
- This reduces false positives in figures

## Implementation Plan

1. **Create `analyze_page_with_ade_structure()`** - combines `analyze_page()` logic with ADE structure context
2. **Modify `app_ade.py` main loop** to:
   - Parse full document with ADE
   - For each page: get ADE chunks
   - Call `analyze_page_with_ade_structure()` with chunks
   - Extract signals from LLM response
   - Call `get_fence_related_text_boxes()` with signals
   - Filter highlights by ADE structure (legend vs figures)
3. **Add helper function** to filter highlights based on ADE regions

## Key Benefits

- ✅ Simpler: Uses proven app.py approach
- ✅ Smarter: ADE structure helps LLM understand layout
- ✅ More precise: Structure-aware filtering reduces false positives
- ✅ Maintainable: Clear separation of concerns

The ADE structure helps the LLM understand WHERE to look (legend vs figures) and helps filter highlights to be more precise, but the extraction itself uses LLM intelligence (like app.py), not complex parsing.

























