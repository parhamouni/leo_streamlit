# ADE Integration Strategy Summary

## Current Implementation Strategy

### Multi-Source Extraction Approach

1. **Primary: ADE Chunks** (if available)
   - Uses ADE structure to identify legend/table/text regions
   - Extracts indicators from ADE markdown/text

2. **Fallback 1: PDF Text Layer** (when ADE unavailable)
   - Extracts PDF text layer words with coordinates
   - Groups words by lines (spatial proximity)
   - Filters lines that might contain indicators (short codes, fence keywords)
   - Uses LLM to extract indicators from filtered text
   - Matches extracted indicators back to text layer words for highlighting

3. **Fallback 2: OCR Text** (when PDF text layer doesn't work)
   - Groups OCR items by spatial proximity
   - Better line-based grouping
   - Filters noise and fragmented text
   - Uses LLM to extract indicators

### Highlighting Strategy

1. **Legend Elements**: Highlight BOTH codes AND descriptions
   - Uses LLM to match OCR/text layer groups to indicators
   - Creates highlights for codes and descriptions separately

2. **Figure Indicators**: Highlight indicator codes in figures
   - Uses indicators extracted from legend
   - Matches them to OCR items in figure regions

### Key Functions

- `extract_legend_keywords_and_indicators()`: Main extraction function
  - Accepts: `page_chunks`, `google_ocr_results`, `pdf_text_layer_words` (NEW)
  - Returns: List of highlight boxes with codes and descriptions

- `extract_indicators_from_text_llm()`: LLM-based indicator extraction
  - No hardcoded patterns - pure LLM
  - Extracts both code and description

## Current Status

✅ PDF text layer extraction implemented
✅ OCR grouping improved
✅ LLM-based extraction (no hardcoded patterns)
✅ Multi-source fallback chain
✅ Highlights both codes and descriptions

## Potential Issues

1. **LLM might not extract indicators from text layer**
   - Text layer might not have structured legend format
   - Need to verify what text is being sent to LLM

2. **Matching logic might not find indicators**
   - Even if indicators extracted, matching to text layer words might fail
   - Coordinate matching might be off

3. **Highlights might be created but not visible**
   - Need to verify highlight format matches what UI expects
   - Check if highlights are being merged correctly

## Next Steps to Debug

1. Add debug logging to see what indicators are extracted
2. Verify text layer contains actual indicator codes
3. Check if highlights are being created but not displayed
4. Test with a known good page to see what's happening

























