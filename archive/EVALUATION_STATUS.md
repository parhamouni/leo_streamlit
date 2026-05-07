# Hybrid Highlighting System - Evaluation Status

## Current Status

The hybrid ADE + Google OCR highlighting system has been implemented and an evaluation framework created. 

### Evaluation Script
- **File**: `evaluate_hybrid_highlighting.py`
- **Tests**: Compares predicted highlights against gold standard annotations
- **Metrics**: Precision, Recall, F1, IoU (Intersection over Union)
- **Gold Standard**: `subset_gold/df_annotations_sub.csv`
- **Test PDF**: `subset_gold/selected_pages_no_annotations.pdf`

### Current Performance Issues

1. **Indicator Extraction**: The system is extracting too many false positives (dimensions, measurements) instead of actual indicator codes (e.g., "3301", "0113", "0402")
2. **Table Parsing**: ADE returns HTML tables, needs proper parsing to extract indicator codes from table cells
3. **Legend Identification**: Need to better identify which table chunks contain the legend/keynote table vs. other tables

### Improvements Made

1. ✅ **Full Document Processing**: ADE now processes entire document once, then chunks are aligned to pages
2. ✅ **Page Alignment**: Chunks are correctly aligned to specific pages based on y-coordinates
3. ✅ **Stricter Indicator Patterns**: Focus on 4-digit codes (0113, 0401, 3301, 3219) and avoid dimensions
4. ✅ **Table Parsing**: Added HTML and markdown table parsing
5. ✅ **Dimension Filtering**: Added checks to skip numbers that are clearly dimensions (with units like ', ", S.F.)

### Next Steps for Improvement

1. **Better Legend Table Detection**: 
   - Identify which table contains the legend/keynote (look for headers like "KEYNOTE", "NUMBER", "DESCRIPTION")
   - Focus indicator extraction on legend tables only

2. **Indicator Code Validation**:
   - Extract only from tables that have a structure suggesting they're legends (e.g., NUMBER | DESCRIPTION format)
   - Prioritize extraction from cells that are clearly indicator codes (standalone 4-digit codes)

3. **Coordinate Verification**:
   - Ensure bounding boxes from Google OCR align properly with gold standard coordinates
   - Check if coordinate system conversion is correct

4. **Testing**:
   - Run evaluation after improvements
   - Analyze false positives and false negatives to refine extraction

## Usage

Run evaluation:
```bash
python evaluate_hybrid_highlighting.py \
  --pdf subset_gold/selected_pages_no_annotations.pdf \
  --gold subset_gold/df_annotations_sub.csv \
  --output subset_gold/evaluation_results.csv
```

View debug extraction:
```bash
python debug_extraction.py
```



























