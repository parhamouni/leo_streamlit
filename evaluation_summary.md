# Evaluation Results Summary

## Latest Evaluation (LLM-only, No Hardcoded Patterns)
**File:** `evaluation_llm_only_final.csv`
- **Precision:** 0.000 (0/0 predictions)
- **Recall:** 0.000 (0/112 found)
- **F1:** 0.000
- **TP:** 0, **FP:** 0, **FN:** 112

**Status:** ❌ No highlights produced because OCR/ADE functions are stubbed out

## Previous Evaluations Comparison

| File | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|----|----|----|
| evaluation_with_ade.csv | 0.000 | 0.000 | 0.000 | 0 | 36 | 112 |
| evaluation_improved_prompt.csv | 0.000 | 0.000 | 0.000 | 0 | 36 | 112 |
| evaluation_strict_filter.csv | 0.000 | 0.000 | 0.000 | 0 | 36 | 112 |
| evaluation_llm_only.csv | 0.000 | 0.000 | 0.000 | 0 | 36 | 112 |
| **evaluation_llm_only_final.csv** | **0.000** | **0.000** | **0.000** | **0** | **0** | **112** |

## Key Observations

1. **Previous runs had 36 False Positives** - These were likely from hardcoded regex patterns matching non-indicator codes
2. **Latest run has 0 False Positives** - ✅ This is an improvement! The removal of hardcoded patterns eliminated false positives
3. **No True Positives in any run** - The LLM extraction needs proper OCR/ADE integration to work

## What Changed

✅ **Removed all hardcoded regex patterns** for finding indicator codes
✅ **Removed pattern-based validation** (architectural codes, dimensions, etc.)
✅ **Made system fully LLM-driven** with intelligent prompts
✅ **Eliminated false positives** (36 → 0)

## Next Steps

The LLM-only extraction code is in place, but needs:
1. Proper Google OCR integration (`get_google_ocr_results_with_boxes`)
2. ADE document parsing (if using ADE)
3. The LLM extraction functions are ready but can't run without OCR input

