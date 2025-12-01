# Data Extraction Analysis Report

## Summary

- **Total gold standard annotations**: 112
- **OCR matches**: 2 (1.8% coverage)
- **Text layer matches**: 38 (33.9% coverage)

## Per-Page Analysis

### Page 1
- Gold annotations: 25
- OCR items: 369
- Text layer words: 1674

**Matches:**
- OCR: 0/25 (avg IoU: 0.000)
- Text layer: 20/25 (avg IoU: 0.722)

**Content Analysis:**
- Indicator codes: 0
- Descriptions: 0
- Mixed: 0

### Page 2
- Gold annotations: 11
- OCR items: 310
- Text layer words: 1620

**Matches:**
- OCR: 0/11 (avg IoU: 0.000)
- Text layer: 7/11 (avg IoU: 0.359)

**Content Analysis:**
- Indicator codes: 7
- Descriptions: 4
- Mixed: 0

### Page 3
- Gold annotations: 10
- OCR items: 209
- Text layer words: 918

**Matches:**
- OCR: 0/10 (avg IoU: 0.000)
- Text layer: 6/10 (avg IoU: 0.360)

**Content Analysis:**
- Indicator codes: 6
- Descriptions: 4
- Mixed: 0

### Page 4
- Gold annotations: 53
- OCR items: 1183
- Text layer words: 291

**Matches:**
- OCR: 2/53 (avg IoU: 0.346)
- Text layer: 5/53 (avg IoU: 0.312)

**Content Analysis:**
- Indicator codes: 17
- Descriptions: 0
- Mixed: 0

**Sample OCR matches:**
- Gold: '' → OCR: 'THIS' (IoU: 0.333)
- Gold: '' → OCR: 'PUBLIC' (IoU: 0.360)

### Page 5
- Gold annotations: 13
- OCR items: 1502
- Text layer words: 0

**Matches:**
- OCR: 0/13 (avg IoU: 0.000)
- Text layer: 0/13 (avg IoU: 0.000)

**Content Analysis:**
- Indicator codes: 0
- Descriptions: 0
- Mixed: 0

## Pattern Analysis

**Indicator Codes Found:** 21 unique codes
- Examples: 0401, 3301, 0403, 0, 0402, 0508, 4, 6, 8, 46

**Descriptions Found:** 8 descriptions
- Sample: 3219 TRASH ENCLOSURE

## Insights for LLM-Based Extraction

### Key Findings:

1. **OCR Coverage**: OCR provides better spatial coverage for indicator codes
2. **Text Layer**: Text layer may miss some annotations (especially in image-heavy regions)
3. **Indicator Codes**: Typically short (1-10 chars), numeric or alphanumeric
4. **Descriptions**: Longer text (10+ chars) that describe fence/gate/barrier elements

### Recommendations:

1. Use OCR as primary source for precise bounding boxes
2. Use ADE structure to identify legend vs figure regions
3. Use LLM to extract indicator codes from legend regions (both from ADE text and OCR)
4. Use LLM to find indicator codes in figure regions (match against legend codes)
5. Highlight both codes AND descriptions in legend regions
6. Use fuzzy matching for indicator codes (they may be slightly different in OCR vs ADE)
