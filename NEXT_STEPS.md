# Next Steps for Fence Indicator Detection

## Current Checkpoint: f9c243a
This is the stable checkpoint where highlighting works correctly for most pages.

## Problem on Page 4
Page 4 has indicators in the figure area that are not being detected. Investigation revealed:

1. **OCR doesn't pick up all indicators** - The full-page OCR misses some small circled numbers in the drawing area
2. **PDF native text has the indicators** - But they may be outside the detected figure chunk boundary

## Attempted Solutions (not yet working)

### 1. Expand Figure Boundaries
- Tried expanding figure chunk boundaries by 100-150 points
- Problem: This also included text from legend areas, causing false positives

### 2. Filter "True Legend" Chunks
- Tried to only exclude chunks with "legend", "keynote" keywords
- Problem: Over-complicated and still had issues with boundary detection

### 3. OCR Figure Regions Separately (recommended approach)
- Create `ocr_figure_regions()` function in `utils_ade.py`
- Crop each figure region and send to OCR at higher resolution (5x zoom, PNG format)
- Add resulting tokens to `all_page_tokens` before instance finding
- **Error encountered**: `Unknown field for Block: paragraphs` - need to check Google Document AI API response structure

## Implementation Notes for Figure Region OCR

```python
def ocr_figure_regions(page_bytes, figure_chunks, google_cloud_config, pdf_width, pdf_height):
    """
    1. For each figure chunk:
       - Crop the region with margin
       - Render at high zoom (5x)
       - Use PNG format (better for line drawings)
       - Send to Google Document AI
       - Convert normalized coords back to full-page PDF coords
    
    2. Return list of tokens with PDF coordinates
    
    3. In app_ade.py, add these tokens to all_page_tokens before find_instances_in_figures()
    """
```

## Key Files
- `utils_ade.py` - Add `ocr_figure_regions()` function after `run_google_ocr_blocks()`
- `app_ade.py` - Call `ocr_figure_regions()` and extend `all_page_tokens`
- `debug_text_sources.ipynb` - Notebook for visualizing PDF vs OCR text

## Debug Notebooks Available
- `debug_coordinates.ipynb` - Visualize coordinate transforms
- `debug_text_sources.ipynb` - Compare PDF native text vs OCR text on page 4
