# ADE-powered Fence Detector

This is an enhanced version of the Fence Detector app that uses LandingAI's Agentic Document Extraction (ADE) instead of Google Cloud Document AI for text extraction and element detection.

## Key Differences from Original App

### Text Extraction
- **Original**: PyMuPDF text layer → Google Cloud Document AI OCR (if sparse) → comprehensive_page_extractor
- **ADE Version**: ADE Parse API → extract text + element boxes → use directly for analysis and highlighting

### Highlighting
- **Original**: `get_fence_related_text_boxes()` with pdfplumber + keyword matching + DocAI OCR
- **ADE Version**: `get_ade_fence_boxes()` filters ADE elements by keywords + OpenAI signals, returns boxes directly

## Files

- `app_ade.py` - ADE-powered version of the main app
- `utils_ade.py` - ADE API wrapper functions and utilities
- `test_ade_integration.py` - Test script for ADE functionality
- `app.py` - Original app (unchanged for comparison)
- `utils.py` - Original utilities (unchanged for comparison)

## Setup

1. **Install dependencies** (same as original):
   ```bash
   pip install streamlit langchain-openai pymupdf pdfplumber requests
   ```

2. **Set up API keys** in your Streamlit secrets or environment:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your-openai-key"
   LANDINGAI_API_KEY = "your-landingai-key"
   ```

3. **Run the ADE-powered app**:
   ```bash
   streamlit run app_ade.py
   ```

4. **Run the original app for comparison**:
   ```bash
   streamlit run app.py
   ```

## Features

### ADE Integration
- **Layout-agnostic parsing**: Extracts text, tables, and form fields with precise coordinates
- **Built-in OCR**: No need for separate Google Cloud Document AI setup
- **Grounded output**: Page and coordinate-level references for each element
- **Zero Data Retention (ZDR)**: Optional mode for sensitive documents

### Enhanced Highlighting
- Uses ADE element bounding boxes for more precise highlighting
- Filters elements by fence keywords and LLM signals
- Maintains same visual appearance as original app

### Configuration
- **ADE API Key**: Loaded from secrets or environment variables
- **ZDR Mode**: Toggle for Zero Data Retention processing
- **Same UI/UX**: Identical interface for easy comparison

## API Usage

The ADE integration uses the LandingAI Parse API:

```python
# Parse a single page
ade_result = ade_parse_page(page_bytes, api_key, zdr=False)

# Extract text and elements
ade_text, ade_elements = extract_ade_text_and_elements(ade_result, page_width, page_height)

# Filter for fence-related elements
fence_boxes = get_ade_fence_boxes(ade_elements, fence_keywords, signals)
```

## Testing

Run the test script to verify ADE integration:

```bash
export LANDINGAI_API_KEY="your-key"
python test_ade_integration.py
```

## Comparison

To compare results between the original and ADE versions:

1. Upload the same PDF to both apps
2. Compare the text extraction quality
3. Compare the highlighting accuracy
4. Compare processing speed and reliability

## Notes

- The ADE version maintains the same OpenAI-based fence classification logic
- Caching is implemented for ADE results to improve performance
- Fallback to basic text extraction if ADE parsing fails
- Memory management and error handling are preserved from the original app

## Troubleshooting

- **ADE API errors**: Check your API key and network connectivity
- **Coordinate issues**: ADE coordinates are automatically converted to PDF points
- **Memory issues**: Same memory management as original app
- **Rate limits**: ADE calls use the same retry logic as OpenAI calls



























