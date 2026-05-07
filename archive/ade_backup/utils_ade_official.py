"""
utils_ade_official.py - Official ADE utility functions

This file now imports from the unified utils_ade.py module for backward compatibility.
All functions are available through this module.
"""

# Import all functions from the unified utils_ade module
from utils_ade import (
    # Core ADE API Functions
    ade_parse_document_official,
    align_ade_chunks_to_page,
    filter_ocr_by_ade_regions,
    
    # OCR and PDF Text Extraction
    get_google_ocr_results_with_boxes,
    get_pdf_native_text,
    
    # Text Processing and Matching
    group_items_by_line,
    combine_bbox,
    make_highlight,
    match_text_in_lines,
    find_text_in_ocr_and_pdf,
    
    # LLM-based Keyword/Indicator Extraction
    extract_keywords_and_indicators_llm,
    extract_keywords_and_indicators_from_chunks,
    
    # Instance Detection
    is_bbox_inside,
    find_indicators_in_figures,
    
    # Layout Analysis
    analyze_page_layout_for_indicators,
    
    # Visualization
    visualize_page_results,
    convert_pdf_page_to_image,
    
    # PDF Utilities
    get_page_dimensions,
    create_single_page_pdf,
    
    # Compatibility Stubs
    extract_legend_keywords_and_indicators,
    
    # Constants
    ADE_PARSE_ENDPOINT,
)

# Re-export everything for backward compatibility
__all__ = [
    'ade_parse_document_official',
    'align_ade_chunks_to_page',
    'filter_ocr_by_ade_regions',
    'get_google_ocr_results_with_boxes',
    'get_pdf_native_text',
    'group_items_by_line',
    'combine_bbox',
    'make_highlight',
    'match_text_in_lines',
    'find_text_in_ocr_and_pdf',
    'extract_keywords_and_indicators_llm',
    'extract_keywords_and_indicators_from_chunks',
    'is_bbox_inside',
    'find_indicators_in_figures',
    'analyze_page_layout_for_indicators',
    'visualize_page_results',
    'convert_pdf_page_to_image',
    'get_page_dimensions',
    'create_single_page_pdf',
    'extract_legend_keywords_and_indicators',
    'ADE_PARSE_ENDPOINT',
]
