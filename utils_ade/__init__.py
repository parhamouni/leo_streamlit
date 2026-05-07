"""utils_ade — package facade.

Historically utils_ade was a single 3,153-line file. It is now split into
focused submodules; this __init__.py re-exports every public symbol so
existing callers (`import utils_ade as ade; ade.<fn>(...)`) keep working
byte-for-byte unchanged.
"""

from __future__ import annotations

# Module-level constants & feature flags (canonical source: ade_api).
from .ade_api import (
    ADE_PARSE_ENDPOINT,
    GOOGLE_CLOUD_AVAILABLE,
    VECTOR_UTILS_AVAILABLE,
    documentai,
)

# Re-exposed common imports for legacy callers that grabbed them off utils_ade.*.
from .ade_api import (
    BytesIO,
    Dict,
    Image,
    ImageDraw,
    List,
    Optional,
    Tuple,
)

# Vector utilities re-exported (originally in utils_ade via star import).
try:
    from utils_vector import (
        VectorLine,
        calculate_total_length,
        detect_scale_with_vision,
        extract_layer_names,
        extract_lines_by_layers,
        extract_vector_lines,
        find_fence_run_from_indicator,
        find_lines_near_bbox,
        group_connected_lines,
        group_lines_by_layer,
        infer_scale_from_page,
    )
except ImportError:
    pass

# From ade_api
from .ade_api import (
    ade_parse_document,
    align_ade_chunks_to_page,
    segment_chunks,
)

# From ocr
from .ocr import (
    get_docai_client,
    run_google_ocr_blocks_multipage,
    run_google_ocr_blocks,
)

# From pdf
from .pdf import (
    get_native_pdf_lines,
    create_single_page_pdf,
    create_multi_page_pdf,
    create_single_page_pdf_from_path,
    create_multi_page_pdf_from_path,
    measure_page_bytes,
    safe_single_page_pdf_from_path,
    safe_multi_page_pdf_from_path,
    create_page_batches,
    create_page_batches_from_path,
)

# From instances
from .instances import (
    is_center_inside,
    find_best_bbox,
    find_instances_in_figures,
    find_instances_in_figures_fast,
)

# From keywords
from .keywords import (
    scan_page_for_keywords,
    scan_page_for_keywords_fast,
    fallback_fence_detection,
    fallback_fence_detection_fast,
)

# From highlight
from .highlight import (
    highlight_page_image,
    highlight_keyword_matches,
    debug_visualize_coordinates,
    highlight_fence_lines,
)

# From llm
from .llm import (
    llm_extract_fence_elements_batch,
    llm_extract_fence_elements,
    extract_legend_entries,
    extract_element_details,
    llm_classify_page,
    llm_classify_pages_batch,
    llm_identify_fence_layers,
    llm_match_layers_to_definitions,
    llm_suggest_filter_params,
)

# From measure
from .measure import (
    detect_dimension_lines,
    measure_fence_elements,
)

__all__ = [
    "ADE_PARSE_ENDPOINT",
    "BytesIO",
    "Dict",
    "GOOGLE_CLOUD_AVAILABLE",
    "Image",
    "ImageDraw",
    "List",
    "Optional",
    "Tuple",
    "VECTOR_UTILS_AVAILABLE",
    "VectorLine",
    "ade_parse_document",
    "align_ade_chunks_to_page",
    "calculate_total_length",
    "create_multi_page_pdf",
    "create_multi_page_pdf_from_path",
    "create_page_batches",
    "create_page_batches_from_path",
    "create_single_page_pdf",
    "create_single_page_pdf_from_path",
    "debug_visualize_coordinates",
    "detect_dimension_lines",
    "detect_scale_with_vision",
    "documentai",
    "extract_element_details",
    "extract_layer_names",
    "extract_legend_entries",
    "extract_lines_by_layers",
    "extract_vector_lines",
    "fallback_fence_detection",
    "fallback_fence_detection_fast",
    "find_best_bbox",
    "find_fence_run_from_indicator",
    "find_instances_in_figures",
    "find_instances_in_figures_fast",
    "find_lines_near_bbox",
    "get_docai_client",
    "get_native_pdf_lines",
    "group_connected_lines",
    "group_lines_by_layer",
    "highlight_fence_lines",
    "highlight_keyword_matches",
    "highlight_page_image",
    "infer_scale_from_page",
    "is_center_inside",
    "llm_classify_page",
    "llm_classify_pages_batch",
    "llm_extract_fence_elements",
    "llm_extract_fence_elements_batch",
    "llm_identify_fence_layers",
    "llm_match_layers_to_definitions",
    "llm_suggest_filter_params",
    "measure_fence_elements",
    "measure_page_bytes",
    "run_google_ocr_blocks",
    "run_google_ocr_blocks_multipage",
    "safe_multi_page_pdf_from_path",
    "safe_single_page_pdf_from_path",
    "scan_page_for_keywords",
    "scan_page_for_keywords_fast",
    "segment_chunks",
]
