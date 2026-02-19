"""Debug script: run the processing pipeline on a PDF without Streamlit UI."""
import os
import sys
import toml
import fitz
import json

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

import utils_ade as ade
from utils_vector import extract_vector_lines, infer_scale_from_page
from langchain_openai import ChatOpenAI

PDF_PATH = "subset_gold/_3_ARCHITECTURAL_BASEBALL_fence_highlights_fence_highlights.pdf"
FENCE_KEYWORDS = [
    'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
    'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
    'adot specifications', 'mag specifications', 'rail', 'railing',
    'bollards', 'handrails', 'wall', 'cmu',
    'operator', 'davis', 'bacon', 'davis-bacon', 'davis – bacon',
    'buy america', 'american', 'dug out',
]

# Setup
llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=secrets["OPENAI_API_KEY"], timeout=180)
ade_key = secrets["LANDINGAI_API_KEY"]

google_cloud_config = {
    "project_number": secrets["google_cloud"]["project_number"],
    "location": secrets["google_cloud"]["location"],
    "processor_id": secrets["google_cloud"]["processor_id"],
    "service_account_info": secrets["gcp_service_account"],
}

file_bytes = open(PDF_PATH, "rb").read()
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
print(f"=== PDF: {PDF_PATH} | {total_pages} pages ===\n")

for page_idx in range(total_pages):
    page_num = page_idx + 1
    page = doc[page_idx]
    pdf_width, pdf_height = page.rect.width, page.rect.height
    rotation = page.rotation
    print(f"\n{'='*80}")
    print(f"PAGE {page_num} | Size: {pdf_width:.0f}x{pdf_height:.0f} | Rotation: {rotation}°")
    print(f"{'='*80}")

    # STEP 1: Text extraction
    pdf_lines = ade.get_native_pdf_lines(page)
    print(f"  PDF native lines: {len(pdf_lines)}")
    
    single_page_pdf = ade.create_single_page_pdf(file_bytes, page_idx)
    ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, pdf_width, pdf_height)
    print(f"  OCR lines: {len(ocr_lines)}")

    # STEP 2: Pre-filter
    prefilter_result = ade.fallback_fence_detection(
        pdf_lines=pdf_lines,
        ocr_lines=ocr_lines,
        fence_keywords=FENCE_KEYWORDS,
        llm=llm,
        use_llm_confirmation=True
    )
    fence_found = prefilter_result["fence_found"]
    method = prefilter_result.get("method", "none")
    print(f"  Pre-filter: fence_found={fence_found}, method={method}")
    if prefilter_result.get("matched_keywords"):
        print(f"    Keywords matched: {prefilter_result['matched_keywords']}")

    if not fence_found:
        print(f"  >> SKIPPED (not fence-related)")
        continue

    # STEP 3: ADE parsing
    ade_response = ade.ade_parse_document(single_page_pdf, ade_key)
    if not ade_response["success"]:
        print(f"  >> ADE FAILED: {ade_response['error']}")
        continue

    chunks = ade.align_ade_chunks_to_page(ade_response, 0, pdf_width, pdf_height)
    legend_chunks, figure_chunks = ade.segment_chunks(chunks)
    print(f"  ADE chunks: {len(chunks)} total, {len(legend_chunks)} legend, {len(figure_chunks)} figure")

    # STEP 4: Extract definitions
    definitions = []
    if legend_chunks:
        definitions = ade.extract_legend_entries(
            legend_chunks=legend_chunks,
            pdf_lines=pdf_lines,
            ocr_lines=ocr_lines,
            fence_keywords=FENCE_KEYWORDS,
            llm=llm,
            figure_chunks=figure_chunks
        )
    print(f"  Definitions found: {len(definitions)}")
    for d in definitions:
        print(f"    - indicator='{d.get('indicator','')}' keyword='{d.get('keyword','')}' desc='{d.get('description','')[:60]}'")

    # STEP 5: Find instances
    native_words = page.get_text("words")
    mediabox_w = page.mediabox.width
    mediabox_h = page.mediabox.height
    
    def transform_for_rotation(x0, y0, x1, y1):
        if rotation == 0:
            return x0, y0, x1, y1
        elif rotation == 90:
            return mediabox_h - y1, x0, mediabox_h - y0, x1
        elif rotation == 180:
            return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
        elif rotation == 270:
            return y0, mediabox_w - x1, y1, mediabox_w - x0
        return x0, y0, x1, y1

    all_page_tokens = []
    for w in native_words:
        nx0, ny0, nx1, ny1 = transform_for_rotation(w[0], w[1], w[2], w[3])
        all_page_tokens.append({"text": w[4], "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1})

    instances = []
    if definitions and figure_chunks:
        instances = ade.find_instances_in_figures(definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines)
    print(f"  Instances found: {len(instances)}")
    for inst in instances:
        print(f"    - indicator='{inst.get('indicator','')}' at ({inst.get('x0',0):.0f},{inst.get('y0',0):.0f})")

    # STEP 6: Measurement
    if definitions or instances:
        try:
            ocr_full_text = "\n".join(line.get('text', '') for line in ocr_lines) if ocr_lines else None
            measurement_result = ade.measure_fence_elements(
                page, definitions, instances,
                figure_chunks=figure_chunks,
                llm=llm,
                ocr_text=ocr_full_text
            )
            mmethod = measurement_result.get('measurement_method', 'none')
            page_info = measurement_result.get('page_info', {})
            scale_factor = page_info.get('scale_factor', 1.0)
            scale_detected = page_info.get('scale_detected', False)
            fence_layers = measurement_result.get('fence_layers', [])
            all_fence_lines = measurement_result.get('all_fence_lines', [])
            layer_to_cat = measurement_result.get('layer_to_category', {})
            prox_totals = measurement_result.get('proximity_totals', {})
            indicator_meas = measurement_result.get('indicator_measurements', {})

            print(f"\n  MEASUREMENT RESULTS:")
            print(f"    Method: {mmethod}")
            print(f"    Scale detected: {scale_detected}, scale_factor: {scale_factor}")
            print(f"    Fence layers: {fence_layers}")
            print(f"    Layer→Category mapping: {layer_to_cat}")
            print(f"    Total fence lines: {len(all_fence_lines)}")
            print(f"    Proximity totals: segments={prox_totals.get('total_segments',0)}, "
                  f"length_pts={prox_totals.get('total_length_pts',0):.0f}, "
                  f"length_ft={prox_totals.get('total_length_feet',0):.1f}")
            
            if indicator_meas:
                print(f"    Per-indicator breakdown:")
                for ind, stats in indicator_meas.items():
                    print(f"      {ind}: {stats.get('total_segments',0)} segs, "
                          f"{stats.get('total_length_feet',0):.1f} ft")
            
            # Show scale detection details
            print(f"\n  SCALE DETECTION DETAILS:")
            print(f"    scale_text: {page_info.get('scale_text', 'N/A')}")
            print(f"    scale_ratio: {page_info.get('scale_ratio', 'N/A')}")
            
            # Also try standalone scale inference
            vector_lines = extract_vector_lines(page)
            print(f"    Vector lines on page: {len(vector_lines)}")
            scale_info = infer_scale_from_page(page)
            print(f"    infer_scale_from_page: {scale_info}")
            
        except Exception as e:
            import traceback
            print(f"  >> MEASUREMENT ERROR: {e}")
            traceback.print_exc()
    else:
        print(f"  >> No definitions or instances, skipping measurement")
        # Still check scale
        scale_info = infer_scale_from_page(page)
        print(f"    infer_scale_from_page: {scale_info}")

doc.close()
print(f"\n{'='*80}")
print("DONE")
