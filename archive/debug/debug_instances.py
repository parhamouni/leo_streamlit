"""
Debug script to diagnose why indicators aren't found in figure areas.
Runs the same pipeline as app_ade.py but prints detailed diagnostics.
"""
import sys
import os
import re
import json
import fitz
import toml

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")

# Setup
sys.path.insert(0, os.path.dirname(__file__))
import utils_ade as ade

PDF_PATH = "LEO - MARKED.pdf"
ADE_KEY = secrets["LANDINGAI_API_KEY"]
GOOGLE_CLOUD_CONFIG = {
    "project_number": secrets["google_cloud"]["project_number"],
    "location": secrets["google_cloud"]["location"],
    "processor_id": secrets["google_cloud"]["processor_id"],
    "service_account_info": secrets["gcp_service_account"],
}
FENCE_KEYWORDS = [
    'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
    'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
    'adot specifications', 'mag specifications', 'rail', 'railing',
    'bollards', 'handrails', 'wall', 'cmu',
    'operator', 'davis', 'bacon', 'davis-bacon', 'davis – bacon',
    'buy america', 'american', 'dug out',
]

# Pick which page to debug (1-indexed)
TARGET_PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def main():
    print(f"=" * 80)
    print(f"DEBUGGING PAGE {TARGET_PAGE} of {PDF_PATH}")
    print(f"=" * 80)
    
    file_bytes = open(PDF_PATH, "rb").read()
    doc = fitz.open(PDF_PATH)
    page_idx = TARGET_PAGE - 1
    page = doc[page_idx]
    pdf_w, pdf_h = page.rect.width, page.rect.height
    print(f"\nPage dimensions: {pdf_w:.1f} x {pdf_h:.1f}")
    print(f"Page rotation: {page.rotation}")
    
    # Step 1: Native PDF text
    print(f"\n{'=' * 40}")
    print("STEP 1: Native PDF Text Extraction")
    print(f"{'=' * 40}")
    pdf_lines = ade.get_native_pdf_lines(page)
    print(f"Found {len(pdf_lines)} native PDF lines")
    
    # Show all native text that looks like indicators (digit.digit)
    indicator_pattern = re.compile(r'^\d+\.\d+$')
    native_indicators = [l for l in pdf_lines if indicator_pattern.match(l['text'].strip())]
    print(f"Native indicator-like tokens: {len(native_indicators)}")
    for t in native_indicators:
        print(f"  '{t['text']}' at ({t['x0']:.1f}, {t['y0']:.1f}) - ({t['x1']:.1f}, {t['y1']:.1f})")
    
    # Step 2: Google OCR
    print(f"\n{'=' * 40}")
    print("STEP 2: Google OCR")
    print(f"{'=' * 40}")
    single_page_pdf = ade.create_single_page_pdf(file_bytes, page_idx)
    ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, GOOGLE_CLOUD_CONFIG, pdf_w, pdf_h)
    print(f"Found {len(ocr_lines)} OCR items")
    
    ocr_indicators = [l for l in ocr_lines if indicator_pattern.match(l['text'].strip())]
    print(f"OCR indicator-like tokens: {len(ocr_indicators)}")
    for t in ocr_indicators:
        print(f"  '{t['text']}' at ({t['x0']:.1f}, {t['y0']:.1f}) - ({t['x1']:.1f}, {t['y1']:.1f}) source={t.get('source','?')}")
    
    # Also show OCR items that contain digits and dots (broader search)
    digit_dot = [l for l in ocr_lines if re.search(r'\d+\.\d+', l['text'])]
    if digit_dot:
        print(f"\nOCR items containing digit.digit patterns ({len(digit_dot)}):")
        for t in digit_dot[:30]:
            print(f"  '{t['text'][:60]}' at ({t['x0']:.1f}, {t['y0']:.1f})")
    
    # Step 3: ADE chunks
    print(f"\n{'=' * 40}")
    print("STEP 3: ADE Document Extraction")
    print(f"{'=' * 40}")
    ade_response = ade.ade_parse_document(single_page_pdf, ADE_KEY)
    if ade_response["success"]:
        chunks = ade.align_ade_chunks_to_page(ade_response, 0, pdf_w, pdf_h)
        print(f"Found {len(chunks)} ADE chunks")
        for c in chunks:
            print(f"  [{c['type']}] '{c['text'][:80]}...' bbox=({c['x0']:.1f}, {c['y0']:.1f})-({c['x1']:.1f}, {c['y1']:.1f})")
    else:
        print(f"ADE FAILED: {ade_response.get('error')}")
        chunks = []
    
    # Step 4: Segment chunks
    print(f"\n{'=' * 40}")
    print("STEP 4: Segment Chunks")
    print(f"{'=' * 40}")
    legend_chunks, figure_chunks = ade.segment_chunks(chunks)
    print(f"Legend chunks: {len(legend_chunks)}")
    for c in legend_chunks:
        print(f"  [{c['type']}] bbox=({c['x0']:.1f}, {c['y0']:.1f})-({c['x1']:.1f}, {c['y1']:.1f}) text='{c['text'][:60]}...'")
    print(f"Figure chunks: {len(figure_chunks)}")
    for c in figure_chunks:
        print(f"  [{c['type']}] bbox=({c['x0']:.1f}, {c['y0']:.1f})-({c['x1']:.1f}, {c['y1']:.1f}) text='{c['text'][:60]}...'")
    
    # Step 5: Build all_page_tokens (same as app_ade.py)
    print(f"\n{'=' * 40}")
    print("STEP 5: Build Token Pool (as app_ade.py does)")
    print(f"{'=' * 40}")
    rotation = page.rotation
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
    for w in page.get_text("words"):
        ox0, oy0, ox1, oy1 = w[0], w[1], w[2], w[3]
        nx0, ny0, nx1, ny1 = transform_for_rotation(ox0, oy0, ox1, oy1)
        all_page_tokens.append({
            "text": w[4],
            "x0": nx0, "y0": ny0,
            "x1": nx1, "y1": ny1
        })
    
    print(f"Native PDF word tokens: {len(all_page_tokens)}")
    native_ind = [t for t in all_page_tokens if indicator_pattern.match(t['text'].strip())]
    print(f"Native indicator-like word tokens: {len(native_ind)}")
    for t in native_ind:
        print(f"  '{t['text']}' at ({t['x0']:.1f}, {t['y0']:.1f}) - ({t['x1']:.1f}, {t['y1']:.1f})")
    
    # Step 6: Check which tokens fall inside figure chunks
    print(f"\n{'=' * 40}")
    print("STEP 6: Tokens Inside Figure Chunks")
    print(f"{'=' * 40}")
    
    for fi, fc in enumerate(figure_chunks):
        cx0, cy0, cx1, cy1 = fc['x0'], fc['y0'], fc['x1'], fc['y1']
        tokens_in = []
        for t in all_page_tokens:
            tx, ty = (t['x0'] + t['x1']) / 2, (t['y0'] + t['y1']) / 2
            if cx0 <= tx <= cx1 and cy0 <= ty <= cy1:
                tokens_in.append(t)
        
        ind_in = [t for t in tokens_in if indicator_pattern.match(t['text'].strip())]
        print(f"\nFigure chunk {fi}: ({cx0:.1f}, {cy0:.1f}) - ({cx1:.1f}, {cy1:.1f})")
        print(f"  Total tokens inside: {len(tokens_in)}")
        print(f"  Indicator-like tokens: {len(ind_in)}")
        for t in ind_in:
            print(f"    '{t['text']}' at ({t['x0']:.1f}, {t['y0']:.1f})")
        
        # Also show OCR tokens that fall inside this figure chunk
        ocr_in = []
        for t in ocr_lines:
            tx, ty = (t['x0'] + t.get('x1', t['x0'])) / 2, (t['y0'] + t.get('y1', t['y0'])) / 2
            if cx0 <= tx <= cx1 and cy0 <= ty <= cy1:
                ocr_in.append(t)
        
        ocr_ind_in = [t for t in ocr_in if indicator_pattern.match(t['text'].strip())]
        print(f"  OCR tokens inside: {len(ocr_in)}")
        print(f"  OCR indicator-like: {len(ocr_ind_in)}")
        for t in ocr_ind_in:
            print(f"    '{t['text']}' at ({t['x0']:.1f}, {t['y0']:.1f}) source={t.get('source','?')}")
    
    # Step 7: Run find_instances_in_figures (if we have definitions)
    print(f"\n{'=' * 40}")
    print("STEP 7: Legend Extraction + Instance Finding")
    print(f"{'=' * 40}")
    
    if legend_chunks:
        # We need an LLM for legend extraction — skip if not available
        # Instead, create mock definitions from the known indicators
        print("Creating mock definitions for known indicators...")
        mock_definitions = [
            {"indicator": "2.1", "text_element": "RAIL FENCE"},
            {"indicator": "2.7", "text_element": "FULL VIEW FENCE"},
            {"indicator": "2.8", "text_element": "PEDESTRIAN GATE"},
            {"indicator": "2.9", "text_element": "DOG PARK FENCE"},
            {"indicator": "2.10", "text_element": "DOG PARK GATE"},
            {"indicator": "2.11", "text_element": "6' PICKLEBALL FENCE"},
            {"indicator": "2.12", "text_element": "4' PICKLEBALL FENCE"},
            {"indicator": "2.13", "text_element": "MAINTENANCE GATE"},
        ]
        
        instances = ade.find_instances_in_figures(mock_definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines)
        print(f"\nInstances found: {len(instances)}")
        for inst in instances:
            print(f"  indicator='{inst['indicator']}' at ({inst['x0']:.1f}, {inst['y0']:.1f})")
        
        found = set(i['indicator'] for i in instances)
        expected = set(d['indicator'] for d in mock_definitions)
        missing = expected - found
        if missing:
            print(f"\n⚠️  MISSING indicators: {sorted(missing)}")
    else:
        print("No legend chunks found!")
    
    # Step 8: Generate highlighted image to visually verify
    print(f"\n{'=' * 40}")
    print("STEP 8: Generate Highlighted Image")
    print(f"{'=' * 40}")
    
    page_img_bytes = page.get_pixmap(dpi=150).tobytes("png")
    highlighted = ade.highlight_page_image(page_img_bytes, mock_definitions, instances, pdf_w, pdf_h)
    
    out_path = f"debug_page_{TARGET_PAGE}_highlighted.png"
    with open(out_path, "wb") as f:
        f.write(highlighted)
    print(f"Saved highlighted image to {out_path}")
    
    doc.close()
    print(f"\n{'=' * 80}")
    print("DEBUG COMPLETE")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
