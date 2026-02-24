"""Deep debug: investigate why pages 1, 2, 4, 5 underperform."""
import os, sys, toml, fitz, re
secrets = toml.load(".streamlit/secrets.toml")
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

import utils_ade as ade
from utils_vector import extract_vector_lines, infer_scale_from_text
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

llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=secrets["OPENAI_API_KEY"], timeout=180)
google_cloud_config = {
    "project_number": secrets["google_cloud"]["project_number"],
    "location": secrets["google_cloud"]["location"],
    "processor_id": secrets["google_cloud"]["processor_id"],
    "service_account_info": secrets["gcp_service_account"],
}

file_bytes = open(PDF_PATH, "rb").read()
doc = fitz.open(PDF_PATH)

# ===================== PAGE 1 DEEP DIVE =====================
print("=" * 80)
print("PAGE 1 DEEP DIVE: Why 0 instances despite 8 definitions?")
print("=" * 80)
page = doc[0]
pdf_w, pdf_h = page.rect.width, page.rect.height

single_pdf = ade.create_single_page_pdf(file_bytes, 0)
ocr_lines = ade.run_google_ocr_blocks(single_pdf, google_cloud_config, pdf_w, pdf_h)
ade_resp = ade.ade_parse_document(single_pdf, secrets["LANDINGAI_API_KEY"])
chunks = ade.align_ade_chunks_to_page(ade_resp, 0, pdf_w, pdf_h)
legend_chunks, figure_chunks = ade.segment_chunks(chunks)

definitions = ade.extract_legend_entries(
    legend_chunks=legend_chunks, pdf_lines=[], ocr_lines=ocr_lines,
    fence_keywords=FENCE_KEYWORDS, llm=llm
)

# What indicators are we looking for?
indicators = set()
for d in definitions:
    ind = d.get('indicator', '').strip()
    if ind:
        indicators.add(ind)
print(f"\nIndicators to find: {indicators}")
print(f"Figure chunks: {len(figure_chunks)}")
for i, fc in enumerate(figure_chunks):
    print(f"  Figure {i}: ({fc['x0']:.0f},{fc['y0']:.0f})-({fc['x1']:.0f},{fc['y1']:.0f})")

# Check OCR tokens inside figure areas
ocr_tokens = []
for line in ocr_lines:
    text = line.get('text', '').strip()
    if not text:
        continue
    for w in text.split():
        ocr_tokens.append({
            'text': w,
            'x0': line.get('x0', 0), 'y0': line.get('y0', 0),
            'x1': line.get('x1', 0), 'y1': line.get('y1', 0),
        })

fig_tokens_matching = []
for fc in figure_chunks:
    for t in ocr_tokens:
        tx = (t['x0'] + t['x1']) / 2
        ty = (t['y0'] + t['y1']) / 2
        if fc['x0'] <= tx <= fc['x1'] and fc['y0'] <= ty <= fc['y1']:
            clean = re.sub(r'[^\w]', '', t['text'])
            if t['text'] in indicators or clean in indicators or t['text'].strip('()[]') in indicators:
                fig_tokens_matching.append(t)

print(f"\nOCR tokens matching indicators in figure areas: {len(fig_tokens_matching)}")
for t in fig_tokens_matching[:20]:
    print(f"  '{t['text']}' at ({t['x0']:.0f},{t['y0']:.0f})")

# Also check: what ARE the indicators? Are they sensible?
print(f"\nDefinitions detail:")
for d in definitions:
    print(f"  indicator='{d.get('indicator','')}' desc='{d.get('description','')[:80]}'")

# ===================== PAGE 2 DEEP DIVE =====================
print("\n" + "=" * 80)
print("PAGE 2 DEEP DIVE: Why 0 fence lines despite 5 instances + scale?")
print("=" * 80)
page2 = doc[1]
pdf_w2, pdf_h2 = page2.rect.width, page2.rect.height
single_pdf2 = ade.create_single_page_pdf(file_bytes, 1)
ocr_lines2 = ade.run_google_ocr_blocks(single_pdf2, google_cloud_config, pdf_w2, pdf_h2)

# Check what scale text exists
ocr_text2 = "\n".join(l.get('text', '') for l in ocr_lines2)
scale = infer_scale_from_text(ocr_text2)
print(f"Scale from OCR: {scale}")

# Search for scale-related text in OCR
for line in ocr_lines2:
    t = line.get('text', '')
    if re.search(r'scale|1\s*["\']?\s*=', t, re.IGNORECASE):
        print(f"  Scale text: '{t}' at ({line.get('x0',0):.0f},{line.get('y0',0):.0f})")

# Check vector lines near instance positions
ade_resp2 = ade.ade_parse_document(single_pdf2, secrets["LANDINGAI_API_KEY"])
chunks2 = ade.align_ade_chunks_to_page(ade_resp2, 0, pdf_w2, pdf_h2)
legend2, figure2 = ade.segment_chunks(chunks2)
defs2 = ade.extract_legend_entries(legend2, [], ocr_lines2, FENCE_KEYWORDS, llm)

# Build OCR tokens
ocr_tokens2 = []
for line in ocr_lines2:
    text = line.get('text', '').strip()
    for w in text.split():
        ocr_tokens2.append({
            'text': w, 'x0': line['x0'], 'y0': line['y0'],
            'x1': line['x1'], 'y1': line['y1'],
        })
instances2 = ade.find_instances_in_figures(defs2, figure2, [], ocr_lines=ocr_lines2)
print(f"\nInstances: {len(instances2)}")
for inst in instances2:
    print(f"  '{inst['indicator']}' at ({inst['x0']:.0f},{inst['y0']:.0f})-({inst['x1']:.0f},{inst['y1']:.0f})")

# Check vector lines near these instances
all_vector = extract_vector_lines(page2)
print(f"\nTotal vector lines on page 2: {len(all_vector)}")

# Check lines near each instance
for inst in instances2:
    ix = (inst['x0'] + inst['x1']) / 2
    iy = (inst['y0'] + inst['y1']) / 2
    margin = 80
    nearby = [l for l in all_vector if (
        min(l.start[0], l.end[0]) - margin <= ix <= max(l.start[0], l.end[0]) + margin and
        min(l.start[1], l.end[1]) - margin <= iy <= max(l.start[1], l.end[1]) + margin
    )]
    long_nearby = [l for l in nearby if l.length_pts > 50]
    print(f"  Near '{inst['indicator']}' ({ix:.0f},{iy:.0f}): {len(nearby)} lines, {len(long_nearby)} > 50pts")

# Check figure area constraint
if figure2:
    fc = figure2[0]
    fig_lines = [l for l in all_vector if (
        min(l.start[0], l.end[0]) >= fc['x0'] and max(l.start[0], l.end[0]) <= fc['x1'] and
        min(l.start[1], l.end[1]) >= fc['y0'] and max(l.start[1], l.end[1]) <= fc['y1']
    )]
    long_fig = [l for l in fig_lines if l.length_pts > 50]
    print(f"\n  Lines inside figure area: {len(fig_lines)}, > 50pts: {len(long_fig)}")
    # What lengths do the long lines have?
    if long_fig:
        lengths = sorted([l.length_pts for l in long_fig], reverse=True)[:20]
        print(f"  Top 20 line lengths (pts): {[f'{l:.0f}' for l in lengths]}")

# ===================== PAGES 4-5: Check legend text =====================
print("\n" + "=" * 80)
print("PAGES 4-5: What text is in legend chunks?")
print("=" * 80)
for pg_idx in [3, 4]:
    page_n = doc[pg_idx]
    pdf_wn, pdf_hn = page_n.rect.width, page_n.rect.height
    single_n = ade.create_single_page_pdf(file_bytes, pg_idx)
    ocr_n = ade.run_google_ocr_blocks(single_n, google_cloud_config, pdf_wn, pdf_hn)
    ade_n = ade.ade_parse_document(single_n, secrets["LANDINGAI_API_KEY"])
    chunks_n = ade.align_ade_chunks_to_page(ade_n, 0, pdf_wn, pdf_hn)
    legend_n, figure_n = ade.segment_chunks(chunks_n)
    
    print(f"\n--- Page {pg_idx+1} ---")
    print(f"Legend chunks: {len(legend_n)}, Figure chunks: {len(figure_n)}")
    ocr_text_n = "\n".join(l.get('text', '') for l in ocr_n)
    
    # Check for fence keywords in OCR text
    found_kw = [kw for kw in FENCE_KEYWORDS if kw.lower() in ocr_text_n.lower()]
    print(f"Fence keywords in OCR: {found_kw}")
    
    # Print legend chunk texts (first 200 chars each)
    for i, lc in enumerate(legend_n):
        txt = lc.get('text', '')[:200]
        print(f"  Legend {i}: '{txt}'")
    
    # Print scale info
    scale_n = infer_scale_from_text(ocr_text_n)
    print(f"  Scale: {scale_n}")
    
    # Print any lines containing fence-related text
    for line in ocr_n:
        t = line.get('text', '').lower()
        if any(kw in t for kw in ['fence', 'gate', 'wall', 'rail', 'chain', 'mesh', 'scale']):
            print(f"  OCR line: '{line['text'][:100]}' at ({line.get('x0',0):.0f},{line.get('y0',0):.0f})")

doc.close()
print("\nDONE")
