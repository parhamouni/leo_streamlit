# app.py
import streamlit as st
from utils import analyze_page, get_fence_related_text_boxes, UnrecoverableRateLimitError, time_it 
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
import io
import time 
import uuid 
import hashlib # For hashing PDF bytes
import json

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)
HIGHLIGHT_WIDTH_UI = 2.0
HIGHLIGHT_COLOR_PDF = (0, 0.9, 0)
HIGHLIGHT_WIDTH_PDF = 2.0
DISPLAY_IMAGE_DPI = 96  

st.set_page_config(page_title="Fence Detector", layout="wide")
st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True) 
st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id

def initialize_session_state(session_id_val):
    print(f"SESSION {session_id_val} LOG: Initializing/checking session state.")
    default_state = {
        'session_id': session_id_val, 
        'fence_pages': [], 'non_fence_pages': [], 'total_pages_processed_count': 0,
        'doc_total_pages': 0, 'processing_complete': False, 'analysis_halted_due_to_error': False,
        'fence_keywords_app': ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link'],
        'run_analysis_triggered': False, 'uploaded_pdf_name': None, 'original_pdf_bytes': None,
        'current_pdf_hash': None, # NEW: To store hash of current PDF
        'highlighted_pdf_bytes_for_download': None, 'last_uploaded_file_id': None,
        'selected_model_for_analysis': "gpt-4o" 
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else \
                                    dict(value) if isinstance(value, dict) else \
                                    value
        elif key == 'session_id' and st.session_state.session_id != session_id_val :
             st.session_state.session_id = session_id_val

current_session_id = get_session_id() 
initialize_session_state(current_session_id) 

# --- Sidebar (Keep as is) ---
with st.sidebar:
    # ... (Your existing sidebar code) ...
    st.header("⚙️ Configuration") # Copied for completeness
    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
        if openai_key_input: openai_key = openai_key_input; st.rerun()
    
    # Load Google Cloud credentials for comprehensive text extraction
    google_cloud_config = None
    try:
        if "google_cloud" in st.secrets and "gcp_service_account" in st.secrets:
            google_cloud_config = {
                "project_number": st.secrets["google_cloud"]["project_number"],
                "location": st.secrets["google_cloud"]["location"], 
                "processor_id": st.secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(st.secrets["gcp_service_account"])
            }
            print(f"SESSION {current_session_id} LOG: Google Cloud config loaded from secrets")
    except Exception as e:
        print(f"SESSION {current_session_id} WARNING: Could not load Google Cloud config: {e}")
        google_cloud_config = None
    
    # Test comprehensive extraction if available
    if google_cloud_config:
        from utils import test_comprehensive_extraction
        test_comprehensive_extraction(google_cloud_config)
    st.subheader("Model Selection")
    model_options = {
        "gpt-4o (128k context, recommended)": "gpt-4o",
        "gpt-4-turbo (128k context)": "gpt-4-turbo",
        "gpt-4 (8k/32k context)": "gpt-4",
        "gpt-3.5-turbo (16k context, fastest)": "gpt-3.5-turbo"}
    current_model_val = st.session_state.selected_model_for_analysis
    if current_model_val not in model_options.values(): current_model_val = list(model_options.values())[0]
    st.session_state.selected_model_for_analysis = current_model_val
    default_model_idx = list(model_options.values()).index(current_model_val)
    selected_label = st.radio("Select LLM:", list(model_options.keys()), key="model_selector_radio", index=default_model_idx)
    st.session_state.selected_model_for_analysis = model_options[selected_label]
    st.info(f"Using: **{st.session_state.selected_model_for_analysis}**.")
 
    highlight_fence_text_app = st.toggle("🔍 Highlight text & indicators", value=True, key="highlight_toggle")
    st.subheader("Fence Keywords")
    if 'fence_keywords_app' not in st.session_state: st.session_state.fence_keywords_app = ['fence']
    custom_keywords_str = st.text_area("Custom keywords (one per line):", "\n".join(st.session_state.fence_keywords_app), height=150, key="kw_text_area")
    if st.button("Update Keywords", key="update_kw_btn"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()
    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


llm_analysis_instance = None
if openai_key:
    try:
        print(f"SESSION {current_session_id} LOG: Initializing LLM instance.")
        llm_analysis_instance = ChatOpenAI(model=st.session_state.selected_model_for_analysis, temperature=0, openai_api_key=openai_key, timeout=180, max_retries=2)
        print(f"SESSION {current_session_id} LOG: LLM instance initialized.")
    except Exception as e: st.error(f"LLM Init Error: {e}"); openai_key = None; print(f"SESSION {current_session_id} ERROR: LLM Init Error: {e}")


def get_image_download_link_html(img_bytes, filename, text):
    """
    Return a download <a> tag using a data URL with the correct MIME type.
    If the bytes are JPEG but the filename ends with .png, rewrite to .jpg
    to avoid mismatches. Call sites do not need to change.
    """
    # MIME sniffing
    mime = "application/octet-stream"
    if img_bytes[:2] == b"\xff\xd8":
        mime = "image/jpeg"
        if filename.lower().endswith(".png"):
            filename = filename[:-4] + ".jpg"
    elif img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"

    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" class="download-button">{text}</a>'

@time_it
@st.cache_data(ttl=900, show_spinner=False, max_entries=60)  # 15 min, at most 60 entries
def _generate_display_images_for_page_cached(page_idx,
                                            pdf_hash_for_cache_key,
                                            fence_text_boxes_details_tuple,
                                            ui_color, ui_width, display_dpi,
                                            session_id_for_log="N/A_CACHE"):
    """
    Memory-aware preview renderer:
      - Renders JPEG previews (much smaller than PNG) at a modest DPI.
      - Cache limited to 60 entries with short TTL to keep RSS low.
      - The 3rd arg should be a tuple of (x0,y0,x1,y1) rects (geometry-only).
        For backward-compatibility, if items look like dict-items we try to
        reconstruct x0..y1 from them.
    """
    if 'original_pdf_bytes' not in st.session_state or st.session_state.original_pdf_bytes is None:
        print(f"SESSION {session_id_for_log} ERROR (_cached): original_pdf_bytes missing for hash {pdf_hash_for_cache_key}")
        return None, None

    # Parse rectangles defensively
    rects = []
    try:
        for it in (fence_text_boxes_details_tuple or ()):
            # Already a rect tuple?
            if isinstance(it, (tuple, list)) and len(it) == 4 and all(isinstance(v, (int, float)) for v in it):
                x0, y0, x1, y1 = map(float, it)
                rects.append((x0, y0, x1, y1))
            else:
                # Possibly tuple of dict-items
                try:
                    d = dict(it)
                    x0 = float(d.get('x0')); y0 = float(d.get('y0')); x1 = float(d.get('x1')); y1 = float(d.get('y1'))
                    rects.append((x0, y0, x1, y1))
                except Exception:
                    continue
    except Exception as e:
        print(f"SESSION {session_id_for_log} WARNING: could not parse rects for cache key: {e}")

    current_pdf_bytes = st.session_state.original_pdf_bytes
    original_image_bytes, highlighted_image_bytes = None, None

    func_call_id = str(uuid.uuid4())[:4]
    print(f"SESSION {session_id_for_log} CACHE_CALL ({func_call_id}): _generate_display_images_for_page_cached "
          f"pg={page_idx}, hash={pdf_hash_for_cache_key[:8]}…, rects={len(rects)}")

    t0 = time.time()
    try:
        # Render original preview (JPEG @ display_dpi)
        with fitz.open(stream=io.BytesIO(current_pdf_bytes), filetype="pdf") as doc_orig:
            page_orig = doc_orig.load_page(page_idx)
            pix_orig = page_orig.get_pixmap(dpi=display_dpi)
            original_image_bytes = pix_orig.tobytes("jpg", jpg_quality=70)
            del pix_orig

        # Render highlighted preview only if we actually have rects
        if rects:
            with fitz.open(stream=io.BytesIO(current_pdf_bytes), filetype="pdf") as doc_hl:
                page_hl = doc_hl.load_page(page_idx)
                derot_matrix = page_hl.derotation_matrix
                for (x0, y0, x1, y1) in rects:
                    r = fitz.Rect(x0, y0, x1, y1)
                    fr = r * derot_matrix if page_hl.rotation != 0 else r
                    fr.normalize()
                    if not fr.is_empty and fr.is_valid:
                        page_hl.draw_rect(fr, color=ui_color, width=ui_width, overlay=True)
                pix_hl = page_hl.get_pixmap(dpi=display_dpi)
                highlighted_image_bytes = pix_hl.tobytes("jpg", jpg_quality=70)
                del pix_hl

    except Exception as e:
        print(f"SESSION {session_id_for_log} ERROR ({func_call_id}) _generate_display_images_for_page_cached pg {page_idx}: {e}")

    print(f"SESSION {session_id_for_log} CACHE_CALL_RENDER_TIME ({func_call_id}): "
          f"pg {page_idx} took {time.time() - t0:.4f}s.")
    return original_image_bytes, highlighted_image_bytes


def generate_display_images_for_page_wrapper(page_result_data, session_id):
    """
    Wraps the cached renderer. Converts the page's box list to a geometry-only
    tuple so the cache key stays small and stable.
    """
    page_idx = page_result_data.get('page_index_in_original_doc')
    pdf_hash = st.session_state.get('current_pdf_hash')

    if page_idx is None or pdf_hash is None:
        print(f"SESSION {session_id} WARNING (wrapper): Missing page_idx or pdf_hash.")
        return None, None

    boxes_details = page_result_data.get('fence_text_boxes_details', []) or []
    # Geometry-only cache signature: ((x0,y0,x1,y1), ...)
    box_rects_tuple = tuple(
        (float(b.get('x0', 0.0)), float(b.get('y0', 0.0)),
         float(b.get('x1', 0.0)), float(b.get('y1', 0.0)))
        for b in boxes_details
    )

    return _generate_display_images_for_page_cached(
        page_idx,
        pdf_hash,
        box_rects_tuple,           # <— small, avoids text in cache key
        HIGHLIGHT_COLOR_UI,
        HIGHLIGHT_WIDTH_UI,
        DISPLAY_IMAGE_DPI,
        session_id
    )

# generate_combined_highlighted_pdf (Keep as is from previous version - it doesn't use this image cache)
# (Ensure it's present in your file)
def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base, session_id):
    # ... (Same as previous full app.py) ...
    print(f"SESSION {session_id} LOG: Generating combined highlighted PDF.")
    if not fence_pages_results_list or not original_pdf_bytes: return None, "No data for PDF."
    output_doc = fitz.open(); input_doc = None
    try: input_doc = fitz.open(stream=io.BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        print(f"SESSION {session_id} ERROR: Opening original PDF for combined: {e}")
        if output_doc: output_doc.close(); return None, f"Error opening original PDF: {e}"
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        if page_idx is None: continue
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            if res_data.get('highlight_fence_text_app_setting', True) and res_data.get('fence_text_boxes_details'):
                derot_matrix = page_out.derotation_matrix
                for box in res_data['fence_text_boxes_details']:
                    r = fitz.Rect(box['x0'], box['y0'], box['x1'], box['y1'])
                    final_r = r * derot_matrix if page_out.rotation != 0 else r; final_r.normalize()
                    if not final_r.is_empty and final_r.is_valid:
                        try: page_out.draw_rect(final_r, color=HIGHLIGHT_COLOR_PDF, width=HIGHLIGHT_WIDTH_PDF, overlay=True)
                        except Exception as e_dr: print(f"SESSION {session_id} Err draw PDF pg {page_idx}: {e_dr}")
        except Exception as e_pi: print(f"SESSION {session_id} Err process pg {page_idx} for PDF: {e_pi}")
    pdf_bytes, fname = None, "error.pdf"
    if len(output_doc) > 0:
        try:
            pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
            base, ext = os.path.splitext(uploaded_pdf_name_base); fname = f"{base}_fence_highlights{ext}"
        except Exception as e_s: print(f"SESSION {session_id} Err PDF tobytes: {e_s}"); fname=f"err_save_{uploaded_pdf_name_base}.pdf"
    if input_doc: input_doc.close()
    if output_doc: output_doc.close()
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)

def compute_doc_legend_and_refs_compact(
    pdf_bytes: bytes,
    llm,
    google_cloud_config=None,
    max_pages: int = 60,
    char_budget_per_page: int = 2400,
    max_workers: int = 4,
):
    """
    Low-memory, document-level pass:
      1) Discover legend-like items per page (LLM).
      2) Merge to a canonical legend index (LLM).
      3) Detect references to those identifiers on pages (LLM).

    Returns:
      {
        "id_list": ["F1", "F-2A", "0113", ...],
        "page_refs": { 3: [{"identifier":"F1","confidence":0.8,"evidence":"F1 CHAIN LINK..."}], ... },
        "index_compact": [ {"identifier":"F1","title":"8' CL FENCE","short_description":"...","definition_pages":[1,2]}, ... ]
      }

    Notes:
      - Uses only the PDF text layer for speed/memory; this is usually enough to
        build a useful identifier pool. You can extend to OCR later if needed.
      - Limits pages and per-page text length to keep both token usage and RAM low.
    """
    import io, json, os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from langchain_core.messages import HumanMessage
    from llm_utils import (
        json_invoke,
        make_legend_discovery_prompt,
        make_legend_merge_prompt,
        make_page_ref_prompt,
    )
    import fitz  # PyMuPDF

    # --- Collect small per-page texts (sequential, very cheap) ---
    page_texts = []
    total_pages = 0
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        total_pages = len(doc)
        n = min(total_pages, int(os.getenv("MAX_PAGES_FOR_LEGEND", str(max_pages))))
        for i in range(n):
            try:
                p = doc.load_page(i)
                t = p.get_text("text") or ""
                if len(t) > char_budget_per_page:
                    t = t[:char_budget_per_page]
                page_texts.append(t)
            except Exception as e:
                print(f"[legend] text read failed on page {i+1}: {e}")
                page_texts.append("")

    # --- 1) Legend discovery per page (in parallel) ---
    candidates = []
    if page_texts:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i, text in enumerate(page_texts):
                pnum = i + 1
                prompt = make_legend_discovery_prompt(pnum, text)
                futures[ex.submit(json_invoke, llm, [HumanMessage(content=prompt)])] = pnum

            for fut in as_completed(futures):
                pnum = futures[fut]
                try:
                    resp = fut.result()
                    content = resp.content if hasattr(resp, "content") else str(resp)
                    data = json.loads(content)
                    items = data.get("legend_items", []) or []
                    for it in items:
                        it.setdefault("page", pnum)
                    candidates.extend(items)
                except Exception as e:
                    print(f"[legend] page {pnum} failed: {e}")

    # --- 2) Merge candidates to canonical index ---
    index = []
    id_list = []
    try:
        cjson = json.dumps(candidates, separators=(",", ":"))
        merge_prompt = make_legend_merge_prompt(cjson)  # <-- correct builder
        merge_resp = json_invoke(llm, [HumanMessage(content=merge_prompt)])
        mc = merge_resp.content if hasattr(merge_resp, "content") else str(merge_resp)
        merged = json.loads(mc)
        index = merged.get("index", []) or []
        id_list = [x.get("identifier") for x in index if x.get("identifier")]
    except Exception as e:
        print(f"[legend-merge] failed: {e}")

    # --- 3) Detect per-page references to identifiers (in parallel) ---
    page_refs = {}
    if id_list and page_texts:
        id_list_preview = ", ".join(id_list[:60]) + (" …" if len(id_list) > 60 else "")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i, text in enumerate(page_texts):
                pnum = i + 1
                prompt = make_page_ref_prompt(id_list_preview, text)
                futures[ex.submit(json_invoke, llm, [HumanMessage(content=prompt)])] = pnum

            for fut in as_completed(futures):
                pnum = futures[fut]
                try:
                    resp = fut.result()
                    content = resp.content if hasattr(resp, "content") else str(resp)
                    data = json.loads(content)
                    refs = data.get("references", []) or []
                    if refs:
                        page_refs[pnum] = [
                            {
                                "identifier": r.get("identifier"),
                                "confidence": float(r.get("confidence", 0.0)),
                                "evidence": r.get("evidence", ""),
                            }
                            for r in refs
                            if r.get("identifier")
                        ]
                except Exception as e:
                    print(f"[xref] page {pnum} failed: {e}")

    # compact index to keep memory tiny
    index_compact = [
        {k: v for k, v in item.items() if k in ("identifier", "title", "short_description", "definition_pages")}
        for item in (index or [])
    ]

    return {"id_list": id_list, "page_refs": page_refs, "index_compact": index_compact}


def merge_extra_keywords(signals: list) -> list:
    """
    Merge page-local 'signals' with document-level legend identifiers
    discovered by compute_doc_legend_and_refs_compact(). Keeps size bounded.
    """
    import itertools
    merged = list(signals or [])
    try:
        id_list = st.session_state.get("legend_id_list") or []
        if id_list:
            merged = list(dict.fromkeys(itertools.chain(merged, id_list)))  # de-dup preserve order
        if len(merged) > 60:
            merged = merged[:60]
    except Exception:
        pass
    return merged


# --- Main App Flow ---
st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}" # Used for detecting new file
    
    if st.session_state.last_uploaded_file_id != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Resetting state for {current_file_id}.")
        # Preserve some settings across resets
        current_selected_model = st.session_state.selected_model_for_analysis
        current_keywords = st.session_state.fence_keywords_app
        
        initialize_session_state(current_session_id) # Reset to defaults
        
        st.session_state.selected_model_for_analysis = current_selected_model # Restore
        st.session_state.fence_keywords_app = current_keywords # Restore

        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        # Generate and store hash of the current PDF
        st.session_state.current_pdf_hash = hashlib.sha256(st.session_state.original_pdf_bytes).hexdigest()
        st.session_state.last_uploaded_file_id = current_file_id
        
        st.cache_data.clear() 
        print(f"SESSION {current_session_id} LOG: Cleared all @st.cache_data caches due to new file.")
        st.rerun() 

    if openai_key and llm_analysis_instance and \
       not st.session_state.run_analysis_triggered and \
       not st.session_state.processing_complete and \
       not st.session_state.analysis_halted_due_to_error:
        print(f"SESSION {current_session_id} LOG: Triggering analysis.")
        st.session_state.run_analysis_triggered = True

# ... (Rest of the app logic: elif for no key, main analysis block, rerun display block)
# Ensure all calls to generate_display_images_for_page_wrapper(res_data_item, current_session_id)
# And generate_combined_highlighted_pdf(..., current_session_id)

# --- Analysis Execution Block --- (Copied and adapted from previous version)
if st.session_state.run_analysis_triggered and \
   st.session_state.original_pdf_bytes and \
   llm_analysis_instance and \
   not st.session_state.analysis_halted_due_to_error and \
   not st.session_state.processing_complete:
    
    print(f"SESSION {current_session_id} LOG: Starting PDF processing loop.")
    doc_proc_loop = None
    try:
        doc_proc_loop = fitz.open(stream=io.BytesIO(st.session_state.original_pdf_bytes), filetype="pdf")
        st.session_state.doc_total_pages = len(doc_proc_loop)
        print(f"SESSION {current_session_id} LOG: PDF opened, {st.session_state.doc_total_pages} pages.")
    except Exception as e:
        st.error(f"Failed to open PDF: {e}"); st.session_state.processing_complete = True; st.session_state.analysis_halted_due_to_error = True
        if doc_proc_loop: doc_proc_loop.close()
        print(f"SESSION {current_session_id} ERROR: Failed to open PDF for processing: {e}")
        st.stop() 
        # --- OPTIONAL: document-level cross analysis (legend + references) ---
    try:
        cross = compute_doc_legend_and_refs_compact(
            st.session_state.original_pdf_bytes,
            llm_analysis_instance,
            google_cloud_config=google_cloud_config,
            max_pages=60,                 # keep budget bounded
            char_budget_per_page=2400,
            max_workers=4,
        )
        st.session_state.legend_id_list = cross.get("id_list", [])
        st.session_state.page_refs = cross.get("page_refs", {})
        st.session_state.legend_index_compact = cross.get("index_compact", [])
        if st.session_state.legend_id_list:
            print(f"[CROSS] legend identifiers: {len(st.session_state.legend_id_list)}")
    except Exception as e:
        print(f"[CROSS] cross analysis skipped due to error: {e}")
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results (Live)</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty(); col_f, col_nf = st.columns(2)
    with col_f: st.subheader("✅ Fence-Related Pages")
    with col_nf: st.subheader("❌ Non-Fence Pages")
    prog_bar = st.progress(0); status_txt_area = st.empty()
    try:
        for i in range(st.session_state.doc_total_pages):
            curr_pg_num = i + 1; st.session_state.total_pages_processed_count = curr_pg_num
            prog_bar.progress(curr_pg_num / st.session_state.doc_total_pages)
            status_txt_area.text(f"Processing Page {curr_pg_num}/{st.session_state.doc_total_pages}...")
            print(f"SESSION {current_session_id} LOG: Processing page {curr_pg_num}.")
            page_obj = doc_proc_loop.load_page(i); text_content = page_obj.get_text("text")
            
            # Create single-page PDF bytes for comprehensive text extraction
            single_page_pdf_bytes = None
            try:
                temp_doc = fitz.open()
                temp_doc.insert_pdf(doc_proc_loop, from_page=i, to_page=i)
                single_page_pdf_bytes = temp_doc.tobytes()
                temp_doc.close()
            except Exception as e:
                print(f"SESSION {current_session_id} WARNING: Could not create single page PDF for page {curr_pg_num}: {e}")
            
            page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
            analysis_res_core = {}; fatal_err_page = False
            try:
                with st.spinner(f"Page {curr_pg_num}: Core analysis..."):
                    analysis_res_core = analyze_page(
                        page_data_an, llm_analysis_instance, FENCE_KEYWORDS_APP, google_cloud_config,
                        recall_mode="strict"   # or "balanced"/"high"
                    )
                    try:
                        jr = json.loads(analysis_res_core["text_response"])
                        signals = jr.get("signals", [])
                    except Exception:
                        pass


            except UnrecoverableRateLimitError as urle:
                msg = f"🛑 API Rate Limit Pg {curr_pg_num}: {urle}. Analysis halted."; status_txt_area.error(msg); st.error(msg)
                st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
            except Exception as e_core: st.error(f"Core analysis error pg {curr_pg_num}: {e_core}"); analysis_res_core = {"fence_found": False}; print(f"SESSION {current_session_id} ERROR: Core analysis pg {curr_pg_num}: {e_core}")
            analysis_result = {**analysis_res_core, 'page_number': curr_pg_num, 'page_index_in_original_doc': i, 'fence_text_boxes_details': [], 'highlight_fence_text_app_setting': highlight_fence_text_app}
            if not fatal_err_page and highlight_fence_text_app and analysis_result.get('text_found'):
                status_txt_area.text(f"Page {curr_pg_num}: Highlighting (text match found)...")
                single_pg_bytes_io = io.BytesIO(); temp_doc_single = None
                try: 
                    temp_doc_single = fitz.open()
                    temp_doc_single.insert_pdf(doc_proc_loop, from_page=i, to_page=i); temp_doc_single.save(single_pg_bytes_io)
                finally: 
                    if temp_doc_single: temp_doc_single.close()
                try:
                    with st.spinner(f"Page {curr_pg_num}: Extracting highlight boxes..."):   
                        boxes,_,_ = get_fence_related_text_boxes(
                            single_pg_bytes_io.getvalue(),
                            llm_analysis_instance,
                            FENCE_KEYWORDS_APP,
                            merge_extra_keywords(signals),         # <— signals + doc-level legend IDs
                            st.session_state.selected_model_for_analysis,
                            google_cloud_config
                        )


                        if boxes: analysis_result['fence_text_boxes_details'] = boxes
                except UnrecoverableRateLimitError as urle_hl:
                    msg = f"🛑 API Rate Limit Highlight Pg {curr_pg_num}: {urle_hl}. Halted."; status_txt_area.error(msg); st.error(msg)
                    st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
                except Exception as e_hl: st.warning(f"Highlight error pg {curr_pg_num}: {e_hl}"); print(f"SESSION {current_session_id} WARNING: Highlight error pg {curr_pg_num}: {e_hl}")
            elif not fatal_err_page and highlight_fence_text_app and analysis_result.get('fence_found'):
                 status_txt_area.text(f"Page {curr_pg_num}: Fence found, no text match for detailed highlighting.")
            if fatal_err_page: break
            target_col = col_f if analysis_result.get('fence_found') else col_nf
            (st.session_state.fence_pages if analysis_result.get('fence_found') else st.session_state.non_fence_pages).append(analysis_result)
            with target_col: # Display Logic (copied from display_page_result_expander for consistency)
                exp_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    reasons = []; 
                    if analysis_result.get('text_found'): reasons.append("Text")

                    if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app: reasons.append("Highlights")
                    if reasons: exp_title += f" ({' & '.join(reasons)} Match)"
                with st.expander(exp_title, expanded=True):
                    img_col, det_col = st.columns([2,1])
                    print(f"SESSION {current_session_id} DEBUG LIVE DISPLAY Page {analysis_result['page_number']}: Num boxes: {len(analysis_result.get('fence_text_boxes_details', []))}")
                    wrapper_call_start_time = time.time()
                    with st.spinner(f"Rendering image for page {analysis_result['page_number']}..."): 
                        orig_b, hl_b = generate_display_images_for_page_wrapper(analysis_result, current_session_id) # Pass session_id
                    wrapper_call_duration = time.time() - wrapper_call_start_time
                    print(f"SESSION {current_session_id} PERF_LOG: generate_display_images_for_page_wrapper Page {curr_pg_num} took {wrapper_call_duration:.4f}s.")
                    with img_col: # Image display
                        disp_img_ui = hl_b if hl_b else orig_b
                        if disp_img_ui: st.image(disp_img_ui, caption=f"Page {analysis_result['page_number']}{' (Highlighted)' if hl_b else ''}")
                        dl_links_html_live = []
                        if hl_b: dl_links_html_live.append(get_image_download_link_html(hl_b, f"page_{analysis_result['page_number']}_hl.png", "DL HL Img"))
                        if orig_b: dl_links_html_live.append(get_image_download_link_html(orig_b, f"page_{analysis_result['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_live: st.markdown(" ".join(dl_links_html_live), unsafe_allow_html=True)
                    with det_col: # Text details display
                        # ... (Same detailed text display as before)
                        st.markdown("##### Analysis Details")
                        if analysis_result.get('fence_found'):
                            pts = []; 
                            if analysis_result.get('text_found'): pts.append("✔️ Text")

                            if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app : pts.append("✔️ Highlights")
                            if not pts: pts.append("Fence flagged")
                            st.markdown("\n".join(f"- {s}" for s in pts))
                        else: st.markdown("No strong fence indicators.")
                        if analysis_result.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{analysis_result['text_response']}_")

                        if analysis_result.get('text_snippet'):
                            st.markdown("---"); st.markdown("**Key Snippet:**"); st.code(analysis_result['text_snippet'],language=None)
                        if analysis_result.get('highlight_fence_text_app_setting', True) and \
                           analysis_result.get('fence_text_boxes_details') and analysis_result.get('fence_found'):
                            details_list = analysis_result['fence_text_boxes_details']
                            st.markdown("---"); st.markdown("**Highlights (from Text):**")
                            disp_set_live = set(); count_live = 0
                            for d_item_live in sorted(details_list, key=lambda x: x.get('y0', 0)):
                                txt_live = d_item_live.get('text', "N/A"); tag_live = d_item_live.get('tag_from_llm', 'N/A'); type_llm_live = d_item_live.get('type_from_llm', 'N/A')
                                display_text_live = f"- `{txt_live}` (Type: {type_llm_live}, Tag: {tag_live})"
                                if display_text_live not in disp_set_live: st.markdown(display_text_live); disp_set_live.add(display_text_live); count_live+=1
                                if count_live >=15 and len(details_list) > 17: st.markdown(f"- ...& {len(details_list)-count_live} more."); break
            summary_placeholder.markdown(f"### Summary (Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages})\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}")
            time.sleep(0.05) 
    finally: 
        if doc_proc_loop:
            doc_proc_loop.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document in finally block.")
        doc_proc_loop = None 
    st.session_state.processing_complete = True 
    if not st.session_state.analysis_halted_due_to_error:
        prog_bar.empty(); status_txt_area.success("All pages processed!")
        if st.session_state.fence_pages and st.session_state.original_pdf_bytes:
            pdf_b, pdf_n = generate_combined_highlighted_pdf(st.session_state.original_pdf_bytes, st.session_state.fence_pages, st.session_state.uploaded_pdf_name, current_session_id)
            if pdf_b: st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download = pdf_b, pdf_n
            else: st.warning(f"Could not generate PDF: {pdf_n}")
    else: prog_bar.empty() 
    final_summary_text = f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    summary_placeholder.markdown(final_summary_text)
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button("⬇️ Download Highlighted Fence Pages (PDF)", st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download, "application/pdf", key="dl_combined_pdf_main")

elif st.session_state.processing_complete: 
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    final_summary_text_rerun = f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    st.markdown(final_summary_text_rerun)
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
         st.download_button("⬇️ Download Highlighted Fence Pages (PDF)", st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download, "application/pdf", key="dl_combined_pdf_rerun")
    col_f_res, col_nf_res = st.columns(2)
    with col_f_res: st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
    with col_nf_res: st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")
    def display_page_result_expander(res_data_list, target_column_res, session_id_for_display):
        for res_data_item in res_data_list:
            with target_column_res:
                exp_title_res = f"Page {res_data_item['page_number']}"
                # ... (expander title construction)
                if res_data_item.get('fence_found'):
                    reasons_res = []; 
                    if res_data_item.get('text_found'): reasons_res.append("Text")

                    if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting', True): reasons_res.append("Highlights")
                    if reasons_res: exp_title_res += f" ({' & '.join(reasons_res)} Match)"
                with st.expander(exp_title_res, expanded=False):
                    img_col_r, det_col_r = st.columns([2,1])
                    with st.spinner(f"Loading image page {res_data_item['page_number']}..."):
                        orig_b_r, hl_b_r = generate_display_images_for_page_wrapper(res_data_item, session_id_for_display) # Pass session_id
                    with img_col_r: # Image display
                        # ... (Same as live loop image display)
                        disp_img_r = hl_b_r if hl_b_r else orig_b_r
                        if disp_img_r: st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}{' (HL)' if hl_b_r else ''}")
                        dl_links_html_rerun = []
                        if hl_b_r: dl_links_html_rerun.append(get_image_download_link_html(hl_b_r, f"page_{res_data_item['page_number']}_hl.png", "DL HL Img"))
                        if orig_b_r: dl_links_html_rerun.append(get_image_download_link_html(orig_b_r, f"page_{res_data_item['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_rerun: st.markdown(" ".join(dl_links_html_rerun), unsafe_allow_html=True)
                    with det_col_r: # Text details
                        # ... (Same as live loop text details display)
                        st.markdown("##### Analysis Details")
                        if res_data_item.get('fence_found'):
                            pts_r = [] 
                            if res_data_item.get('text_found'): pts_r.append("✔️ Text")

                            if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting',True) : pts_r.append("✔️ Highlights")
                            if not pts_r: pts_r.append("Fence flagged")
                            st.markdown("\n".join(f"- {s}" for s in pts_r))
                        else: st.markdown("No strong fence indicators.")
                        if res_data_item.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{res_data_item['text_response']}_")

                        if res_data_item.get('text_snippet'): st.markdown("---"); st.code(res_data_item['text_snippet'],language=None)
                        if res_data_item.get('highlight_fence_text_app_setting', True) and \
                           res_data_item.get('fence_text_boxes_details') and res_data_item.get('fence_found'):
                            details_list_r = res_data_item['fence_text_boxes_details']
                            st.markdown("---"); st.markdown("**Highlights (from Text):**")
                            disp_set_r = set(); count_r = 0
                            for d_item_r in sorted(details_list_r, key=lambda x: x.get('y0', 0)):
                                txt_r = d_item_r.get('text', "N/A"); tag_r = d_item_r.get('tag_from_llm', 'N/A'); type_llm_r = d_item_r.get('type_from_llm', 'N/A')
                                display_text_r = f"- `{txt_r}` (Type: {type_llm_r}, Tag: {tag_r})"
                                if display_text_r not in disp_set_r: st.markdown(display_text_r); disp_set_r.add(display_text_r); count_r+=1
                                if count_r >=15 and len(details_list_r) > 17: st.markdown(f"- ...& {len(details_list_r)-count_r} more."); break
    display_page_result_expander(st.session_state.fence_pages, col_f_res, current_session_id)
    display_page_result_expander(st.session_state.non_fence_pages, col_nf_res, current_session_id)

elif not st.session_state.original_pdf_bytes : st.info("Upload PDF.")
elif not (openai_key and llm_analysis_instance): st.error("OpenAI models not initialized. Check API key.")
elif st.session_state.analysis_halted_due_to_error: st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---"); st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App</p>", unsafe_allow_html=True)
