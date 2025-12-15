# app_ade_v2.py - ADE Fence Detector with app.py UI
import streamlit as st
import os
import toml
import json
import time
import uuid
import hashlib
import base64
from pathlib import Path
from io import BytesIO
import pandas as pd
import fitz  # PyMuPDF

# Import our consolidated ADE utilities
import utils_ade as ade

# Optional: LLM client
from langchain_openai import ChatOpenAI

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)  # Green for definitions
HIGHLIGHT_COLOR_INSTANCE = (0.9, 0, 0.9)  # Purple for instances
HIGHLIGHT_WIDTH_UI = 2.0
DISPLAY_IMAGE_DPI = 96

st.set_page_config(page_title="ADE Fence Detector", layout="wide")
st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>🔍 ADE Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)


# ==============================================================================
# Session Management (matching app.py)
# ==============================================================================

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def initialize_session_state(session_id_val):
    print(f"SESSION {session_id_val} LOG: Initializing/checking session state.")
    default_state = {
        'session_id': session_id_val,
        'fence_pages': [],
        'non_fence_pages': [],
        'total_pages_processed_count': 0,
        'doc_total_pages': 0,
        'processing_complete': False,
        'analysis_halted_due_to_error': False,
        'fence_keywords_app': [
            'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
            'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
            'adot specifications', 'mag specifications', 'rail', 'railing',
            'bollards', 'handrails', 'wall', 'cmu', 'keynote'
        ],
        'run_analysis_triggered': False,
        'uploaded_pdf_name': None,
        'original_pdf_bytes': None,
        'current_pdf_hash': None,
        'highlighted_pdf_bytes_for_download': None,
        'last_uploaded_file_id': None,
        'selected_model_for_analysis': "gpt-5.1",
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else \
                                    dict(value) if isinstance(value, dict) else \
                                    value
        elif key == 'session_id' and st.session_state.session_id != session_id_val:
            st.session_state.session_id = session_id_val


current_session_id = get_session_id()
initialize_session_state(current_session_id)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_image_download_link_html(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'


def generate_page_images(page_idx, pdf_bytes, definitions, instances, pdf_width, pdf_height):
    """Generate original and highlighted images for a page."""
    try:
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
            page = doc.load_page(page_idx)
            
            # Original image
            pix_orig = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI)
            original_bytes = pix_orig.tobytes("png")
            
            # Highlighted image
            highlighted_bytes = ade.highlight_page_image(
                original_bytes,
                definitions,
                instances,
                pdf_width,
                pdf_height
            )
            
            return original_bytes, highlighted_bytes
    except Exception as e:
        print(f"SESSION {current_session_id} ERROR: Image generation failed: {e}")
        return None, None


def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base, session_id):
    """Generate a combined PDF with only fence-related pages highlighted."""
    print(f"SESSION {session_id} LOG: Generating combined highlighted PDF.")
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No data for PDF."
    
    output_doc = fitz.open()
    input_doc = None
    
    try:
        input_doc = fitz.open(stream=BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        print(f"SESSION {session_id} ERROR: Opening original PDF for combined: {e}")
        if output_doc:
            output_doc.close()
        return None, f"Error opening original PDF: {e}"
    
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        if page_idx is None:
            continue
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            
            # Get page rotation and MediaBox dimensions for coordinate transform
            # Coordinates in definitions/instances are in DISPLAY space (after rotation)
            # but draw_rect expects MediaBox space, so we need to reverse the transform
            rotation = page_out.rotation
            mediabox_w = page_out.mediabox.width
            mediabox_h = page_out.mediabox.height
            
            def reverse_rotation_transform(x0, y0, x1, y1):
                """Transform display coords back to MediaBox coords for PDF annotation."""
                if rotation == 0:
                    return x0, y0, x1, y1
                elif rotation == 90:
                    # Display->MediaBox: (x,y) -> (y, mediabox_h - x)
                    return y0, mediabox_h - x1, y1, mediabox_h - x0
                elif rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                elif rotation == 270:
                    # Display->MediaBox: (x,y) -> (mediabox_w - y, x)
                    return mediabox_w - y1, x0, mediabox_w - y0, x1
                return x0, y0, x1, y1
            
            # Draw definition boxes (green)
            definitions = res_data.get('definitions', [])
            for d in definitions:
                mx0, my0, mx1, my1 = reverse_rotation_transform(d['x0'], d['y0'], d['x1'], d['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)
            
            # Draw instance boxes (purple)
            instances = res_data.get('instances', [])
            for inst in instances:
                mx0, my0, mx1, my1 = reverse_rotation_transform(inst['x0'], inst['y0'], inst['x1'], inst['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)
            
            # Draw keyword match boxes (orange) - for fallback detection
            keyword_matches = res_data.get('keyword_matches', [])
            for kw in keyword_matches:
                if all(k in kw for k in ['x0', 'y0', 'x1', 'y1']):
                    mx0, my0, mx1, my1 = reverse_rotation_transform(kw['x0'], kw['y0'], kw['x1'], kw['y1'])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
                    
        except Exception as e_pi:
            print(f"SESSION {session_id} Err process pg {page_idx} for PDF: {e_pi}")
    
    pdf_bytes, fname = None, "error.pdf"
    if len(output_doc) > 0:
        try:
            pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
            base, ext = os.path.splitext(uploaded_pdf_name_base)
            fname = f"{base}_fence_highlights{ext}"
        except Exception as e_s:
            print(f"SESSION {session_id} Err PDF tobytes: {e_s}")
            fname = f"err_save_{uploaded_pdf_name_base}.pdf"
    
    if input_doc:
        input_doc.close()
    if output_doc:
        output_doc.close()
    
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)


# ==============================================================================
# Sidebar (matching app.py structure)
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load secrets if available
    secrets = {}
    if os.path.exists(".streamlit/secrets.toml"):
        secrets = toml.load(".streamlit/secrets.toml")
    
    # 1. OpenAI Key
    openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
    
    # 2. LandingAI (ADE) Key
    ade_key = secrets.get("LANDINGAI_API_KEY", os.getenv("LANDINGAI_API_KEY"))
    if not ade_key:
        ade_key = st.text_input("Enter LandingAI API Key", type="password", key="ade_key_input_sidebar")
    
    # 3. Google Cloud Config (JSON) - load silently without UI messages
    google_cloud_config = None
    try:
        if "google_cloud" in secrets and "gcp_service_account" in secrets:
            google_cloud_config = {
                "project_number": secrets["google_cloud"]["project_number"],
                "location": secrets["google_cloud"]["location"],
                "processor_id": secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(secrets["gcp_service_account"])
            }
            print(f"SESSION {current_session_id} LOG: Google Cloud config loaded from secrets")
    except Exception as e:
        print(f"SESSION {current_session_id} WARNING: Could not load Google Cloud config: {e}")
    
    # Highlight toggle
    st.markdown("---")
    highlight_fence_text_app = st.toggle("🔍 Highlight text & indicators", value=True, key="highlight_toggle")
    
    # ADE usage toggle
    use_ade = st.toggle("🧠 Use ADE (LandingAI)", value=True, key="use_ade_toggle")
    
    # Fence Measurement toggle
    enable_fence_measurement = st.toggle("📏 Measure fence elements", value=True, key="measurement_toggle")
    
    # Debug mode (disabled in UI)
    DEBUG_MODE = False
    
    # Fence Keywords
    st.markdown("---")
    st.subheader("Fence Keywords")
    if 'fence_keywords_app' not in st.session_state:
        st.session_state.fence_keywords_app = ['fence']
    custom_keywords_str = st.text_area(
        "Custom keywords (one per line):",
        "\n".join(st.session_state.fence_keywords_app),
        height=150,
        key="kw_text_area"
    )
    if st.button("Update Keywords", key="update_kw_btn"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()
    
    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


# ==============================================================================
# Initialize LLM
# ==============================================================================

llm_analysis_instance = None
if openai_key:
    try:
        print(f"SESSION {current_session_id} LOG: Initializing LLM instance.")
        llm_analysis_instance = ChatOpenAI(
            model=st.session_state.selected_model_for_analysis,
            temperature=0,
            openai_api_key=openai_key,
            timeout=180,
            max_retries=2
        )
        print(f"SESSION {current_session_id} LOG: LLM instance initialized.")
    except Exception as e:
        st.error(f"LLM Init Error: {e}")
        openai_key = None
        print(f"SESSION {current_session_id} ERROR: LLM Init Error: {e}")


# ==============================================================================
# Main App Flow
# ==============================================================================

st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}"
    
    if st.session_state.last_uploaded_file_id != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Resetting state for {current_file_id}.")
        # Preserve some settings across resets
        current_selected_model = st.session_state.selected_model_for_analysis
        current_keywords = st.session_state.fence_keywords_app
        
        initialize_session_state(current_session_id)
        
        st.session_state.selected_model_for_analysis = current_selected_model
        st.session_state.fence_keywords_app = current_keywords
        
        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.current_pdf_hash = hashlib.sha256(st.session_state.original_pdf_bytes).hexdigest()
        st.session_state.last_uploaded_file_id = current_file_id
        
        st.cache_data.clear()
        print(f"SESSION {current_session_id} LOG: Cleared all @st.cache_data caches due to new file.")
        st.rerun()
    
    if openai_key and llm_analysis_instance and \
       (ade_key or not use_ade) and \
       not st.session_state.run_analysis_triggered and \
       not st.session_state.processing_complete and \
       not st.session_state.analysis_halted_due_to_error:
        print(f"SESSION {current_session_id} LOG: Triggering analysis.")
        st.session_state.run_analysis_triggered = True


# ==============================================================================
# Analysis Execution Block
# ==============================================================================

if st.session_state.run_analysis_triggered and \
   st.session_state.original_pdf_bytes and \
   llm_analysis_instance and \
   (ade_key or not use_ade) and \
   not st.session_state.analysis_halted_due_to_error and \
   not st.session_state.processing_complete:
    
    print(f"SESSION {current_session_id} LOG: Starting ADE-based PDF processing.")
    file_bytes = st.session_state.original_pdf_bytes
    
    # Open PDF to get page count
    doc_proc = None
    try:
        doc_proc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
        st.session_state.doc_total_pages = len(doc_proc)
        print(f"SESSION {current_session_id} LOG: PDF opened, {st.session_state.doc_total_pages} pages.")
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        st.session_state.processing_complete = True
        st.session_state.analysis_halted_due_to_error = True
        if doc_proc:
            doc_proc.close()
        print(f"SESSION {current_session_id} ERROR: Failed to open PDF for processing: {e}")
        st.stop()
    
    # UI Setup (matching app.py)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results (Live)</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty()
    col_f, col_nf = st.columns(2)
    with col_f:
        st.subheader("✅ Fence-Related Pages")
    with col_nf:
        st.subheader("❌ Non-Fence Pages")
    prog_bar = st.progress(0)
    status_txt_area = st.empty()
    
    try:
        total_pages = st.session_state.doc_total_pages
        
        # Process each page with pre-filtering
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            st.session_state.total_pages_processed_count = page_num
            prog_bar.progress(page_num / total_pages)
            status_txt_area.text(f"Page {page_num}/{total_pages}: Checking for fence content...")
            print(f"SESSION {current_session_id} LOG: Processing page {page_num}.")
            
            # Get page dimensions and image
            page = doc_proc[page_idx]
            pdf_width, pdf_height = page.rect.width, page.rect.height
            page_img_bytes = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI).tobytes("png")
            
            # =====================================================================
            # STEP 1: Extract text sources (PDF native + OCR)
            # =====================================================================
            pdf_lines = ade.get_native_pdf_lines(page)
            
            single_page_pdf = ade.create_single_page_pdf(file_bytes, page_idx)
            ocr_lines = []
            if google_cloud_config:
                ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, pdf_width, pdf_height)
            
            # =====================================================================
            # STEP 2: PRE-FILTER - Check if page is fence-related
            # =====================================================================
            status_txt_area.text(f"Page {page_num}/{total_pages}: Pre-filtering...")
            prefilter_result = ade.fallback_fence_detection(
                pdf_lines=pdf_lines,
                ocr_lines=ocr_lines,
                fence_keywords=FENCE_KEYWORDS_APP,
                llm=llm_analysis_instance,
                use_llm_confirmation=True
            )
            
            # Initialize variables
            chunks = []
            legend_chunks = []
            figure_chunks = []
            definitions = []
            instances = []
            fallback_result = None
            keyword_matches = []
            # Default detection method based on prefilter; can be overridden later
            detection_method = prefilter_result.get("method", "none")
            fence_found = prefilter_result["fence_found"]
            
            if not fence_found:
                # =====================================================================
                # NON-FENCE PAGE: Skip ADE, build minimal result
                # =====================================================================
                print(f"[APP] Page {page_num}: Pre-filter says NOT fence-related, skipping ADE.")
                detection_method = "none"
            else:
                # =====================================================================
                # STEP 3: FENCE PAGE - optionally send single page to ADE
                # =====================================================================
                if use_ade and ade_key:
                    print(f"[APP] Page {page_num}: Pre-filter detected fence content via {prefilter_result['method']}, sending to ADE...")
                    status_txt_area.text(f"Page {page_num}/{total_pages}: Sending to ADE for detailed extraction...")
                    
                    ade_response = ade.ade_parse_document(single_page_pdf, ade_key)
                    
                    if not ade_response["success"]:
                        # ADE failed - use pre-filter results as fallback
                        print(f"[APP] Page {page_num}: ADE failed ({ade_response['error']}), using pre-filter results.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                        detection_method = prefilter_result["method"]
                    else:
                        # =====================================================================
                        # STEP 4: Process ADE results (page_idx=0 since single-page PDF)
                        # =====================================================================
                        status_txt_area.text(f"Page {page_num}/{total_pages}: Extracting definitions...")
                        
                        chunks = ade.align_ade_chunks_to_page(ade_response, 0, pdf_width, pdf_height)
                        legend_chunks, figure_chunks = ade.segment_chunks(chunks)
                else:
                    # ADE disabled or no key: rely solely on pre-filter result
                    print(f"[APP] Page {page_num}: ADE is disabled or missing key; using pre-filter result only.")
                    fallback_result = prefilter_result
                    keyword_matches = prefilter_result.get("matched_lines", [])
                
                # =====================================================================
                # STEP 5: Process chunks (runs for ALL fence pages)
                # =====================================================================
                # Debug visualization
                if DEBUG_MODE and (legend_chunks or pdf_lines or ocr_lines):
                    debug_bytes = ade.debug_visualize_coordinates(
                        page_img_bytes, legend_chunks, pdf_lines, ocr_lines, pdf_width, pdf_height
                    )
                    st.image(debug_bytes, caption=f"DEBUG: Layers Page {page_num}", use_container_width=True)
                
                # Extract fence-related definitions from legend chunks
                if highlight_fence_text_app and legend_chunks:
                    definitions = ade.extract_legend_entries(
                        legend_chunks=legend_chunks,
                        pdf_lines=pdf_lines,
                        ocr_lines=ocr_lines,
                        fence_keywords=FENCE_KEYWORDS_APP,
                        llm=llm_analysis_instance
                    )
                
                # Get all page tokens for instance finding
                # IMPORTANT: Transform MediaBox coords to display coords for rotated pages
                native_words = page.get_text("words")
                rotation = page.rotation
                mediabox_w = page.mediabox.width
                mediabox_h = page.mediabox.height
                print(f"[DEBUG] Page {page_num} rotation: {rotation}°, MediaBox: {mediabox_w:.0f}x{mediabox_h:.0f}")
                
                def transform_for_rotation(x0, y0, x1, y1):
                    """Transform MediaBox coords to display coords based on page rotation"""
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
                    all_page_tokens.append({
                        "text": w[4], 
                        "x0": nx0, "y0": ny0, 
                        "x1": nx1, "y1": ny1
                    })
                
                if all_page_tokens:
                    sample = all_page_tokens[0]
                    print(f"[DEBUG] Sample token after transform: '{sample['text']}' at ({sample['x0']:.1f}, {sample['y0']:.1f})")
                
                # Find instances in figures
                if definitions and figure_chunks:
                    instances = ade.find_instances_in_figures(definitions, figure_chunks, all_page_tokens)
                
                # =====================================================================
                # STEP 6: Smart Fence Measurement (if enabled)
                # =====================================================================
                measurement_result = {}
                if enable_fence_measurement and (definitions or instances):
                    try:
                        status_txt_area.text(f"Page {page_num}/{total_pages}: Measuring fence elements...")
                        measurement_result = ade.measure_fence_elements(
                            page, definitions, instances, 
                            figure_chunks=figure_chunks,  # Pass figure areas for boundary
                            llm=llm_analysis_instance
                        )
                    except Exception as e:
                        print(f"[APP] Measurement error: {e}")

                
                # DEBUG: Show coordinate info if enabled
                if DEBUG_MODE:
                    with st.expander(f"🔧 DEBUG Page {page_num}", expanded=True):
                        st.markdown(f"**PDF size:** {pdf_width:.1f} x {pdf_height:.1f}")
                        st.markdown(f"**All ADE Chunks:** {len(chunks)}")
                        for i, c in enumerate(chunks[:10]):
                            st.markdown(f"  - `{c.get('type')}`: ({c.get('x0'):.1f}, {c.get('y0'):.1f}) - ({c.get('x1'):.1f}, {c.get('y1'):.1f})")
                        st.markdown(f"**Figure/Architectural Chunks:** {len(figure_chunks)}")
                        for i, fc in enumerate(figure_chunks):
                            st.markdown(f"  - `{fc.get('type')}`: ({fc.get('x0'):.1f}, {fc.get('y0'):.1f}) - ({fc.get('x1'):.1f}, {fc.get('y1'):.1f})")
                        st.markdown(f"**Legend-like Chunks:** {len(legend_chunks)}")
                        st.markdown(f"**Definitions found:** {len(definitions)}")
                        for i, d in enumerate(definitions[:5]):
                            kw = d.get('keyword', '')[:30]
                            st.markdown(f"  - `{d.get('indicator')}`: {kw}... @ ({d.get('x0'):.1f}, {d.get('y0'):.1f})")
                        st.markdown(f"**Instances found:** {len(instances)}")
                        for i, inst in enumerate(instances[:10]):
                            st.markdown(f"  - `{inst.get('indicator')}` @ ({inst.get('x0'):.1f}, {inst.get('y0'):.1f})")
                        st.markdown(f"**Total page tokens:** {len(all_page_tokens)}")
                        if all_page_tokens:
                            st.markdown(f"**Sample tokens (first 20):**")
                            for t in all_page_tokens[:20]:
                                st.markdown(f"  - `{t.get('text')}` @ ({t.get('x0'):.1f}, {t.get('y0'):.1f})")
                
                # Determine detection method
                if definitions or instances:
                    detection_method = "ade"
                    fence_found = True
                else:
                    # No structured data found, fall back to pre-filter results
                    if not fallback_result:
                        print(f"[APP] Page {page_num}: No definitions found, using pre-filter results.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                    detection_method = prefilter_result.get("method", "none")
            
            # Build text snippet from definitions or fallback keywords
            text_snippet = None
            if definitions:
                snippets = [f"{d.get('indicator', '')} - {d.get('keyword', '')}" for d in definitions[:3]]
                text_snippet = "; ".join(snippets)
            elif fallback_result and fallback_result.get("matched_keywords"):
                text_snippet = "Keywords: " + ", ".join(fallback_result["matched_keywords"][:5])
            
            # Generate images
            original_img_bytes = page_img_bytes
            highlighted_img_bytes = None
            
            if highlight_fence_text_app:
                if definitions or instances:
                    # Primary highlighting: definitions (green) + instances (purple)
                    highlighted_img_bytes = ade.highlight_page_image(
                        page_img_bytes, definitions, instances, pdf_width, pdf_height
                    )
                elif keyword_matches:
                    # Fallback highlighting: keyword matches (orange)
                    highlighted_img_bytes = ade.highlight_keyword_matches(
                        page_img_bytes, keyword_matches, pdf_width, pdf_height
                    )
                
                # Highlight measured fence lines (cyan)
                if measurement_result and measurement_result.get('all_fence_lines'):
                    highlighted_img_bytes = ade.highlight_fence_lines(
                        highlighted_img_bytes or page_img_bytes,
                        measurement_result['all_fence_lines'],
                        pdf_width, pdf_height
                    )
            
            # Build result structure (matching app.py format)
            # detection_method already set above based on flow
            llm_result = fallback_result.get("llm_result") if fallback_result else None
            
            analysis_result = {
                'page_number': page_num,
                'page_index_in_original_doc': page_idx,
                'fence_found': fence_found,
                'text_found': fence_found,
                'text_response': json.dumps({
                    "answer": "yes" if fence_found else "no",
                    "confidence": 0.9 if definitions else (llm_result["confidence"] if llm_result else 0.6),
                    "signals": [d.get('keyword', '') for d in definitions[:5]] if definitions else (fallback_result.get("matched_keywords", []) if fallback_result else []),
                    "reason": f"Found {len(definitions)} definitions, {len(instances)} instances" if definitions else (llm_result["reason"] if llm_result else f"Keyword match: {fallback_result.get('matched_keywords', [])}" if fallback_result else "No fence content")
                }),
                'text_snippet': text_snippet,
                'definitions': definitions,
                'instances': instances,
                'instances': instances,
                'keyword_matches': keyword_matches,  # NEW: Store keyword matches
                'fallback_result': fallback_result,  # NEW: Store fallback result
                'measurements': measurement_result,  # NEW: Store measurements
                'detection_method': detection_method,  # NEW: Track how fence was detected
                'fence_text_boxes_details': definitions + instances + keyword_matches,  # Combined for compatibility
                'highlight_fence_text_app_setting': highlight_fence_text_app,
                'original_image_bytes': original_img_bytes,
                'highlighted_image_bytes': highlighted_img_bytes,
                'pdf_width': pdf_width,
                'pdf_height': pdf_height,
                'chunk_count': len(chunks),
                'legend_count': len(legend_chunks),
                'figure_count': len(figure_chunks),
            }
            
            # Add to appropriate list
            if fence_found:
                st.session_state.fence_pages.append(analysis_result)
            else:
                st.session_state.non_fence_pages.append(analysis_result)
            
            # Display in appropriate column (matching app.py)
            target_col = col_f if fence_found else col_nf
            
            with target_col:
                exp_title = f"Page {page_num}"
                if fence_found:
                    reasons = []
                    if definitions:
                        reasons.append("Definitions")
                    if instances:
                        reasons.append("Instances")
                    if keyword_matches and not definitions:
                        reasons.append("Keywords")
                    if highlight_fence_text_app and (definitions or instances or keyword_matches):
                        reasons.append("Highlights")
                    if reasons:
                        exp_title += f" ({' & '.join(reasons)})"
                
                with st.expander(exp_title, expanded=True):
                    img_col, det_col = st.columns([2, 1])
                    
                    with img_col:
                        disp_img = highlighted_img_bytes if highlighted_img_bytes else original_img_bytes
                        if disp_img:
                            st.image(disp_img, caption=f"Page {page_num}{' (Highlighted)' if highlighted_img_bytes else ''}")
                        
                        # Download links
                        dl_links = []
                        if highlighted_img_bytes:
                            dl_links.append(get_image_download_link_html(highlighted_img_bytes, f"page_{page_num}_hl.png", "DL HL Img"))
                        if original_img_bytes:
                            dl_links.append(get_image_download_link_html(original_img_bytes, f"page_{page_num}_orig.png", "DL Orig Img"))
                        if dl_links:
                            st.markdown(" ".join(dl_links), unsafe_allow_html=True)
                    
                    with det_col:
                        # Detection method badge
                        if detection_method == "ade":
                            st.success("🎯 ADE Detection")
                        elif detection_method == "llm_confirmed":
                            st.warning("🔍 Keyword + LLM")
                        elif detection_method == "keyword_only":
                            st.warning("🔤 Keyword Match")
                        else:
                            st.info("❌ No Detection")
                        
                        # ADE Stats (compact)
                        st.metric("ADE Chunks", len(chunks))
                        col_leg, col_fig = st.columns(2)
                        with col_leg:
                            st.metric("Legend", len(legend_chunks))
                        with col_fig:
                            st.metric("Figure", len(figure_chunks))
                        
                        # Text response popover
                        if analysis_result.get('text_response'):
                            with st.popover("Analysis Log"):
                                st.markdown(f"_{analysis_result['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # NEW: Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (Fallback)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                    
                    # Show Measurements (for ALL fence pages, not just keyword matches)
                    if measurement_result and (measurement_result.get('indicator_measurements') or measurement_result.get('proximity_totals', {}).get('total_segments', 0) > 0):
                        st.markdown("---")
                        st.markdown("### 📏 Fence Measurements")
                        
                        page_info = measurement_result.get('page_info', {})
                        scale_factor = page_info.get('scale_factor', 1.0)
                        method = measurement_result.get('measurement_method', 'unknown')
                        
                        # Show method badge
                        if method == "layer":
                            st.info("📂 Method: Layer-based (fence layers detected)")
                        elif method == "proximity":
                            st.info("🎯 Method: Proximity-based (fallback)")
                        
                        # Show scale info
                        if page_info.get('scale_detected'):
                            st.success(f"✅ Scale: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                        else:
                            st.warning("⚠️ Scale not detected - raw measurements")
                        
                        # Show totals
                        prox_totals = measurement_result.get('proximity_totals', {})
                        if prox_totals.get('total_segments', 0) > 0:
                            col_pts, col_ft = st.columns(2)
                            with col_pts:
                                st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                            with col_ft:
                                st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                            
                            # Per-indicator breakdown
                            indicator_meas = measurement_result.get('indicator_measurements', {})
                            if indicator_meas:
                                st.markdown("**Per-Indicator:**")
                                for ind, stats in indicator_meas.items():
                                    pts = stats.get('run_length_pts', 0)
                                    ft = stats.get('run_length_feet', 0)
                                    segs = stats.get('run_segment_count', 0)
                                    count = stats.get('instance_count', 0)
                                    st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                        
                        # Layer breakdown (secondary)
                        if measurement_result.get('fence_layers'):
                            with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                totals = measurement_result.get('totals', {})
                                st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                for layer in measurement_result['fence_layers']:
                                    l_stats = measurement_result['layer_measurements'].get(layer, {})
                                    segs = l_stats.get('total_segments', 0)
                                    ft = l_stats.get('total_length_feet', 0)
                                    runs = l_stats.get('connected_runs', 0)
                                    st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
            
            # Update summary
            summary_placeholder.markdown(
                f"### Summary (Processed: {st.session_state.total_pages_processed_count}/{total_pages})\n"
                f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
            )
            
            time.sleep(0.05)
        
        # Processing complete
        st.session_state.processing_complete = True
        prog_bar.empty()
        status_txt_area.success("All pages processed!")
        
        # Generate combined PDF
        if st.session_state.fence_pages and st.session_state.original_pdf_bytes:
            pdf_b, pdf_n = generate_combined_highlighted_pdf(
                st.session_state.original_pdf_bytes,
                st.session_state.fence_pages,
                st.session_state.uploaded_pdf_name,
                current_session_id
            )
            if pdf_b:
                st.session_state.highlighted_pdf_bytes_for_download = pdf_b
                st.session_state.highlighted_pdf_filename_for_download = pdf_n
            else:
                st.warning(f"Could not generate PDF: {pdf_n}")
        
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.session_state.analysis_halted_due_to_error = True
        print(f"SESSION {current_session_id} ERROR: {e}")
    finally:
        if doc_proc:
            doc_proc.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document.")
    
    # Final summary
    final_summary_text = (
        f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    summary_placeholder.markdown(final_summary_text)
    
    # Download button
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_main"
        )


# ==============================================================================
# Display Previously Processed Results (on rerun)
# ==============================================================================

elif st.session_state.processing_complete:
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    
    final_summary_text_rerun = (
        f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    st.markdown(final_summary_text_rerun)
    
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_rerun"
        )
    
    col_f_res, col_nf_res = st.columns(2)
    with col_f_res:
        st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
    with col_nf_res:
        st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")
    
    def display_page_result_expander(res_data_list, target_column_res):
        for res_data_item in res_data_list:
            with target_column_res:
                exp_title_res = f"Page {res_data_item['page_number']}"
                definitions = res_data_item.get('definitions', [])
                instances = res_data_item.get('instances', [])
                keyword_matches = res_data_item.get('keyword_matches', [])
                detection_method = res_data_item.get('detection_method', 'none')
                fallback_result = res_data_item.get('fallback_result')
                
                if res_data_item.get('fence_found'):
                    reasons_res = []
                    if definitions:
                        reasons_res.append("Definitions")
                    if instances:
                        reasons_res.append("Instances")
                    if keyword_matches and not definitions:
                        reasons_res.append("Keywords")
                    if res_data_item.get('highlight_fence_text_app_setting', True) and \
                       (definitions or instances or keyword_matches or res_data_item.get('measurements')):
                        reasons_res.append("Highlights")
                    if reasons_res:
                        exp_title_res += f" ({' & '.join(reasons_res)})"
                
                with st.expander(exp_title_res, expanded=False):
                    img_col_r, det_col_r = st.columns([2, 1])
                    
                    with img_col_r:
                        disp_img_r = res_data_item.get('highlighted_image_bytes') or res_data_item.get('original_image_bytes')
                        if disp_img_r:
                            st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}")
                        
                        dl_links_rerun = []
                        if res_data_item.get('highlighted_image_bytes'):
                            dl_links_rerun.append(get_image_download_link_html(
                                res_data_item['highlighted_image_bytes'],
                                f"page_{res_data_item['page_number']}_hl.png",
                                "DL HL Img"
                            ))
                        if res_data_item.get('original_image_bytes'):
                            dl_links_rerun.append(get_image_download_link_html(
                                res_data_item['original_image_bytes'],
                                f"page_{res_data_item['page_number']}_orig.png",
                                "DL Orig Img"
                            ))
                        if dl_links_rerun:
                            st.markdown(" ".join(dl_links_rerun), unsafe_allow_html=True)
                    
                    with det_col_r:
                        # Detection method badge
                        if detection_method == "ade":
                            st.success("🎯 ADE Detection")
                        elif detection_method == "llm_confirmed":
                            st.warning("🔍 Keyword + LLM")
                        elif detection_method == "keyword_only":
                            st.warning("🔤 Keyword Match")
                        else:
                            st.info("❌ No Detection")
                        
                        # ADE Stats (compact)
                        st.metric("ADE Chunks", res_data_item.get('chunk_count', 0))
                        col_leg_r, col_fig_r = st.columns(2)
                        with col_leg_r:
                            st.metric("Legend", res_data_item.get('legend_count', 0))
                        with col_fig_r:
                            st.metric("Figure", res_data_item.get('figure_count', 0))
                        
                        if res_data_item.get('text_response'):
                            with st.popover("Analysis Log"):
                                st.markdown(f"_{res_data_item['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (Fallback)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                        
                        # Show Measurements
                        measurements = res_data_item.get('measurements')
                        if measurements and (measurements.get('indicator_measurements') or measurements.get('totals', {}).get('total_length_feet', 0) > 0):
                            st.markdown("---")
                            st.markdown("### 📏 Fence Measurements")
                            
                            page_info = measurements.get('page_info', {})
                            scale_factor = page_info.get('scale_factor', 1.0)
                            
                            # Show scale info prominently
                            if page_info.get('scale_detected'):
                                st.success(f"✅ Scale Auto-Detected: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                            else:
                                st.warning("⚠️ Scale not detected - showing raw measurements")
                            
                            # Show proximity-based measurements (primary)
                            prox_totals = measurements.get('proximity_totals', {})
                            if prox_totals.get('total_segments', 0) > 0:
                                st.markdown("#### 🎯 Near Detected Indicators:")
                                col_pts, col_ft = st.columns(2)
                                with col_pts:
                                    st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                                with col_ft:
                                    st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                                
                                # Per-indicator breakdown
                                indicator_meas = measurements.get('indicator_measurements', {})
                                if indicator_meas:
                                    st.markdown("**Per-Indicator:**")
                                    for ind, stats in indicator_meas.items():
                                        pts = stats.get('run_length_pts', 0)
                                        ft = stats.get('run_length_feet', 0)
                                        segs = stats.get('run_segment_count', 0)
                                        count = stats.get('instance_count', 0)
                                        st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                            
                            # Layer breakdown (secondary)
                            if measurements.get('fence_layers'):
                                with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                    totals = measurements.get('totals', {})
                                    st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                    for layer in measurements['fence_layers']:
                                        l_stats = measurements['layer_measurements'].get(layer, {})
                                        segs = l_stats.get('total_segments', 0)
                                        ft = l_stats.get('total_length_feet', 0)
                                        runs = l_stats.get('connected_runs', 0)
                                        st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
    
    display_page_result_expander(st.session_state.fence_pages, col_f_res)
    display_page_result_expander(st.session_state.non_fence_pages, col_nf_res)


# ==============================================================================
# Fallback Messages
# ==============================================================================

elif not st.session_state.original_pdf_bytes:
    st.info("Upload a PDF to begin analysis.")
elif not (openai_key and llm_analysis_instance):
    st.error("OpenAI models not initialized. Check API key.")
elif not ade_key:
    st.error("LandingAI API key required for ADE analysis.")
elif st.session_state.analysis_halted_due_to_error:
    st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>ADE Fence Detector App</p>", unsafe_allow_html=True)
