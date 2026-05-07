# app_ade.py - ADE-powered Fence Detector
import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, .env file won't be loaded automatically")
    print("   Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Error loading .env: {e}")

# Import ADE functions with fallback
try:
    from utils_ade_official import (
        ade_parse_page_official, extract_ade_text_and_elements_official, get_ade_fence_boxes_official, 
        create_single_page_pdf, get_page_dimensions,
        ade_parse_page_official_cached, get_google_ocr_results_with_boxes,
        extract_legend_keywords_and_indicators, find_indicators_in_figures,
        extract_indicators_from_legend_text, ade_parse_document_official,
        align_ade_chunks_to_page, filter_ocr_by_ade_regions
    )
    ADE_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ADE functions from utils_ade_official: {e}")
    ADE_FUNCTIONS_AVAILABLE = False
    # Create stub functions
    def ade_parse_page_official(*args, **kwargs):
        return {"success": False, "error": "ADE functions not available"}
    def extract_ade_text_and_elements_official(*args, **kwargs):
        return "", []
    def get_ade_fence_boxes_official(*args, **kwargs):
        return []
    def create_single_page_pdf(*args, **kwargs):
        import fitz
        from io import BytesIO
        pdf_bytes, page_idx = args[0], args[1]
        doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        temp_doc = fitz.open()
        temp_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        result = temp_doc.tobytes()
        temp_doc.close()
        doc.close()
        return result
    def get_page_dimensions(*args, **kwargs):
        import fitz
        from io import BytesIO
        page_bytes = args[0]
        doc = fitz.open(stream=BytesIO(page_bytes), filetype="pdf")
        page = doc[0]
        width, height = page.rect.width, page.rect.height
        doc.close()
        return width, height
    def ade_parse_page_official_cached(*args, **kwargs):
        return {"success": False, "error": "ADE functions not available"}
    def get_google_ocr_results_with_boxes(*args, **kwargs):
        return []
    def extract_legend_keywords_and_indicators(*args, **kwargs):
        return []
    def find_indicators_in_figures(*args, **kwargs):
        return []
    def extract_indicators_from_legend_text(*args, **kwargs):
        return []
    def ade_parse_document_official(*args, **kwargs):
        return {"success": False, "error": "ADE functions not available"}
    def align_ade_chunks_to_page(*args, **kwargs):
        return []
    def filter_ocr_by_ade_regions(*args, **kwargs):
        return []
from utils import UnrecoverableRateLimitError, time_it, analyze_page, get_fence_related_text_boxes
import re 
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
import io
from io import BytesIO
import time 
import uuid 
import hashlib # For hashing PDF bytes

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)
HIGHLIGHT_WIDTH_UI = 2.0
HIGHLIGHT_COLOR_PDF = (0, 0.9, 0)
HIGHLIGHT_WIDTH_PDF = 2.0
DISPLAY_IMAGE_DPI = 96  

st.set_page_config(page_title="Fence Detector (ADE)", layout="wide")
st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True) 
st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings (ADE-powered)</h1>", unsafe_allow_html=True)

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
        'fence_keywords_app': ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link', 'masonry', 'fence details', 'canopy shading', 'adot specifications', 'mag specifications', 'rail', 'railing', 'bollards', 'handrails', 'wall', 'cmu', 'keynote'],
        'run_analysis_triggered': False, 'uploaded_pdf_name': None, 'original_pdf_bytes': None,
        'current_pdf_hash': None, # NEW: To store hash of current PDF
        'highlighted_pdf_bytes_for_download': None, 'last_uploaded_file_id': None,
        'selected_model_for_analysis': "gpt-4o",
        # Memory management for 413 error prevention
        'max_pages_in_memory': 20,  # Limit pages stored in memory
        'session_size_mb': 0,  # Track session size
        # ADE-specific state
        'ade_api_key': None,
        'ade_zdr_mode': False
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else \
                                    dict(value) if isinstance(value, dict) else \
                                    value
        elif key == 'session_id' and st.session_state.session_id != session_id_val :
             st.session_state.session_id = session_id_val

def manage_session_memory():
    """Simple memory management to prevent 413 errors."""
    try:
        import sys
        
        # Check if we have too many pages in memory
        total_pages = len(st.session_state.fence_pages) + len(st.session_state.non_fence_pages)
        max_pages = st.session_state.get('max_pages_in_memory', 20)
        
        if total_pages > max_pages:
            print(f"SESSION {get_session_id()} WARNING: Too many pages in memory ({total_pages}). Clearing oldest pages.")
            
            # Keep only the most recent pages
            if len(st.session_state.fence_pages) > max_pages // 2:
                st.session_state.fence_pages = st.session_state.fence_pages[-max_pages//2:]
            if len(st.session_state.non_fence_pages) > max_pages // 2:
                st.session_state.non_fence_pages = st.session_state.non_fence_pages[-max_pages//2:]
            
            # Clear cache to free memory
            st.cache_data.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print(f"SESSION {get_session_id()} LOG: Memory cleared. Pages: fence={len(st.session_state.fence_pages)}, non_fence={len(st.session_state.non_fence_pages)}")
        
        # Check PDF size
        if st.session_state.original_pdf_bytes:
            pdf_size_mb = len(st.session_state.original_pdf_bytes) / (1024 * 1024)
            st.session_state.session_size_mb = pdf_size_mb
        
        return True
    except Exception as e:
        print(f"SESSION {get_session_id()} ERROR: Memory management failed: {e}")
        return True

current_session_id = get_session_id() 
initialize_session_state(current_session_id) 

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key
    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
        if openai_key_input: openai_key = openai_key_input; st.rerun()
    
    # ADE API Key - prioritize secrets.toml (as user requested), then .env, then env vars
    ade_key = (
        st.secrets.get("LANDINGAI_API_KEY") or  # First: from secrets.toml (user's preference)
        st.secrets.get("VISION_AGENT_API_KEY") or  # Alternative name in secrets
        os.getenv("LANDINGAI_API_KEY") or  # Fallback: from .env file (via dotenv)
        os.getenv("VISION_AGENT_API_KEY")  # Alternative name from .env
    )
    if not ade_key:
        ade_key_input = st.text_input("Enter LandingAI ADE API Key", type="password", key="ade_key_input_sidebar", 
                                     help="API key should start with 'land_sk_' - get it from https://landing.ai/")
        if ade_key_input: 
            ade_key = ade_key_input.strip()  # Remove any whitespace
            st.session_state.ade_api_key = ade_key
            st.rerun()
    else:
        st.session_state.ade_api_key = ade_key.strip()  # Remove any whitespace
    
    # ADE Configuration
    st.subheader("🔧 ADE Configuration")
    st.session_state.ade_zdr_mode = st.toggle("Zero Data Retention (ZDR)", value=False, help="Process data without storing it on LandingAI servers")
    
    if ade_key:
        st.success("✅ ADE API Key loaded")
    else:
        st.error("❌ ADE API Key required")
    
    # Model Selection
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
    
    # Display memory usage
    if st.session_state.get('session_size_mb', 0) > 0:
        st.metric("📊 Memory Usage", f"{st.session_state.session_size_mb:.1f} MB")
        if st.session_state.session_size_mb > 20:
            st.warning("⚠️ High memory usage - may cause 413 errors")
    
    # Load Google Cloud credentials for OCR (required for hybrid highlighting)
    google_cloud_config = None
    try:
        if "google_cloud" in st.secrets and "gcp_service_account" in st.secrets:
            google_cloud_config = {
                "project_number": st.secrets["google_cloud"]["project_number"],
                "location": st.secrets["google_cloud"]["location"], 
                "processor_id": st.secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(st.secrets["gcp_service_account"])
            }
            print(f"SESSION {current_session_id} LOG: Google Cloud config loaded from secrets for hybrid OCR")
    except Exception as e:
        print(f"SESSION {current_session_id} WARNING: Could not load Google Cloud config: {e}")
        google_cloud_config = None
    
    # ADE Information
    st.subheader("📄 Hybrid ADE + Google OCR Processing")
    if google_cloud_config:
        st.info("🔍 Using ADE for structure detection + Google OCR for precise highlighting.")
    else:
        st.warning("⚠️ Google Cloud config not found. Hybrid highlighting requires both ADE and Google OCR.")
 
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
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

@time_it 
@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def _generate_display_images_for_page_cached(page_idx, 
                                            pdf_hash_for_cache_key,
                                            fence_text_boxes_details_tuple,
                                            ui_color, ui_width, display_dpi, 
                                            session_id_for_log="N/A_CACHE"):
    if 'original_pdf_bytes' not in st.session_state or st.session_state.original_pdf_bytes is None:
        print(f"SESSION {session_id_for_log} ERROR (_cached): original_pdf_bytes not in session_state for hash {pdf_hash_for_cache_key}")
        return None, None
    
    current_pdf_bytes = st.session_state.original_pdf_bytes
    original_image_bytes, highlighted_image_bytes = None, None
    func_call_id = str(uuid.uuid4())[:4] 
    print(f"SESSION {session_id_for_log} CACHE_CALL ({func_call_id}): _generate_display_images_for_page_cached for Page {page_idx}, PDF Hash: {pdf_hash_for_cache_key}. Num boxes: {len(fence_text_boxes_details_tuple)}")
    render_start_time = time.time()
    try:
        # Use the PDF bytes from session state
        with fitz.open(stream=io.BytesIO(current_pdf_bytes), filetype="pdf") as doc_orig:
            page_orig = doc_orig.load_page(page_idx)
            pix_orig = page_orig.get_pixmap(dpi=display_dpi); original_image_bytes = pix_orig.tobytes("png"); del pix_orig

            if fence_text_boxes_details_tuple: # Check the tuple directly
                fence_text_boxes_details = [dict(item_tuple) for item_tuple in fence_text_boxes_details_tuple]
                with fitz.open(stream=io.BytesIO(current_pdf_bytes), filetype="pdf") as doc_hl:
                    page_hl = doc_hl.load_page(page_idx)
                    derot_matrix = page_hl.derotation_matrix
                    for box_detail in fence_text_boxes_details:
                        rot_rect = fitz.Rect(box_detail['x0'], box_detail['y0'], box_detail['x1'], box_detail['y1'])
                        final_rect = rot_rect * derot_matrix if page_hl.rotation != 0 else rot_rect
                        final_rect.normalize()
                        if not final_rect.is_empty and final_rect.is_valid:
                            page_hl.draw_rect(final_rect, color=ui_color, width=ui_width, overlay=True)
                    pix_hl = page_hl.get_pixmap(dpi=display_dpi); highlighted_image_bytes = pix_hl.tobytes("png"); del pix_hl
    except Exception as e: print(f"SESSION {session_id_for_log} ERROR ({func_call_id}) in _generate_display_images_for_page_cached for page {page_idx}: {e}")
    render_duration = time.time() - render_start_time
    print(f"SESSION {session_id_for_log} CACHE_CALL_RENDER_TIME ({func_call_id}): Page {page_idx} took {render_duration:.4f}s for PyMuPDF rendering.")
    return original_image_bytes, highlighted_image_bytes

def generate_display_images_for_page_wrapper(page_result_data, session_id):
    page_idx = page_result_data.get('page_index_in_original_doc')
    pdf_hash = st.session_state.get('current_pdf_hash')

    if page_idx is None or pdf_hash is None: 
        print(f"SESSION {session_id} WARNING (wrapper): Missing page_idx or pdf_hash for on-demand image gen.")
        return None, None
        
    boxes_details = page_result_data.get('fence_text_boxes_details', [])
    details_tuple = tuple(tuple(sorted(d.items())) for d in sorted(boxes_details, key=lambda x: x.get('id', str(x)))) if boxes_details else tuple()
    
    return _generate_display_images_for_page_cached(
        page_idx, 
        pdf_hash,
        details_tuple, 
        HIGHLIGHT_COLOR_UI, 
        HIGHLIGHT_WIDTH_UI, 
        DISPLAY_IMAGE_DPI, 
        session_id
    )

def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base, session_id):
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
            base, ext = os.path.splitext(uploaded_pdf_name_base); fname = f"{base}_fence_highlights_ade{ext}"
        except Exception as e_s: print(f"SESSION {session_id} Err PDF tobytes: {e_s}"); fname=f"err_save_{uploaded_pdf_name_base}.pdf"
    if input_doc: input_doc.close()
    if output_doc: output_doc.close()
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)

# --- Main App Flow ---
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

    if openai_key and llm_analysis_instance and ade_key and google_cloud_config and \
       not st.session_state.run_analysis_triggered and \
       not st.session_state.processing_complete and \
       not st.session_state.analysis_halted_due_to_error:
        print(f"SESSION {current_session_id} LOG: Triggering analysis.")
        st.session_state.run_analysis_triggered = True

# --- Analysis Execution Block ---
if st.session_state.run_analysis_triggered and \
   st.session_state.original_pdf_bytes and \
   llm_analysis_instance and \
   ade_key and \
   google_cloud_config and \
   not st.session_state.analysis_halted_due_to_error and \
   not st.session_state.processing_complete:
    
    print(f"SESSION {current_session_id} LOG: Starting ADE-powered PDF processing loop.")
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
    
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results (Live) - ADE-powered</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty(); col_f, col_nf = st.columns(2)
    with col_f: st.subheader("✅ Fence-Related Pages")
    with col_nf: st.subheader("❌ Non-Fence Pages")
    prog_bar = st.progress(0); status_txt_area = st.empty()
    
    # Step 1: Process entire document with ADE once
    # Use session state API key if available (user might have entered it manually)
    ade_api_key_to_use = st.session_state.get('ade_api_key') or ade_key
    if not ade_api_key_to_use:
        st.error("❌ ADE API Key is missing! Please enter it in the sidebar.")
        st.session_state.analysis_halted_due_to_error = True
        st.stop()
    
    # Debug: Check key format (mask it for security)
    key_preview = ade_api_key_to_use[:10] + "..." + ade_api_key_to_use[-5:] if len(ade_api_key_to_use) > 15 else "***"
    print(f"SESSION {current_session_id} LOG: Using ADE API key: {key_preview} (length: {len(ade_api_key_to_use)})")
    
    full_doc_ade_result = None
    try:
        status_txt_area.text(f"Processing entire document with ADE (this may take a moment)...")
        with st.spinner("Running ADE on full document..."):
            full_doc_ade_result = ade_parse_document_official(
                st.session_state.original_pdf_bytes,
                ade_api_key_to_use,
                st.session_state.ade_zdr_mode
            )
        if full_doc_ade_result.get("success"):
            print(f"SESSION {current_session_id} LOG: ADE processed full document with {full_doc_ade_result['data'].get('total_pages', 0)} pages")
            st.success(f"✅ ADE successfully processed document ({full_doc_ade_result['data'].get('total_pages', 0)} pages)")
        else:
            error_msg = full_doc_ade_result.get('error', 'Unknown error')
            print(f"SESSION {current_session_id} ERROR: ADE full document processing failed: {error_msg}")
            if "401" in str(error_msg) or "Unauthorized" in str(error_msg):
                st.error(f"❌ ADE API Key is INVALID or EXPIRED!")
                st.warning(f"Error: {error_msg}")
                st.info("ℹ️ The app will continue using PDF text layer + OCR (fallback mode). ADE features will be disabled.")
                st.info("💡 To fix: Check your LandingAI API key in the sidebar - it may be wrong, expired, or have incorrect permissions.")
            else:
                st.warning(f"⚠️ ADE processing failed: {error_msg}")
                st.info("ℹ️ The app will continue using PDF text layer + OCR (fallback mode).")
            full_doc_ade_result = None
    except Exception as e:
        error_str = str(e)
        print(f"SESSION {current_session_id} ERROR: Failed to process full document with ADE: {e}")
        if "401" in error_str or "Unauthorized" in error_str:
            st.error(f"❌ ADE API Key authentication FAILED!")
            st.warning(f"Error: {error_str}")
            st.info("ℹ️ Continuing with PDF text layer + OCR extraction (ADE disabled).")
            st.info("💡 Your API key may be invalid, expired, or missing required permissions.")
        else:
            st.error(f"❌ ADE processing error: {error_str}")
            st.info("ℹ️ Continuing with PDF text layer + OCR extraction (fallback mode).")
        full_doc_ade_result = None
    
    try:
        for i in range(st.session_state.doc_total_pages):
            curr_pg_num = i + 1; st.session_state.total_pages_processed_count = curr_pg_num
            prog_bar.progress(curr_pg_num / st.session_state.doc_total_pages)
            status_txt_area.text(f"Processing Page {curr_pg_num}/{st.session_state.doc_total_pages} with ADE...")
            print(f"SESSION {current_session_id} LOG: Processing page {curr_pg_num} with ADE.")
            
            # Create single-page PDF for ADE processing
            single_page_pdf_bytes = create_single_page_pdf(st.session_state.original_pdf_bytes, i)
            if not single_page_pdf_bytes:
                print(f"SESSION {current_session_id} WARNING: Could not create single page PDF for page {curr_pg_num}")
                continue
            
            # Get page dimensions for coordinate conversion
            page_width, page_height = get_page_dimensions(single_page_pdf_bytes)
            
            # Extract page-specific chunks from full document ADE result (for indicator extraction later)
            page_chunks = []
            if full_doc_ade_result and full_doc_ade_result.get("success"):
                page_chunks = align_ade_chunks_to_page(
                    full_doc_ade_result,
                    i,
                    page_width,
                    page_height
                )
                print(f"SESSION {current_session_id} LOG: Aligned {len(page_chunks)} ADE chunks to page {curr_pg_num}")
            else:
                print(f"SESSION {current_session_id} LOG: No ADE chunks available for page {curr_pg_num} (using PDF text layer + OCR fallback)")
            
            # Get text content for fence detection (same as app.py)
            page_obj = doc_proc_loop.load_page(i)
            text_content = page_obj.get_text("text")
            
            # Prepare page data for analysis (same as app.py)
            page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
            analysis_res_core = {}; fatal_err_page = False
            
            try:
                with st.spinner(f"Page {curr_pg_num}: Core analysis..."):
                    analysis_res_core = analyze_page(
                        page_data_an, llm_analysis_instance, FENCE_KEYWORDS_APP, google_cloud_config
                    )

            except UnrecoverableRateLimitError as urle:
                msg = f"🛑 API Rate Limit Pg {curr_pg_num}: {urle}. Analysis halted."; status_txt_area.error(msg); st.error(msg)
                st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
            except Exception as e_core: 
                st.error(f"Core analysis error pg {curr_pg_num}: {e_core}"); 
                analysis_res_core = {"fence_found": False}; 
                print(f"SESSION {current_session_id} ERROR: Core analysis pg {curr_pg_num}: {e_core}")
            
            analysis_result = {
                **analysis_res_core, 
                'page_number': curr_pg_num, 
                'page_index_in_original_doc': i, 
                'fence_text_boxes_details': [], 
                'highlight_fence_text_app_setting': highlight_fence_text_app
            }
            
            # Extract highlight boxes using hybrid ADE + Google OCR approach
            # First, use the same highlighting as app.py for legend/text sections
            if not fatal_err_page and highlight_fence_text_app and analysis_result.get('text_found'):
                status_txt_area.text(f"Page {curr_pg_num}: Highlighting (text match found)...")
                single_pg_bytes_io = io.BytesIO(); temp_doc_single = None
                try: 
                    temp_doc_single = fitz.open()
                    temp_doc_single.insert_pdf(doc_proc_loop, from_page=i, to_page=i); temp_doc_single.save(single_pg_bytes_io)
                finally: 
                    if temp_doc_single: temp_doc_single.close()
                
                # Extract signals from analysis result to use for highlighting
                signals_for_highlighting = []
                try:
                    import json
                    text_resp = json.loads(analysis_result.get('text_response', '{}'))
                    signals_for_highlighting = text_resp.get('signals', [])
                    print(f"SESSION {current_session_id} LOG: Page {curr_pg_num} using signals for highlighting: {signals_for_highlighting}")
                except Exception as e:
                    print(f"SESSION {current_session_id} WARNING: Could not extract signals from analysis: {e}")
                
                try:
                    with st.spinner(f"Page {curr_pg_num}: Extracting highlight boxes..."):   
                        boxes,_,_ = get_fence_related_text_boxes(
                            single_pg_bytes_io.getvalue(),
                            llm_analysis_instance,
                            FENCE_KEYWORDS_APP,
                            signals_for_highlighting,  # Pass signals from LLM analysis
                            st.session_state.selected_model_for_analysis,
                            google_cloud_config
                        )

                        if boxes: 
                            analysis_result['fence_text_boxes_details'] = boxes
                            
                except UnrecoverableRateLimitError as urle_hl:
                    msg = f"🛑 API Rate Limit Highlight Pg {curr_pg_num}: {urle_hl}. Halted."; status_txt_area.error(msg); st.error(msg)
                    st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
                except Exception as e_hl: 
                    st.warning(f"Highlight error pg {curr_pg_num}: {e_hl}"); 
                    print(f"SESSION {current_session_id} WARNING: Highlight error pg {curr_pg_num}: {e_hl}")
                
            # ADE-based indicator extraction runs independently of text_found
            # This allows finding indicators even if LLM didn't detect text initially
            if not fatal_err_page and highlight_fence_text_app:
                print(f"SESSION {current_session_id} LOG: 🚀 STARTING ADE extraction for page {curr_pg_num}")
                status_txt_area.text(f"Page {curr_pg_num}: Running ADE-based indicator extraction...")
                
                # Use single_page_pdf_bytes that was already created at line 433
                # No need to recreate it - it's already available in this scope
                
                try:
                    with st.spinner(f"Page {curr_pg_num}: Running Google OCR for ADE extraction..."):
                        # Step 1: Run Google OCR on entire page
                        google_ocr_results = get_google_ocr_results_with_boxes(
                            single_page_pdf_bytes,
                            google_cloud_config,
                            curr_pg_num
                        )
                        print(f"SESSION {current_session_id} LOG: Google OCR found {len(google_ocr_results)} text elements on page {curr_pg_num}")
                    
                    if google_ocr_results:
                        print(f"SESSION {current_session_id} LOG: ✅ Google OCR successful, proceeding with extraction...")
                        st.info(f"✅ Page {curr_pg_num}: Google OCR found {len(google_ocr_results)} text elements")
                        # Extract PDF text layer words as fallback source (works even without ADE)
                        pdf_text_layer_words = []
                        try:
                            text_layer_doc = fitz.open(stream=BytesIO(single_page_pdf_bytes), filetype="pdf")
                            if len(text_layer_doc) > 0:
                                page = text_layer_doc[0]
                                words = page.get_text("words")
                                for word_tuple in words:
                                    x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
                                    pdf_text_layer_words.append({
                                        "text": text,
                                        "x0": float(x0),
                                        "y0": float(y0),
                                        "x1": float(x1),
                                        "y1": float(y1),
                                        "block_no": block_no,
                                        "line_no": line_no
                                    })
                            text_layer_doc.close()
                        except Exception as e:
                            print(f"SESSION {current_session_id} LOG: Error extracting PDF text layer: {e}")
                        
                        with st.spinner(f"Page {curr_pg_num}: Extracting legend keywords and indicators..."):
                            # Step 2: Highlight keywords and indicators in legend (using Google OCR boxes + PDF text layer)
                            # This function also extracts indicators internally
                            # Works with or without ADE chunks - now uses PDF text layer as fallback
                            legend_highlights = extract_legend_keywords_and_indicators(
                                page_chunks if page_chunks else [],
                                google_ocr_results,
                                FENCE_KEYWORDS_APP,
                                page_width,
                                page_height,
                                llm=llm_analysis_instance,
                                pdf_text_layer_words=pdf_text_layer_words if pdf_text_layer_words else None
                            )
                            print(f"SESSION {current_session_id} LOG: ✅ FOUND {len(legend_highlights)} legend highlights on page {curr_pg_num}")
                            if legend_highlights:
                                print(f"SESSION {current_session_id} LOG: Sample legend highlights: {[(h.get('text', '')[:20], h.get('tag_from_llm', '')) for h in legend_highlights[:5]]}")
                                st.success(f"✅ Page {curr_pg_num}: Found {len(legend_highlights)} legend highlights!")
                            else:
                                st.warning(f"⚠️ Page {curr_pg_num}: No legend highlights found")
                        
                        # Step 3: Extract indicators from legend for figure search
                        # Extract indicators separately from the chunks (they were extracted in step 2 but we need them for figures)
                        from utils_ade_official import extract_indicators_from_table_llm, extract_indicators_from_text_llm
                        
                        indicators_from_legend = []
                        if llm_analysis_instance:
                            # Process tables first (only if we have ADE chunks)
                            if page_chunks:
                                for chunk in page_chunks:
                                    chunk_type = chunk.get("type", "").lower()
                                    if chunk_type == "table":
                                        table_markdown = chunk.get("markdown", chunk.get("text", ""))
                                        if table_markdown:
                                            table_box = {
                                                "x0": chunk.get("x0", 0),
                                                "y0": chunk.get("y0", 0),
                                                "x1": chunk.get("x1", 0),
                                                "y1": chunk.get("y1", 0)
                                            }
                                            if table_box.get("x1", 0) > 0:
                                                table_ocr = filter_ocr_by_ade_regions(google_ocr_results, [table_box])
                                                table_indicators = extract_indicators_from_table_llm(table_markdown, FENCE_KEYWORDS_APP, llm_analysis_instance, table_ocr)
                                            else:
                                                table_indicators = extract_indicators_from_table_llm(table_markdown, FENCE_KEYWORDS_APP, llm_analysis_instance)
                                            indicators_from_legend.extend(table_indicators)
                                
                                # Process text chunks
                                for chunk in page_chunks:
                                    chunk_type = chunk.get("type", "").lower()
                                    if chunk_type == "text":
                                        text_markdown = chunk.get("markdown", chunk.get("text", ""))
                                        if text_markdown:
                                            text_lower = text_markdown.lower()
                                            if any(kw in text_lower for kw in ["keynote", "legend", "note", "symbol"]) or len(text_markdown) < 500:
                                                text_indicators = extract_indicators_from_text_llm(text_markdown, FENCE_KEYWORDS_APP, llm_analysis_instance)
                                                indicators_from_legend.extend(text_indicators)
                            
                            # If no ADE chunks, indicators were extracted inside extract_legend_keywords_and_indicators
                            # We need to extract them from the text layer or OCR for figure matching
                            if not page_chunks and pdf_text_layer_words:
                                # Group text layer into lines and extract indicators
                                text_layer_lines = []
                                current_line = []
                                prev_y = None
                                
                                for word in sorted(pdf_text_layer_words, key=lambda x: (x.get("y0", 0), x.get("x0", 0))):
                                    text = word.get("text", "").strip()
                                    if not text:
                                        continue
                                    
                                    y = word.get("y0", 0)
                                    if prev_y is None or abs(y - prev_y) < 10:
                                        current_line.append(text)
                                    else:
                                        if current_line:
                                            text_layer_lines.append(" ".join(current_line))
                                        current_line = [text]
                                    prev_y = y
                                
                                if current_line:
                                    text_layer_lines.append(" ".join(current_line))
                                
                                # Filter lines that might contain indicators
                                filtered_lines = []
                                for line in text_layer_lines:
                                    line_lower = line.lower()
                                    if (len(line) <= 15 or 
                                        any(kw in line_lower for kw in FENCE_KEYWORDS_APP) or
                                        re.search(r'\b(0\d{3}|[3-9]\d{3})\b', line)):
                                        filtered_lines.append(line)
                                
                                if filtered_lines:
                                    text_layer_text = "\n".join(filtered_lines[:100])
                                    ocr_indicators = extract_indicators_from_text_llm(text_layer_text, FENCE_KEYWORDS_APP, llm_analysis_instance)
                                    indicators_from_legend.extend(ocr_indicators)
                                    print(f"SESSION {current_session_id} LOG: Extracted {len(ocr_indicators)} indicators from text layer (no ADE chunks)")
                        
                        print(f"SESSION {current_session_id} LOG: Extracted {len(indicators_from_legend)} indicators from legend on page {curr_pg_num}: {[ind['indicator'] for ind in indicators_from_legend[:10]]}")
                        
                        with st.spinner(f"Page {curr_pg_num}: Finding indicators in figures..."):
                            # Step 4: Highlight indicators inside figures/architectural_drawings (using Google OCR boxes)
                            # Pass LLM instance for intelligent matching
                            figure_highlights = find_indicators_in_figures(
                                indicators_from_legend,
                                page_chunks,
                                google_ocr_results,
                                page_width,
                                page_height,
                                llm=llm_analysis_instance,
                                fence_keywords=FENCE_KEYWORDS_APP
                            )
                            print(f"SESSION {current_session_id} LOG: Found {len(figure_highlights)} figure highlights on page {curr_pg_num}")
                        
                        # Combine ADE-based highlights with existing highlights
                        ade_highlights = legend_highlights + figure_highlights
                        print(f"SESSION {current_session_id} LOG: ADE highlights summary - legend: {len(legend_highlights)}, figure: {len(figure_highlights)}, total: {len(ade_highlights)}")
                        if ade_highlights:
                            # Merge with existing highlights from get_fence_related_text_boxes
                            existing_boxes = analysis_result.get('fence_text_boxes_details', [])
                            # Combine and deduplicate (simple merge for now)
                            all_highlights = existing_boxes + ade_highlights
                            analysis_result['fence_text_boxes_details'] = all_highlights
                            print(f"SESSION {current_session_id} LOG: ✅ MERGED ADE HIGHLIGHTS! Total {len(all_highlights)} highlights (standard: {len(existing_boxes)}, ADE indicators: {len(ade_highlights)}) for page {curr_pg_num}")
                            # Show visible UI message
                            st.success(f"🎯 Page {curr_pg_num}: ADE extraction SUCCESS! Found {len(ade_highlights)} new highlights (merged with {len(existing_boxes)} existing). Total: {len(all_highlights)} highlights")
                            # Debug: Show first few highlights
                            if ade_highlights:
                                print(f"SESSION {current_session_id} LOG: First 3 ADE highlights: {[(h.get('text', '')[:20], h.get('tag_from_llm', '')) for h in ade_highlights[:3]]}")
                                with st.expander(f"🔍 View ADE highlights for page {curr_pg_num}", expanded=False):
                                    for i, h in enumerate(ade_highlights[:10]):
                                        st.text(f"{i+1}. '{h.get('text', '')[:30]}' - {h.get('tag_from_llm', 'N/A')}")
                        else:
                            print(f"SESSION {current_session_id} LOG: ⚠️ No ADE-based indicator highlights found on page {curr_pg_num}")
                            st.warning(f"⚠️ Page {curr_pg_num}: No ADE highlights found")
                    else:
                        print(f"SESSION {current_session_id} WARNING: No Google OCR results for ADE indicator extraction on page {curr_pg_num}")
                        st.warning(f"⚠️ Page {curr_pg_num}: No Google OCR results - cannot run ADE extraction")
                        
                except Exception as e_ade: 
                    st.error(f"❌ ADE extraction ERROR on page {curr_pg_num}: {e_ade}")
                    print(f"SESSION {current_session_id} WARNING: ADE indicator extraction error pg {curr_pg_num}: {e_ade}")
                    import traceback
                    traceback.print_exc()
            elif not fatal_err_page and highlight_fence_text_app and analysis_result.get('fence_found'):
                 status_txt_area.text(f"Page {curr_pg_num}: Fence found, no text match for detailed highlighting.")
            
            if fatal_err_page: break
            target_col = col_f if analysis_result.get('fence_found') else col_nf
            (st.session_state.fence_pages if analysis_result.get('fence_found') else st.session_state.non_fence_pages).append(analysis_result)
            
            # Manage memory to prevent 413 errors
            if curr_pg_num % 5 == 0:  # Check every 5 pages
                manage_session_memory()
            
            with target_col: # Display Logic
                exp_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    reasons = []; 
                    if analysis_result.get('text_found'): reasons.append("Text")
                    if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app: reasons.append("Hybrid Highlights")
                    if reasons: exp_title += f" ({' & '.join(reasons)} Match)"
                with st.expander(exp_title, expanded=True):
                    img_col, det_col = st.columns([2,1])
                    print(f"SESSION {current_session_id} DEBUG LIVE DISPLAY Page {analysis_result['page_number']}: Num boxes: {len(analysis_result.get('fence_text_boxes_details', []))}")
                    wrapper_call_start_time = time.time()
                    with st.spinner(f"Rendering image for page {analysis_result['page_number']}..."): 
                        orig_b, hl_b = generate_display_images_for_page_wrapper(analysis_result, current_session_id)
                    wrapper_call_duration = time.time() - wrapper_call_start_time
                    print(f"SESSION {current_session_id} PERF_LOG: generate_display_images_for_page_wrapper Page {curr_pg_num} took {wrapper_call_duration:.4f}s.")
                    with img_col: # Image display
                        disp_img_ui = hl_b if hl_b else orig_b
                        if disp_img_ui: st.image(disp_img_ui, caption=f"Page {analysis_result['page_number']}{' (ADE Highlighted)' if hl_b else ''}")
                        dl_links_html_live = []
                        if hl_b: dl_links_html_live.append(get_image_download_link_html(hl_b, f"page_{analysis_result['page_number']}_hl_ade.png", "DL ADE HL Img"))
                        if orig_b: dl_links_html_live.append(get_image_download_link_html(orig_b, f"page_{analysis_result['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_live: st.markdown(" ".join(dl_links_html_live), unsafe_allow_html=True)
                    with det_col: # Text details display
                        st.markdown("##### Analysis Details")
                        if analysis_result.get('fence_found'):
                            pts = []; 
                            if analysis_result.get('text_found'): pts.append("✔️ Text")
                            if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app : pts.append("✔️ ADE Highlights")
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
                            st.markdown("---"); st.markdown("**Highlights (from ADE):**")
                            disp_set_live = set(); count_live = 0
                            for d_item_live in sorted(details_list, key=lambda x: x.get('y0', 0)):
                                txt_live = d_item_live.get('text', "N/A"); tag_live = d_item_live.get('tag_from_llm', 'N/A'); type_llm_live = d_item_live.get('type_from_llm', 'N/A')
                                display_text_live = f"- `{txt_live}` (Type: {type_llm_live}, Tag: {tag_live})"
                                if display_text_live not in disp_set_live: st.markdown(display_text_live); disp_set_live.add(display_text_live); count_live+=1
                                if count_live >=15 and len(details_list) > 17: st.markdown(f"- ...& {len(details_list)-count_live} more."); break
            summary_placeholder.markdown(f"### Summary (Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages})\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}")
            time.sleep(0.05) 
    except Exception as e:
        if "413" in str(e) or "Request Entity Too Large" in str(e):
            st.error("🚨 413 ERROR: Request too large. Clearing large data and retrying...")
            # Clear large data
            st.session_state.original_pdf_bytes = None
            st.session_state.highlighted_pdf_bytes_for_download = None
            st.cache_data.clear()
            import gc
            gc.collect()
            st.error("Please refresh the page and try with a smaller PDF.")
            st.stop()
        else:
            st.error(f"Processing error: {e}")
            st.stop()
    finally: 
        if doc_proc_loop:
            doc_proc_loop.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document in finally block.")
        doc_proc_loop = None 
    st.session_state.processing_complete = True 
    if not st.session_state.analysis_halted_due_to_error:
        prog_bar.empty(); status_txt_area.success("All pages processed with ADE!")
        if st.session_state.fence_pages and st.session_state.original_pdf_bytes:
            pdf_b, pdf_n = generate_combined_highlighted_pdf(st.session_state.original_pdf_bytes, st.session_state.fence_pages, st.session_state.uploaded_pdf_name, current_session_id)
            if pdf_b: st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download = pdf_b, pdf_n
            else: st.warning(f"Could not generate PDF: {pdf_n}")
    else: prog_bar.empty() 
    final_summary_text = f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    summary_placeholder.markdown(final_summary_text)
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button("⬇️ Download Highlighted Fence Pages (ADE) (PDF)", st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download, "application/pdf", key="dl_combined_pdf_main")

elif st.session_state.processing_complete: 
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results - ADE-powered</h2>", unsafe_allow_html=True)
    final_summary_text_rerun = f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    st.markdown(final_summary_text_rerun)
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
         st.download_button("⬇️ Download Highlighted Fence Pages (ADE) (PDF)", st.session_state.highlighted_pdf_bytes_for_download, st.session_state.highlighted_pdf_filename_for_download, "application/pdf", key="dl_combined_pdf_rerun")
    col_f_res, col_nf_res = st.columns(2)
    with col_f_res: st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
    with col_nf_res: st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")
    def display_page_result_expander(res_data_list, target_column_res, session_id_for_display):
        for res_data_item in res_data_list:
            with target_column_res:
                exp_title_res = f"Page {res_data_item['page_number']}"
                if res_data_item.get('fence_found'):
                    reasons_res = []; 
                    if res_data_item.get('text_found'): reasons_res.append("Text")
                    if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting', True): reasons_res.append("ADE Highlights")
                    if reasons_res: exp_title_res += f" ({' & '.join(reasons_res)} Match)"
                with st.expander(exp_title_res, expanded=False):
                    img_col_r, det_col_r = st.columns([2,1])
                    with st.spinner(f"Loading image page {res_data_item['page_number']}..."):
                        orig_b_r, hl_b_r = generate_display_images_for_page_wrapper(res_data_item, session_id_for_display)
                    with img_col_r: # Image display
                        disp_img_r = hl_b_r if hl_b_r else orig_b_r
                        if disp_img_r: st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}{' (ADE HL)' if hl_b_r else ''}")
                        dl_links_html_rerun = []
                        if hl_b_r: dl_links_html_rerun.append(get_image_download_link_html(hl_b_r, f"page_{res_data_item['page_number']}_hl_ade.png", "DL ADE HL Img"))
                        if orig_b_r: dl_links_html_rerun.append(get_image_download_link_html(orig_b_r, f"page_{res_data_item['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_rerun: st.markdown(" ".join(dl_links_html_rerun), unsafe_allow_html=True)
                    with det_col_r: # Text details
                        st.markdown("##### Analysis Details")
                        if res_data_item.get('fence_found'):
                            pts_r = [] 
                            if res_data_item.get('text_found'): pts_r.append("✔️ Text")
                            if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting',True) : pts_r.append("✔️ ADE Highlights")
                            if not pts_r: pts_r.append("Fence flagged")
                            st.markdown("\n".join(f"- {s}" for s in pts_r))
                        else: st.markdown("No strong fence indicators.")
                        if res_data_item.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{res_data_item['text_response']}_")
                        if res_data_item.get('text_snippet'): st.markdown("---"); st.code(res_data_item['text_snippet'],language=None)
                        if res_data_item.get('highlight_fence_text_app_setting', True) and \
                           res_data_item.get('fence_text_boxes_details') and res_data_item.get('fence_found'):
                            details_list_r = res_data_item['fence_text_boxes_details']
                            st.markdown("---"); st.markdown("**Highlights (from ADE):**")
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
elif not ade_key: st.error("LandingAI ADE API key not found. Check configuration.")
elif st.session_state.analysis_halted_due_to_error: st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---"); st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App (ADE-powered)</p>", unsafe_allow_html=True)
