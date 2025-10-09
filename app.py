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
import traceback
import sys

try:
    import psutil
    import os
    import resource
    def _rss_mb():
        try:
            # Try psutil first
            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except:
            try:
                # Fallback to resource module (works on Linux/Unix)
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on Linux
            except:
                return 0.0
except ImportError:
    try:
        import resource
        def _rss_mb():
            try:
                # resource.getrusage works on Unix systems
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
            except:
                return 0.0
    except ImportError:
        def _rss_mb(): 
            return 0.0

# Setup comprehensive error logging
def log_exception(session_id, context, exception):
    """Log detailed exception information"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"\n{'='*80}")
    print(f"SESSION {session_id} FATAL ERROR in {context}")
    print(f"Exception Type: {type(exception).__name__}")
    print(f"Exception Message: {str(exception)}")
    print(f"Memory Usage: {_rss_mb():.1f} MB")
    print(f"Traceback:\n{tb_str}")
    print(f"{'='*80}\n")
    return tb_str

# ============================================
# MEMORY PROFILER - Track every function call
# ============================================
class MemoryProfiler:
    """Track memory per function to find leaks"""
    def __init__(self):
        self.page_data = []
        self.current_page = None
    
    def start_page(self, page_num):
        self.current_page = {'page': page_num, 'steps': [], 'start_mem': _rss_mb()}
        print(f"\n📄 Page {page_num} START: {self.current_page['start_mem']:.1f}MB")
    
    def record_step(self, step_name, details=""):
        if not self.current_page:
            return
        mem_now = _rss_mb()
        last_mem = self.current_page['steps'][-1]['mem'] if self.current_page['steps'] else self.current_page['start_mem']
        delta = mem_now - last_mem
        self.current_page['steps'].append({
            'name': step_name,
            'mem': mem_now,
            'delta': delta,
            'details': details
        })
        sign = "+" if delta >= 0 else ""
        print(f"      {step_name}: {mem_now:.1f}MB ({sign}{delta:.1f}MB) {details}")
    
    def end_page(self):
        if not self.current_page:
            return 0
        end_mem = _rss_mb()
        net = end_mem - self.current_page['start_mem']
        self.page_data.append(self.current_page)
        print(f"   📊 Page {self.current_page['page']} NET: {net:+.1f}MB (start={self.current_page['start_mem']:.1f}MB, end={end_mem:.1f}MB)")
        return net

# Global profiler instance
profiler = MemoryProfiler()

# Memory profiling decorator
def profile_memory(func_name):
    """Decorator to profile memory usage of a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import gc
            gc.collect()  # Clean up before measuring
            mem_before = _rss_mb()
            result = func(*args, **kwargs)
            gc.collect()  # Clean up after
            mem_after = _rss_mb()
            delta = mem_after - mem_before
            if delta > 5:  # Only log if significant memory change
                print(f"🔍 MEMORY PROFILE [{func_name}]: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ {delta:+.1f}MB)")
            return result
        return wrapper
    return decorator

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)
HIGHLIGHT_WIDTH_UI = 1.5
HIGHLIGHT_COLOR_PDF = (0, 0.9, 0)
HIGHLIGHT_WIDTH_PDF = 1.5
DISPLAY_IMAGE_DPI = 72  

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
        'fence_keywords_app': [
            'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link',
            'screen wall', 'privacy screen', 'CMU wall', 'masonry wall', 'wall', 'bollard',
            'railing', 'handrail', 'security barrier', 'perimeter'
        ],
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
    # Configuration and API key loading
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
    
    # Memory monitoring
    st.caption(f"RAM: {_rss_mb():.1f} MB")
    
    # Memory flush button
    st.markdown("---")
    if st.button("🧹 Flush Memory & Reset", key="flush_memory_btn", help="Clean up temp files and clear all session data"):
        import gc
        
        # Delete temp file if exists
        if st.session_state.get('temp_pdf_path') and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.unlink(st.session_state.temp_pdf_path)
                print(f"SESSION {current_session_id} LOG: Flushed temp file: {st.session_state.temp_pdf_path}")
            except Exception as e:
                print(f"SESSION {current_session_id} WARNING: Could not delete temp file during flush: {e}")
        
        # Clear all caches
        st.cache_data.clear()
        
        # Reset session state (keep only model preference and keywords)
        model_backup = st.session_state.get('selected_model_for_analysis')
        keywords_backup = st.session_state.get('fence_keywords_app')
        
        for key in list(st.session_state.keys()):
            if key not in ['selected_model_for_analysis', 'fence_keywords_app', 'model_selector_radio', 'highlight_toggle', 'kw_text_area', 'update_kw_btn', 'flush_memory_btn']:
                del st.session_state[key]
        
        # Restore preferences
        if model_backup:
            st.session_state.selected_model_for_analysis = model_backup
        if keywords_backup:
            st.session_state.fence_keywords_app = keywords_backup
        
        # Aggressive GC
        for _ in range(10):
            gc.collect()
        
        st.success(f"✅ Memory flushed! RAM: {_rss_mb():.1f} MB")
        st.rerun()


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
@st.cache_data(ttl=180, show_spinner=False, max_entries=10)  # 3 min, ~10 pages cached (reduced from 18 to save 120MB)
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
    # Get PDF bytes from temp file (during processing) or session state (before processing)
    pdf_bytes_source = None
    if st.session_state.get('temp_pdf_path'):
        # Read from temp file
        try:
            with open(st.session_state.temp_pdf_path, 'rb') as f:
                pdf_bytes_source = f.read()
        except:
            pass
    if not pdf_bytes_source:
        pdf_bytes_source = st.session_state.get('original_pdf_bytes')
    
    if not pdf_bytes_source:
        print(f"SESSION {session_id_for_log} ERROR (_cached): PDF not available for hash {pdf_hash_for_cache_key}")
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

    current_pdf_bytes = pdf_bytes_source
    original_image_bytes, highlighted_image_bytes = None, None

    func_call_id = str(uuid.uuid4())[:4]
    print(f"SESSION {session_id_for_log} CACHE_CALL ({func_call_id}): _generate_display_images_for_page_cached "
          f"pg={page_idx}, hash={pdf_hash_for_cache_key[:8]}…, rects={len(rects)}")

    t0 = time.time()
    try:
        # Render original preview is disabled by default to reduce memory
        # If you want it as an option, render on demand in the UI instead.
        # (Keep original_image_bytes=None)

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
                pix_hl = page_hl.get_pixmap(dpi=display_dpi, alpha=False)
                highlighted_image_bytes = pix_hl.tobytes("jpg", jpg_quality=58)
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


# DELETED: compute_doc_legend_and_refs_compact() function (lines 367-495)
# This was 130 lines of dead code for cross-reference analysis (never called)

def merge_extra_keywords(signals: list) -> list:
    """Return page-local signals only (cross-refs deleted)."""
    return list(signals or [])



# --- Main App Flow ---
st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}" # Used for detecting new file
    
    if st.session_state.last_uploaded_file_id != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Resetting state for {current_file_id}.")
        # Preserve user preferences
        current_selected_model = st.session_state.get('selected_model_for_analysis', "gpt-4o")
        current_keywords = st.session_state.get('fence_keywords_app', ['fence'])

        # Hard reset memory-heavy state
        st.session_state.update({
            'fence_pages': [],
            'non_fence_pages': [],
            'total_pages_processed_count': 0,
            'doc_total_pages': 0,
            'processing_complete': False,
            'analysis_halted_due_to_error': False,
            'legend_id_list': [],
            'page_refs': {},
            'legend_index_compact': [],
            'highlighted_pdf_bytes_for_download': None,
            'highlighted_pdf_filename_for_download': None,
            'run_analysis_triggered': False,
        })

        # CLEANUP: Delete old temp file if exists
        if st.session_state.get('temp_pdf_path') and os.path.exists(st.session_state.temp_pdf_path):
            try:
                os.unlink(st.session_state.temp_pdf_path)
                print(f"SESSION {current_session_id} LOG: Cleaned up old temp file on new upload: {st.session_state.temp_pdf_path}")
            except Exception as e_cleanup:
                print(f"SESSION {current_session_id} WARNING: Could not delete old temp file: {e_cleanup}")
        
        # Store new file bytes & hash
        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.current_pdf_hash = hashlib.sha256(st.session_state.original_pdf_bytes).hexdigest()
        st.session_state.last_uploaded_file_id = current_file_id

        # Restore user prefs
        st.session_state.selected_model_for_analysis = current_selected_model
        st.session_state.fence_keywords_app = current_keywords
        
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
    print(f"SESSION {current_session_id} LOG: Memory BEFORE temp file: {_rss_mb():.1f} MB")
    
    # CRITICAL: Write PDF to temp file FIRST, then open from file (not from memory!)
    import tempfile
    temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_pdf_path.write(st.session_state.original_pdf_bytes)
    temp_pdf_path.close()
    st.session_state.temp_pdf_path = temp_pdf_path.name
    print(f"SESSION {current_session_id} LOG: Wrote PDF to temp file: {st.session_state.temp_pdf_path}")
    
    # NOW free the 149MB from session state BEFORE opening
    st.session_state.original_pdf_bytes = None
    import gc
    gc.collect()
    gc.collect()  # Call twice
    print(f"SESSION {current_session_id} LOG: Freed 149MB from session state. RAM: {_rss_mb():.1f} MB")
    
    # OPEN PDF FROM FILE (not from memory!)
    doc_proc_loop = None
    try:
        doc_proc_loop = fitz.open(st.session_state.temp_pdf_path)  # Open from FILE, not memory!
        st.session_state.doc_total_pages = len(doc_proc_loop)
        print(f"SESSION {current_session_id} LOG: PDF opened from file, {st.session_state.doc_total_pages} pages. RAM: {_rss_mb():.1f} MB")
    except Exception as e:
        st.error(f"Failed to open PDF: {e}"); st.session_state.processing_complete = True; st.session_state.analysis_halted_due_to_error = True
        if doc_proc_loop: doc_proc_loop.close()
        print(f"SESSION {current_session_id} ERROR: Failed to open PDF for processing: {e}")
        st.stop() 
        # --- Cross-reference analysis disabled to save memory ---
    # Skip cross-reference analysis entirely to prevent memory crashes
    st.session_state.legend_id_list = []
    st.session_state.page_refs = {}
    st.session_state.legend_index_compact = []
    print(f"[CROSS] Cross-reference analysis disabled for memory optimization. RAM: {_rss_mb():.1f} MB")
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results (Live)</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty(); col_f, col_nf = st.columns(2)
    with col_f: st.subheader("✅ Fence-Related Pages")
    with col_nf: st.subheader("❌ Non-Fence Pages")
    prog_bar = st.progress(0); status_txt_area = st.empty()
    try:
        print(f"SESSION {current_session_id} LOG: Starting processing loop for {st.session_state.doc_total_pages} pages")
        print(f"SESSION {current_session_id} LOG: Initial memory: {_rss_mb():.1f} MB")
        for i in range(st.session_state.doc_total_pages):
            curr_pg_num = i + 1; st.session_state.total_pages_processed_count = curr_pg_num
            prog_bar.progress(curr_pg_num / st.session_state.doc_total_pages)
            status_txt_area.text(f"Processing Page {curr_pg_num}/{st.session_state.doc_total_pages}...")
            memory_usage = _rss_mb()
            if memory_usage == 0.0:
                # Fallback memory monitoring if psutil fails
                import sys
                memory_usage = sys.getsizeof(st.session_state) / (1024**2)
                memory_usage += sum(sys.getsizeof(v) for v in st.session_state.values() if hasattr(v, '__len__')) / (1024**2)
            print(f"SESSION {current_session_id} LOG: Processing page {curr_pg_num}. RAM: {memory_usage:.1f} MB")
            
            # START PROFILING THIS PAGE
            profiler.start_page(curr_pg_num)
            
            # AGGRESSIVE MEMORY MANAGEMENT - Force cleanup BEFORE processing
            # Profiling shows DocAI responses are tiny (< 12KB), but Python objects accumulate
            import gc
            
            # Step 1: Force immediate garbage collection (5x for thorough cleanup)
            for _ in range(8):
                gc.collect()
            profiler.record_step("1. GC cleanup (8×)")
            
            # Step 2: Clear Streamlit caches more aggressively
            if i >= 14 and i < 30:
                # Critical zone - clear EVERY page
                st.cache_data.clear()
                profiler.record_step("2. Cache clear (critical zone)")
            elif i % 3 == 0 and i > 0:  # Every 3 pages (was 5)
                st.cache_data.clear()
                profiler.record_step("2. Cache clear")
            
            # Step 3: Explicitly null out previous page data if it exists
            if i > 0:
                # Force release of any lingering references
                if 'page_obj' in locals():
                    try:
                        del page_obj
                    except:
                        pass
                if 'text_content' in locals():
                    try:
                        del text_content
                    except:
                        pass
                if 'single_page_pdf_bytes' in locals():
                    try:
                        del single_page_pdf_bytes
                    except:
                        pass
                # Force GC again after deletions
                gc.collect()
                gc.collect()
            
            # Step 4: CRITICAL - Close and reopen PDF document every 2 pages
            # PyMuPDF keeps loaded pages in C-level memory that Python GC can't free
            # Image generation on complex pages creates 100+ MB temporary buffers
            # Server shows higher memory usage than local - reopen more frequently
            if i > 0 and i % 2 == 0:
                try:
                    doc_proc_loop.close()
                    # Force GC after close to ensure Python objects are freed
                    for _ in range(5):
                        gc.collect()
                    doc_proc_loop = fitz.open(st.session_state.temp_pdf_path)
                    profiler.record_step("2b. Close/Reopen PDF", f"page {i} (free C memory)")
                    # Clear cache after reopen to maximize memory release
                    st.cache_data.clear()
                    # Force GC after document reload
                    for _ in range(5):
                        gc.collect()
                except Exception as e:
                    print(f"SESSION {current_session_id} WARNING: Could not reopen PDF: {e}")
            
            # Check memory usage and halt if too high
            current_memory = _rss_mb()
            # Streamlit Cloud has 1GB (1024MB) hard limit
            memory_limit = 950  # Safe limit with OCR enabled for all pages
            
            if current_memory > memory_limit:
                error_msg = f"⚠️ Memory usage too high ({current_memory:.1f} MB). Stopping analysis to prevent crash."
                st.error(error_msg)
                st.warning("💡 Tip: You can download the partial results below and resume processing later.")
                status_txt_area.error(error_msg)
                st.session_state.analysis_halted_due_to_error = True
                print(f"SESSION {current_session_id} ERROR: {error_msg}")
                break
            # Load page
            page_obj = doc_proc_loop.load_page(i)
            profiler.record_step("3. load_page()")
            
            # Extract text
            text_content = page_obj.get_text("text")
            profiler.record_step("4. get_text()", f"len={len(text_content)}")
            
            # CRITICAL FIX: For large pages, skip page_bytes entirely to prevent memory accumulation
            # Testing shows memory grows from 730MB → 1427MB peak (memory leak!)
            # Generate page_bytes for ALL pages (including large) for better OCR accuracy
            # Testing showed 233% improvement with only 7.7MB additional memory cost
            single_page_pdf_bytes = None
            page_width = page_obj.rect.width
            page_height = page_obj.rect.height
            is_large_page = page_width > 2000 or page_height > 2000
            
            # Set DPI based on page size
            # Profiling proves DocAI responses are tiny (< 12 KB even for complex pages)
            # The memory issue is Python object retention, not OCR data size
            # Therefore: Keep DPI=30 for quality, focus on aggressive GC instead
            
            if is_large_page:
                dpi = 30  # Optimal DPI - responses are only ~2-12 KB
                profiler.record_step("→ Large page", f"{page_width:.0f}×{page_height:.0f}, DPI={dpi}")
            else:
                dpi = 45  # Normal DPI for small pages
                profiler.record_step("→ Normal page", f"{page_width:.0f}×{page_height:.0f}, DPI={dpi}")
            
            try:
                # Generate page_bytes for ALL pages with adaptive DPI
                pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
                pix_width, pix_height = pix.width, pix.height
                profiler.record_step("5. get_pixmap()", f"size={pix_width}x{pix_height}")
                
                # Convert to PNG bytes and FREE pixmap immediately
                img_bytes = pix.tobytes("png")
                pix_size_mb = len(img_bytes) / (1024*1024)
                del pix
                gc.collect()
                gc.collect()
                profiler.record_step("6. tobytes() + free pixmap", f"{pix_size_mb:.2f}MB PNG")
                
                # Wrap PNG in minimal PDF wrapper
                from io import BytesIO
                temp_img_doc = fitz.open()
                temp_page = temp_img_doc.new_page(width=pix_width, height=pix_height)
                temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
                single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
                temp_img_doc.close()
                profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")
                
                # Cleanup img_bytes
                del img_bytes
                gc.collect()
                gc.collect()
                profiler.record_step("8. Cleanup img_bytes")
                
                # EXTRA aggressive cleanup for large pages to prevent memory accumulation
                if is_large_page:
                    del temp_page
                    gc.collect()
                    gc.collect()
                    gc.collect()
                    profiler.record_step("8b. Extra cleanup (large page)")
            except Exception as e:
                print(f"SESSION {current_session_id} WARNING: Could not create page image for page {curr_pg_num}: {e}")
                single_page_pdf_bytes = None  # Fallback to text-only if image generation fails
            
            page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
            analysis_res_core = {}; fatal_err_page = False
            try:
                with st.spinner(f"Page {curr_pg_num}: Core analysis..."):
                    try:
                    analysis_res_core = analyze_page(
                        page_data_an, llm_analysis_instance, FENCE_KEYWORDS_APP, google_cloud_config,
                        recall_mode="strict"   # or "balanced"/"high"
                    )
                        profiler.record_step("9. analyze_page()", f"fence={analysis_res_core.get('fence_found')}")
                        
                        jr = json.loads(analysis_res_core["text_response"])
                        signals = jr.get("signals", [])
                    except Exception:
                        signals = []
                    profiler.record_step("10. Extract signals", f"count={len(signals)}")
            except MemoryError as me:
                tb = log_exception(current_session_id, f"Core Analysis Page {curr_pg_num} (MemoryError)", me)
                st.error(f"💥 Memory error processing page {curr_pg_num}. Skipping OCR analysis.")
                analysis_res_core = {"fence_found": False, "text_found": False}
                signals = []
            except UnrecoverableRateLimitError as urle:
                msg = f"🛑 API Rate Limit Pg {curr_pg_num}: {urle}. Analysis halted."; status_txt_area.error(msg); st.error(msg)
                st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
            except Exception as e:
                tb = log_exception(current_session_id, f"Core Analysis Page {curr_pg_num}", e)
                st.warning(f"⚠️ Analysis error on page {curr_pg_num}: {str(e)[:100]}...")
                analysis_res_core = {"fence_found": False, "text_found": False}
                signals = []
            analysis_result = {**analysis_res_core, 'page_number': curr_pg_num, 'page_index_in_original_doc': i, 'fence_text_boxes_details': [], 'highlight_fence_text_app_setting': highlight_fence_text_app}
            # OCR HIGHLIGHTING (ALWAYS RUN - no memory-based skipping per user request)
            if not fatal_err_page and highlight_fence_text_app and analysis_result.get('text_found'):
                status_txt_area.text(f"Page {curr_pg_num}: Highlighting (text match found)...")
                try:
                    with st.spinner(f"Page {curr_pg_num}: Extracting highlight boxes..."):   
                        # Page_bytes now available for all pages (OCR enabled globally)
                        if single_page_pdf_bytes:
                            # Validate signals against page text before using as keywords
                            validated_signals = []
                            page_text_lower = text_content.lower()
                            for sig in signals:
                                if sig and sig.lower() in page_text_lower:
                                    validated_signals.append(sig)
                            if len(validated_signals) < len(signals):
                                print(f"SESSION {current_session_id} LOG: Filtered signals {len(signals)}→{len(validated_signals)} (only those in page text)")
                            
                            boxes,_,_ = get_fence_related_text_boxes(
                                single_page_pdf_bytes,
                                llm_analysis_instance,
                                FENCE_KEYWORDS_APP,
                                merge_extra_keywords(validated_signals),
                                st.session_state.selected_model_for_analysis,
                                google_cloud_config
                            )

                            # Note: No coordinate scaling needed - page_bytes already at correct DPI
                            # Large pages use DPI=30, small pages use DPI=45
                            # OCR coordinates match the page_bytes dimensions, no conversion needed
                            
                            if boxes:
                                analysis_result['fence_text_boxes_details'] = boxes
                            profiler.record_step("11. OCR highlighting", f"boxes={len(boxes) if boxes else 0}")
                    else:
                        # Fallback: no page_bytes available (image generation failed)
                        profiler.record_step("11. OCR highlighting", "skipped (no page_bytes)")
                except MemoryError as me:
                    tb = log_exception(current_session_id, f"OCR Processing Page {curr_pg_num} (MemoryError)", me)
                    st.warning(f"💥 Memory error during OCR on page {curr_pg_num}. Skipping highlights.")
                    analysis_result['fence_text_boxes_details'] = []
                    profiler.record_step("11. OCR highlighting (FAILED)", "MemoryError")
                except UnrecoverableRateLimitError as urle_hl:
                    msg = f"🛑 API Rate Limit Highlight Pg {curr_pg_num}: {urle_hl}. Halted."; status_txt_area.error(msg); st.error(msg)
                    st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
                except Exception as e_hl: 
                    tb = log_exception(current_session_id, f"OCR Processing Page {curr_pg_num}", e_hl)
                    st.warning(f"⚠️ OCR error on page {curr_pg_num}: {str(e_hl)[:100]}...")
                    analysis_result['fence_text_boxes_details'] = []
                    profiler.record_step("11. OCR highlighting (FAILED)", str(e_hl)[:50])
            elif not fatal_err_page and highlight_fence_text_app and analysis_result.get('fence_found'):
                 status_txt_area.text(f"Page {curr_pg_num}: Fence found, no text match for detailed highlighting.")
            if fatal_err_page: break
            
            # Store result in session state (ULTRA minimal - only essential fields)
            # Remove ALL large data structures to save memory
            analysis_result_minimal = {
                'page_number': analysis_result.get('page_number'),
                'page_index_in_original_doc': analysis_result.get('page_index_in_original_doc'),
                'fence_found': analysis_result.get('fence_found'),
                'text_found': analysis_result.get('text_found'),
                'text_snippet': analysis_result.get('text_snippet', '')[:200] if analysis_result.get('text_snippet') else '',  # Limit snippet
                'fence_text_boxes_details': analysis_result.get('fence_text_boxes_details', [])[:50],  # Limit to 50 boxes max
                'highlight_fence_text_app_setting': analysis_result.get('highlight_fence_text_app_setting')
            }
            target_col = col_f if analysis_result.get('fence_found') else col_nf
            (st.session_state.fence_pages if analysis_result.get('fence_found') else st.session_state.non_fence_pages).append(analysis_result_minimal)
            profiler.record_step("12. Store result (minimal)", f"size={len(str(analysis_result_minimal))/(1024):.1f}KB")
            with target_col: # Display Logic (copied from display_page_result_expander for consistency)
                exp_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    reasons = []; 
                    if analysis_result.get('text_found'): reasons.append("Text")

                    if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app: reasons.append("Highlights")
                    if reasons: exp_title += f" ({' & '.join(reasons)} Match)"
                # Don't expand any pages during live processing to save maximum memory
                with st.expander(exp_title, expanded=False):
                    img_col, det_col = st.columns([2,1])
                    
                    # Don't generate images during live processing - too memory intensive
                    # Images will be available after processing completes
                    with img_col:
                        st.info("🚀 Processing... Images will be available after completion")
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
            
            # ULTRA-AGGRESSIVE memory cleanup after each page
            try:
                # Delete everything from this page
                del page_obj, text_content, single_page_pdf_bytes, page_data_an, analysis_res_core, analysis_result, analysis_result_minimal
                if 'analysis_result_compact' in locals():
                    del analysis_result_compact
                if 'single_pg_bytes_io' in locals():
                    del single_pg_bytes_io
                if 'temp_doc_single' in locals():
                    del temp_doc_single
                if 'boxes' in locals():
                    del boxes  # OCR boxes can be large
                if 'signals' in locals():
                    del signals
                if 'jr' in locals():
                    del jr
                if 'pix' in locals():
                    del pix
                if 'img_bytes' in locals():
                    del img_bytes
                if 'temp_img_doc' in locals():
                    del temp_img_doc
                if 'temp_page' in locals():
                    del temp_page
                if 'validated_signals' in locals():
                    del validated_signals
                if 'page_text_lower' in locals():
                    del page_text_lower
                    
                # Force IMMEDIATE garbage collection (10× for maximum cleanup)
                # Profiling showed Python object retention is the real issue, not DocAI responses
                # Server needs more aggressive GC than local
                for _ in range(10):
                    gc.collect()
                profiler.record_step("13. Cleanup variables (ultra-aggressive + 10× GC)")
            except Exception as cleanup_err:
                print(f"Warning: cleanup error: {cleanup_err}")
            
            # Clear Streamlit's internal cache more aggressively
            if i % 1 == 0:  # EVERY PAGE
                try:
                    st.cache_data.clear()
                except:
                    pass
            
            # End page profiling
            net_change = profiler.end_page()
            
            time.sleep(0.05) 
    except Exception as fatal_error:
        # Catch any unhandled exceptions in the main loop
        tb = log_exception(current_session_id, f"FATAL ERROR in main processing loop at page {st.session_state.total_pages_processed_count}", fatal_error)
        st.error(f"🔥 Fatal error during processing: {str(fatal_error)[:200]}")
        with st.expander("⚠️ Click to see full error details", expanded=False):
            st.code(tb, language="python")
        st.session_state.analysis_halted_due_to_error = True
    finally: 
        if doc_proc_loop:
            doc_proc_loop.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document in finally block.")
        doc_proc_loop = None 
        
        print(f"SESSION {current_session_id} LOG: Processing loop ended. Final memory: {_rss_mb():.1f} MB")
        
        # PRINT PROFILING SUMMARY
        print("\n" + "="*80)
        print("PROFILING SUMMARY - Memory Delta Per Function")
        print("="*80)
        if profiler.page_data:
            summary = {}
            for page_data in profiler.page_data:
                for step in page_data['steps']:
                    name = step['name']
                    if name not in summary:
                        summary[name] = {'count': 0, 'total': 0, 'max': 0, 'min': 999}
                    summary[name]['count'] += 1
                    summary[name]['total'] += step['delta']
                    summary[name]['max'] = max(summary[name]['max'], step['delta'])
                    summary[name]['min'] = min(summary[name]['min'], step['delta'])
            
            print(f"\n{'Step':<35} {'Calls':<8} {'Avg Δ':<10} {'Max Δ':<10} {'Min Δ':<10}")
            print("-"*80)
            for name in sorted(summary.keys()):
                s = summary[name]
                avg = s['total'] / s['count'] if s['count'] > 0 else 0
                print(f"{name:<35} {s['count']:<8} {avg:>+9.1f}MB {s['max']:>+9.1f}MB {s['min']:>+9.1f}MB")
            
            total_net = sum((p['steps'][-1]['mem'] if p['steps'] else p['start_mem']) - p['start_mem'] for p in profiler.page_data)
            avg_net = total_net / len(profiler.page_data) if profiler.page_data else 0
            print(f"\nTotal pages profiled: {len(profiler.page_data)}")
            print(f"Total NET memory change: {total_net:+.1f}MB")
            print(f"Average NET per page: {avg_net:+.1f}MB")
            print("="*80 + "\n")
        else:
            print("No profiling data collected")
            print("="*80 + "\n")
    
    st.session_state.processing_complete = True 
    if not st.session_state.analysis_halted_due_to_error:
        prog_bar.empty(); status_txt_area.success("All pages processed!")
        
        # Debug logging
        print(f"SESSION {current_session_id} LOG: Fence pages found: {len(st.session_state.fence_pages)}")
        print(f"SESSION {current_session_id} LOG: Temp PDF path exists: {st.session_state.get('temp_pdf_path') and os.path.exists(st.session_state.temp_pdf_path)}")
        
        if st.session_state.fence_pages and st.session_state.temp_pdf_path:
            # Read PDF from temp file
            try:
                with open(st.session_state.temp_pdf_path, 'rb') as f:
                    pdf_bytes_for_final = f.read()
                print(f"SESSION {current_session_id} LOG: Read {len(pdf_bytes_for_final)} bytes from temp file")
                
                pdf_b, pdf_n = generate_combined_highlighted_pdf(
                    pdf_bytes_for_final,
                    st.session_state.fence_pages,
                    st.session_state.uploaded_pdf_name,
                    current_session_id
                )
                if pdf_b:
                    # cache by hash to avoid keeping duplicate big blobs in session
                    @st.cache_data
                    def _store_combined_pdf(pdf_hash, pdf_bytes, pdf_name):
                        return pdf_bytes, pdf_name
                    st.session_state.combined_pdf_ref = _store_combined_pdf(
                        st.session_state.current_pdf_hash, pdf_b, pdf_n
                    )
                    print(f"SESSION {current_session_id} LOG: Combined PDF generated successfully: {pdf_n}")
                else:
                    st.warning(f"Could not generate PDF: {pdf_n}")
                    print(f"SESSION {current_session_id} WARNING: PDF generation failed: {pdf_n}")
            except Exception as e_pdf:
                st.error(f"Error generating combined PDF: {e_pdf}")
                print(f"SESSION {current_session_id} ERROR: Exception generating PDF: {e_pdf}")
        elif not st.session_state.fence_pages:
            st.info("ℹ️ No fence-related pages found in this document.")
            print(f"SESSION {current_session_id} INFO: No fence pages found")
        
        # NOTE: Keep temp file for image generation in results display
        # It will be cleaned up when a new file is uploaded or session ends
        print(f"SESSION {current_session_id} LOG: Keeping temp file for results display: {st.session_state.temp_pdf_path}")
    else: 
        prog_bar.empty()
        # NOTE: Keep temp file even on error for potential results display
        print(f"SESSION {current_session_id} LOG: Keeping temp file after error (if exists): {st.session_state.get('temp_pdf_path')}") 
    final_summary_text = f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    summary_placeholder.markdown(final_summary_text)
    
    # Show download button if combined PDF was generated
    if st.session_state.get('combined_pdf_ref') and not st.session_state.analysis_halted_due_to_error:
        data, fname = st.session_state.combined_pdf_ref
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            data,
            fname,
            "application/pdf",
            key="dl_combined_pdf_main"
        )
        st.caption(f"📄 {len(st.session_state.fence_pages)} fence page(s) included with highlights")
    elif len(st.session_state.fence_pages) == 0 and not st.session_state.analysis_halted_due_to_error:
        st.info("ℹ️ No fence-related pages found. No PDF to download.")
    elif st.session_state.analysis_halted_due_to_error and len(st.session_state.fence_pages) > 0:
        st.warning("⚠️ Processing was halted. You can view partial results below, but no combined PDF was generated.")

elif st.session_state.processing_complete: 
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True); st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    final_summary_text_rerun = f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n- ✅ Fence: {len(st.session_state.fence_pages)}\n- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    st.markdown(final_summary_text_rerun)
    
    # Show download button if combined PDF was generated
    if st.session_state.get('combined_pdf_ref') and not st.session_state.analysis_halted_due_to_error:
        data, fname = st.session_state.combined_pdf_ref
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            data,
            fname,
            "application/pdf",
            key="dl_combined_pdf_rerun"
        )
        st.caption(f"📄 {len(st.session_state.fence_pages)} fence page(s) included with highlights")
    elif len(st.session_state.fence_pages) == 0 and not st.session_state.analysis_halted_due_to_error:
        st.info("ℹ️ No fence-related pages found. No PDF to download.")
    elif st.session_state.analysis_halted_due_to_error and len(st.session_state.fence_pages) > 0:
        st.warning("⚠️ Processing was halted. You can view partial results below, but no combined PDF was generated.")
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
                    
                    # Always generate and display images (uses cache)
                    orig_b_r, hl_b_r = generate_display_images_for_page_wrapper(res_data_item, session_id_for_display)
                    
                    with img_col_r: # Image display
                        # ... (Same as live loop image display)
                        disp_img_r = hl_b_r if hl_b_r else orig_b_r
                        if disp_img_r: st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}{' (HL)' if hl_b_r else ''}")
                        if hl_b_r:
                            st.download_button(
                                "Download highlighted JPG",
                                data=hl_b_r,
                                file_name=f"page_{res_data_item['page_number']}_hl.jpg",
                                mime="image/jpeg",
                                key=f"dl_hl_r_{res_data_item['page_number']}",
                            )
                        if orig_b_r:
                            st.download_button(
                                "Download original JPG",
                                data=orig_b_r,
                                file_name=f"page_{res_data_item['page_number']}_orig.jpg",
                                mime="image/jpeg",
                                key=f"dl_orig_r_{res_data_item['page_number']}",
                            )
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
