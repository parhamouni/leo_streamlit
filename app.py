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

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)
HIGHLIGHT_WIDTH_UI = 2.0
HIGHLIGHT_COLOR_PDF = (0, 0.9, 0)
HIGHLIGHT_WIDTH_PDF = 2.0
DISPLAY_IMAGE_DPI = 96 
VISION_IMAGE_DPI = 72  

st.set_page_config(page_title="Fence Detector", layout="wide")
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; margin-bottom: 1rem; color: #1E3A8A;}
    .section-header {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 1rem; margin-bottom: 1rem;}
    .stExpander {border-left: 5px solid #ccc; margin-bottom: 10px;}
    .download-button {margin-top: 10px; display: inline-block; margin-right: 10px; padding: 8px 12px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; font-size: 0.9rem;}
    .download-button:hover {background-color: #0056b3; color: white; text-decoration: none;}
    .stDownloadButton>button { background-color: #28a745; color:white; border: none; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; transition-duration: 0.4s; cursor: pointer; border-radius: 5px; }
    .stDownloadButton>button:hover { background-color: #218838; color: white; }
    .centered-button { display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)
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

with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
        if openai_key_input: openai_key = openai_key_input; st.rerun()
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
    process_images_vision = st.toggle("🖼️ Enable visual analysis", value=False, key="vision_toggle")
    vision_model_name_option = "gpt-4-turbo" 
    highlight_fence_text_app = st.toggle("🔍 Highlight text & indicators", value=True, key="highlight_toggle")
    st.subheader("Fence Keywords")
    if 'fence_keywords_app' not in st.session_state: st.session_state.fence_keywords_app = ['fence']
    custom_keywords_str = st.text_area("Custom keywords (one per line):", "\n".join(st.session_state.fence_keywords_app), height=150, key="kw_text_area")
    if st.button("Update Keywords", key="update_kw_btn"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()
    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app

llm_analysis_instance, llm_vision_instance = None, None
if openai_key:
    try:
        print(f"SESSION {current_session_id} LOG: Initializing LLM instances.")
        llm_analysis_instance = ChatOpenAI(model=st.session_state.selected_model_for_analysis, temperature=0, openai_api_key=openai_key, timeout=180, max_retries=2)
        if process_images_vision: llm_vision_instance = ChatOpenAI(model=vision_model_name_option, temperature=0, openai_api_key=openai_key, timeout=180, max_retries=2)
        print(f"SESSION {current_session_id} LOG: LLM instances initialized.")
    except Exception as e: st.error(f"LLM Init Error: {e}"); openai_key = None; print(f"SESSION {current_session_id} ERROR: LLM Init Error: {e}")

def get_image_download_link_html(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

@time_it 
@st.cache_data(ttl=3600, show_spinner=False, max_entries=100)
def _generate_display_images_for_page_cached(page_idx, original_pdf_doc_bytes_tuple, fence_text_boxes_details_tuple,
                                            ui_color, ui_width, display_dpi, session_id_for_log="N/A_CACHE"):
    original_pdf_doc_bytes = bytes(original_pdf_doc_bytes_tuple)
    fence_text_boxes_details = [dict(item_tuple) for item_tuple in fence_text_boxes_details_tuple]
    original_image_bytes, highlighted_image_bytes = None, None
    func_call_id = str(uuid.uuid4())[:4] 
    print(f"SESSION {session_id_for_log} CACHE_CALL ({func_call_id}): _generate_display_images_for_page_cached for Page {page_idx}. Num boxes: {len(fence_text_boxes_details)}")
    render_start_time = time.time()
    try:
        with fitz.open(stream=io.BytesIO(original_pdf_doc_bytes), filetype="pdf") as doc_orig:
            page_orig = doc_orig.load_page(page_idx)
            pix_orig = page_orig.get_pixmap(dpi=display_dpi); original_image_bytes = pix_orig.tobytes("png"); del pix_orig
            if fence_text_boxes_details:
                with fitz.open(stream=io.BytesIO(original_pdf_doc_bytes), filetype="pdf") as doc_hl:
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

def generate_display_images_for_page_wrapper(page_result_data, original_pdf_doc_bytes, session_id):
    page_idx = page_result_data.get('page_index_in_original_doc')
    if page_idx is None or original_pdf_doc_bytes is None: return None, None
    boxes_details = page_result_data.get('fence_text_boxes_details', [])
    details_tuple = tuple(tuple(sorted(d.items())) for d in sorted(boxes_details, key=lambda x: x.get('id', str(x)))) if boxes_details else tuple()
    return _generate_display_images_for_page_cached(page_idx, tuple(original_pdf_doc_bytes), details_tuple, HIGHLIGHT_COLOR_UI, HIGHLIGHT_WIDTH_UI, DISPLAY_IMAGE_DPI, session_id)

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
            base, ext = os.path.splitext(uploaded_pdf_name_base); fname = f"{base}_fence_highlights{ext}"
        except Exception as e_s: print(f"SESSION {session_id} Err PDF tobytes: {e_s}"); fname=f"err_save_{uploaded_pdf_name_base}.pdf"
    if input_doc: input_doc.close()
    if output_doc: output_doc.close()
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)

st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}"
    if st.session_state.last_uploaded_file_id != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Resetting state for {current_file_id}.")
        current_selected_model = st.session_state.selected_model_for_analysis
        current_keywords = st.session_state.fence_keywords_app
        initialize_session_state(current_session_id)
        st.session_state.selected_model_for_analysis = current_selected_model
        st.session_state.fence_keywords_app = current_keywords
        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.last_uploaded_file_id = current_file_id
        st.cache_data.clear() # CORRECTED: Clear all @st.cache_data caches
        print(f"SESSION {current_session_id} LOG: Cleared all @st.cache_data caches.")
        st.rerun() 
    if openai_key and llm_analysis_instance and not st.session_state.run_analysis_triggered and not st.session_state.processing_complete and not st.session_state.analysis_halted_due_to_error:
        print(f"SESSION {current_session_id} LOG: Triggering analysis.")
        st.session_state.run_analysis_triggered = True
elif not openai_key and not uploaded_pdf_file_obj: st.info("Upload PDF and set API key in sidebar.")
elif not openai_key and uploaded_pdf_file_obj: st.warning("OpenAI API Key needed for analysis.")

if st.session_state.run_analysis_triggered and st.session_state.original_pdf_bytes and llm_analysis_instance and not st.session_state.analysis_halted_due_to_error and not st.session_state.processing_complete:
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
            page_data_an = {"page_number": curr_pg_num, "text": text_content}
            if process_images_vision:
                print(f"SESSION {current_session_id} LOG: Page {curr_pg_num} - Generating image for vision (DPI: {VISION_IMAGE_DPI}).")
                pix_vis = page_obj.get_pixmap(alpha=False, dpi=VISION_IMAGE_DPI) 
                img_b_vis = pix_vis.tobytes("png"); del pix_vis 
                page_data_an["image_b64"] = base64.b64encode(img_b_vis).decode("utf-8")
            analysis_res_core = {}; fatal_err_page = False
            try:
                with st.spinner(f"Page {curr_pg_num}: Core analysis..."):
                    analysis_res_core = analyze_page(page_data_an, llm_analysis_instance, llm_vision_instance if process_images_vision else None, FENCE_KEYWORDS_APP)
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
                        boxes,_,_ = get_fence_related_text_boxes(single_pg_bytes_io.getvalue(), llm_analysis_instance, FENCE_KEYWORDS_APP, st.session_state.selected_model_for_analysis)
                        if boxes: analysis_result['fence_text_boxes_details'] = boxes
                except UnrecoverableRateLimitError as urle_hl:
                    msg = f"🛑 API Rate Limit Highlight Pg {curr_pg_num}: {urle_hl}. Halted."; status_txt_area.error(msg); st.error(msg)
                    st.session_state.analysis_halted_due_to_error = True; fatal_err_page = True; print(f"SESSION {current_session_id} ERROR: {msg}"); break
                except Exception as e_hl: st.warning(f"Highlight error pg {curr_pg_num}: {e_hl}"); print(f"SESSION {current_session_id} WARNING: Highlight error pg {curr_pg_num}: {e_hl}")
            elif not fatal_err_page and highlight_fence_text_app and analysis_result.get('fence_found'):
                 status_txt_area.text(f"Page {curr_pg_num}: Fence found (e.g. vision), no text match for detailed highlighting.")
            if fatal_err_page: break
            target_col = col_f if analysis_result.get('fence_found') else col_nf
            (st.session_state.fence_pages if analysis_result.get('fence_found') else st.session_state.non_fence_pages).append(analysis_result)
            with target_col:
                exp_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    reasons = []; 
                    if analysis_result.get('text_found'): reasons.append("Text")
                    if analysis_result.get('vision_found'): reasons.append("Image")
                    if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app: reasons.append("Highlights")
                    if reasons: exp_title += f" ({' & '.join(reasons)} Match)"
                with st.expander(exp_title, expanded=True):
                    img_col, det_col = st.columns([2,1])
                    print(f"SESSION {current_session_id} DEBUG LIVE DISPLAY Page {analysis_result['page_number']}: fence_found: {analysis_result.get('fence_found')}, Highlight toggle: {highlight_fence_text_app}, Num boxes: {len(analysis_result.get('fence_text_boxes_details', []))}")
                    if analysis_result.get('fence_text_boxes_details'): print(f"SESSION {current_session_id} DEBUG LIVE: First box detail: {analysis_result['fence_text_boxes_details'][0] if analysis_result.get('fence_text_boxes_details') else 'No boxes'}")
                    wrapper_call_start_time = time.time()
                    with st.spinner(f"Rendering image for page {analysis_result['page_number']}..."): orig_b, hl_b = generate_display_images_for_page_wrapper(analysis_result, st.session_state.original_pdf_bytes, current_session_id)
                    wrapper_call_duration = time.time() - wrapper_call_start_time
                    print(f"SESSION {current_session_id} PERF_LOG: generate_display_images_for_page_wrapper Page {curr_pg_num} took {wrapper_call_duration:.4f}s.")
                    with img_col:
                        disp_img_ui = hl_b if hl_b else orig_b
                        if disp_img_ui: st.image(disp_img_ui, caption=f"Page {analysis_result['page_number']}{' (Highlighted)' if hl_b else ''}")
                        # ... (Download links)
                        dl_links_html_live = []
                        if hl_b: dl_links_html_live.append(get_image_download_link_html(hl_b, f"page_{analysis_result['page_number']}_hl.png", "DL HL Img"))
                        if orig_b: dl_links_html_live.append(get_image_download_link_html(orig_b, f"page_{analysis_result['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_live: st.markdown(" ".join(dl_links_html_live), unsafe_allow_html=True)
                    with det_col: # Text details display
                        st.markdown("##### Analysis Details")
                        if analysis_result.get('fence_found'):
                            pts = []; 
                            if analysis_result.get('text_found'): pts.append("✔️ Text")
                            if analysis_result.get('vision_found'): pts.append("✔️ Image")
                            if analysis_result.get('fence_text_boxes_details') and highlight_fence_text_app : pts.append("✔️ Highlights")
                            if not pts: pts.append("Fence flagged")
                            st.markdown("\n".join(f"- {s}" for s in pts))
                        else: st.markdown("No strong fence indicators.")
                        if analysis_result.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{analysis_result['text_response']}_")
                        if analysis_result.get('vision_response'):
                            with st.popover("Image Log"): st.markdown(f"_{analysis_result['vision_response']}_")
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
                if res_data_item.get('fence_found'):
                    reasons_res = []; 
                    if res_data_item.get('text_found'): reasons_res.append("Text")
                    if res_data_item.get('vision_found'): reasons_res.append("Image")
                    if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting', True): reasons_res.append("Highlights")
                    if reasons_res: exp_title_res += f" ({' & '.join(reasons_res)} Match)"
                with st.expander(exp_title_res, expanded=False):
                    img_col_r, det_col_r = st.columns([2,1])
                    with st.spinner(f"Loading image page {res_data_item['page_number']}..."):
                        orig_b_r, hl_b_r = generate_display_images_for_page_wrapper(res_data_item, st.session_state.original_pdf_bytes, session_id_for_display)
                    with img_col_r:
                        disp_img_r = hl_b_r if hl_b_r else orig_b_r
                        if disp_img_r: st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}{' (HL)' if hl_b_r else ''}")
                        dl_links_html_rerun = []
                        if hl_b_r: dl_links_html_rerun.append(get_image_download_link_html(hl_b_r, f"page_{res_data_item['page_number']}_hl.png", "DL HL Img"))
                        if orig_b_r: dl_links_html_rerun.append(get_image_download_link_html(orig_b_r, f"page_{res_data_item['page_number']}_orig.png", "DL Orig Img"))
                        if dl_links_html_rerun: st.markdown(" ".join(dl_links_html_rerun), unsafe_allow_html=True)
                    with det_col_r: 
                        st.markdown("##### Analysis Details")
                        if res_data_item.get('fence_found'):
                            pts_r = [] 
                            if res_data_item.get('text_found'): pts_r.append("✔️ Text")
                            if res_data_item.get('vision_found'): pts_r.append("✔️ Image")
                            if res_data_item.get('fence_text_boxes_details') and res_data_item.get('highlight_fence_text_app_setting',True) : pts_r.append("✔️ Highlights")
                            if not pts_r: pts_r.append("Fence flagged")
                            st.markdown("\n".join(f"- {s}" for s in pts_r))
                        else: st.markdown("No strong fence indicators.")
                        if res_data_item.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{res_data_item['text_response']}_")
                        if res_data_item.get('vision_response'):
                            with st.popover("Image Log"): st.markdown(f"_{res_data_item['vision_response']}_")
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

elif not st.session_state.original_pdf_bytes : st.info("Upload PDF and ensure API key is set in sidebar.")
elif not (openai_key and llm_analysis_instance): st.error("OpenAI models not initialized. Check API key.")
elif st.session_state.analysis_halted_due_to_error: st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---"); st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App</p>", unsafe_allow_html=True)