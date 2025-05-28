import streamlit as st
from utils import analyze_page, get_fence_related_text_boxes 
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
# from PIL import Image, ImageDraw # No longer needed
import io
import time 

st.set_page_config(
    page_title="Fence Detector for Engineering Drawings",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; margin-bottom: 1rem; color: #1E3A8A;}
    .section-header {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 1rem; margin-bottom: 1rem;}
    .stExpander {border-left: 5px solid #ccc; margin-bottom: 10px;}
    .download-button {margin-top: 10px; display: inline-block; margin-right: 10px; padding: 8px 12px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; font-size: 0.9rem;}
    .download-button:hover {background-color: #0056b3; color: white; text-decoration: none;}
    .stDownloadButton>button { background-color: #28a745; color:white; border: none; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; transition-duration: 0.4s; cursor: pointer; border-radius: 5px; }
    .stDownloadButton>button:hover { background-color: #218838; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)

# Initialize session state
if 'fence_pages' not in st.session_state: st.session_state.fence_pages = []
if 'non_fence_pages' not in st.session_state: st.session_state.non_fence_pages = []
if 'total_pages_processed' not in st.session_state: st.session_state.total_pages_processed = 0
if 'doc_total_pages' not in st.session_state: st.session_state.doc_total_pages = 0
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'fence_keywords_app' not in st.session_state:
    st.session_state.fence_keywords_app = ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link']
if 'run_analysis_triggered' not in st.session_state:
    st.session_state.run_analysis_triggered = False
if 'uploaded_pdf_name' not in st.session_state:
    st.session_state.uploaded_pdf_name = None
if 'original_pdf_bytes' not in st.session_state: st.session_state.original_pdf_bytes = None
if 'highlighted_pdf_bytes_for_download' not in st.session_state: st.session_state.highlighted_pdf_bytes_for_download = None
if 'last_uploaded_file_id' not in st.session_state: st.session_state.last_uploaded_file_id = None


with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
        if openai_key_input:
            openai_key = openai_key_input
            st.rerun() 

    st.subheader("Model Selection")
    model_options = {
        "gpt-4o (128k context, recommended for highlighting)": "gpt-4o",
        "gpt-4-turbo (128k context, powerful)": "gpt-4-turbo",
        "gpt-4 (8k/32k context, check your API access)": "gpt-4",
        "gpt-3.5-turbo (16k context, fastest, lower accuracy)": "gpt-3.5-turbo"
    }
    default_model_value = "gpt-4o"
    if default_model_value not in model_options.values(): default_model_value = "gpt-4-turbo"
    if default_model_value not in model_options.values(): default_model_value = "gpt-4"

    if "selected_model_for_analysis" not in st.session_state:
        st.session_state.selected_model_for_analysis = default_model_value
    
    default_model_label = [label for label, value in model_options.items() if value == st.session_state.selected_model_for_analysis][0]

    selected_label = st.radio(
        "Select LLM for Text Analysis & Highlighting:",
        list(model_options.keys()),
        key="model_selector_radio",
        index=list(model_options.keys()).index(default_model_label)
    )
    if st.session_state.selected_model_for_analysis != model_options[selected_label]:
        st.session_state.selected_model_for_analysis = model_options[selected_label]
    st.info(f"Using: **{st.session_state.selected_model_for_analysis}** for main analysis.")

    process_images_vision = st.toggle("🖼️ Enable visual analysis (GPT-4 Vision)", value=False, key="vision_toggle")
    vision_model_name_option = "gpt-4-turbo" 

    highlight_fence_text_app = st.toggle("🔍 Highlight fence-related text & indicators", value=True, key="highlight_toggle")
    
    st.subheader("Fence Keywords")
    custom_keywords_str = st.text_area("Add custom keywords (one per line):",
                                     "\n".join(st.session_state.fence_keywords_app), height=150, key="fence_keywords_text_area")
    if st.button("Update Keywords", key="update_keywords_button"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()

    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


llm_analysis_instance = None
llm_vision_instance = None

if openai_key:
    try:
        llm_analysis_instance = ChatOpenAI(model=st.session_state.selected_model_for_analysis, temperature=0, openai_api_key=openai_key, timeout=120)
        if process_images_vision:
            llm_vision_instance = ChatOpenAI(model=vision_model_name_option, temperature=0, openai_api_key=openai_key, timeout=120)
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {type(e).__name__}: {e}")
        openai_key = None 
else:
    pass


def get_image_download_link(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base):
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No fence pages to include or original PDF missing."
    
    HIGHLIGHT_COLOR_PDF = (0, 0.9, 0)
    HIGHLIGHT_WIDTH_PDF = 2.0

    output_pdf_doc = fitz.open() 
    input_pdf_doc_for_combine = None

    try:
        input_pdf_doc_for_combine = fitz.open(stream=io.BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        if output_pdf_doc: output_pdf_doc.close()
        print(f"Error opening original PDF for highlighting: {e}")
        return None, f"Error opening original PDF for highlighting: {e}"

    sorted_fence_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    for res_data in sorted_fence_pages:
        page_idx_original = res_data.get('page_index_in_original_doc')
        if page_idx_original is None:
            print(f"Warning: Skipping page due to missing 'page_index_in_original_doc': {res_data.get('page_number')}")
            continue
        try:
            output_pdf_doc.insert_pdf(input_pdf_doc_for_combine, from_page=page_idx_original, to_page=page_idx_original)
            page_in_output_pdf = output_pdf_doc.load_page(len(output_pdf_doc) - 1)
            if res_data.get('highlight_fence_text_app', True) and 'fence_text_boxes_details' in res_data and res_data['fence_text_boxes_details']:
                derotation_matrix = page_in_output_pdf.derotation_matrix
                for box in res_data['fence_text_boxes_details']:
                    rotated_rect = fitz.Rect(box['x0'], box['y0'], box['x1'], box['y1'])
                    if page_in_output_pdf.rotation != 0:
                        unrotated_rect = rotated_rect * derotation_matrix
                        unrotated_rect.normalize()
                    else:
                        unrotated_rect = rotated_rect
                    if not unrotated_rect.is_empty and unrotated_rect.is_valid:
                        try:
                            page_in_output_pdf.draw_rect(
                                unrotated_rect,
                                color=HIGHLIGHT_COLOR_PDF,
                                width=HIGHLIGHT_WIDTH_PDF,
                                overlay=True
                            )
                        except Exception as e_draw:
                            print(f"Error drawing rect on page {page_idx_original} (coords: {unrotated_rect}): {e_draw}")
                    else:
                        print(f"Warning: Skipping invalid/empty rect for page {page_idx_original}: {unrotated_rect}")
        except Exception as e_page_insert:
            print(f"Error processing page {page_idx_original} for combined PDF: {e_page_insert}")


    pdf_bytes = None
    final_pdf_name = "error_generating_pdf.pdf" 

    if len(output_pdf_doc) == 0:
        print("No pages were successfully added to the highlighted PDF.")
    else:
        try:
            # REMOVED linear=True
            pdf_bytes = output_pdf_doc.tobytes(garbage=2, deflate=True) 
            base_name, ext = os.path.splitext(uploaded_pdf_name_base)
            final_pdf_name = f"{base_name}_fence_highlights{ext}"
            print(f"Successfully generated PDF bytes for {final_pdf_name}")
        except fitz.mupdf.FzError as fzea: # Catching base FzError as FzErrorArgument might be a subclass
            print(f"PyMuPDF FzError during tobytes(): {fzea}. Error code might provide more clues if it's not about linearisation again.")
            pdf_bytes = None 
            final_pdf_name = f"error_saving_fz_{uploaded_pdf_name_base}.pdf"
        except Exception as e_save:
            print(f"Generic error during PDF tobytes() on Streamlit Cloud: {e_save}")
            pdf_bytes = None
            final_pdf_name = f"error_saving_generic_{uploaded_pdf_name_base}.pdf"

    if input_pdf_doc_for_combine:
        input_pdf_doc_for_combine.close()
    if output_pdf_doc:
        output_pdf_doc.close()
    
    if pdf_bytes:
        return pdf_bytes, final_pdf_name
    else:
        return None, final_pdf_name
    
    
st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload Engineering PDF Document", type=["pdf"], key="pdf_uploader_main_auto")

if uploaded_pdf_file_obj is not None:
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}_{uploaded_pdf_file_obj.type}"
    is_new_file_upload = (st.session_state.last_uploaded_file_id != current_file_id)

    if is_new_file_upload:
        st.session_state.fence_pages = []
        st.session_state.non_fence_pages = []
        st.session_state.total_pages_processed = 0
        st.session_state.doc_total_pages = 0
        st.session_state.processing_complete = False
        st.session_state.run_analysis_triggered = False 
        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.highlighted_pdf_bytes_for_download = None
        st.session_state.last_uploaded_file_id = current_file_id

    if openai_key and llm_analysis_instance:
        if is_new_file_upload or \
           (not st.session_state.get('run_analysis_triggered') and \
            not st.session_state.get('processing_complete') and \
            st.session_state.get('original_pdf_bytes')):
            st.session_state.run_analysis_triggered = True
            if is_new_file_upload:
                 st.session_state.total_pages_processed = 0
                 st.session_state.processing_complete = False
    else: 
        if is_new_file_upload: 
            st.session_state.run_analysis_triggered = False 
            if not openai_key: st.warning("OpenAI API key missing in sidebar. Analysis cannot start.")
            elif not llm_analysis_instance: st.warning("LLM initialization failed. Analysis cannot start. Check API key or model selection.")


if st.session_state.get('run_analysis_triggered') and \
   st.session_state.get('original_pdf_bytes') and \
   llm_analysis_instance:

    if not st.session_state.get('processing_complete') and st.session_state.get('total_pages_processed', 0) == 0 :
        doc_for_processing_loop = None
        try:
            doc_for_processing_loop = fitz.open(stream=io.BytesIO(st.session_state.original_pdf_bytes), filetype="pdf")
            st.session_state.doc_total_pages = len(doc_for_processing_loop)
        except Exception as e:
            st.error(f"Failed to open PDF for processing: {e}")
            st.session_state.processing_complete = True 
            if 'doc_for_processing_loop' in locals() and doc_for_processing_loop: doc_for_processing_loop.close()
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h2>📊 Analysis Results (Live Updates)</h2></div>", unsafe_allow_html=True)
        summary_placeholder = st.empty() 
        col_fence, col_non_fence = st.columns(2)
        with col_fence: st.subheader("✅ Fence-Related Pages")
        with col_non_fence: st.subheader("❌ Non-Fence Pages")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # --- Parameters for more visible UI highlights ---
        HIGHLIGHT_COLOR_UI = (0, 0.9, 0)  # Brighter Green (RGB for PyMuPDF: 0.0-1.0 range)
        HIGHLIGHT_WIDTH_UI = 2.0          # Increased border width

        for i in range(st.session_state.doc_total_pages):
            current_page_num_loop = i + 1
            st.session_state.total_pages_processed = current_page_num_loop
            progress = current_page_num_loop / st.session_state.doc_total_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing Page {current_page_num_loop} of {st.session_state.doc_total_pages}...")

            page_obj_from_doc = doc_for_processing_loop.load_page(i)
            text_content = page_obj_from_doc.get_text("text")
            
            original_pix = page_obj_from_doc.get_pixmap(dpi=200) 
            original_img_bytes = original_pix.tobytes("png")
            img_b64_for_vision = base64.b64encode(original_img_bytes).decode("utf-8")

            page_data = {
                "page_number": current_page_num_loop, 
                "text": text_content, 
                "image_bytes": original_img_bytes, 
                "image_b64": img_b64_for_vision,
                "page_index_in_original_doc": i 
            }

            analysis_result = {}
            try:
                with st.spinner(f"Page {current_page_num_loop}: Core analysis..."):
                    analysis_result = analyze_page(page_data, llm_analysis_instance, 
                                                   llm_vision_instance if process_images_vision else None, 
                                                   FENCE_KEYWORDS_APP)
            except Exception as e:
                st.error(f"Core analysis error page {current_page_num_loop}: {e}")
                analysis_result = {"page_number": current_page_num_loop, "fence_found": False, "image": original_img_bytes}
            
            analysis_result['page_index_in_original_doc'] = i 
            analysis_result['image'] = original_img_bytes 
            analysis_result['highlighted_image'] = None 
            analysis_result['fence_text_details_for_display'] = []
            analysis_result['fence_text_boxes_details'] = [] 
            analysis_result['highlight_fence_text_app'] = highlight_fence_text_app


            if highlight_fence_text_app and analysis_result.get('fence_found'):
                status_text.text(f"Page {current_page_num_loop}: Identifying text locations for highlighting...")
                single_page_pdf_bytes_io = io.BytesIO()
                with fitz.open() as temp_single_page_doc:
                    temp_single_page_doc.insert_pdf(doc_for_processing_loop, from_page=i, to_page=i)
                    temp_single_page_doc.save(single_page_pdf_bytes_io)
                single_page_bytes = single_page_pdf_bytes_io.getvalue()
                
                try:
                    with st.spinner(f"Page {current_page_num_loop}: Extracting text boxes..."):
                        fence_text_boxes_details, _, _ = get_fence_related_text_boxes(
                            single_page_bytes, llm_analysis_instance, FENCE_KEYWORDS_APP, 
                            st.session_state.selected_model_for_analysis 
                        )
                    
                    if fence_text_boxes_details:
                        analysis_result['fence_text_boxes_details'] = fence_text_boxes_details 
                        analysis_result['fence_text_details_for_display'] = fence_text_boxes_details 

                        with fitz.open(stream=io.BytesIO(st.session_state.original_pdf_bytes), filetype="pdf") as temp_doc_for_ui_highlight:
                            page_for_ui_highlight = temp_doc_for_ui_highlight.load_page(i)
                            ui_derotation_matrix = page_for_ui_highlight.derotation_matrix
                            for box_detail in analysis_result['fence_text_boxes_details']:
                                rotated_coords_rect = fitz.Rect(box_detail['x0'], box_detail['y0'], box_detail['x1'], box_detail['y1'])
                                if page_for_ui_highlight.rotation != 0:
                                    final_ui_rect = rotated_coords_rect * ui_derotation_matrix
                                    final_ui_rect.normalize()
                                else:
                                    final_ui_rect = rotated_coords_rect
                                page_for_ui_highlight.draw_rect(
                                    final_ui_rect, 
                                    color=HIGHLIGHT_COLOR_UI,  # Use defined color
                                    width=HIGHLIGHT_WIDTH_UI,  # Use defined width
                                    overlay=True
                                )
                            analysis_result['highlighted_image'] = page_for_ui_highlight.get_pixmap(dpi=200).tobytes("png")
                except Exception as e_hl: 
                    st.warning(f"Highlighting preparation error page {current_page_num_loop}: {e_hl}")
            
            target_column = col_fence if analysis_result.get('fence_found') else col_non_fence
            if analysis_result.get('fence_found'): 
                st.session_state.fence_pages.append(analysis_result)
            else: 
                st.session_state.non_fence_pages.append(analysis_result)

            with target_column:
                expander_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    reason = []
                    if analysis_result.get('text_found'): reason.append("Text")
                    if analysis_result.get('vision_found'): reason.append("Image")
                    if analysis_result.get('highlighted_image'): reason.append("Highlights")
                    if reason : expander_title += f" ({' & '.join(reason)} Match)"
                
                with st.expander(expander_title, expanded=True): 
                    inner_col1, inner_col2 = st.columns([2, 1]) 
                    with inner_col1:
                        st.markdown("##### Drawing View")
                        display_image_bytes = analysis_result.get('highlighted_image') or analysis_result.get('image')
                        if display_image_bytes:
                            try: 
                                st.image(display_image_bytes, caption=f"Page {analysis_result['page_number']}{' (Highlighted)' if analysis_result.get('highlighted_image') else ''}")
                            except Exception as img_e: 
                                st.error(f"Error displaying image for page {analysis_result['page_number']}: {img_e}")
                        else: 
                            st.warning(f"No image available for page {analysis_result['page_number']}.")
                        
                        dl_links = []
                        if analysis_result.get('highlighted_image'): 
                            dl_links.append(get_image_download_link(analysis_result['highlighted_image'], f"page_{analysis_result['page_number']}_highlighted.png", "DL Highlighted IMG"))
                        if analysis_result.get('image'): 
                            dl_links.append(get_image_download_link(analysis_result['image'], f"page_{analysis_result['page_number']}_original.png", "DL Original IMG"))
                        if dl_links: 
                            st.markdown(" ".join(dl_links), unsafe_allow_html=True)
                    
                    with inner_col2:
                        st.markdown("##### Analysis Details")
                        if analysis_result.get('fence_found'):
                            summary_points = []
                            if analysis_result.get('text_found'): summary_points.append("✔️ Text indicates fences.")
                            if analysis_result.get('vision_found'): summary_points.append("✔️ Image shows fences.")
                            if not summary_points: summary_points.append(" Fence flagged by other means.") 
                            st.markdown("\n".join(f"- {s}" for s in summary_points))
                        else: 
                            st.markdown("No strong fence indicators found.")

                        if analysis_result.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{analysis_result['text_response']}_")
                        if analysis_result.get('vision_response'):
                            with st.popover("Image Log"): st.markdown(f"_{analysis_result['vision_response']}_")
                        if analysis_result.get('text_snippet'):
                            st.markdown("---"); st.markdown("**Key Snippet:**"); st.code(analysis_result['text_snippet'],language=None)
                        
                        if highlight_fence_text_app and 'fence_text_details_for_display' in analysis_result and analysis_result.get('fence_found'):
                            details = analysis_result['fence_text_details_for_display']
                            if details:
                                st.markdown("---"); st.markdown("**Highlights (from Text):**")
                                disp_set = set(); count = 0
                                for d_item in sorted(details, key=lambda x: x.get('y0', 0)):
                                    txt = d_item.get('text', f"Indicator @ ({d_item.get('x0',0):.0f},{d_item.get('y0',0):.0f})")
                                    tag = d_item.get('tag_from_llm', 'N/A')
                                    type_llm = d_item.get('type_from_llm', 'N/A')
                                    display_text = f"- `{txt}` (Type: {type_llm}, Tag: {tag})"
                                    if display_text not in disp_set:
                                        st.markdown(display_text)
                                        disp_set.add(display_text); count += 1
                                    if count >= 15 and len(details) > 17: st.markdown(f"- ...& {len(details)-count} more."); break
            
            summary_placeholder.markdown(f"""
            ### Summary (Processed: {st.session_state.total_pages_processed}/{st.session_state.doc_total_pages})
            - ✅ **Fence Pages:** {len(st.session_state.fence_pages)}
            - ❌ **Non-Fence Pages:** {len(st.session_state.non_fence_pages)}
            """)

        st.session_state.processing_complete = True
        progress_bar.empty()
        status_text.success("All pages processed!")
        if doc_for_processing_loop: doc_for_processing_loop.close() 

        if st.session_state.fence_pages and st.session_state.original_pdf_bytes:
            pdf_bytes, pdf_name = generate_combined_highlighted_pdf(
                st.session_state.original_pdf_bytes,
                st.session_state.fence_pages,
                st.session_state.uploaded_pdf_name
            )
            if pdf_bytes:
                st.session_state.highlighted_pdf_bytes_for_download = pdf_bytes
                st.session_state.highlighted_pdf_filename_for_download = pdf_name
            else:
                st.warning(f"Could not generate combined highlighted PDF: {pdf_name}") 

        final_summary_text = f"""
        ### Final Summary
        - **Total Pages Processed:** {st.session_state.doc_total_pages}
        - ✅ **Fence-Related Pages:** {len(st.session_state.fence_pages)} ({int(len(st.session_state.fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        - ❌ **Non-Fence Pages:** {len(st.session_state.non_fence_pages)} ({int(len(st.session_state.non_fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        """
        summary_placeholder.markdown(final_summary_text)
        
        if st.session_state.get('highlighted_pdf_bytes_for_download'):
            st.download_button(
                label="⬇️ Download Highlighted Fence Pages (PDF)",
                data=st.session_state.highlighted_pdf_bytes_for_download,
                file_name=st.session_state.highlighted_pdf_filename_for_download,
                mime="application/pdf",
                key="download_combined_pdf_button_main"
            )

    elif st.session_state.get('processing_complete'):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h2>📊 Analysis Results</h2></div>", unsafe_allow_html=True)
        
        final_summary_text = f"""
        ### Final Summary
        - **Total Pages Processed:** {st.session_state.doc_total_pages}
        - ✅ **Fence-Related Pages:** {len(st.session_state.fence_pages)} ({int(len(st.session_state.fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        - ❌ **Non-Fence Pages:** {len(st.session_state.non_fence_pages)} ({int(len(st.session_state.non_fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        """
        st.markdown(final_summary_text)

        if st.session_state.get('highlighted_pdf_bytes_for_download'):
             st.download_button(
                label="⬇️ Download Highlighted Fence Pages (PDF)",
                data=st.session_state.highlighted_pdf_bytes_for_download,
                file_name=st.session_state.highlighted_pdf_filename_for_download,
                mime="application/pdf",
                key="download_combined_pdf_button_rerun"
            )
        else: 
            if st.session_state.fence_pages and st.session_state.original_pdf_bytes and st.session_state.uploaded_pdf_name:
                pdf_bytes, pdf_name = generate_combined_highlighted_pdf(
                    st.session_state.original_pdf_bytes,
                    st.session_state.fence_pages,
                    st.session_state.uploaded_pdf_name
                )
                if pdf_bytes:
                    st.session_state.highlighted_pdf_bytes_for_download = pdf_bytes
                    st.session_state.highlighted_pdf_filename_for_download = pdf_name
                    st.download_button(
                        label="⬇️ Download Highlighted Fence Pages (PDF)",
                        data=st.session_state.highlighted_pdf_bytes_for_download,
                        file_name=st.session_state.highlighted_pdf_filename_for_download,
                        mime="application/pdf",
                        key="download_combined_pdf_button_rerun_generated"
                    )

        col_fence, col_non_fence = st.columns(2)
        with col_fence: st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
        with col_non_fence: st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")

        for res_data in st.session_state.fence_pages:
            with col_fence:
                expander_title = f"Page {res_data['page_number']}"
                reason = []
                if res_data.get('text_found'): reason.append("Text")
                if res_data.get('vision_found'): reason.append("Image")
                if res_data.get('highlighted_image'): reason.append("Highlights")
                if reason : expander_title += f" ({' & '.join(reason)} Match)"

                with st.expander(expander_title, expanded=False): 
                    inner_col1, inner_col2 = st.columns([2, 1])
                    with inner_col1:
                        st.markdown("##### Drawing View")
                        display_image_bytes = res_data.get('highlighted_image') or res_data.get('image')
                        if display_image_bytes:
                            st.image(display_image_bytes, caption=f"Page {res_data['page_number']}{' (Highlighted)' if res_data.get('highlighted_image') else ''}")
                        dl_links = []
                        if res_data.get('highlighted_image'): dl_links.append(get_image_download_link(res_data['highlighted_image'], f"page_{res_data['page_number']}_highlighted.png", "DL Highlighted IMG"))
                        if res_data.get('image'): dl_links.append(get_image_download_link(res_data['image'], f"page_{res_data['page_number']}_original.png", "DL Original IMG"))
                        if dl_links: st.markdown(" ".join(dl_links), unsafe_allow_html=True)
                    with inner_col2:
                        st.markdown("##### Analysis Details")
                        if res_data.get('fence_found'):
                            summary_points = []
                            if res_data.get('text_found'): summary_points.append("✔️ Text indicates fences.")
                            if res_data.get('vision_found'): summary_points.append("✔️ Image shows fences.")
                            if not summary_points: summary_points.append(" Fence flagged.")
                            st.markdown("\n".join(f"- {s}" for s in summary_points))
                        if res_data.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{res_data['text_response']}_")
                        if res_data.get('vision_response'):
                            with st.popover("Image Log"): st.markdown(f"_{res_data['vision_response']}_")
                        if res_data.get('text_snippet'):
                            st.markdown("---"); st.markdown("**Key Snippet:**"); st.code(res_data['text_snippet'],language=None)
                        if res_data.get('highlight_fence_text_app', True) and 'fence_text_details_for_display' in res_data and res_data.get('fence_found'):
                            details = res_data['fence_text_details_for_display']
                            if details:
                                st.markdown("---"); st.markdown("**Highlights (from Text):**")
                                disp_set = set(); count = 0
                                for d_item in sorted(details, key=lambda x: x.get('y0', 0)):
                                    txt = d_item.get('text', "N/A"); tag = d_item.get('tag_from_llm', 'N/A'); type_llm = d_item.get('type_from_llm', 'N/A')
                                    display_text = f"- `{txt}` (Type: {type_llm}, Tag: {tag})"
                                    if display_text not in disp_set: st.markdown(display_text); disp_set.add(display_text); count+=1
                                    if count >=15 and len(details) > 17: st.markdown(f"- ...& {len(details)-count} more."); break
        
        for res_data in st.session_state.non_fence_pages:
            with col_non_fence:
                with st.expander(f"Page {res_data['page_number']}", expanded=False):
                    st.image(res_data.get('image'), caption=f"Page {res_data['page_number']}")
                    if res_data.get('text_response'):
                        with st.popover("Text Log"): st.markdown(f"_{res_data['text_response']}_")
                    if res_data.get('vision_response'):
                        with st.popover("Image Log"): st.markdown(f"_{res_data['vision_response']}_")
                    st.markdown("No strong fence indicators found.")

elif not st.session_state.get('original_pdf_bytes') and not (openai_key and llm_analysis_instance) :
     st.info("Please upload an engineering PDF document and ensure the OpenAI API key is configured in the sidebar to begin analysis.")
elif not st.session_state.get('original_pdf_bytes'):
    st.info("Upload an engineering PDF document to begin analysis.")
elif not (openai_key and llm_analysis_instance):
    st.error("OpenAI models not initialized. Please check your API key in the sidebar. Analysis cannot proceed.")


st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App</p>", unsafe_allow_html=True)