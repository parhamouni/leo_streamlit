import streamlit as st
from utils import analyze_page, get_fence_related_text_boxes 
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
from PIL import Image, ImageDraw
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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)

# Initialize session state
if 'fence_pages' not in st.session_state: st.session_state.fence_pages = []
if 'non_fence_pages' not in st.session_state: st.session_state.non_fence_pages = []
if 'total_pages_processed' not in st.session_state: st.session_state.total_pages_processed = 0 # Changed name for clarity
if 'doc_total_pages' not in st.session_state: st.session_state.doc_total_pages = 0
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'fence_keywords_app' not in st.session_state: # Use a different key to avoid conflict if needed
    st.session_state.fence_keywords_app = ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link']
if 'run_analysis_triggered' not in st.session_state:
    st.session_state.run_analysis_triggered = False
if 'uploaded_pdf_name' not in st.session_state:
    st.session_state.uploaded_pdf_name = None


with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key =  st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key_input = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
        openai_key = openai_key_input

    st.subheader("Model Selection")
    # Ensure these are valid model identifiers for the desired context lengths
    # For GPT-4 Turbo 128k, "gpt-4-turbo" or "gpt-4-turbo-preview" or "gpt-4-0125-preview"
    # For GPT-4o (128k), "gpt-4o"
    model_options = {
        "gpt-4o (128k context, recommended for highlighting)": "gpt-4o",
        "gpt-4-turbo (128k context, powerful)": "gpt-4-turbo",
        "gpt-4 (8k/32k context, check your API access)": "gpt-4",
        "gpt-3.5-turbo (16k context, fastest, lower accuracy)": "gpt-3.5-turbo"
    }
    
    default_model_value = "gpt-4o" # Default to a large context model
    if default_model_value not in model_options.values(): # Fallback if gpt-4o is not an option string
        default_model_value = "gpt-4-turbo"
    if default_model_value not in model_options.values():
        default_model_value = "gpt-4" # Further fallback

    if "selected_model_for_analysis" not in st.session_state:
        st.session_state.selected_model_for_analysis = default_model_value

    # Find the key (label) for the default model value to set the index
    default_model_label = [label for label, value in model_options.items() if value == st.session_state.selected_model_for_analysis][0]

    selected_label = st.radio(
        "Select LLM for Text Analysis & Highlighting:",
        list(model_options.keys()),
        key="model_selector_radio",
        index=list(model_options.keys()).index(default_model_label)
    )
    st.session_state.selected_model_for_analysis = model_options[selected_label]
    st.info(f"Using: **{st.session_state.selected_model_for_analysis}** for main analysis.")


    process_images_vision = st.toggle("🖼️ Enable visual analysis (GPT-4 Vision)", value=False, key="vision_toggle")
    vision_model_name_option = "gpt-4-turbo" # Generally points to latest vision capable GPT-4

    highlight_fence_text_app = st.toggle("🔍 Highlight fence-related text & indicators", value=True, key="highlight_toggle")
    
    st.subheader("Fence Keywords")
    custom_keywords_str = st.text_area("Add custom keywords (one per line):",
                                     "\n".join(st.session_state.fence_keywords_app), height=150, key="fence_keywords_text_area")
    if st.button("Update Keywords", key="update_keywords_button"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun() # Rerun to reflect updated keywords immediately

    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


llm_analysis_instance = None # This will be used for both analyze_page text and get_fence_related_text_boxes
llm_vision_instance = None

if openai_key:
    try:
        print(f"DEBUG APP: Initializing llm_analysis_instance with model: {st.session_state.selected_model_for_analysis}")
        llm_analysis_instance = ChatOpenAI(model=st.session_state.selected_model_for_analysis, temperature=0, openai_api_key=openai_key, timeout=120) # Increased timeout
        
        if process_images_vision:
            print(f"DEBUG APP: Initializing llm_vision_instance with model: {vision_model_name_option}")
            llm_vision_instance = ChatOpenAI(model=vision_model_name_option, temperature=0, openai_api_key=openai_key, timeout=120)
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {type(e).__name__}: {e}"); st.stop()
else:
    st.warning("Please provide an OpenAI API key in the sidebar."); st.stop()


def get_image_download_link(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
newly_uploaded_pdf = st.file_uploader("Upload Engineering PDF Document", type=["pdf"], key="pdf_uploader_main")

# Logic to reset if a new PDF is uploaded
if newly_uploaded_pdf is not None and newly_uploaded_pdf.name != st.session_state.get('uploaded_pdf_name'):
    print(f"DEBUG APP: New PDF uploaded: {newly_uploaded_pdf.name}. Resetting analysis state.")
    st.session_state.fence_pages = []
    st.session_state.non_fence_pages = []
    st.session_state.total_pages_processed = 0
    st.session_state.doc_total_pages = 0
    st.session_state.processing_complete = False
    st.session_state.run_analysis_triggered = False # Reset trigger
    st.session_state.uploaded_pdf_name = newly_uploaded_pdf.name

if newly_uploaded_pdf:
    if st.button("Start Analysis", key="start_analysis_button"):
        st.session_state.run_analysis_triggered = True
        # Clear previous results when starting a new analysis on the same file explicitly
        if newly_uploaded_pdf.name == st.session_state.get('uploaded_pdf_name_processed'):
             st.session_state.fence_pages = []
             st.session_state.non_fence_pages = []
             st.session_state.total_pages_processed = 0
             st.session_state.doc_total_pages = 0
             st.session_state.processing_complete = False
        st.session_state.uploaded_pdf_name_processed = newly_uploaded_pdf.name


if st.session_state.run_analysis_triggered and newly_uploaded_pdf and llm_analysis_instance:
    if not st.session_state.processing_complete and st.session_state.total_pages_processed == 0 : # Start processing only if not already done and not started
        pdf_bytes_io = io.BytesIO(newly_uploaded_pdf.read())
        doc = fitz.open(stream=pdf_bytes_io, filetype="pdf")
        st.session_state.doc_total_pages = len(doc)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h2>📊 Analysis Results (Live Updates)</h2></div>", unsafe_allow_html=True)
        summary_placeholder = st.empty() 
        col_fence, col_non_fence = st.columns(2)
        with col_fence: st.subheader("✅ Fence-Related Pages")
        with col_non_fence: st.subheader("❌ Non-Fence Pages")

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(st.session_state.doc_total_pages):
            current_page_num_loop = i + 1
            st.session_state.total_pages_processed = current_page_num_loop # Update total processed
            progress = current_page_num_loop / st.session_state.doc_total_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing Page {current_page_num_loop} of {st.session_state.doc_total_pages}...")

            page_obj = doc.load_page(i)
            text_content = page_obj.get_text("text")
            pix = page_obj.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            page_data = {"page_number": current_page_num_loop, "text": text_content, "image_bytes": img_bytes, "image_b64": img_b64}

            analysis_result = {}
            try:
                with st.spinner(f"Page {current_page_num_loop}: Core analysis..."):
                    analysis_result = analyze_page(page_data, llm_analysis_instance, 
                                                   llm_vision_instance if process_images_vision else None, 
                                                   FENCE_KEYWORDS_APP)
            except Exception as e:
                st.error(f"Core analysis error page {current_page_num_loop}: {e}")
                analysis_result = {"page_number": current_page_num_loop, "fence_found": False, "image": img_bytes}

            if highlight_fence_text_app and analysis_result.get('fence_found'):
                status_text.text(f"Page {current_page_num_loop}: Highlighting...")
                single_page_pdf_bytes_io = io.BytesIO()
                single_page_doc_fitz = fitz.open()
                single_page_doc_fitz.insert_pdf(doc, from_page=i, to_page=i)
                single_page_doc_fitz.save(single_page_pdf_bytes_io); single_page_doc_fitz.close()
                single_page_bytes = single_page_pdf_bytes_io.getvalue()
                
                analysis_result['highlighted_image'] = None
                analysis_result['fence_text_details_for_display'] = []
                try:
                    with st.spinner(f"Page {current_page_num_loop}: Identifying highlights..."):
                        # Use llm_analysis_instance for highlighting text logic
                        # Pass the selected model NAME for potential internal use by utils (e.g. tiktoken)
                        fence_text_boxes_details, page_width_pdf, page_height_pdf = get_fence_related_text_boxes(
                            single_page_bytes, llm_analysis_instance, FENCE_KEYWORDS_APP, 
                            st.session_state.selected_model_for_analysis 
                        ) # Removed llm_vision from here as per previous utils
                    if fence_text_boxes_details and page_width_pdf > 0:
                        img_pil = Image.open(io.BytesIO(analysis_result.get('image', img_bytes)))
                        draw = ImageDraw.Draw(img_pil)
                        scale_x = img_pil.width / page_width_pdf; scale_y = img_pil.height / page_height_pdf
                        for box in fence_text_boxes_details:
                            x0,y0,x1,y1 = box['x0']*scale_x,box['y0']*scale_y,box['x1']*scale_x,box['y1']*scale_y
                            draw.rectangle([x0-3,y0-3,x1+3,y1+3],outline="lime",width=3)
                        buf = io.BytesIO(); img_pil.save(buf, format="PNG"); analysis_result['highlighted_image'] = buf.getvalue()
                        analysis_result['fence_text_details_for_display'] = fence_text_boxes_details
                except Exception as e_hl: st.warning(f"Highlighting error page {current_page_num_loop}: {e_hl}")
            
            target_column = col_fence if analysis_result.get('fence_found') else col_non_fence
            if analysis_result.get('fence_found'): st.session_state.fence_pages.append(analysis_result)
            else: st.session_state.non_fence_pages.append(analysis_result)

            with target_column:
                # ... (Your existing expander and display logic for a single page result - unchanged)
                expander_title = f"Page {analysis_result['page_number']}"
                if analysis_result.get('fence_found'):
                    if analysis_result.get('text_found') and analysis_result.get('vision_found'): expander_title += " (Text & Image)"
                    elif analysis_result.get('text_found'): expander_title += " (Text Match)"
                    elif analysis_result.get('vision_found'): expander_title += " (Image Match)"
                
                with st.expander(expander_title, expanded=True): 
                    inner_col1, inner_col2 = st.columns([2, 1]) 
                    with inner_col1:
                        st.markdown("##### Drawing View")
                        display_image_bytes = analysis_result.get('highlighted_image') or analysis_result.get('image')
                        if display_image_bytes:
                            try: st.image(display_image_bytes, caption=f"Page {analysis_result['page_number']}{' (Highlighted)' if analysis_result.get('highlighted_image') else ''}")
                            except Exception as img_e: st.error(f"Err display img p{analysis_result['page_number']}: {img_e}")
                        else: st.warning(f"No image for page {analysis_result['page_number']}.")
                        
                        dl_links = []
                        if analysis_result.get('highlighted_image'): dl_links.append(get_image_download_link(analysis_result['highlighted_image'], f"page_{analysis_result['page_number']}_highlighted.png", "DL Highlighted"))
                        if analysis_result.get('image'): dl_links.append(get_image_download_link(analysis_result['image'], f"page_{analysis_result['page_number']}_original.png", "DL Original"))
                        if dl_links: st.markdown(" ".join(dl_links), unsafe_allow_html=True)
                    
                    with inner_col2:
                        st.markdown("##### Analysis Details")
                        if analysis_result.get('fence_found'):
                            summary_points = []
                            if analysis_result.get('text_found'): summary_points.append("✔️ Text indicates fences.")
                            if analysis_result.get('vision_found'): summary_points.append("✔️ Image shows fences.")
                            if not summary_points: summary_points.append(" Fence flagged.")
                            st.markdown("\n".join(f"- {s}" for s in summary_points))
                        else: st.markdown("No strong fence indicators found.")

                        if analysis_result.get('text_response'):
                            with st.popover("Text Log"): st.markdown(f"_{analysis_result['text_response']}_")
                        if analysis_result.get('vision_response'):
                            with st.popover("Image Log"): st.markdown(f"_{analysis_result['vision_response']}_")
                        if analysis_result.get('text_snippet'):
                            st.markdown("---"); st.markdown("**Key Snippet:**"); st.code(analysis_result['text_snippet'],language=None)
                        
                        if highlight_fence_text_app and 'fence_text_details_for_display' in analysis_result and analysis_result.get('fence_found'):
                            details = analysis_result['fence_text_details_for_display']
                            if details:
                                st.markdown("---"); st.markdown("**Highlights:**")
                                disp_set = set(); count = 0
                                for d_item in sorted(details, key=lambda x: x.get('y0', 0)):
                                    txt = d_item.get('text', f"Indicator @ ({d_item.get('x0',0):.0f},{d_item.get('y0',0):.0f})")
                                    tag = d_item.get('tag_from_llm', 'N/A')
                                    type_llm = d_item.get('type_from_llm', 'N/A')
                                    display_text = f"- `{txt}` (Type: {type_llm}, Tag: {tag})"
                                    if display_text not in disp_set: # Check uniqueness of the full display string
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
        summary_placeholder.markdown(f"""
        ### Final Summary
        - **Total Pages Processed:** {st.session_state.doc_total_pages}
        - ✅ **Fence-Related Pages:** {len(st.session_state.fence_pages)} ({int(len(st.session_state.fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        - ❌ **Non-Fence Pages:** {len(st.session_state.non_fence_pages)} ({int(len(st.session_state.non_fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
        """)

elif st.session_state.get('run_analysis_triggered') and st.session_state.get('processing_complete'):
    # If processing is complete and triggered, just display results (handles reruns)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h2>📊 Analysis Results</h2></div>", unsafe_allow_html=True)
    summary_placeholder = st.empty() 
    col_fence, col_non_fence = st.columns(2)
    with col_fence: st.subheader("✅ Fence-Related Pages")
    with col_non_fence: st.subheader("❌ Non-Fence Pages")
    
    summary_placeholder.markdown(f"""
    ### Final Summary
    - **Total Pages Processed:** {st.session_state.doc_total_pages}
    - ✅ **Fence-Related Pages:** {len(st.session_state.fence_pages)} ({int(len(st.session_state.fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
    - ❌ **Non-Fence Pages:** {len(st.session_state.non_fence_pages)} ({int(len(st.session_state.non_fence_pages)/st.session_state.doc_total_pages*100) if st.session_state.doc_total_pages > 0 else 0}%)
    """)

    for res_data in st.session_state.fence_pages: # Display stored fence pages
        with col_fence:
            # ... (Your existing expander and display logic for a single page result)
            expander_title = f"Page {res_data['page_number']}" # Example title
            # Add more logic as per your detailed display
            with st.expander(expander_title, expanded=False): # Default to collapsed on rerun
                st.image(res_data.get('highlighted_image') or res_data.get('image'))
                # Add more details from res_data

    for res_data in st.session_state.non_fence_pages: # Display stored non-fence pages
        with col_non_fence:
            # ... (Your existing expander and display logic for a single page result)
            expander_title = f"Page {res_data['page_number']}" # Example title
            with st.expander(expander_title, expanded=False):
                st.image(res_data.get('image'))
                # Add more details from res_data

elif newly_uploaded_pdf and not llm_analysis_instance:
    st.error("OpenAI models not initialized. Check API key.")
elif not newly_uploaded_pdf:
    st.info("Upload a PDF document and click 'Start Analysis' to begin.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App</p>", unsafe_allow_html=True)