import streamlit as st
from utils import analyze_page, get_fence_related_text_boxes # Ensure utils.py is updated
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
from PIL import Image, ImageDraw
import io

# Set page config to wide mode for better space utilization
st.set_page_config(
    page_title="Fence Detector for Engineering Drawings",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E3A8A; /* Dark blue */
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card { /* This class seems unused, can be removed if not applied */
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Classes for expander borders - might need !important or more specific selectors if overridden */
    .stExpander {
        border-left: 5px solid #ccc; /* Default border */
        margin-bottom: 10px;
    }
    /* Dynamic styling for expanders will be harder with just CSS, Streamlit controls this.
       We can try to set a class on the expander container if possible, or adjust content within.
       For now, the tabs serve as the primary visual distinction.
    */
    .download-button {
        margin-top: 10px;
        display: inline-block; /* Ensure buttons are on the same line if space allows */
        margin-right: 10px; /* Add some space between buttons */
        padding: 8px 12px;
        background-color: #007bff;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .download-button:hover {
        background-color: #0056b3;
        color: white;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    openai_key_input = st.text_input("Enter OpenAI API Key", type="password", help="Your API key is stored securely if using Streamlit secrets.")
    openai_key = openai_key_input or st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

    st.subheader("Model Selection")
    model_options = {
        "gpt-3.5-turbo (cheaper & faster - ~$0.001/page)": "gpt-3.5-turbo",
        "gpt-4 (more accurate & expensive - ~$0.03/page)": "gpt-4",
        "gpt-4o (latest, fast & accurate - ~$0.005/page input)": "gpt-4o", # Added GPT-4o
    }

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(model_options.values())[0]

    selected_label = st.radio(
        "Select language model for text analysis:",
        list(model_options.keys()),
        index=list(model_options.values()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = model_options[selected_label]

    process_images = st.toggle("🖼️ Enable image analysis (GPT-4-Vision)", value=False, # Default to False for cost
                              help="Uses GPT-4-Vision model (~$0.01/image). Slower but can find visual fences.")

    highlight_fence_text = st.toggle("🔍 Highlight fence-related text & indicators", value=True,
                                    help="Identifies and highlights fence legends and their drawing indicators.")

    st.subheader("Fence Keywords")
    default_keywords = ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh', 'panel', 'chain link']
    custom_keywords_str = st.text_area("Add custom keywords (one per line):",
                                     "\n".join(default_keywords), height=150)
    FENCE_KEYWORDS = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]

# --- Initialize Language Models ---
llm_text = None
llm_vision = None

if openai_key:
    try:
        llm_text = ChatOpenAI(model=st.session_state.selected_model, temperature=0, openai_api_key=openai_key)
        if process_images:
            # GPT-4-turbo is an alias that points to the latest vision-capable GPT-4 model
            # Using "gpt-4-vision-preview" or a specific version like "gpt-4-1106-vision-preview" was common
            # "gpt-4-turbo" with vision is now often the preferred model.
            # If "gpt-4-turbo" itself isn't recognized as vision-capable by older langchain,
            # you might need "gpt-4-vision-preview" or check latest recommendations.
            # For now, assuming "gpt-4-turbo" is correctly configured for vision by Langchain.
            llm_vision = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {e}")
        st.stop()
else:
    st.warning("Please provide an OpenAI API key in the sidebar to use this application.")
    st.stop()


# --- Helper Function for Download Links ---
def get_image_download_link(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'

# --- File Uploader ---
st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf = st.file_uploader("Upload Engineering PDF Document", type=["pdf"])

if uploaded_pdf and llm_text: # Ensure llm_text is initialized
    pdf_bytes_io = io.BytesIO(uploaded_pdf.read()) # Use BytesIO for PyMuPDF
    doc = fitz.open(stream=pdf_bytes_io, filetype="pdf")

    fence_pages = []
    non_fence_pages = []
    total_pages = len(doc)

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initializing analysis...")

    # --- Process Each Page ---
    for i in range(total_pages):
        current_page_num = i + 1
        progress = current_page_num / total_pages
        progress_bar.progress(progress)
        status_text.text(f"Processing Page {current_page_num} of {total_pages}...")

        page_obj = doc.load_page(i)
        text_content = page_obj.get_text("text") # Get plain text

        # Higher DPI for better quality images for vision and display
        pix = page_obj.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        page_data = {
            "page_number": current_page_num,
            "text": text_content,
            "image_bytes": img_bytes, # Original image bytes for display/download
            "image_b64": img_b64,   # Base64 for vision model
            # No need to pass full pdf_bytes for each page analysis
        }

        analysis_result = {}
        try:
            with st.spinner(f"Analyzing Page {current_page_num} with LLM..."):
                analysis_result = analyze_page(page_data, llm_text, llm_vision if process_images else None, FENCE_KEYWORDS)
        except Exception as e:
            st.error(f"Error analyzing page {current_page_num}: {e}")
            analysis_result = { # Default structure on error
                "page_number": current_page_num, "fence_found": False, "text_found": False,
                "vision_found": False, "text_response": f"Error: {e}", "vision_response": None,
                "text_snippet": None, "image": img_bytes
            }


        # --- Text Highlighting Logic ---
        if highlight_fence_text and analysis_result.get('fence_found'):
            status_text.text(f"Page {current_page_num}: Identifying text locations for highlighting...")
            single_page_pdf_bytes_io = io.BytesIO()
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=i, to_page=i)
            single_page_doc.save(single_page_pdf_bytes_io)
            single_page_doc.close()
            single_page_bytes = single_page_pdf_bytes_io.getvalue()

            fence_text_boxes_details = [] # Renamed for clarity
            page_width, page_height = 0, 0

            try:
                with st.spinner(f"Page {current_page_num}: Highlighting fence-related text..."):
                    fence_text_boxes_details, page_width, page_height = get_fence_related_text_boxes(
                        single_page_bytes, llm_text, FENCE_KEYWORDS
                    )
            except Exception as e:
                st.warning(f"Could not perform highlighting for page {current_page_num}: {e}")


            if fence_text_boxes_details and page_width > 0 and page_height > 0:
                img_pil = Image.open(io.BytesIO(analysis_result['image'])) # Use original image from analysis_result
                draw = ImageDraw.Draw(img_pil)
                
                scale_x = img_pil.width / page_width
                scale_y = img_pil.height / page_height
                
                for box_detail in fence_text_boxes_details:
                    x0 = box_detail['x0'] * scale_x
                    y0 = box_detail['y0'] * scale_y
                    x1 = box_detail['x1'] * scale_x
                    y1 = box_detail['y1'] * scale_y
                    
                    padding = 3 # pixels
                    draw.rectangle([x0 - padding, y0 - padding, x1 + padding, y1 + padding],
                                     outline="lime", width=3) # Bright green, thick line
                
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format="PNG")
                analysis_result['highlighted_image'] = img_buffer.getvalue()
                analysis_result['fence_text_details_for_display'] = fence_text_boxes_details
            else:
                 analysis_result['highlighted_image'] = None # Ensure key exists
                 analysis_result['fence_text_details_for_display'] = []


        if analysis_result.get('fence_found'):
            fence_pages.append(analysis_result)
        else:
            non_fence_pages.append(analysis_result)

    progress_bar.empty()
    status_text.empty()

    # --- Display Results ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><h2>📊 Analysis Results</h2></div>", unsafe_allow_html=True)

    st.markdown(f"""
    ### Summary Statistics
    - **Total Pages Analyzed:** {total_pages}
    - ✅ **Fence-Related Pages:** {len(fence_pages)} ({int(len(fence_pages)/total_pages*100) if total_pages > 0 else 0}%)
    - ❌ **Non-Fence Pages:** {len(non_fence_pages)} ({int(len(non_fence_pages)/total_pages*100) if total_pages > 0 else 0}%)
    """)

    fence_tab, non_fence_tab = st.tabs(["✅ Fence-Related Pages", "❌ Non-Fence Pages"])

    with fence_tab:
        if not fence_pages:
            st.info("No fence-related pages were identified in this document.")
        else:
            st.subheader(f"Found {len(fence_pages)} Fence-Related Page(s)")
            for res in sorted(fence_pages, key=lambda x: x['page_number']): # Sort by page number
                expander_title = f"Page {res['page_number']}"
                if res['text_found'] and res['vision_found']:
                    expander_title += " (Text & Image Match)"
                elif res['text_found']:
                    expander_title += " (Text Match)"
                elif res['vision_found']:
                    expander_title += " (Image Match)"

                with st.expander(expander_title):
                    col1, col2 = st.columns([2, 1]) # Image column wider

                with col1:
                    st.markdown("##### Drawing View")

                    # Determine which image to display
                    image_to_display_bytes = None
                    caption_suffix = ""

                    if res.get('highlighted_image'): # Check if highlighted image exists and is not None
                        image_to_display_bytes = res['highlighted_image']
                        caption_suffix = " (Highlighted)"
                    elif res.get('image'): # Fallback to original image if no highlighted one
                        image_to_display_bytes = res['image']
                    
                    if image_to_display_bytes: # Ensure there are bytes to display
                        try:
                            st.image(image_to_display_bytes, caption=f"Page {res['page_number']}{caption_suffix}")
                        except Exception as img_e:
                            st.error(f"Error displaying image for page {res['page_number']}: {img_e}")
                            # Optionally, try to display the original image if highlighted one failed
                            if caption_suffix and res.get('image') and res['image'] != image_to_display_bytes :
                                st.info("Attempting to display original image instead.")
                                try:
                                    st.image(res['image'], caption=f"Page {res['page_number']} (Original)")
                                except Exception as orig_img_e:
                                    st.error(f"Error displaying original image as well: {orig_img_e}")
                    else:
                        st.warning(f"No image data available for page {res['page_number']}.")

                    # Download Links
                    download_links = []
                    if res.get('highlighted_image'): # Check again for existence for download link
                        download_links.append(get_image_download_link(
                            res['highlighted_image'],
                            f"page_{res['page_number']}_fence_highlighted.png",
                            "Download Highlighted"
                        ))
                    if res.get('image'): # Original image download link
                        download_links.append(get_image_download_link(
                            res['image'],
                            f"page_{res['page_number']}_fence_original.png",
                            "Download Original"
                        ))
                    
                    if download_links:
                        st.markdown(" ".join(download_links), unsafe_allow_html=True)

                    with col2:
                        st.markdown("##### Analysis Details")
                        summary_points = []
                        if res.get('text_found'): summary_points.append("✔️ Text indicates fences.")
                        if res.get('vision_found'): summary_points.append("✔️ Image shows fences.")
                        if not summary_points: summary_points.append("❓ No direct fence indicators found by primary analysis, but flagged by secondary process.")
                        st.markdown("\n".join(f"- {s}" for s in summary_points))

                        if res.get('text_response'):
                            with st.popover("Text Analysis Log"):
                                st.markdown(f"_{res['text_response']}_")
                        if res.get('vision_response'):
                            with st.popover("Image Analysis Log"):
                                st.markdown(f"_{res['vision_response']}_")
                        
                        if res.get('text_snippet'):
                            st.markdown("---")
                            st.markdown("**Key Text Snippet (General):**")
                            st.code(res['text_snippet'], language=None)

                        if highlight_fence_text and 'fence_text_details_for_display' in res:
                            details_to_display = res['fence_text_details_for_display']
                            if details_to_display:
                                st.markdown("---")
                                st.markdown("**Highlighted Text/Indicators:**")
                                displayed_texts_set = set()
                                count = 0
                                for detail in sorted(details_to_display, key=lambda x: x.get('y0', 0)): # Sort by y-position
                                    text_val = detail.get('text', f"Indicator @ ({detail.get('x0',0):.0f}, {detail.get('y0',0):.0f})")
                                    if text_val not in displayed_texts_set:
                                        st.markdown(f"- `{text_val}`")
                                        displayed_texts_set.add(text_val)
                                        count += 1
                                    if count >= 10 and len(details_to_display) > 12 : # Show a few more if many
                                        st.markdown(f"- ... and {len(details_to_display) - count} more.")
                                        break


    with non_fence_tab:
        if not non_fence_pages:
            st.info("All pages in this document were identified as potentially fence-related.")
        else:
            st.subheader(f"Found {len(non_fence_pages)} Non-Fence Page(s)")
            for res in sorted(non_fence_pages, key=lambda x: x['page_number']):
                with st.expander(f"Page {res['page_number']}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("##### Drawing View")
                        st.image(res['image'], caption=f"Page {res['page_number']}")
                        st.markdown(get_image_download_link(
                            res['image'],
                            f"page_{res['page_number']}_non_fence.png",
                            "Download Image"
                        ), unsafe_allow_html=True)

                    with col2:
                        st.markdown("##### Analysis Details")
                        st.markdown("No strong fence indicators found by the current analysis settings.")
                        if res.get('text_response'):
                            with st.popover("Text Analysis Log (Why it was NOT flagged)"):
                                st.markdown(f"_{res['text_response']}_")
                        if res.get('vision_response'):
                             with st.popover("Image Analysis Log (Why it was NOT flagged)"):
                                st.markdown(f"_{res['vision_response']}_")


elif uploaded_pdf and not llm_text:
    st.error("OpenAI models could not be initialized. Please check your API key and configuration.")

else:
    st.info("Upload a PDF document to begin the fence detection process.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Fence Detector App</p>", unsafe_allow_html=True)