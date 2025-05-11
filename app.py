import streamlit as st
from utils import analyze_page, get_fence_related_text_boxes
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
from PIL import Image, ImageDraw
import io

# Set page config to wide mode for better space utilization
st.set_page_config(
    page_title="Fence Detector for Engineering Drawings",
    layout="wide"  # This makes the app use the full width of the screen
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fence-card {
        border-left: 5px solid #28a745;
    }
    .non-fence-card {
        border-left: 5px solid #dc3545;
    }
    .download-button {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("<h1 class='main-header'>🔍 Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)

# Create a sidebar for configuration options
with st.sidebar:
    st.header("Configuration")
    
    # API Key input (still using secrets if available)
    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

    
    # Model selection with pricing info and session state persistence
    st.subheader("Model Selection")
    model_options = {
        "gpt-3.5-turbo (cheap & fast - ~$0.001/page)": "gpt-3.5-turbo",
        "gpt-4 (accurate & expensive - ~$0.03/page)": "gpt-4"
    }

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(model_options.values())[0]  # default to gpt-3.5

    selected_label = st.radio(
        "Select language model for text analysis:", 
        list(model_options.keys()), 
        index=list(model_options.values()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = model_options[selected_label]

    # Vision toggle and conditional loading
    process_images = st.toggle("🖼️ Enable image analysis", value=True, 
                              help="Uses GPT-4-turbo vision model (~$0.01/image)")
    
    # Text highlighting toggle
    highlight_fence_text = st.toggle("🔍 Highlight fence-related text", value=True,
                                    help="Identifies and highlights fence-related text on drawings")
    
    # Customize fence keywords
    st.subheader("Fence Keywords")
    default_keywords = ['fence', 'fencing', 'gate', 'barrier', 'guardrail']
    custom_keywords = st.text_area("Add custom keywords (one per line)", 
                                 "\n".join(default_keywords))
    FENCE_KEYWORDS = [k.strip() for k in custom_keywords.split("\n") if k.strip()]

# Initialize the language models
if openai_key:
    llm_text = ChatOpenAI(model=st.session_state.selected_model, temperature=0, openai_api_key=openai_key)
    llm_vision = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key) if process_images else None
else:
    st.error("Please provide an OpenAI API key in the sidebar to use this application.")
    st.stop()

# Create a function to download the image
def get_image_download_link(img_bytes, filename, text):
    """Generate a link to download an image"""
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

# File uploader in the main area
st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf = st.file_uploader("Upload Engineering PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Status counters
    fence_pages = []
    non_fence_pages = []
    
    # Progress bar for analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process all pages
    for i in range(len(doc)):
        # Update progress
        progress = (i + 1) / len(doc)
        progress_bar.progress(progress)
        status_text.text(f"Processing page {i+1} of {len(doc)}...")
        
        # Extract page data
        page_obj = doc.load_page(i)
        text = page_obj.get_text()
        pix = page_obj.get_pixmap(dpi=150)  # Higher DPI for better quality
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        page = {
            "page_number": i + 1,
            "text": text,
            "image_bytes": img_bytes,
            "image_b64": img_b64,
            "pdf_bytes": pdf_bytes
        }

        # Analyze the page
        with st.spinner(f"Analyzing Page {page['page_number']}..."):
            result = analyze_page(page, llm_text, llm_vision, FENCE_KEYWORDS)

        # Process text highlighting if enabled and fence was found
        highlighted_image = None
        fence_text_boxes = []
        
        if highlight_fence_text and result['fence_found']:
            with st.spinner(f"Highlighting fence text on Page {page['page_number']}..."):
                # Extract just this page from the PDF
                page_pdf_bytes = fitz.open()
                page_pdf_bytes.insert_pdf(doc, from_page=i, to_page=i)
                temp_buffer = io.BytesIO()
                page_pdf_bytes.save(temp_buffer)
                single_page_bytes = temp_buffer.getvalue()
                
                # Get fence-related text boxes
                fence_text_boxes, page_width, page_height = get_fence_related_text_boxes(
                    single_page_bytes, llm_text, FENCE_KEYWORDS
                )
                
                # Create PIL Image from PyMuPDF image
                img = Image.open(io.BytesIO(img_bytes))
                draw = ImageDraw.Draw(img)
                
                # Calculate scale factor between pdfplumber and image coordinates
                scale_x = img.width / page_width
                scale_y = img.height / page_height
                
                # Draw highlighted boxes
                for box in fence_text_boxes:
                    # Scale coordinates
                    x0 = box['x0'] * scale_x
                    y0 = box['y0'] * scale_y
                    x1 = box['x1'] * scale_x
                    y1 = box['y1'] * scale_y
                    
                    # Draw rectangle AROUND the text (not covering it)
                    # Add small padding around the text (3 pixels)
                    padding = 3
                    draw.rectangle([x0-padding, y0-padding, x1+padding, y1+padding], 
                                 outline="green", width=2)
                
                # Convert back to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                highlighted_image_bytes = img_buffer.getvalue()
                result['highlighted_image'] = highlighted_image_bytes

        # Add to appropriate list
        if result['fence_found']:
            fence_pages.append(result)
        else:
            non_fence_pages.append(result)
    
    # Clear progress elements after processing is complete
    progress_bar.empty()
    status_text.empty()
    
    # Display results in an organized manner
    st.markdown("<div class='section-header'><h2>🔍 Analysis Results</h2></div>", unsafe_allow_html=True)
    
    # Summary statistics at the top
    st.markdown(f"""
    ### 📊 Summary
    - ✅ **Fence-Related Pages:** {len(fence_pages)} of {len(doc)} ({int(len(fence_pages)/len(doc)*100)}%)
    - ❌ **Non-Fence Pages:** {len(non_fence_pages)} of {len(doc)} ({int(len(non_fence_pages)/len(doc)*100)}%)
    """)
    
    # Create tabs for Fence Pages and Non-Fence Pages
    fence_tab, non_fence_tab = st.tabs(["✅ Fence-Related Pages", "❌ Non-Fence Pages"])
    
    # Display fence-related pages
    with fence_tab:
        if not fence_pages:
            st.info("No fence-related pages found in this document.")
        else:
            for result in fence_pages:
                with st.expander(f"Page {result['page_number']}"):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Display image (original or highlighted)
                        if highlight_fence_text and 'highlighted_image' in result:
                            st.image(result['highlighted_image'], caption=f"Page {result['page_number']} (with highlighted fence text)")
                            
                            # Download buttons
                            st.markdown(get_image_download_link(
                                result['highlighted_image'], 
                                f"fence_page_{result['page_number']}_highlighted.png",
                                "⬇️ Download Highlighted Image"
                            ), unsafe_allow_html=True)
                            
                            st.markdown(get_image_download_link(
                                result['image'], 
                                f"fence_page_{result['page_number']}_original.png",
                                "⬇️ Download Original Image"
                            ), unsafe_allow_html=True)
                        else:
                            st.image(result['image'], caption=f"Page {result['page_number']}")
                            
                            # Download button
                            st.markdown(get_image_download_link(
                                result['image'], 
                                f"fence_page_{result['page_number']}.png",
                                "⬇️ Download Image"
                            ), unsafe_allow_html=True)
                    
                    with col2:
                        # Display analysis details
                        st.markdown("#### Detection Summary")
                        summary = []
                        if result['text_found']: summary.append("📝 Text mentions fences")
                        if result['vision_found']: summary.append("🖼️ Drawing shows fences")
                        st.markdown("**Results:** " + ", ".join(summary))
                        
                        if result['text_response']: 
                            st.markdown("#### Text Analysis")
                            st.markdown(f"{result['text_response']}")
                        
                        if result['vision_response']: 
                            st.markdown("#### Image Analysis")
                            st.markdown(f"{result['vision_response']}")
                        
                        if result['text_snippet']: 
                            st.markdown("#### Key Text Snippet")
                            st.code(result['text_snippet'])
                        
                        # Display detected fence elements
                        if highlight_fence_text and 'highlighted_image' in result:
                            fence_text_boxes, _, _ = get_fence_related_text_boxes(
                                single_page_bytes, llm_text, FENCE_KEYWORDS
                            )
                            
                            if fence_text_boxes:
                                st.markdown("#### Detected Fence Text Elements")
                                for idx, box in enumerate(fence_text_boxes):
                                    st.markdown(f"- **{box['text']}**")
    
    # Display non-fence pages
    with non_fence_tab:
        if not non_fence_pages:
            st.info("All pages in this document contain fence-related content.")
        else:
            for result in non_fence_pages:
                with st.expander(f"Page {result['page_number']}"):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.image(result['image'], caption=f"Page {result['page_number']}")
                        
                        # Download button
                        st.markdown(get_image_download_link(
                            result['image'], 
                            f"non_fence_page_{result['page_number']}.png",
                            "⬇️ Download Image"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        if result['text_response']: 
                            st.markdown("#### Text Analysis")
                            st.markdown(f"{result['text_response']}")
                        
                        if result['vision_response']: 
                            st.markdown("#### Image Analysis")
                            st.markdown(f"{result['vision_response']}")