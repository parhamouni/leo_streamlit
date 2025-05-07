import streamlit as st
from utils import analyze_page
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64
import pandas as pd

st.set_page_config(page_title="Fence Indicator Detector", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .fence-header {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 8px;
        color: #155724;
        margin-bottom: 10px;
    }
    .non-fence-header {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 8px;
        color: #721c24;
        margin-bottom: 10px;
    }
    .download-button {
        display: inline-block;
        padding: 8px 12px;
        background-color: #4CAF50;
        color: white !important;
        text-decoration: none;
        border-radius: 4px;
        text-align: center;
        margin: 5px;
    }
    .download-button:hover {
        background-color: #45a049;
        text-decoration: none;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 12px;
        margin: 15px 0;
    }
    .item-1 {
        color: #28a745;
        font-weight: bold;
    }
    .item-2 {
        color: #dc3545;
        font-weight: bold;
    }
    .item-3 {
        color: #007bff;
        font-weight: bold;
    }
    .item-other {
        color: #fd7e14;
        font-weight: bold;
    }
    .debug-box {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🔍 Fence Indicator Detector for Engineering Drawings</div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
This tool detects and highlights fence-related elements in engineering drawings.
It uses OCR to detect text and an AI to identify which elements are related to fences.
</div>
""", unsafe_allow_html=True)

# Get API key from secrets or environment variable
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Set up OpenAI model
@st.cache_resource
def get_llm_models(api_key):
    try:
        return {
            "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
            "gpt-4": ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key),
            "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        }
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {str(e)}")
        return {}

# Model selection options
model_options = {
    "gpt-4o (recommended - ~$0.01/page)": "gpt-4o",
    "gpt-4 (more expensive - ~$0.03/page)": "gpt-4",
    "gpt-3.5-turbo (faster but less accurate - ~$0.001/page)": "gpt-3.5-turbo"
}

# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o"

# Get models
if openai_key:
    models = get_llm_models(openai_key)
else:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    models = {}

# UI for model selection
selected_label = st.radio("Select model for analysis:", 
                        list(model_options.keys()), 
                        index=list(model_options.values()).index(st.session_state.selected_model))
st.session_state.selected_model = model_options[selected_label]

# Enable vision analysis (required for OCR)
process_images = st.toggle("🖼️ Enable image analysis (required for indicator detection)", value=True)

# Set up the selected models
if models and st.session_state.selected_model in models:
    llm_text = models[st.session_state.selected_model]
    # Always use GPT-4o for vision tasks as it's best for engineering drawings
    llm_vision = models["gpt-4o"] if process_images else None
else:
    llm_text = None
    llm_vision = None
    if openai_key:
        st.warning("Selected model not available. Please choose another model.")

# File uploader
uploaded_pdf = st.file_uploader("Upload Engineering PDF", type=["pdf"])

FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail', 'chain link', 'enclosure']

# Main analysis logic
if uploaded_pdf and llm_text:
    try:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        
        fence_pages = []
        non_fence_pages = []
        
        col_results1, col_results2 = st.columns(2)
        
        with col_results1:
            st.markdown("""<div class="fence-header"><h4>✅ Pages with Fence Elements</h4></div>""", unsafe_allow_html=True)
        with col_results2:
            st.markdown("""<div class="non-fence-header"><h4>❌ Pages without Fence Elements</h4></div>""", unsafe_allow_html=True)
        
        # Progress bar for processing
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Process each page
        for i in range(len(doc)):
            try:
                status.text(f"Processing page {i+1} of {len(doc)}...")
                
                # Load page
                page_obj = doc.load_page(i)
                text = page_obj.get_text()
                
                # Higher DPI for better detail
                pix = page_obj.get_pixmap(dpi=200)  
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                page = {
                    "page_number": i + 1,
                    "text": text,
                    "image_bytes": img_bytes,
                    "image_b64": img_b64
                }
                
                # Analyze page
                result = analyze_page(page, llm_vision, FENCE_KEYWORDS)
                
                # Sort into appropriate category
                if result['fence_found']:
                    fence_pages.append(result)
                else:
                    non_fence_pages.append(result)
                
                # Update progress
                progress_bar.progress((i + 1) / len(doc))
            
            except Exception as e:
                st.error(f"Error processing page {i+1}: {str(e)}")
        
        # Clear status
        status.empty()
        progress_bar.empty()
        
        # Display fence-related pages
        for result in fence_pages:
            with col_results1.expander(f"Page {result['page_number']}"):
                # Show highlighted image if available, otherwise original
                if result['highlighted_image'] is not None:
                    st.image(result['highlighted_image'], caption="OCR Text Detection with Fence Analysis", use_column_width=True)
                else:
                    st.image(result['image'], caption="Page Image")
                
                st.markdown(f"✅ **Fence Elements Detected**")
                
                # Display OCR detection results
                if "ocr_text_elements" in result:
                    ocr_count = len(result["ocr_text_elements"])
                    st.write(f"OCR detected {ocr_count} text elements")
                
                # Show detected fence references from text
                if result.get("text_references"):
                    st.markdown("### 📝 Text References to Fences")
                    for ref in result["text_references"]:
                        st.markdown(f"- \"{ref['text']}\"")
                
                # Show LLM analysis
                if result.get("llm_analysis"):
                    st.markdown("### 🔍 AI Analysis")
                    st.markdown(f"""<div class="debug-box">{result["llm_analysis"]}</div>""", unsafe_allow_html=True)
                
                # Show legend items (if in the old format)
                if result.get('legend_items', []):
                    st.markdown("### 📋 Fence Items in Legend")
                    for item in result['legend_items']:
                        item_num = item.get('item_number', '')
                        desc = item.get('description', '')
                        
                        # Style based on item number
                        css_class = f"item-{item_num}" if item_num in ['1', '2', '3'] else "item-other"
                        st.markdown(f"<span class='{css_class}'>Item {item_num}: {desc}</span>", unsafe_allow_html=True)
                    
                    st.divider()
                
                # Show detected indicators (if in the old format)
                if result.get('fence_indicators', []):
                    st.markdown("### 📍 Detected Fence Indicators")
                    
                    # Create table data
                    table_data = []
                    for idx, indicator in enumerate(result['fence_indicators']):
                        item_num = indicator.get('item_number', '')
                        center = indicator.get('center', [0, 0])
                        
                        # Determine CSS class based on item number
                        css_class = f"item-{item_num}" if item_num in ['1', '2', '3'] else "item-other"
                        
                        # Add to table with styling
                        table_data.append({
                            "#": idx+1,
                            "Item": f'<span class="{css_class}">Item {item_num}</span>',
                            "Position": f'({center[0]:.1f}%, {center[1]:.1f}%)'
                        })
                    
                    if table_data:
                        # Convert to DataFrame for display
                        df = pd.DataFrame(table_data)
                        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
                
                # Download link for highlighted image
                if result.get('highlighted_image'):
                    highlighted_b64 = base64.b64encode(result['highlighted_image']).decode()
                    download_link = f'<a href="data:image/png;base64,{highlighted_b64}" download="fence_analysis_page_{result["page_number"]}.png" class="download-button">⬇️ Download Analysis Image</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
        
        # Display non-fence pages
        for result in non_fence_pages:
            with col_results2.expander(f"Page {result['page_number']}"):
                st.image(result['image'], caption="Page Image", width=300)
                st.markdown("❌ **No fence elements detected**")
                
                # Show OCR analysis for debugging
                if st.checkbox(f"Show OCR analysis for page {result['page_number']}", key=f"show_ocr_{result['page_number']}", value=False):
                    if "ocr_text_elements" in result:
                        ocr_count = len(result["ocr_text_elements"])
                        st.write(f"OCR detected {ocr_count} text elements")
                    
                    if result.get("llm_analysis"):
                        st.markdown("### 🔍 AI Analysis")
                        st.markdown(f"""<div class="debug-box">{result["llm_analysis"]}</div>""", unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Analysis Summary")
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Pages with Fence Elements", len(fence_pages))
        with col_stats2:
            st.metric("Pages without Fence Elements", len(non_fence_pages))
            
        # Count indicators by type (old format support)
        if fence_pages and any('fence_indicators' in page for page in fence_pages):
            st.markdown("### 📋 Fence Indicator Summary")
            
            # Count indicators by item number
            item_counts = {}
            
            for page in fence_pages:
                for indicator in page.get('fence_indicators', []):
                    item_num = indicator.get('item_number', '')
                    if item_num:
                        item_counts[item_num] = item_counts.get(item_num, 0) + 1
            
            # Display counts
            if item_counts:
                cols = st.columns(len(item_counts))
                for i, (item_num, count) in enumerate(item_counts.items()):
                    # Get description if available
                    desc = ""
                    if fence_pages[0].get('legend_items'):
                        for item in fence_pages[0].get('legend_items', []):
                            if item.get('item_number') == item_num:
                                desc = item.get('description', '').split('.')[0]
                                break
                    
                    title = f"Item {item_num}"
                    if desc:
                        title += f": {desc[:20]}..."
                    
                    cols[i].metric(title, count)
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
else:
    if uploaded_pdf and not llm_text:
        st.warning("Please ensure OpenAI API key is properly configured.")