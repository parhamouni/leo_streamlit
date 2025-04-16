# 🚀 Optimized Fence Detector for Engineering Drawings (Streamlit)

import streamlit as st
from utils import analyze_page
from langchain_openai import ChatOpenAI
import os
import fitz  # PyMuPDF
import base64

st.set_page_config(page_title="Fence Detector for Engineering Drawings")
st.title("🔍 Fence Detection in Engineering Drawings")

openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Model selection with pricing info and session state persistence
model_options = {
    "gpt-3.5-turbo (cheap & fast - ~$0.001/page)": "gpt-3.5-turbo",
    "gpt-4 (accurate & expensive - ~$0.03/page)": "gpt-4"
}

if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(model_options.values())[0]  # default to gpt-3.5

selected_label = st.radio("Select language model for text analysis:", list(model_options.keys()), index=list(model_options.values()).index(st.session_state.selected_model))
st.session_state.selected_model = model_options[selected_label]

llm_text = ChatOpenAI(model=st.session_state.selected_model, temperature=0, openai_api_key=openai_key)

# Vision toggle and conditional loading
process_images = st.toggle("🖼️ Enable image (vision) analysis (uses GPT-4-turbo, ~$0.01/image)", value=True)
llm_vision = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key) if process_images else None

uploaded_pdf = st.file_uploader("Upload Engineering PDF", type=["pdf"])

FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail']

if uploaded_pdf:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

    fence_pages = []
    non_fence_pages = []

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div style='background-color:#d4edda; padding:10px; border-radius:8px;'><h4 style='color:#155724;'>✅ Fence-Related Pages</h4></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style='background-color:#f8d7da; padding:10px; border-radius:8px;'><h4 style='color:#721c24;'>❌ Non-Fence Pages</h4></div>""", unsafe_allow_html=True)

    for i in range(len(doc)):
        page_obj = doc.load_page(i)
        text = page_obj.get_text()
        pix = page_obj.get_pixmap(dpi=100)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        page = {
            "page_number": i + 1,
            "text": text,
            "image_bytes": img_bytes,
            "image_b64": img_b64
        }

        with st.spinner(f"Analyzing Page {page['page_number']}..."):
            result = analyze_page(page, llm_text, llm_vision, FENCE_KEYWORDS)

        if result['fence_found']:
            fence_pages.append(result)
            with col1.expander(f"Page {result['page_number']}"):
                st.image(result['image'], caption="Page Image")
                st.markdown(f"✅ **Fence Detected!**")
                summary = []
                if result['text_found']: summary.append("📝 Text mentions fences")
                if result['vision_found']: summary.append("🖼️ Drawing shows fences")
                st.markdown("**Summary:** " + ", ".join(summary))
                if result['text_response']: st.markdown(f"**Text Reasoning:** {result['text_response']}")
                if result['vision_response']: st.markdown(f"**Image Reasoning:** {result['vision_response']}")
                if result['text_snippet']: st.markdown(f"**Text Snippet:** `{result['text_snippet']}`")
        else:
            non_fence_pages.append(result)
            with col2.expander(f"Page {result['page_number']}"):
                st.image(result['image'], caption="Page Image")
                st.markdown("❌ **No fence-related content found.**")
                if result['text_response']: st.markdown(f"**Text Reasoning:** {result['text_response']}")
                if result['vision_response']: st.markdown(f"**Image Reasoning:** {result['vision_response']}")
                if result['text_snippet']: st.markdown(f"**Text Snippet:** `{result['text_snippet']}`")

    st.markdown("""
    ---
    ### 📊 Summary
    - ✅ Fence-Related Pages: **{0}**
    - ❌ Non-Fence Pages: **{1}**
    """.format(len(fence_pages), len(non_fence_pages)))
