
import streamlit as st
from utils import extract_pdf_data, analyze_page
from langchain_openai import ChatOpenAI
import os

st.set_page_config(page_title="Fence Detector for Engineering Drawings")
st.title("🔍 Fence Detection in Engineering Drawings")

openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

llm_text = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_key)
llm_vision = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=openai_key)

uploaded_pdf = st.file_uploader("Upload Engineering PDF", type=["pdf"])

process_images = st.toggle("🖼️ Enable image (vision) analysis", value=True)

if uploaded_pdf:
    with st.spinner("Extracting pages..."):
        pages = extract_pdf_data(uploaded_pdf)

    st.success("Extraction complete. Beginning analysis...")

    fence_pages = []
    non_fence_pages = []

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div style='background-color:#d4edda; padding:10px; border-radius:8px;'><h4 style='color:#155724;'>✅ Fence-Related Pages</h4></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div style='background-color:#f8d7da; padding:10px; border-radius:8px;'><h4 style='color:#721c24;'>❌ Non-Fence Pages</h4></div>""", unsafe_allow_html=True)

    for page in pages:
        with st.spinner(f"Analyzing Page {page['page_number']}..."):
            result = analyze_page(page, llm_text, llm_vision if process_images else None)

            if result['fence_found']:
                fence_pages.append(result)
                with col1.expander(f"Page {result['page_number']}"):
                    st.image(result['image'], caption="Page Image")
                    st.markdown(f"✅ **Fence Detected!**")
                    summary = []
                    if result['text_found']:
                        summary.append("📝 Text mentions fences")
                    if result['vision_found']:
                        summary.append("🖼️ Drawing shows fences")
                    st.markdown("**Summary:** " + ", ".join(summary))
                    if result['text_response']:
                        st.markdown(f"**Text Reasoning:** {result['text_response']}")
                    if result['vision_response']:
                        st.markdown(f"**Image Reasoning:** {result['vision_response']}")
                    if result['text_snippet']:
                        st.markdown(f"**Text Snippet:** `{result['text_snippet']}`")
            else:
                non_fence_pages.append(result)
                with col2.expander(f"Page {result['page_number']}"):
                    st.image(result['image'], caption="Page Image")
                    st.markdown("❌ **No fence-related content found.**")
                    if result['text_response']:
                        st.markdown(f"**Text Reasoning:** {result['text_response']}")
                    if result['vision_response']:
                        st.markdown(f"**Image Reasoning:** {result['vision_response']}")
                    if result['text_snippet']:
                        st.markdown(f"**Text Snippet:** `{result['text_snippet']}`")

    st.markdown("""
    ---
    ### 📊 Summary
    - ✅ Fence-Related Pages: **{0}**
    - ❌ Non-Fence Pages: **{1}**
    """.format(len(fence_pages), len(non_fence_pages)))