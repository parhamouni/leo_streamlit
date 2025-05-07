import streamlit as st
from utils import analyze_page
from langchain_openai import ChatOpenAI
import os, fitz, base64, pandas as pd
from PIL import Image
import io

# Config
st.set_page_config(page_title="Fence Detector", layout="wide")
FENCE_KEYWORDS = ['fence', 'fencing', 'gate', 'barrier', 'guardrail',
                  'chain link', 'enclosure']

# Styles
st.markdown("""
<style>
.main-header { font-size:28px; font-weight:bold; margin:20px 0; }
.fence {background:#d4edda;color:#155724;padding:8px;border-radius:6px;}
.non {background:#f8d7da;color:#721c24;padding:8px;border-radius:6px;}
</style>""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🔍 Fence Indicator Detector</div>", unsafe_allow_html=True)
st.markdown("> Upload an engineering PDF; OCR+AI marks fence-related text.", unsafe_allow_html=True)

# Secrets
openai_key = st.secrets["OPENAI_API_KEY"]
ocr_key    = st.secrets["OCR_API_KEY"]
models = {}
if openai_key:
    try:
        models = {
            "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key),
            "gpt-4":  ChatOpenAI(model="gpt-4",  temperature=0, openai_api_key=openai_key),
        }
    except Exception as e:
        st.error("LLM init error: "+str(e))
else:
    st.error("🔑 Set OPENAI_API_KEY in secrets")

# UI
sel = st.radio("Model for vision analysis:",
               ["None","gpt-4o","gpt-4"], index=1)
llm_vision = models.get(sel) if sel!="None" else None

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
if uploaded_pdf and openai_key and ocr_key:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    fence_pages, non_pages = [], []
    c1, c2 = st.columns(2)
    c1.markdown("<div class='fence'><h4>✅ Fence Pages</h4></div>", unsafe_allow_html=True)
    c2.markdown("<div class='non'><h4>❌ Non-Fence Pages</h4></div>", unsafe_allow_html=True)
    progress = st.progress(0)
    status = st.empty()

    for i in range(len(doc)):
        status.text(f"Page {i+1}/{len(doc)}")
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        res = analyze_page(
            {"page_number":i+1, "image_bytes":img_bytes, "ocr_api_key":ocr_key},
            llm_vision, FENCE_KEYWORDS
        )
        (fence_pages if res["fence_found"] else non_pages).append(res)
        progress.progress((i+1)/len(doc))
    status.empty(); progress.empty()

    # Display fence pages
    for res in fence_pages:
        with c1.expander(f"Page {res['page_number']}"):
            st.image(res["highlighted_image"], use_column_width=True)
            st.markdown("**Text matches:**")
            for m in res["text_references"]:
                st.write(f"- \"{m['text']}\" at {m['bbox']}")
            if res["vision_found"]:
                st.markdown("**AI Analysis:**")
                st.write(res["llm_analysis"])
            b64 = base64.b64encode(res["highlighted_image"]).decode()
            st.markdown(f'<a href="data:image/png;base64,{b64}" download="page_{res["page_number"]}.png">⬇️ Download</a>', unsafe_allow_html=True)

    # Non-fence pages
    for res in non_pages:
        with c2.expander(f"Page {res['page_number']}"):
            st.image(res["image"], width=250)
            st.write("No fence text detected.")

    # Summary
    st.markdown("---")
    s1, s2 = st.columns(2)
    s1.metric("Pages with fences", len(fence_pages))
    s2.metric("Pages without", len(non_pages))

else:
    st.info("Upload a PDF and set both API keys in secrets to begin.")
