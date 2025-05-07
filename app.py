import os, base64, fitz, pandas as pd
import streamlit as st
from utils import analyze_page
from langchain_openai import ChatOpenAI

# Before importing utils, ensure env vars are set from secrets
st.set_page_config(page_title="Fence Detector", layout="wide")
# (Streamlit loads secrets into environment automatically here)
# If not, you can explicitly do:
# os.environ["AZURE_CV_ENDPOINT"] = st.secrets["AZURE_CV_ENDPOINT"]
# os.environ["AZURE_CV_KEY"]      = st.secrets["AZURE_CV_KEY"]

FENCE_KEYWORDS = ['fence','fencing','gate','barrier','guardrail','chain link','enclosure']

# Styling
st.markdown("""
<style>
.main{font-size:28px;font-weight:bold;margin:20px 0;}
.fence{background:#d4edda;color:#155724;padding:8px;border-radius:5px;}
.non{background:#f8d7da;color:#721c24;padding:8px;border-radius:5px;}
</style>""", unsafe_allow_html=True)
st.markdown("<div class='main'>🔍 Fence Indicator Detector</div>", unsafe_allow_html=True)

# API keys
openai_key = st.secrets.get("OPENAI_API_KEY")
if not openai_key:
    st.error("Set OPENAI_API_KEY in .streamlit/secrets.toml")
ocr_endpoint = st.secrets.get("AZURE_CV_ENDPOINT")
ocr_key      = st.secrets.get("AZURE_CV_KEY")
if not (ocr_endpoint and ocr_key):
    st.error("Set AZURE_CV_ENDPOINT & AZURE_CV_KEY in secrets")

# Init models
models = {"None": None}
if openai_key:
    try:
        models.update({
            "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key),
            "gpt-4":  ChatOpenAI(model="gpt-4",  temperature=0, openai_api_key=openai_key),
        })
    except Exception as e:
        st.error("LLM init error: " + str(e))

sel = st.radio("Vision model:", list(models.keys()), index=1)
llm_vision = models[sel]

# PDF upload
uploaded = st.file_uploader("Upload Engineering PDF", type="pdf")
if uploaded and openai_key and ocr_endpoint and ocr_key:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    fence_pages, non_pages = [], []
    c1, c2 = st.columns(2)
    c1.markdown("<div class='fence'><h4>✅ Pages with fences</h4></div>", unsafe_allow_html=True)
    c2.markdown("<div class='non'><h4>❌ Pages without fences</h4></div>", unsafe_allow_html=True)

    progress = st.progress(0)
    status = st.empty()

    for i in range(len(doc)):
        status.text(f"Page {i+1}/{len(doc)}")
        pg = doc.load_page(i)
        pix = pg.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        res = analyze_page(
            {"page_number": i+1, "image_bytes": img_bytes},
            llm_vision, FENCE_KEYWORDS
        )
        if res["fence_found"]:
            fence_pages.append(res)
        else:
            non_pages.append(res)
        progress.progress((i+1)/len(doc))

    status.empty(); progress.empty()

    # Show fence pages
    for res in fence_pages:
        with c1.expander(f"Page {res['page_number']}"):
            st.image(res["highlighted_image"], use_column_width=True)
            st.markdown("**LLM verdict:** " + res["llm_analysis"])
            st.markdown("**Keyword matches:**")
            for m in res["text_references"]:
                st.write(f"- {m['text']} at {m['bbox']}")
            b64 = base64.b64encode(res["highlighted_image"]).decode()
            link = f'<a href="data:image/png;base64,{b64}" download="page_{res["page_number"]}.png">⬇️ Download</a>'
            st.markdown(link, unsafe_allow_html=True)

    # Show non-fence pages
    for res in non_pages:
        with c2.expander(f"Page {res['page_number']}"):
            st.image(res["image"], width=250)
            st.write("LLM said NO fences.")

    # Summary
    st.markdown("---")
    s1, s2 = st.columns(2)
    s1.metric("Pages with fences",    len(fence_pages))
    s2.metric("Pages without fences", len(non_pages))

else:
    st.info("Upload a PDF and configure both API keys in secrets.")
