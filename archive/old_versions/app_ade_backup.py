import streamlit as st
import os
import toml
import json
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF

# Import our consolidated ADE utilities
import utils_ade as ade

# Optional: LLM client
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="ADE Fence Detector (v2)", layout="wide")

# ==============================================================================
# 1. Configuration & Sidebar
# ==============================================================================

st.title("🔍 ADE Fence Detector (Refactored)")

with st.sidebar:
    st.header("🔑 Configuration")

    # Load secrets if available
    secrets = {}
    if os.path.exists(".streamlit/secrets.toml"):
        secrets = toml.load(".streamlit/secrets.toml")

    # 1. OpenAI Key
    openai_key = st.text_input("OpenAI API Key", value=secrets.get("OPENAI_API_KEY", ""), type="password")

    # 2. LandingAI (ADE) Key
    ade_key = st.text_input("LandingAI API Key", value=secrets.get("LANDINGAI_API_KEY", ""), type="password")

    # 3. Google Cloud Config (JSON)
    st.markdown("---")
    st.markdown("**Google Cloud OCR** (Optional)")
    google_cloud_config = None
    if "google_cloud" in secrets and "gcp_service_account" in secrets:
        google_cloud_config = {
            "project_number": secrets["google_cloud"]["project_number"],
            "location": secrets["google_cloud"]["location"],
            "processor_id": secrets["google_cloud"]["processor_id"],
            "service_account_info": dict(secrets["gcp_service_account"])
        }
        st.success("✅ Google Cloud Config Loaded")
    else:
        st.warning("⚠️ Google Cloud Config Missing (OCR disabled)")

    st.markdown("---")
    fence_keywords_str = st.text_area(
        "Fence Keywords",
        value="fence, gate, barrier, guardrail, post, mesh, panel, chain link, masonry, cmu",
        height=100
    )
    FENCE_KEYWORDS = [k.strip() for k in fence_keywords_str.split(",") if k.strip()]

    # Debug mode (disabled in UI)
    DEBUG_MODE = False

# ==============================================================================
# 2. Main Logic
# ==============================================================================

uploaded_file = st.file_uploader("Upload PDF Drawing", type=["pdf"])

if uploaded_file and openai_key and ade_key:
    # Save file bytes to session state to avoid re-reading
    if "file_bytes" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name
        st.session_state.ade_result = None  # Reset analysis
        st.session_state.page_results = {}

    file_bytes = st.session_state.file_bytes
    st.info(f"Loaded: {st.session_state.file_name} ({len(file_bytes)/1024:.1f} KB)")

    # Initialize LLM
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0)

    # Step 1: ADE Analysis (Full Document)
    if st.button("🚀 Run Analysis"):
        with st.status("Running ADE Analysis...", expanded=True) as status:

            # 1. Call ADE
            if not st.session_state.ade_result:
                st.write("Sending to LandingAI ADE...")
                print("[APP] Sending document to LandingAI ADE...")
                ade_response = ade.ade_parse_document(file_bytes, ade_key)

                if not ade_response["success"]:
                    st.error(f"ADE Failed: {ade_response['error']}")
                    print(f"[APP] ADE Failed: {ade_response['error']}")
                    st.stop()

                st.session_state.ade_result = ade_response
                st.write("✅ ADE Analysis Complete!")

            ade_data = st.session_state.ade_result["data"]
            total_pages = ade_data["total_pages"]

            # 2. Process Pages
            progress_bar = st.progress(0)

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                print(f"=== [APP] Starting Page {page_num} ===")
                st.write(f"Processing Page {page_num} / {total_pages}...")

                # A. Get Page Image & Dimensions
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc[page_idx]
                w, h = page.rect.width, page.rect.height
                print(f" [APP] Page Dimensions: {w:.2f} x {h:.2f} points")
                page_img_bytes = page.get_pixmap(dpi=150).tobytes("png")

                # B. Align ADE Chunks
                chunks = ade.align_ade_chunks_to_page(st.session_state.ade_result, page_idx, w, h)
                legend_chunks, figure_chunks = ade.segment_chunks(chunks)

                # C. Get Lines (Native & OCR) for precise Legend Matching
                print(" [APP] Extracting PDF Native Lines...")
                pdf_lines = ade.get_native_pdf_lines(page)

                ocr_lines = []
                if google_cloud_config:
                    print(" [APP] Preparing Google OCR request...")
                    single_page_pdf = ade.create_single_page_pdf(file_bytes, page_idx)
                    ocr_lines = ade.run_google_ocr_blocks(single_page_pdf, google_cloud_config, w, h)

                # --- DEBUG VISUALIZATION START ---
                if DEBUG_MODE:
                    print(" [APP] Generating Debug Layer Image...")
                    # Render the page image (re-using page_img_bytes)
                    debug_bytes = ade.debug_visualize_coordinates(
                        page_img_bytes,
                        legend_chunks,  # Visualize Legend Chunks
                        pdf_lines,
                        ocr_lines,
                        w, h
                    )
                    st.image(debug_bytes, caption=f"DEBUG: Layers Page {page_num}", use_container_width=True)
                # --- DEBUG VISUALIZATION END ---

                # D. Extract Definitions (Legend Items)
                print(" [APP] Matching Legend Entries...")
                definitions = ade.extract_legend_entries(
                    legend_chunks=legend_chunks,
                    pdf_lines=pdf_lines,
                    ocr_lines=ocr_lines,
                    fence_keywords=FENCE_KEYWORDS,
                    llm=llm
                )

                # E. Find Instances (For figures, we still need simple tokens)
                print(" [APP] Finding Instances in Figures...")
                native_words = page.get_text("words")
                all_figure_tokens = [{"text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3]} for w in native_words]

                instances = ade.find_instances_in_figures(
                    definitions, figure_chunks, all_figure_tokens
                )

                # F. Visualize
                print(" [APP] Generating Final Highlight Image...")
                viz_bytes = ade.highlight_page_image(
                    page_img_bytes,
                    definitions,
                    instances,
                    pdf_width=w,  # <--- Pass PDF Width
                    pdf_height=h  # <--- Pass PDF Height
                )

                # Store Results
                st.session_state.page_results[page_num] = {
                    "definitions": definitions,
                    "instances": instances,
                    "image": viz_bytes,
                    "chunk_count": len(chunks),
                    "legend_count": len(legend_chunks),
                    "figure_count": len(figure_chunks)
                }

                progress_bar.progress((page_idx + 1) / total_pages)
                print(f"=== [APP] Finished Page {page_num} ===")

            status.update(label="Analysis Complete!", state="complete", expanded=False)

# ==========================================================================
# 3. Display Results
# ==========================================================================

if st.session_state.get("page_results"):
    st.header("📊 Results")

    tabs = st.tabs([f"Page {p}" for p in st.session_state.page_results.keys()])

    for i, page_num in enumerate(st.session_state.page_results.keys()):
        res = st.session_state.page_results[page_num]
        with tabs[i]:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(res["image"], use_container_width=True, caption=f"Page {page_num} Analysis")

            with col2:
                st.metric("ADE Chunks", res["chunk_count"])
                st.metric("Legend Regions", res["legend_count"])
                st.metric("Figure Regions", res["figure_count"])

            st.subheader("Found Items")

            if res["definitions"]:
                st.markdown("### 🟢 Definitions (Legend)")
                df_def = pd.DataFrame(res["definitions"])

                # FILTER: Only show the main description rows, hide the "Indicator Code" helper rows
                # (These helper rows exist only to draw the extra green box on the number)
                if "description" in df_def.columns:
                    df_display = df_def[df_def["description"] != "Indicator Code"]
                    st.dataframe(df_display[["indicator", "keyword", "description"]], hide_index=True)
                else:
                    st.dataframe(df_def, hide_index=True)
            else:
                st.info("No Legend Items found.")

            if res["instances"]:
                st.markdown("### 🟣 Instances (Drawings)")
                df_inst = pd.DataFrame(res["instances"])
                st.dataframe(df_inst[["indicator"]], hide_index=True)
            else:
                st.info("No Instances found in drawings.")

elif not openai_key or not ade_key:
    st.warning("Please provide OpenAI and LandingAI API keys in the sidebar to continue.")
