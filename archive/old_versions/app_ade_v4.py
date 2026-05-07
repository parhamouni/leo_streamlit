"""
ADE Fence Detector v4 - Three-Level Extraction Approach

Based on three_level_extraction_merge.ipynb:
1. Extract from ADE (layout-aware chunks)
2. Extract from PDF text layer (native text)
3. Extract from OCR (Google Document AI)
4. Merge PDF + OCR tokens
5. Assign each token to an ADE chunk
6. Filter for fence-related content
7. Highlight using token bounding boxes
"""

import streamlit as st
import os
import re
import toml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image, ImageDraw

from langchain_openai import ChatOpenAI

# Import the hybrid extractor
from hybrid_page_extractor import (
    extract_page_sources,
    normalize_document_ai_config,
    extract_pdf_text_words,
    run_document_ai_ocr,
    normalize_text_for_matching,
)

st.set_page_config(page_title="ADE Fence Detector (v4)", layout="wide")

# ==============================================================================
# Helper Functions
# ==============================================================================

def merge_pdf_ocr_tokens(
    pdf_words: List[Dict],
    ocr_tokens: List[Dict],
    overlap_threshold: float = 0.2,
    text_similarity_threshold: float = 0.5,
) -> List[Dict]:
    """Merge PDF and OCR tokens by finding overlaps."""
    merged = []
    ocr_used = set()
    
    for pdf_word in pdf_words:
        pdf_bbox = {k: float(pdf_word.get(k, 0)) for k in ["x0", "y0", "x1", "y1"]}
        pdf_text = pdf_word.get("text", "")
        
        # Find matching OCR tokens
        matching_ocr = []
        for idx, ocr_token in enumerate(ocr_tokens):
            if idx in ocr_used:
                continue
            
            ocr_bbox = {k: float(ocr_token.get(k, 0)) for k in ["x0", "y0", "x1", "y1"]}
            
            # Calculate overlap
            x_left = max(pdf_bbox["x0"], ocr_bbox["x0"])
            y_top = max(pdf_bbox["y0"], ocr_bbox["y0"])
            x_right = min(pdf_bbox["x1"], ocr_bbox["x1"])
            y_bottom = min(pdf_bbox["y1"], ocr_bbox["y1"])
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                pdf_area = (pdf_bbox["x1"] - pdf_bbox["x0"]) * (pdf_bbox["y1"] - pdf_bbox["y0"])
                ocr_area = (ocr_bbox["x1"] - ocr_bbox["x0"]) * (ocr_bbox["y1"] - ocr_bbox["y0"])
                smaller_area = min(pdf_area, ocr_area) if min(pdf_area, ocr_area) > 0 else 1
                overlap_ratio = intersection / smaller_area
                
                if overlap_ratio >= overlap_threshold:
                    # Check text similarity
                    norm1 = normalize_text_for_matching(pdf_text)
                    norm2 = normalize_text_for_matching(ocr_token.get("text", ""))
                    if norm1 == norm2 or (norm1 and norm2 and (norm1 in norm2 or norm2 in norm1)):
                        matching_ocr.append((idx, ocr_token))
        
        if matching_ocr:
            # Merge bounding boxes
            all_x0 = [pdf_bbox["x0"]] + [ocr_tokens[idx]["x0"] for idx, _ in matching_ocr]
            all_y0 = [pdf_bbox["y0"]] + [ocr_tokens[idx]["y0"] for idx, _ in matching_ocr]
            all_x1 = [pdf_bbox["x1"]] + [ocr_tokens[idx]["x1"] for idx, _ in matching_ocr]
            all_y1 = [pdf_bbox["y1"]] + [ocr_tokens[idx]["y1"] for idx, _ in matching_ocr]
            
            for idx, _ in matching_ocr:
                ocr_used.add(idx)
            
            merged.append({
                "text": pdf_text,
                "x0": min(all_x0), "y0": min(all_y0),
                "x1": max(all_x1), "y1": max(all_y1),
                "source": "pdf+ocr"
            })
        else:
            merged.append({
                "text": pdf_text,
                "x0": pdf_bbox["x0"], "y0": pdf_bbox["y0"],
                "x1": pdf_bbox["x1"], "y1": pdf_bbox["y1"],
                "source": "pdf_only"
            })
    
    # Add remaining OCR tokens
    for idx, ocr_token in enumerate(ocr_tokens):
        if idx not in ocr_used:
            merged.append({
                "text": ocr_token.get("text", ""),
                "x0": float(ocr_token.get("x0", 0)),
                "y0": float(ocr_token.get("y0", 0)),
                "x1": float(ocr_token.get("x1", 0)),
                "y1": float(ocr_token.get("y1", 0)),
                "source": "ocr_only"
            })
    
    return merged


def assign_tokens_to_chunks(tokens: List[Dict], ade_chunks: List[Dict]) -> List[Dict]:
    """Assign each token to the ADE chunk that contains it."""
    for token in tokens:
        token_cx = (token["x0"] + token["x1"]) / 2
        token_cy = (token["y0"] + token["y1"]) / 2
        
        best_chunk = None
        best_score = -1
        
        for chunk in ade_chunks:
            # Check if token center is inside chunk
            if (chunk["x0"] <= token_cx <= chunk["x1"] and 
                chunk["y0"] <= token_cy <= chunk["y1"]):
                # Calculate containment score
                chunk_area = (chunk["x1"] - chunk["x0"]) * (chunk["y1"] - chunk["y0"])
                score = 1.0 / (chunk_area + 1)  # Prefer smaller chunks
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
        
        if best_chunk:
            token["ade_type"] = best_chunk.get("type", "unknown")
            token["ade_id"] = best_chunk.get("id", "")
        else:
            token["ade_type"] = None
            token["ade_id"] = None
    
    return tokens


def filter_fence_tokens(tokens: List[Dict], fence_keywords: List[str]) -> List[Dict]:
    """Filter tokens that match fence keywords."""
    fence_tokens = []
    keywords_lower = [kw.lower() for kw in fence_keywords]
    
    for token in tokens:
        text_lower = token.get("text", "").lower()
        for kw in keywords_lower:
            if kw in text_lower:
                token["matched_keyword"] = kw
                fence_tokens.append(token)
                break
    
    return fence_tokens


def highlight_tokens(page_image_bytes: bytes, tokens: List[Dict], 
                    pdf_width: float, pdf_height: float) -> bytes:
    """Draw highlights on the page image for the given tokens."""
    img = Image.open(BytesIO(page_image_bytes))
    draw = ImageDraw.Draw(img, "RGBA")
    
    img_w, img_h = img.size
    scale_x = img_w / pdf_width
    scale_y = img_h / pdf_height
    
    for token in tokens:
        x0 = token["x0"] * scale_x
        y0 = token["y0"] * scale_y
        x1 = token["x1"] * scale_x
        y1 = token["y1"] * scale_y
        
        # Green for legend/table, purple for figure
        if token.get("ade_type") in ["table", "text"]:
            color = (0, 255, 0, 100)  # Green
            outline = (0, 255, 0, 255)
        else:
            color = (255, 0, 255, 100)  # Purple
            outline = (255, 0, 255, 255)
        
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=2)
    
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ==============================================================================
# Sidebar Configuration
# ==============================================================================

st.title("🔍 ADE Fence Detector (v4 - Three-Level)")

with st.sidebar:
    st.header("🔑 Configuration")
    
    secrets = {}
    if os.path.exists(".streamlit/secrets.toml"):
        secrets = toml.load(".streamlit/secrets.toml")
    
    openai_key = st.text_input("OpenAI API Key", value=secrets.get("OPENAI_API_KEY", ""), type="password")
    ade_key = st.text_input("LandingAI API Key", value=secrets.get("LANDINGAI_API_KEY", ""), type="password")
    
    st.markdown("---")
    google_cloud_config = None
    if "google_cloud" in secrets and "gcp_service_account" in secrets:
        google_cloud_config = {
            "project_number": secrets["google_cloud"]["project_number"],
            "location": secrets["google_cloud"]["location"],
            "processor_id": secrets["google_cloud"]["processor_id"],
            "service_account_info": dict(secrets["gcp_service_account"])
        }
        st.success("✅ Google Cloud OCR Loaded")
    else:
        st.warning("⚠️ No OCR config")
    
    st.markdown("---")
    fence_keywords_str = st.text_area(
        "Fence Keywords",
        value="fence, gate, barrier, guardrail, wall, cmu, screen, bollard, chain link, mesh",
        height=80
    )
    FENCE_KEYWORDS = [k.strip() for k in fence_keywords_str.split(",") if k.strip()]
    
    # Debug mode (disabled in UI)
    DEBUG_MODE = False

# ==============================================================================
# Main Processing
# ==============================================================================

uploaded_file = st.file_uploader("Upload PDF Drawing", type=["pdf"])

if uploaded_file and ade_key:
    if "file_bytes" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name
        st.session_state.results = {}
    
    file_bytes = st.session_state.file_bytes
    st.info(f"Loaded: {uploaded_file.name} ({len(file_bytes)/1024:.1f} KB)")
    
    if st.button("🚀 Analyze"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # Normalize Google Cloud config if available
        doc_ai_config = None
        if google_cloud_config:
            doc_ai_config = normalize_document_ai_config(google_cloud_config)
        
        progress = st.progress(0)
        
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            st.write(f"Processing page {page_num}/{total_pages}...")
            
            page = doc[page_idx]
            w, h = page.rect.width, page.rect.height
            page_img = page.get_pixmap(dpi=150).tobytes("png")
            
            # Step 1: Extract from all three levels
            try:
                sources = extract_page_sources(
                    pdf=file_bytes,
                    page_number=page_num,
                    ade_api_key=ade_key,
                    doc_ai_config=doc_ai_config
                )
            except Exception as e:
                st.error(f"Page {page_num} extraction failed: {e}")
                continue
            
            pdf_words = sources.get("pdf_words", [])
            ocr_tokens = sources.get("ocr_tokens", [])
            ade_chunks = sources.get("ade_chunks", [])
            
            if DEBUG_MODE:
                st.write(f"  PDF words: {len(pdf_words)}, OCR tokens: {len(ocr_tokens)}, ADE chunks: {len(ade_chunks)}")
            
            # Step 2: Merge PDF + OCR
            merged_tokens = merge_pdf_ocr_tokens(pdf_words, ocr_tokens)
            
            # Step 3: Assign to ADE chunks
            merged_tokens = assign_tokens_to_chunks(merged_tokens, ade_chunks)
            
            # Step 4: Filter for fence keywords
            fence_tokens = filter_fence_tokens(merged_tokens, FENCE_KEYWORDS)
            
            if DEBUG_MODE:
                st.write(f"  Merged tokens: {len(merged_tokens)}, Fence matches: {len(fence_tokens)}")
            
            # Step 5: Highlight
            highlighted_img = highlight_tokens(page_img, fence_tokens, w, h)
            
            st.session_state.results[page_num] = {
                "image": highlighted_img,
                "tokens": fence_tokens,
                "total_merged": len(merged_tokens),
                "ade_chunks": len(ade_chunks)
            }
            
            progress.progress((page_idx + 1) / total_pages)
        
        st.success("✅ Analysis complete!")

# ==============================================================================
# Display Results
# ==============================================================================

if st.session_state.get("results"):
    st.header("📊 Results")
    
    tabs = st.tabs([f"Page {p}" for p in st.session_state.results.keys()])
    
    for i, page_num in enumerate(st.session_state.results.keys()):
        res = st.session_state.results[page_num]
        with tabs[i]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(res["image"], use_container_width=True)
            
            with col2:
                st.metric("ADE Chunks", res["ade_chunks"])
                st.metric("Merged Tokens", res["total_merged"])
                st.metric("Fence Matches", len(res["tokens"]))
                
                if res["tokens"]:
                    st.markdown("### Found Items")
                    df = pd.DataFrame(res["tokens"])[["text", "matched_keyword", "ade_type", "source"]]
                    st.dataframe(df, hide_index=True)
