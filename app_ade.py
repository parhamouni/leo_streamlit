# app_ade_v2.py - ADE Fence Detector with app.py UI
import streamlit as st
import os
import toml
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import base64
from pathlib import Path
from io import BytesIO
import gc
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

# Import our consolidated ADE utilities
import utils_ade as ade

# Interactive image click for measurement
from streamlit_image_coordinates import streamlit_image_coordinates

# Vector measurement utilities
from utils_vector import (
    measure_lines_in_selection,
    measure_at_click_point,
    infer_scale_from_page,
    extract_vector_lines,
    verify_scale_with_bar
)

# Optional: LLM client
from langchain_openai import ChatOpenAI

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)  # Green for definitions
HIGHLIGHT_COLOR_INSTANCE = (0.9, 0, 0.9)  # Purple for instances
HIGHLIGHT_WIDTH_UI = 2.0
DISPLAY_IMAGE_DPI = 150

st.set_page_config(page_title="ADE Fence Detector", layout="wide")
st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>🔍 ADE Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)


# ==============================================================================
# Session Management (matching app.py)
# ==============================================================================

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def initialize_session_state(session_id_val):
    print(f"SESSION {session_id_val} LOG: Initializing/checking session state.")
    default_state = {
        'session_id': session_id_val,
        'fence_pages': [],
        'non_fence_pages': [],
        'total_pages_processed_count': 0,
        'doc_total_pages': 0,
        'processing_complete': False,
        'analysis_halted_due_to_error': False,
        'fence_keywords_app': [
            'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
            'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
            'adot specifications', 'mag specifications', 'rail', 'railing',
            'bollards', 'handrails', 'wall', 'cmu',
            'operator', 'davis', 'bacon', 'davis-bacon', 'davis – bacon',
            'buy america', 'american', 'dug out',
        ],
        'run_analysis_triggered': False,
        'uploaded_pdf_name': None,
        'original_pdf_bytes': None,
        'current_pdf_hash': None,
        'highlighted_pdf_bytes_for_download': None,
        'last_uploaded_file_id': None,
        'selected_model_for_analysis': "gpt-5.1",
        # Unified measurement storage
        'unified_measurements': {},  # {page_key: {'auto_lines': [...], 'manual_lines': [...], 'drawn_lines': [...]}}
        'per_page_scale_info': {},
        'page_categories': {},
        'active_category_per_page': {},
        'element_details': {},  # {element_name: {height, post_spacing, material, ...}}
        'fence_page_texts': {},  # {page_number: full_text} for cross-page detail extraction
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else \
                                    dict(value) if isinstance(value, dict) else \
                                    value
        elif key == 'session_id' and st.session_state.session_id != session_id_val:
            st.session_state.session_id = session_id_val


current_session_id = get_session_id()
initialize_session_state(current_session_id)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_image_download_link_html(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'


def generate_page_images(page_idx, pdf_bytes, definitions, instances, pdf_width, pdf_height):
    """Generate original and highlighted images for a page."""
    try:
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
            page = doc.load_page(page_idx)
            
            # Original image
            pix_orig = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI)
            original_bytes = pix_orig.tobytes("png")
            
            # Highlighted image
            highlighted_bytes = ade.highlight_page_image(
                original_bytes,
                definitions,
                instances,
                pdf_width,
                pdf_height
            )
            
            return original_bytes, highlighted_bytes
    except Exception as e:
        print(f"SESSION {current_session_id} ERROR: Image generation failed: {e}")
        return None, None


@st.cache_data(show_spinner=False, max_entries=5)
def get_page_image_on_demand(_pdf_bytes_hash, pdf_bytes, page_idx, definitions, instances, keyword_matches,
                              pdf_width, pdf_height, highlight, measurement_lines=None):
    """Regenerate page images on demand instead of storing in session_state.
    Uses st.cache_data with max_entries to bound memory usage."""
    try:
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI)
            original_bytes = pix.tobytes("png")
            del pix
            
            highlighted_bytes = None
            if highlight:
                if definitions or instances:
                    highlighted_bytes = ade.highlight_page_image(
                        original_bytes, definitions, instances, pdf_width, pdf_height
                    )
                elif keyword_matches:
                    highlighted_bytes = ade.highlight_keyword_matches(
                        original_bytes, keyword_matches, pdf_width, pdf_height
                    )
            
            return original_bytes, highlighted_bytes
    except Exception as e:
        print(f"ERROR: On-demand image generation failed for page {page_idx}: {e}")
        return None, None


def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base, session_id):
    """Generate a combined PDF with only fence-related pages highlighted."""
    print(f"SESSION {session_id} LOG: Generating combined highlighted PDF.")
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No data for PDF."
    
    output_doc = fitz.open()
    input_doc = None
    
    try:
        input_doc = fitz.open(stream=BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        print(f"SESSION {session_id} ERROR: Opening original PDF for combined: {e}")
        if output_doc:
            output_doc.close()
        return None, f"Error opening original PDF: {e}"
    
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        if page_idx is None:
            continue
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            
            # Get page rotation and MediaBox dimensions for coordinate transform
            # Coordinates in definitions/instances are in DISPLAY space (after rotation)
            # but draw_rect expects MediaBox space, so we need to reverse the transform
            rotation = page_out.rotation
            mediabox_w = page_out.mediabox.width
            mediabox_h = page_out.mediabox.height
            
            def reverse_rotation_transform(x0, y0, x1, y1):
                """Transform display coords back to MediaBox coords for PDF annotation."""
                if rotation == 0:
                    return x0, y0, x1, y1
                elif rotation == 90:
                    # Display->MediaBox: (x,y) -> (y, mediabox_h - x)
                    return y0, mediabox_h - x1, y1, mediabox_h - x0
                elif rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                elif rotation == 270:
                    # Display->MediaBox: (x,y) -> (mediabox_w - y, x)
                    return mediabox_w - y1, x0, mediabox_w - y0, x1
                return x0, y0, x1, y1
            
            # Draw definition boxes (green)
            definitions = res_data.get('definitions', [])
            for d in definitions:
                mx0, my0, mx1, my1 = reverse_rotation_transform(d['x0'], d['y0'], d['x1'], d['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)
            
            # Draw instance boxes (purple)
            instances = res_data.get('instances', [])
            for inst in instances:
                mx0, my0, mx1, my1 = reverse_rotation_transform(inst['x0'], inst['y0'], inst['x1'], inst['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)
            
            # Draw keyword match boxes (orange) - for fallback detection
            keyword_matches = res_data.get('keyword_matches', [])
            for kw in keyword_matches:
                if all(k in kw for k in ['x0', 'y0', 'x1', 'y1']):
                    mx0, my0, mx1, my1 = reverse_rotation_transform(kw['x0'], kw['y0'], kw['x1'], kw['y1'])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
                    
        except Exception as e_pi:
            print(f"SESSION {session_id} Err process pg {page_idx} for PDF: {e_pi}")
    
    pdf_bytes, fname = None, "error.pdf"
    if len(output_doc) > 0:
        try:
            pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
            base, ext = os.path.splitext(uploaded_pdf_name_base)
            fname = f"{base}_fence_highlights{ext}"
        except Exception as e_s:
            print(f"SESSION {session_id} Err PDF tobytes: {e_s}")
            fname = f"err_save_{uploaded_pdf_name_base}.pdf"
    
    if input_doc:
        input_doc.close()
    if output_doc:
        output_doc.close()
    
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)


def generate_measurement_pdf(original_pdf_bytes, fence_pages_results_list, line_assignments, user_drawn_lines, 
                             page_categories, session_state, min_line_pts, uploaded_pdf_name_base):
    """Generate PDF with measurement lines highlighted by category."""
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No data for PDF."
    
    output_doc = fitz.open()
    input_doc = None
    
    try:
        input_doc = fitz.open(stream=BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        if output_doc:
            output_doc.close()
        return None, f"Error opening original PDF: {e}"
    
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        page_num = res_data.get('page_number')
        page_key = f"page_{page_num}"
        
        if page_idx is None:
            continue
        
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            
            rotation = page_out.rotation
            mediabox_w = page_out.mediabox.width
            mediabox_h = page_out.mediabox.height
            
            def reverse_rotation_transform(x0, y0, x1, y1):
                if rotation == 0:
                    return x0, y0, x1, y1
                elif rotation == 90:
                    return y0, mediabox_h - x1, y1, mediabox_h - x0
                elif rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                elif rotation == 270:
                    return mediabox_w - y1, x0, mediabox_w - y0, x1
                return x0, y0, x1, y1
            
            # Draw definition boxes (green)
            definitions = res_data.get('definitions', [])
            for d in definitions:
                mx0, my0, mx1, my1 = reverse_rotation_transform(d['x0'], d['y0'], d['x1'], d['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)
            
            # Draw instance boxes (purple)
            instances = res_data.get('instances', [])
            for inst in instances:
                mx0, my0, mx1, my1 = reverse_rotation_transform(inst['x0'], inst['y0'], inst['x1'], inst['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)
            
            # Draw keyword match boxes (orange)
            keyword_matches = res_data.get('keyword_matches', [])
            for kw in keyword_matches:
                if all(k in kw for k in ['x0', 'y0', 'x1', 'y1']):
                    mx0, my0, mx1, my1 = reverse_rotation_transform(kw['x0'], kw['y0'], kw['x1'], kw['y1'])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
            
            # Get categories for this page
            categories = page_categories.get(page_key, {})
            
            # Auto-detected lines are now included in line_assignments via coordinate matching,
            # so they'll be drawn with category colors below (no separate cyan pass needed)
            
            # Get lines from session state - try multiple key formats
            lines = []
            for key in session_state.keys():
                if key.startswith(f"lines_{page_num}_"):
                    lines = session_state[key]
                    break
            
            page_assignments = line_assignments.get(page_key, {})
            
            for line_idx, category in page_assignments.items():
                idx = int(line_idx) if isinstance(line_idx, str) else line_idx
                if idx < len(lines):
                    line = lines[idx]
                    cat_info = categories.get(category, {})
                    color_rgb = cat_info.get('color', (0, 255, 0))
                    # Convert 0-255 to 0-1
                    color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)
                    
                    sx, sy = line.start
                    ex, ey = line.end
                    mx0, my0, mx1, my1 = reverse_rotation_transform(sx, sy, ex, ey)
                    
                    page_out.draw_line((mx0, my0), (mx1, my1), color=color, width=3.0, overlay=True)
            
            # Draw user-drawn lines
            user_lines = user_drawn_lines.get(page_key, [])
            for ul in user_lines:
                category = ul.get('category')
                cat_info = categories.get(category, {})
                color_rgb = cat_info.get('color', (0, 255, 0))
                color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)
                
                sx, sy = ul['start']
                ex, ey = ul['end']
                mx0, my0, mx1, my1 = reverse_rotation_transform(sx, sy, ex, ey)
                
                page_out.draw_line((mx0, my0), (mx1, my1), color=color, width=3.0, overlay=True)
                # Draw endpoints
                page_out.draw_circle((mx0, my0), 3, color=color, fill=color, overlay=True)
                page_out.draw_circle((mx1, my1), 3, color=color, fill=color, overlay=True)
                
        except Exception as e:
            print(f"Error processing page {page_idx} for measurement PDF: {e}")
    
    base, ext = os.path.splitext(uploaded_pdf_name_base)
    fname = f"{base}_measurements{ext}"
    
    try:
        pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
    except Exception as e:
        print(f"Error generating PDF bytes: {e}")
        pdf_bytes = None
    
    if input_doc:
        input_doc.close()
    if output_doc:
        output_doc.close()
    
    return (pdf_bytes, fname)


def _lookup_element_details(category: str, element_details: dict) -> dict:
    """Look up element details for a category name, trying exact and fuzzy matching."""
    if not element_details:
        return {}
    # Exact match
    if category in element_details:
        return element_details[category]
    # Case-insensitive match
    cat_lower = category.lower()
    for name, details in element_details.items():
        if name.lower() == cat_lower:
            return details
    # Partial match (category contains element name or vice versa)
    for name, details in element_details.items():
        if name.lower() in cat_lower or cat_lower in name.lower():
            return details
    return {}


def generate_measurement_spreadsheet(fence_pages, line_assignments, user_drawn_lines, page_categories, 
                                     session_state, per_page_scale_info, min_line_pts):
    """Generate Excel workbook (bytes) with multiple sheets: Measurements, Summary, Element Specifications."""
    import io
    rows = []
    element_details = session_state.get('element_details', {})
    
    # Debug: print what we're working with
    print(f"CSV Debug: line_assignments = {line_assignments}")
    print(f"CSV Debug: user_drawn_lines = {user_drawn_lines}")
    print(f"CSV Debug: element_details keys = {list(element_details.keys())}")
    
    # Detail columns to include
    DETAIL_COLS = ['Height', 'Post Type', 'Post Spacing', 'Material', 'Gauge', 'Mesh Size', 'Detail Page', 'Full Details']
    
    def _build_row(page_num, category, row_type, length_feet, length_pts, page_scale):
        """Build a row dict with measurement + detail columns."""
        row = {
            'Page': page_num,
            'Category': category,
            'Type': row_type,
            'Length (ft)': round(length_feet, 2),
            'Length (pts)': round(length_pts, 2),
            'Scale': page_scale,
        }
        # Look up details for this category
        details = _lookup_element_details(category, element_details)
        row['Height'] = details.get('height', '')
        row['Post Type'] = details.get('post_type', '')
        row['Post Spacing'] = details.get('post_spacing', '')
        row['Material'] = details.get('material', '')
        row['Gauge'] = details.get('gauge', '')
        row['Mesh Size'] = details.get('mesh_size', '')
        row['Detail Page'] = details.get('detail_page', '')
        row['Full Details'] = details.get('full_details', '')
        return row
    
    for page_data in fence_pages:
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        
        # Get scale
        scale_info = per_page_scale_info.get(page_key, {})
        page_scale = scale_info.get('verified_scale') or scale_info.get('text_scale') or 360.0
        
        # Get lines from session state - try multiple key formats
        lines = []
        lines_key_found = None
        for key in list(session_state.keys()):
            if key.startswith(f"lines_{page_num}_"):
                lines = session_state[key]
                lines_key_found = key
                break
        
        print(f"CSV Debug: page {page_num}, lines_key={lines_key_found}, num_lines={len(lines)}")
        
        categories = page_categories.get(page_key, {})
        
        # Selected lines (includes auto-matched + manually selected)
        auto_matched = session_state.get(f"auto_matched_indices_{page_key}", set())
        page_assignments = line_assignments.get(page_key, {})
        for line_idx, category in page_assignments.items():
            idx = int(line_idx) if isinstance(line_idx, str) else line_idx
            if idx < len(lines):
                line = lines[idx]
                length_inches = line.length_pts / 72.0
                length_feet = (length_inches * page_scale) / 12.0
                rows.append(_build_row(
                    page_num, category,
                    'Auto' if idx in auto_matched else 'Selected',
                    length_feet, line.length_pts, page_scale
                ))
        
        # User-drawn lines
        user_lines = user_drawn_lines.get(page_key, [])
        for ul in user_lines:
            rows.append(_build_row(
                page_num, ul.get('category', 'Uncategorized'),
                'Drawn',
                ul.get('length_feet', 0), ul.get('length_pts', 0), page_scale
            ))
    
    # Define all columns (measurement + detail)
    all_columns = ['Page', 'Category', 'Type', 'Length (ft)', 'Length (pts)', 'Scale'] + DETAIL_COLS
    
    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows)
        
        # Add summary rows
        summary_rows = []
        for cat in df['Category'].unique():
            cat_df = df[df['Category'] == cat]
            summary_row = {
                'Page': 'TOTAL',
                'Category': cat,
                'Type': 'Summary',
                'Length (ft)': round(cat_df['Length (ft)'].sum(), 2),
                'Length (pts)': '',
                'Scale': ''
            }
            # Include details in summary row too
            details = _lookup_element_details(cat, element_details)
            summary_row['Height'] = details.get('height', '')
            summary_row['Post Type'] = details.get('post_type', '')
            summary_row['Post Spacing'] = details.get('post_spacing', '')
            summary_row['Material'] = details.get('material', '')
            summary_row['Gauge'] = details.get('gauge', '')
            summary_row['Mesh Size'] = details.get('mesh_size', '')
            summary_row['Detail Page'] = details.get('detail_page', '')
            summary_row['Full Details'] = details.get('full_details', '')
            summary_rows.append(summary_row)
        
        # Grand total
        grand_row = {
            'Page': 'GRAND',
            'Category': 'TOTAL',
            'Type': 'Summary',
            'Length (ft)': round(df['Length (ft)'].sum(), 2),
            'Length (pts)': '',
            'Scale': ''
        }
        for col in DETAIL_COLS:
            grand_row[col] = ''
        summary_rows.append(grand_row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Ensure column order for measurements sheet (no detail cols cluttering it)
        meas_cols = ['Page', 'Category', 'Type', 'Length (ft)', 'Length (pts)', 'Scale']
        meas_final = [c for c in meas_cols if c in df.columns]
        df_meas = df[meas_final]
        
        # Summary sheet: totals per category + grand total
        summ_final = [c for c in all_columns if c in summary_df.columns]
        df_summ = summary_df[summ_final]
        
        # Element Specifications sheet
        el_details = element_details or {}
        spec_rows = []
        for elem_name, details in el_details.items():
            if any(v for v in details.values() if v):
                spec_rows.append({
                    'Element': elem_name,
                    'Height': details.get('height', ''),
                    'Post Type': details.get('post_type', ''),
                    'Post Spacing': details.get('post_spacing', ''),
                    'Material': details.get('material', ''),
                    'Gauge': details.get('gauge', ''),
                    'Mesh Size': details.get('mesh_size', ''),
                    'Foundation': details.get('foundation', ''),
                    'Gate Info': details.get('gate_info', ''),
                    'Detail Page': details.get('detail_page', ''),
                    'Full Details': details.get('full_details', ''),
                    'Notes': details.get('notes', ''),
                })
        df_specs = pd.DataFrame(spec_rows) if spec_rows else pd.DataFrame()
        
        # Write to Excel with multiple sheets
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_meas.to_excel(writer, sheet_name='Measurements', index=False)
            df_summ.to_excel(writer, sheet_name='Summary', index=False)
            if not df_specs.empty:
                df_specs.to_excel(writer, sheet_name='Element Specifications', index=False)
        return buf.getvalue()
    
    # Return empty Excel with headers if no data
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pd.DataFrame(columns=all_columns).to_excel(writer, sheet_name='Measurements', index=False)
    return buf.getvalue()


# ==============================================================================
# Sidebar (matching app.py structure)
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load secrets if available
    secrets = {}
    if os.path.exists(".streamlit/secrets.toml"):
        secrets = toml.load(".streamlit/secrets.toml")
    
    # 1. OpenAI Key
    openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
    
    # 2. LandingAI (ADE) Key
    ade_key = secrets.get("LANDINGAI_API_KEY", os.getenv("LANDINGAI_API_KEY"))
    if not ade_key:
        ade_key = st.text_input("Enter LandingAI API Key", type="password", key="ade_key_input_sidebar")
    
    # 3. Google Cloud Config (JSON) - load silently without UI messages
    google_cloud_config = None
    try:
        if "google_cloud" in secrets and "gcp_service_account" in secrets:
            google_cloud_config = {
                "project_number": secrets["google_cloud"]["project_number"],
                "location": secrets["google_cloud"]["location"],
                "processor_id": secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(secrets["gcp_service_account"])
            }
            print(f"SESSION {current_session_id} LOG: Google Cloud config loaded from secrets")
    except Exception as e:
        print(f"SESSION {current_session_id} WARNING: Could not load Google Cloud config: {e}")
    
    # Highlight toggle
    st.markdown("---")
    highlight_fence_text_app = st.toggle("🔍 Highlight text & indicators", value=True, key="highlight_toggle")
    
    # ADE usage toggle
    use_ade = st.toggle("🧠 Use ADE (LandingAI)", value=True, key="use_ade_toggle")
    
    # Unified Measurement toggle (auto-detection + interactive editing)
    enable_unified_measurement = st.toggle("📏 Unified Measurements", value=True, key="unified_measurement_toggle",
                                           help="Auto-detect fence lines and interactively select/draw additional lines")
    enable_nonlayer_suggestions = st.toggle("🔬 Non-layer suggestions", value=False, key="nonlayer_suggestions_toggle",
                                            help="Show auto-detected suggestions even when no fence layers found (less reliable)")
    
    # Debug mode (disabled in UI)
    DEBUG_MODE = False
    
    # Fence Keywords
    st.markdown("---")
    st.subheader("Fence Keywords")
    if 'fence_keywords_app' not in st.session_state:
        st.session_state.fence_keywords_app = ['fence']
    custom_keywords_str = st.text_area(
        "Custom keywords (one per line):",
        "\n".join(st.session_state.fence_keywords_app),
        height=150,
        key="kw_text_area"
    )
    if st.button("Update Keywords", key="update_kw_btn"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()
    
    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


# ==============================================================================
# Initialize LLM (cached to avoid re-init on every rerun)
# ==============================================================================

@st.cache_resource
def get_llm_instance(api_key: str, model: str):
    """Cache LLM instance to avoid slow re-initialization on every page load."""
    print(f"LOG: Creating cached LLM instance for model {model}")
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
        timeout=180,
        max_retries=2
    )

llm_analysis_instance = None
if openai_key:
    try:
        llm_analysis_instance = get_llm_instance(openai_key, st.session_state.selected_model_for_analysis)
    except Exception as e:
        st.error(f"LLM Init Error: {e}")
        openai_key = None
        print(f"SESSION {current_session_id} ERROR: LLM Init Error: {e}")


# ==============================================================================
# Main App Flow
# ==============================================================================

st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}"
    
    # Guard: reject excessively large files to prevent OOM
    MAX_FILE_SIZE_MB = 500
    file_size_mb = uploaded_pdf_file_obj.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.0f} MB). Maximum is {MAX_FILE_SIZE_MB} MB to prevent memory issues.")
        st.stop()
    
    if st.session_state.last_uploaded_file_id != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Resetting state for {current_file_id}.")
        # Preserve some settings across resets
        current_selected_model = st.session_state.selected_model_for_analysis
        current_keywords = st.session_state.fence_keywords_app
        
        initialize_session_state(current_session_id)
        
        st.session_state.selected_model_for_analysis = current_selected_model
        st.session_state.fence_keywords_app = current_keywords
        
        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        st.session_state.original_pdf_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.current_pdf_hash = hashlib.sha256(st.session_state.original_pdf_bytes).hexdigest()
        st.session_state.last_uploaded_file_id = current_file_id
        
        st.cache_data.clear()
        print(f"SESSION {current_session_id} LOG: Cleared all @st.cache_data caches due to new file.")
        st.rerun()
    
    if openai_key and llm_analysis_instance and \
       (ade_key or not use_ade) and \
       not st.session_state.run_analysis_triggered and \
       not st.session_state.processing_complete and \
       not st.session_state.analysis_halted_due_to_error:
        print(f"SESSION {current_session_id} LOG: Triggering analysis.")
        st.session_state.run_analysis_triggered = True


# ==============================================================================
# Analysis Execution Block
# ==============================================================================

if st.session_state.run_analysis_triggered and \
   st.session_state.original_pdf_bytes and \
   llm_analysis_instance and \
   (ade_key or not use_ade) and \
   not st.session_state.analysis_halted_due_to_error and \
   not st.session_state.processing_complete:
    
    print(f"SESSION {current_session_id} LOG: Starting ADE-based PDF processing.")
    file_bytes = st.session_state.original_pdf_bytes
    
    # Open PDF to get page count
    doc_proc = None
    try:
        doc_proc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
        st.session_state.doc_total_pages = len(doc_proc)
        print(f"SESSION {current_session_id} LOG: PDF opened, {st.session_state.doc_total_pages} pages.")
        
        MAX_PAGES = 300
        if st.session_state.doc_total_pages > MAX_PAGES:
            st.error(f"PDF has {st.session_state.doc_total_pages} pages (max {MAX_PAGES}). Please split the document.")
            st.session_state.processing_complete = True
            st.session_state.analysis_halted_due_to_error = True
            doc_proc.close()
            st.stop()
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        st.session_state.processing_complete = True
        st.session_state.analysis_halted_due_to_error = True
        if doc_proc:
            doc_proc.close()
        print(f"SESSION {current_session_id} ERROR: Failed to open PDF for processing: {e}")
        st.stop()
    
    # UI Setup (matching app.py)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results (Live)</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty()
    col_f, col_nf = st.columns(2)
    with col_f:
        st.subheader("✅ Fence-Related Pages")
    with col_nf:
        st.subheader("❌ Non-Fence Pages")
    prog_bar = st.progress(0)
    status_txt_area = st.empty()
    
    try:
        total_pages = st.session_state.doc_total_pages
        
        # =================================================================
        # PHASE 1: Pre-filter ALL pages to identify fence pages
        # Text extraction + keyword/LLM scan (no ADE yet)
        #
        # OPTIMIZATION: Split into 3 sub-steps:
        #   1a. Extract native PDF text (CPU, fast, sequential — needs fitz page)
        #   1b. Run Google OCR in parallel (I/O-bound network calls)
        #   1c. Run fence detection (may need LLM, sequential)
        # =================================================================
        _page_cache = {}
        _fence_page_indices = []
        
        # --- Step 1a: Extract native PDF text + prepare OCR inputs (fast, sequential) ---
        _pdf_lines_by_page = {}
        _page_dims = {}
        _single_page_pdfs = {}  # Cache for reuse in Phase 2 fallback
        
        status_txt_area.text(f"Phase 1a — Extracting text from {total_pages} pages...")
        for page_idx in range(total_pages):
            page = doc_proc[page_idx]
            _page_dims[page_idx] = (page.rect.width, page.rect.height)
            _pdf_lines_by_page[page_idx] = ade.get_native_pdf_lines(page)
        
        # Batch-create single-page PDFs for OCR (reused in Phase 2 fallback)
        # Uses one fitz.open() instead of N separate open/close cycles
        if google_cloud_config:
            for page_idx in range(total_pages):
                _tmp = fitz.open()
                _tmp.insert_pdf(doc_proc, from_page=page_idx, to_page=page_idx)
                _single_page_pdfs[page_idx] = _tmp.tobytes()
                _tmp.close()
        
        # --- Step 1b: Run Google OCR in parallel across all pages ---
        _ocr_lines_by_page = {i: [] for i in range(total_pages)}
        
        if google_cloud_config:
            status_txt_area.text(f"Phase 1b — Running OCR on {total_pages} pages in parallel...")
            
            def _run_ocr_for_page(page_idx):
                """Worker function for parallel OCR."""
                pdf_w, pdf_h = _page_dims[page_idx]
                return page_idx, ade.run_google_ocr_blocks(
                    _single_page_pdfs[page_idx], google_cloud_config, pdf_w, pdf_h
                )
            
            # Use up to 4 parallel OCR workers (API rate-limited, don't overwhelm)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_run_ocr_for_page, pi): pi for pi in range(total_pages)}
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    prog_bar.progress(completed / total_pages * 0.15)
                    try:
                        pi, ocr_result = future.result()
                        _ocr_lines_by_page[pi] = ocr_result
                    except Exception as e:
                        pi = futures[future]
                        print(f"SESSION {current_session_id} WARNING: OCR failed for page {pi + 1}: {e}")
            
            print(f"SESSION {current_session_id} LOG: Phase 1b complete — OCR done for {total_pages} pages")
            
            # Free single-page PDFs after OCR — they can be recreated on-demand for ADE fallback
            _single_page_pdfs.clear()
            gc.collect()
            print(f"SESSION {current_session_id} LOG: Freed single-page PDF cache after OCR")
        
        # --- Step 1c: Run fence detection (keyword scan + optional LLM, sequential) ---
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            prog_bar.progress(0.15 + page_num / total_pages * 0.15)
            status_txt_area.text(f"Phase 1c — Classifying page {page_num}/{total_pages}...")
            print(f"SESSION {current_session_id} LOG: Pre-filtering page {page_num}.")
            
            pdf_lines = _pdf_lines_by_page[page_idx]
            ocr_lines = _ocr_lines_by_page[page_idx]
            
            prefilter_result = ade.fallback_fence_detection(
                pdf_lines=pdf_lines,
                ocr_lines=ocr_lines,
                fence_keywords=FENCE_KEYWORDS_APP,
                llm=llm_analysis_instance,
                use_llm_confirmation=True
            )
            
            _page_cache[page_idx] = {
                'pdf_lines': pdf_lines,
                'ocr_lines': ocr_lines,
                'prefilter_result': prefilter_result,
            }
            
            if prefilter_result["fence_found"]:
                _fence_page_indices.append(page_idx)
        
        print(f"SESSION {current_session_id} LOG: Phase 1 complete — "
              f"{len(_fence_page_indices)}/{total_pages} fence pages detected")
        
        # =================================================================
        # PHASE 2: Batch ADE for fence pages (smart batching by size)
        # =================================================================
        _ade_chunks_by_page = {}
        
        if use_ade and ade_key and _fence_page_indices:
            _batches = ade.create_page_batches(file_bytes, _fence_page_indices)
            
            for _batch_idx, _batch in enumerate(_batches):
                _batch_pages_str = ", ".join(str(i + 1) for i in _batch)
                status_txt_area.text(
                    f"Phase 2 — ADE batch {_batch_idx + 1}/{len(_batches)}: "
                    f"pages {_batch_pages_str}..."
                )
                prog_bar.progress(0.3 + (_batch_idx + 1) / len(_batches) * 0.3)
                
                _batch_pdf = ade.create_multi_page_pdf(file_bytes, _batch)
                print(f"SESSION {current_session_id} LOG: ADE batch {_batch_idx + 1}/{len(_batches)}: "
                      f"{len(_batch)} pages, {len(_batch_pdf) / 1024:.0f}KB")
                
                _ade_response = ade.ade_parse_document(_batch_pdf, ade_key)
                
                if _ade_response["success"]:
                    for _local_idx, _orig_idx in enumerate(_batch):
                        _p = doc_proc[_orig_idx]
                        _chunks = ade.align_ade_chunks_to_page(
                            _ade_response, _local_idx,
                            _p.rect.width, _p.rect.height
                        )
                        _ade_chunks_by_page[_orig_idx] = _chunks
                        print(f"[APP] Page {_orig_idx + 1}: {len(_chunks)} ADE chunks from batch")
                else:
                    # Batch failed — fall back to per-page ADE for this batch
                    print(f"[APP] ADE batch {_batch_idx + 1} failed: {_ade_response.get('error')}")
                    status_txt_area.text(
                        f"Phase 2 — Batch {_batch_idx + 1} failed, retrying pages individually..."
                    )
                    for _orig_idx in _batch:
                        try:
                            _single_pdf = ade.create_single_page_pdf(file_bytes, _orig_idx)
                            _single_resp = ade.ade_parse_document(_single_pdf, ade_key)
                            if _single_resp["success"]:
                                _p = doc_proc[_orig_idx]
                                _chunks = ade.align_ade_chunks_to_page(
                                    _single_resp, 0,
                                    _p.rect.width, _p.rect.height
                                )
                                _ade_chunks_by_page[_orig_idx] = _chunks
                                print(f"[APP] Page {_orig_idx + 1}: {len(_chunks)} ADE chunks (individual retry)")
                            else:
                                _ade_chunks_by_page[_orig_idx] = None
                                print(f"[APP] Page {_orig_idx + 1}: individual ADE also failed")
                        except Exception as _e:
                            _ade_chunks_by_page[_orig_idx] = None
                            print(f"[APP] Page {_orig_idx + 1}: individual ADE error: {_e}")
            
            _ok = sum(1 for v in _ade_chunks_by_page.values() if v is not None)
            print(f"SESSION {current_session_id} LOG: Phase 2 complete — "
                  f"ADE results for {_ok}/{len(_fence_page_indices)} fence pages")
        
        # =================================================================
        # PHASE 3: Process each page using cached pre-filter + ADE results
        # =================================================================
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            st.session_state.total_pages_processed_count = page_num
            prog_bar.progress(0.6 + page_num / total_pages * 0.4)
            status_txt_area.text(f"Phase 3 — Processing page {page_num}/{total_pages}...")
            print(f"SESSION {current_session_id} LOG: Processing page {page_num}.")
            
            # Get page dimensions (cheap) — defer image rendering to fence pages only
            page = doc_proc[page_idx]
            pdf_width, pdf_height = page.rect.width, page.rect.height
            page_img_bytes = None  # Rendered lazily below only for fence pages
            
            # Load cached pre-filter results
            _cached = _page_cache[page_idx]
            pdf_lines = _cached['pdf_lines']
            ocr_lines = _cached['ocr_lines']
            prefilter_result = _cached['prefilter_result']
            
            # Initialize variables
            chunks = []
            legend_chunks = []
            figure_chunks = []
            definitions = []
            instances = []
            fallback_result = None
            keyword_matches = []
            measurement_result = {}  # Initialize to empty dict to prevent undefined errors
            # Default detection method based on prefilter; can be overridden later
            detection_method = prefilter_result.get("method", "none")
            fence_found = prefilter_result["fence_found"]
            
            if not fence_found:
                # =====================================================================
                # NON-FENCE PAGE: Build minimal result and skip to next page
                # =====================================================================
                print(f"[APP] Page {page_num}: Not fence-related, skipping.")
                st.session_state.non_fence_pages.append({
                    'page_number': page_num,
                    'page_index_in_original_doc': page_idx,
                    'fence_found': False,
                })
                with col_nf:
                    with st.expander(f"Page {page_num}", expanded=False):
                        st.info("No fence-related items found on this page.")
                # Update summary
                summary_placeholder.markdown(
                    f"### Summary (Processed: {page_num}/{total_pages})\n"
                    f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                    f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
                )
                continue
            else:
                # =====================================================================
                # FENCE PAGE: Use pre-computed ADE chunks from batch
                # =====================================================================
                if use_ade and ade_key:
                    _ade_chunks = _ade_chunks_by_page.get(page_idx)
                    if _ade_chunks is None:
                        print(f"[APP] Page {page_num}: ADE failed for this page, using pre-filter fallback.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                        detection_method = prefilter_result["method"]
                    else:
                        status_txt_area.text(f"Phase 3 — Page {page_num}/{total_pages}: Extracting definitions...")
                        chunks = _ade_chunks
                        legend_chunks, figure_chunks = ade.segment_chunks(chunks)
                else:
                    # ADE disabled or no key: rely solely on pre-filter result
                    print(f"[APP] Page {page_num}: ADE is disabled or missing key; using pre-filter result only.")
                    fallback_result = prefilter_result
                    keyword_matches = prefilter_result.get("matched_lines", [])
                
                # =====================================================================
                # STEP 5: Process chunks (runs for ALL fence pages)
                # =====================================================================
                # Debug visualization
                if DEBUG_MODE and (legend_chunks or pdf_lines or ocr_lines):
                    debug_bytes = ade.debug_visualize_coordinates(
                        page_img_bytes, legend_chunks, pdf_lines, ocr_lines, pdf_width, pdf_height
                    )
                    st.image(debug_bytes, caption=f"DEBUG: Layers Page {page_num}", use_container_width=True)
                
                # Extract fence-related definitions from legend chunks
                if highlight_fence_text_app and legend_chunks:
                    definitions = ade.extract_legend_entries(
                        legend_chunks=legend_chunks,
                        pdf_lines=pdf_lines,
                        ocr_lines=ocr_lines,
                        fence_keywords=FENCE_KEYWORDS_APP,
                        llm=llm_analysis_instance,
                        figure_chunks=figure_chunks
                    )
                
                # Get all page tokens for instance finding
                # IMPORTANT: Transform MediaBox coords to display coords for rotated pages
                native_words = page.get_text("words")
                rotation = page.rotation
                mediabox_w = page.mediabox.width
                mediabox_h = page.mediabox.height
                print(f"[DEBUG] Page {page_num} rotation: {rotation}°, MediaBox: {mediabox_w:.0f}x{mediabox_h:.0f}")
                
                def transform_for_rotation(x0, y0, x1, y1):
                    """Transform MediaBox coords to display coords based on page rotation"""
                    if rotation == 0:
                        return x0, y0, x1, y1
                    elif rotation == 90:
                        return mediabox_h - y1, x0, mediabox_h - y0, x1
                    elif rotation == 180:
                        return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                    elif rotation == 270:
                        return y0, mediabox_w - x1, y1, mediabox_w - x0
                    return x0, y0, x1, y1
                
                all_page_tokens = []
                for w in native_words:
                    nx0, ny0, nx1, ny1 = transform_for_rotation(w[0], w[1], w[2], w[3])
                    all_page_tokens.append({
                        "text": w[4], 
                        "x0": nx0, "y0": ny0, 
                        "x1": nx1, "y1": ny1
                    })
                
                if all_page_tokens:
                    sample = all_page_tokens[0]
                    print(f"[DEBUG] Sample token after transform: '{sample['text']}' at ({sample['x0']:.1f}, {sample['y0']:.1f})")
                
                # Find instances in figures
                if definitions and figure_chunks:
                    instances = ade.find_instances_in_figures(definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines)
                
                # =====================================================================
                # STEP 6: Scale Detection + Smart Fence Measurement
                # =====================================================================
                page_key = f"page_{page_num}"
                
                # Detect scale ONCE using the full chain (vision GPT -> text LLM -> regex)
                detected_scale = None
                if page_key not in st.session_state.per_page_scale_info:
                    try:
                        status_txt_area.text(f"Phase 3 — Page {page_num}/{total_pages}: Detecting scale...")
                        scale_info = verify_scale_with_bar(page, llm=llm_analysis_instance)
                        st.session_state.per_page_scale_info[page_key] = scale_info
                        if scale_info.get('success') and scale_info.get('verified_scale'):
                            detected_scale = scale_info['verified_scale']
                            print(f"[APP] Page {page_num}: Scale detected = {detected_scale} "
                                  f"({scale_info.get('confidence', '?')}: {scale_info.get('scale_text', '')})")
                        else:
                            print(f"[APP] Page {page_num}: Scale not detected — {scale_info.get('message', '')}")
                    except Exception as e:
                        print(f"[APP] Page {page_num}: Scale detection error: {e}")
                        st.session_state.per_page_scale_info[page_key] = {
                            'success': False, 'verified_scale': None, 'message': str(e)
                        }
                else:
                    scale_info = st.session_state.per_page_scale_info[page_key]
                    detected_scale = scale_info.get('verified_scale')
                
                # Measure fence elements (pass detected scale so it skips its own detection)
                # OPTIMIZATION: Always pass a value (default 1.0) to prevent
                # measure_fence_elements from re-running scale detection (redundant vision LLM call)
                measurement_result = {}
                if enable_unified_measurement and (definitions or instances):
                    try:
                        status_txt_area.text(f"Phase 3 — Page {page_num}/{total_pages}: Measuring fence elements...")
                        ocr_full_text = "\n".join(line.get('text', '') for line in ocr_lines) if ocr_lines else None
                        measurement_result = ade.measure_fence_elements(
                            page, definitions, instances, 
                            figure_chunks=figure_chunks,
                            llm=llm_analysis_instance,
                            scale_factor=detected_scale or 1.0,
                            ocr_text=ocr_full_text
                        )
                    except Exception as e:
                        print(f"[APP] Measurement error: {e}")
                
                # Store auto-detected lines in unified measurement structure
                # ONLY for layer-based detection (reliable) - skip LLM-guided fallback
                measurement_method = measurement_result.get('measurement_method', 'none') if measurement_result else 'none'
                
                if measurement_result and measurement_result.get('all_fence_lines') and measurement_method == 'layer':
                    auto_lines = []
                    all_fence_lines = measurement_result.get('all_fence_lines', [])
                    scale_factor = measurement_result.get('page_info', {}).get('scale_factor', 1.0)
                    layer_to_category = measurement_result.get('layer_to_category', {})
                    
                    # Map each line to its category using layer→category mapping
                    for line in all_fence_lines:
                        length_pts = line.length_pts
                        length_inches = length_pts / 72.0
                        length_feet = (length_inches * scale_factor) / 12.0
                        
                        line_layer = getattr(line, 'layer', None) or ''
                        # Use LLM-matched layer→category mapping
                        category = layer_to_category.get(line_layer)
                        
                        # Fallback: if layer not in mapping, try partial match
                        if not category and line_layer:
                            for mapped_layer, cat in layer_to_category.items():
                                if mapped_layer in line_layer or line_layer in mapped_layer:
                                    category = cat
                                    break
                        
                        if category:
                            auto_lines.append({
                                'start': line.start,
                                'end': line.end,
                                'length_pts': length_pts,
                                'length_feet': length_feet,
                                'layer': line_layer,
                                'category': category,
                                'source': 'auto'
                            })
                    
                    if auto_lines:
                        if page_key not in st.session_state.unified_measurements:
                            st.session_state.unified_measurements[page_key] = {
                                'auto_lines': [], 'manual_lines': [], 'drawn_lines': [], 'accepted_auto': set()
                            }
                        st.session_state.unified_measurements[page_key]['auto_lines'] = auto_lines
                        # Auto-accept all detected lines by default
                        st.session_state.unified_measurements[page_key]['accepted_auto'] = set(range(len(auto_lines)))
                        print(f"[AUTO] Page {page_num}: {len(auto_lines)} layer-based lines stored with categories from {len(layer_to_category)} layer mappings")
                    else:
                        print(f"[AUTO] Page {page_num}: layer-based but no lines matched to categories")
                elif measurement_method != 'layer':
                    print(f"[AUTO] Page {page_num}: skipping suggestions (method={measurement_method}, not layer-based)")
                
                # Scale info already stored above (before measurement)
                
                # Initialize categories from definitions (for all pages)
                if page_key not in st.session_state.page_categories:
                    categories = {}
                    CATEGORY_COLORS = [
                        (0, 255, 0), (255, 165, 0), (0, 191, 255), (255, 0, 255),
                        (255, 255, 0), (0, 255, 255), (255, 105, 180), (173, 255, 47),
                    ]
                    for d in definitions:
                        indicator = d.get('indicator', '')
                        keyword = d.get('keyword', '')
                        if keyword:
                            cat_name = f"{indicator}: {keyword}" if indicator else keyword
                            if cat_name not in categories:
                                color_idx = len(categories)
                                categories[cat_name] = {
                                    'indicator': indicator,
                                    'keyword': keyword,
                                    'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                                }
                    st.session_state.page_categories[page_key] = categories
                
                # DEBUG: Show coordinate info if enabled
                if DEBUG_MODE:
                    with st.expander(f"🔧 DEBUG Page {page_num}", expanded=True):
                        st.markdown(f"**PDF size:** {pdf_width:.1f} x {pdf_height:.1f}")
                        st.markdown(f"**All ADE Chunks:** {len(chunks)}")
                        for i, c in enumerate(chunks[:10]):
                            st.markdown(f"  - `{c.get('type')}`: ({c.get('x0'):.1f}, {c.get('y0'):.1f}) - ({c.get('x1'):.1f}, {c.get('y1'):.1f})")
                        st.markdown(f"**Figure/Architectural Chunks:** {len(figure_chunks)}")
                        for i, fc in enumerate(figure_chunks):
                            st.markdown(f"  - `{fc.get('type')}`: ({fc.get('x0'):.1f}, {fc.get('y0'):.1f}) - ({fc.get('x1'):.1f}, {fc.get('y1'):.1f})")
                        st.markdown(f"**Legend-like Chunks:** {len(legend_chunks)}")
                        st.markdown(f"**Definitions found:** {len(definitions)}")
                        for i, d in enumerate(definitions[:5]):
                            kw = d.get('keyword', '')[:30]
                            st.markdown(f"  - `{d.get('indicator')}`: {kw}... @ ({d.get('x0'):.1f}, {d.get('y0'):.1f})")
                        st.markdown(f"**Instances found:** {len(instances)}")
                        for i, inst in enumerate(instances[:10]):
                            st.markdown(f"  - `{inst.get('indicator')}` @ ({inst.get('x0'):.1f}, {inst.get('y0'):.1f})")
                        st.markdown(f"**Total page tokens:** {len(all_page_tokens)}")
                        if all_page_tokens:
                            st.markdown(f"**Sample tokens (first 20):**")
                            for t in all_page_tokens[:20]:
                                st.markdown(f"  - `{t.get('text')}` @ ({t.get('x0'):.1f}, {t.get('y0'):.1f})")
                
                # Determine detection method
                if definitions or instances:
                    detection_method = "ade"
                    fence_found = True
                else:
                    # No structured data found, fall back to pre-filter results
                    if not fallback_result:
                        print(f"[APP] Page {page_num}: No definitions found, using pre-filter results.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                    detection_method = prefilter_result.get("method", "none")
            
            # Collect full page text for cross-page detail extraction
            if fence_found:
                page_text_parts = []
                for pl in pdf_lines:
                    t = pl.get('text', '').strip()
                    if t:
                        page_text_parts.append(t)
                for ol in ocr_lines:
                    t = ol.get('text', '').strip()
                    if t:
                        page_text_parts.append(t)
                full_page_text = "\n".join(page_text_parts)
                if full_page_text.strip():
                    st.session_state.fence_page_texts[page_num] = full_page_text
            
            # Build text snippet from definitions or fallback keywords
            text_snippet = None
            if definitions:
                snippets = [f"{d.get('indicator', '')} - {d.get('keyword', '')}" for d in definitions[:3]]
                text_snippet = "; ".join(snippets)
            elif fallback_result and fallback_result.get("matched_keywords"):
                text_snippet = "Keywords: " + ", ".join(fallback_result["matched_keywords"][:5])
            
            # Generate images — render only for fence pages (lazy)
            if fence_found and page_img_bytes is None:
                page_img_bytes = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI).tobytes("png")
            original_img_bytes = page_img_bytes
            highlighted_img_bytes = None
            
            if highlight_fence_text_app and page_img_bytes:
                if definitions or instances:
                    # Primary highlighting: definitions (green) + instances (purple)
                    highlighted_img_bytes = ade.highlight_page_image(
                        page_img_bytes, definitions, instances, pdf_width, pdf_height
                    )
                elif keyword_matches:
                    # Fallback highlighting: keyword matches (orange)
                    highlighted_img_bytes = ade.highlight_keyword_matches(
                        page_img_bytes, keyword_matches, pdf_width, pdf_height
                    )
                
                # Highlight measured fence lines (cyan) - only for layer-based or if non-layer toggle enabled
                if measurement_result and measurement_result.get('all_fence_lines') and (measurement_method == 'layer' or enable_nonlayer_suggestions):
                    highlighted_img_bytes = ade.highlight_fence_lines(
                        highlighted_img_bytes or page_img_bytes,
                        measurement_result['all_fence_lines'],
                        pdf_width, pdf_height
                    )
            
            # Build result structure (matching app.py format)
            # detection_method already set above based on flow
            llm_result = fallback_result.get("llm_result") if fallback_result else None
            
            # Strip heavy VectorLine objects from measurement_result before storing
            # Keep only summary data (totals, indicator_measurements, layer info)
            measurement_result_light = {}
            if measurement_result:
                measurement_result_light = {
                    k: v for k, v in measurement_result.items()
                    if k != 'all_fence_lines'
                }
            
            analysis_result = {
                'page_number': page_num,
                'page_index_in_original_doc': page_idx,
                'fence_found': fence_found,
                'text_found': fence_found,
                'text_response': json.dumps({
                    "answer": "yes" if fence_found else "no",
                    "confidence": 0.9 if definitions else (llm_result["confidence"] if llm_result else 0.6),
                    "signals": [d.get('keyword', '') for d in definitions[:5]] if definitions else (fallback_result.get("matched_keywords", []) if fallback_result else []),
                    "reason": f"Found {len(definitions)} definitions, {len(instances)} instances" if definitions else (llm_result["reason"] if llm_result else f"Keyword match: {fallback_result.get('matched_keywords', [])}" if fallback_result else "No fence content")
                }),
                'text_snippet': text_snippet,
                'definitions': definitions,
                'instances': instances,
                'keyword_matches': keyword_matches,
                'fallback_result': fallback_result,
                'measurements': measurement_result_light,
                'detection_method': detection_method,
                'highlight_fence_text_app_setting': highlight_fence_text_app,
                'original_image_bytes': None,
                'highlighted_image_bytes': None,
                'pdf_width': pdf_width,
                'pdf_height': pdf_height,
                'chunk_count': len(chunks),
                'legend_count': len(legend_chunks),
                'figure_count': len(figure_chunks),
            }
            
            # Add to fence pages list (non-fence pages handled via early continue above)
            st.session_state.fence_pages.append(analysis_result)
            
            # Display in fence column (non-fence pages handled via early continue above)
            with col_f:
                exp_title = f"Page {page_num}"
                if True:
                    reasons = []
                    if definitions:
                        reasons.append("Definitions")
                    if instances:
                        reasons.append("Instances")
                    if keyword_matches and not definitions:
                        reasons.append("Keywords")
                    if highlight_fence_text_app and (definitions or instances or keyword_matches):
                        reasons.append("Highlights")
                    if reasons:
                        exp_title += f" ({' & '.join(reasons)})"
                
                with st.expander(exp_title, expanded=True):
                    img_col, det_col = st.columns([2, 1])
                    
                    with img_col:
                        disp_img = highlighted_img_bytes if highlighted_img_bytes else original_img_bytes
                        if disp_img:
                            st.image(disp_img, caption=f"Page {page_num}{' (Highlighted)' if highlighted_img_bytes else ''}")
                        
                        # Download links
                        dl_links = []
                        if highlighted_img_bytes:
                            dl_links.append(get_image_download_link_html(highlighted_img_bytes, f"page_{page_num}_hl.png", "DL HL Img"))
                        if original_img_bytes:
                            dl_links.append(get_image_download_link_html(original_img_bytes, f"page_{page_num}_orig.png", "DL Orig Img"))
                        if dl_links:
                            st.markdown(" ".join(dl_links), unsafe_allow_html=True)
                    
                    with det_col:
                        # Detection method badge
                        if detection_method == "ade":
                            st.success("🎯 ADE Detection")
                        elif detection_method == "llm_confirmed":
                            st.warning("🔍 Keyword + LLM")
                        elif detection_method == "keyword_only":
                            st.warning("🔤 Keyword Match")
                        else:
                            st.info("❌ No Detection")
                        
                        # ADE Stats (compact)
                        st.metric("ADE Chunks", len(chunks))
                        col_leg, col_fig = st.columns(2)
                        with col_leg:
                            st.metric("Legend", len(legend_chunks))
                        with col_fig:
                            st.metric("Figure", len(figure_chunks))
                        
                        # Text response popover
                        if analysis_result.get('text_response'):
                            with st.popover("Analysis Log"):
                                st.markdown(f"_{analysis_result['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # NEW: Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (Fallback)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                    
                    # Show Measurements (for ALL fence pages, not just keyword matches)
                    meas_method_stored = measurement_result.get('measurement_method', 'none') if measurement_result else 'none'
                    if measurement_result and (meas_method_stored == 'layer' or enable_nonlayer_suggestions) and (measurement_result.get('indicator_measurements') or measurement_result.get('proximity_totals', {}).get('total_segments', 0) > 0):
                        st.markdown("---")
                        st.markdown("### 📏 Fence Measurements")
                        
                        page_info = measurement_result.get('page_info', {})
                        scale_factor = page_info.get('scale_factor', 1.0)
                        method = measurement_result.get('measurement_method', 'unknown')
                        
                        # Show method badge
                        if method == "layer":
                            st.info("📂 Method: Layer-based (fence layers detected)")
                        elif method == "proximity":
                            st.info("🎯 Method: Proximity-based (fallback)")
                        elif method == "llm_guided":
                            st.info("🤖 Method: LLM-guided (adaptive filtering)")
                        elif method == "length_filter":
                            st.info("📏 Method: Length-filtered (no layers, using segment length)")
                        elif method == "no_layers":
                            st.error("❌ No fence layers found in PDF - measurement not available")
                        
                        # Show scale info
                        if page_info.get('scale_detected'):
                            st.success(f"✅ Scale: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                        else:
                            st.warning("⚠️ Scale not detected - raw measurements")
                        
                        # Show totals
                        prox_totals = measurement_result.get('proximity_totals', {})
                        if prox_totals.get('total_segments', 0) > 0:
                            col_pts, col_ft = st.columns(2)
                            with col_pts:
                                st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                            with col_ft:
                                st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                            
                            # Per-indicator breakdown
                            indicator_meas = measurement_result.get('indicator_measurements', {})
                            if indicator_meas:
                                st.markdown("**Per-Indicator:**")
                                for ind, stats in indicator_meas.items():
                                    pts = stats.get('run_length_pts', 0)
                                    ft = stats.get('run_length_feet', 0)
                                    segs = stats.get('run_segment_count', 0)
                                    count = stats.get('instance_count', 0)
                                    st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                        
                        # Layer breakdown (secondary)
                        if measurement_result.get('fence_layers'):
                            with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                totals = measurement_result.get('totals', {})
                                st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                for layer in measurement_result['fence_layers']:
                                    l_stats = measurement_result['layer_measurements'].get(layer, {})
                                    segs = l_stats.get('total_segments', 0)
                                    ft = l_stats.get('total_length_feet', 0)
                                    runs = l_stats.get('connected_runs', 0)
                                    st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                        
                        # Dimension line measurements
                        dim_measurements = measurement_result.get('dimension_measurements', [])
                        if dim_measurements:
                            with st.expander("📐 Dimension Line Measurements", expanded=False):
                                st.caption(f"Found {len(dim_measurements)} dimension annotations")
                                for dm in dim_measurements[:10]:
                                    ft = dm.get('actual_ft', 0)
                                    txt = dm.get('measurement_text', '')
                                    st.markdown(f"- **{txt}**: {ft:.1f} ft")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
            
            # Update summary
            summary_placeholder.markdown(
                f"### Summary (Processed: {st.session_state.total_pages_processed_count}/{total_pages})\n"
                f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
            )
            
            time.sleep(0.05)
        
        # Free Phase 1/2 caches to reduce memory (especially for large PDFs)
        del _page_cache, _pdf_lines_by_page, _ocr_lines_by_page, _ade_chunks_by_page
        _single_page_pdfs.clear()
        gc.collect()
        print(f"SESSION {current_session_id} LOG: Freed processing caches, gc.collect() done")
        
        # =====================================================================
        # CROSS-PAGE DETAIL EXTRACTION
        # After all pages processed, extract detailed specs for each element
        # =====================================================================
        if st.session_state.fence_pages and st.session_state.fence_page_texts:
            status_txt_area.text("Extracting element details across pages...")
            # Collect unique element names from all definitions
            all_element_names = []
            seen_elements = set()
            for fp in st.session_state.fence_pages:
                for d in fp.get('definitions', []):
                    kw = d.get('keyword', '').strip()
                    desc_val = d.get('description', '').strip()
                    if kw and kw not in seen_elements and desc_val != "Indicator Code":
                        ind = d.get('indicator', '').strip()
                        element_label = f"{ind}: {kw}" if ind else kw
                        all_element_names.append(element_label)
                        seen_elements.add(kw)
            
            if all_element_names:
                print(f"[APP] Extracting details for {len(all_element_names)} elements: {all_element_names}")
                try:
                    element_details = ade.extract_element_details(
                        llm=llm_analysis_instance,
                        element_names=all_element_names,
                        page_texts=st.session_state.fence_page_texts,
                    )
                    st.session_state.element_details = element_details
                    print(f"[APP] Element details extracted: {len(element_details)} elements")
                except Exception as e:
                    print(f"[APP] Element detail extraction error: {e}")
                    st.session_state.element_details = {}
        
        # Processing complete
        st.session_state.processing_complete = True
        prog_bar.empty()
        status_txt_area.success("All pages processed!")
        
        # Generate combined PDF
        if st.session_state.fence_pages and st.session_state.original_pdf_bytes:
            pdf_b, pdf_n = generate_combined_highlighted_pdf(
                st.session_state.original_pdf_bytes,
                st.session_state.fence_pages,
                st.session_state.uploaded_pdf_name,
                current_session_id
            )
            if pdf_b:
                st.session_state.highlighted_pdf_bytes_for_download = pdf_b
                st.session_state.highlighted_pdf_filename_for_download = pdf_n
            else:
                st.warning(f"Could not generate PDF: {pdf_n}")
        
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.session_state.analysis_halted_due_to_error = True
        print(f"SESSION {current_session_id} ERROR: {e}")
    finally:
        if doc_proc:
            doc_proc.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document.")
    
    # Final summary
    final_summary_text = (
        f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    summary_placeholder.markdown(final_summary_text)
    
    # Download button
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_main"
        )


# ==============================================================================
# Display Previously Processed Results (on rerun)
# ==============================================================================

elif st.session_state.processing_complete:
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    
    final_summary_text_rerun = (
        f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    st.markdown(final_summary_text_rerun)
    
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_rerun"
        )
    
    col_f_res, col_nf_res = st.columns(2)
    with col_f_res:
        st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
    with col_nf_res:
        st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")
    
    def display_page_result_expander(res_data_list, target_column_res):
        for res_data_item in res_data_list:
            with target_column_res:
                exp_title_res = f"Page {res_data_item['page_number']}"
                definitions = res_data_item.get('definitions', [])
                instances = res_data_item.get('instances', [])
                keyword_matches = res_data_item.get('keyword_matches', [])
                detection_method = res_data_item.get('detection_method', 'none')
                fallback_result = res_data_item.get('fallback_result')
                
                if res_data_item.get('fence_found'):
                    reasons_res = []
                    if definitions:
                        reasons_res.append("Definitions")
                    if instances:
                        reasons_res.append("Instances")
                    if keyword_matches and not definitions:
                        reasons_res.append("Keywords")
                    if res_data_item.get('highlight_fence_text_app_setting', True) and \
                       (definitions or instances or keyword_matches or res_data_item.get('measurements')):
                        reasons_res.append("Highlights")
                    if reasons_res:
                        exp_title_res += f" ({' & '.join(reasons_res)})"
                
                with st.expander(exp_title_res, expanded=False):
                    img_col_r, det_col_r = st.columns([2, 1])
                    
                    with img_col_r:
                        # Regenerate images on demand (not stored in session_state)
                        _orig_r, _hl_r = None, None
                        if res_data_item.get('fence_found') and st.session_state.original_pdf_bytes:
                            _defs_hashable = tuple(tuple(sorted(d.items())) for d in definitions) if definitions else ()
                            _inst_hashable = tuple(tuple(sorted(i.items())) for i in instances) if instances else ()
                            _kw_hashable = tuple(tuple(sorted(k.items())) for k in keyword_matches if all(key in k for key in ['x0','y0','x1','y1'])) if keyword_matches else ()
                            _orig_r, _hl_r = get_page_image_on_demand(
                                st.session_state.current_pdf_hash,
                                st.session_state.original_pdf_bytes,
                                res_data_item['page_index_in_original_doc'],
                                _defs_hashable, _inst_hashable, _kw_hashable,
                                res_data_item.get('pdf_width', 792),
                                res_data_item.get('pdf_height', 612),
                                res_data_item.get('highlight_fence_text_app_setting', True),
                            )
                        disp_img_r = _hl_r or _orig_r
                        if disp_img_r:
                            st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}")
                        
                        dl_links_rerun = []
                        if _hl_r:
                            dl_links_rerun.append(get_image_download_link_html(
                                _hl_r,
                                f"page_{res_data_item['page_number']}_hl.png",
                                "DL HL Img"
                            ))
                        if _orig_r:
                            dl_links_rerun.append(get_image_download_link_html(
                                _orig_r,
                                f"page_{res_data_item['page_number']}_orig.png",
                                "DL Orig Img"
                            ))
                        if dl_links_rerun:
                            st.markdown(" ".join(dl_links_rerun), unsafe_allow_html=True)
                    
                    with det_col_r:
                        # Detection method badge
                        if detection_method == "ade":
                            st.success("🎯 ADE Detection")
                        elif detection_method == "llm_confirmed":
                            st.warning("🔍 Keyword + LLM")
                        elif detection_method == "keyword_only":
                            st.warning("🔤 Keyword Match")
                        else:
                            st.info("❌ No Detection")
                        
                        # ADE Stats (compact)
                        st.metric("ADE Chunks", res_data_item.get('chunk_count', 0))
                        col_leg_r, col_fig_r = st.columns(2)
                        with col_leg_r:
                            st.metric("Legend", res_data_item.get('legend_count', 0))
                        with col_fig_r:
                            st.metric("Figure", res_data_item.get('figure_count', 0))
                        
                        if res_data_item.get('text_response'):
                            with st.popover("Analysis Log"):
                                st.markdown(f"_{res_data_item['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                        
                        # Show element details if available
                        el_details = st.session_state.get('element_details', {})
                        if el_details:
                            st.markdown("### 📋 Element Specifications")
                            detail_rows = []
                            seen_kw = set()
                            for d in definitions:
                                kw = d.get('keyword', '').strip()
                                if not kw or kw in seen_kw or d.get('description', '') == 'Indicator Code':
                                    continue
                                seen_kw.add(kw)
                                ind = d.get('indicator', '').strip()
                                cat_label = f"{ind}: {kw}" if ind else kw
                                details = _lookup_element_details(cat_label, el_details)
                                if details and any(v for v in details.values() if v):
                                    detail_rows.append({
                                        'Element': cat_label,
                                        'Height': details.get('height', ''),
                                        'Post Type': details.get('post_type', ''),
                                        'Post Spacing': details.get('post_spacing', ''),
                                        'Material': details.get('material', ''),
                                        'Gauge': details.get('gauge', ''),
                                        'Mesh Size': details.get('mesh_size', ''),
                                        'Detail Page': details.get('detail_page', ''),
                                    })
                            if detail_rows:
                                st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
                                # Full details in expandable section
                                with st.expander("📝 Full Detail Text", expanded=False):
                                    for dr in detail_rows:
                                        elem = dr['Element']
                                        full = _lookup_element_details(elem, el_details).get('full_details', '')
                                        if full:
                                            st.markdown(f"**{elem}:** {full}")
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (Fallback)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                        
                        # Show Measurements
                        measurements = res_data_item.get('measurements')
                        if measurements and (measurements.get('indicator_measurements') or measurements.get('totals', {}).get('total_length_feet', 0) > 0):
                            st.markdown("---")
                            st.markdown("### 📏 Fence Measurements")
                            
                            page_info = measurements.get('page_info', {})
                            scale_factor = page_info.get('scale_factor', 1.0)
                            
                            # Show scale info prominently
                            if page_info.get('scale_detected'):
                                st.success(f"✅ Scale Auto-Detected: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                            else:
                                st.warning("⚠️ Scale not detected - showing raw measurements")
                            
                            # Show proximity-based measurements (primary)
                            prox_totals = measurements.get('proximity_totals', {})
                            if prox_totals.get('total_segments', 0) > 0:
                                st.markdown("#### 🎯 Near Detected Indicators:")
                                col_pts, col_ft = st.columns(2)
                                with col_pts:
                                    st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                                with col_ft:
                                    st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                                
                                # Per-indicator breakdown
                                indicator_meas = measurements.get('indicator_measurements', {})
                                if indicator_meas:
                                    st.markdown("**Per-Indicator:**")
                                    for ind, stats in indicator_meas.items():
                                        pts = stats.get('run_length_pts', 0)
                                        ft = stats.get('run_length_feet', 0)
                                        segs = stats.get('run_segment_count', 0)
                                        count = stats.get('instance_count', 0)
                                        st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                            
                            # Layer breakdown (secondary)
                            if measurements.get('fence_layers'):
                                with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                    totals = measurements.get('totals', {})
                                    st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                    for layer in measurements['fence_layers']:
                                        l_stats = measurements['layer_measurements'].get(layer, {})
                                        segs = l_stats.get('total_segments', 0)
                                        ft = l_stats.get('total_length_feet', 0)
                                        runs = l_stats.get('connected_runs', 0)
                                        st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                            
                            # Dimension line measurements
                            dim_measurements = measurements.get('dimension_measurements', [])
                            if dim_measurements:
                                with st.expander("📐 Dimension Line Measurements", expanded=False):
                                    st.caption(f"Found {len(dim_measurements)} dimension annotations")
                                    for dm in dim_measurements[:10]:
                                        ft = dm.get('actual_ft', 0)
                                        txt = dm.get('measurement_text', '')
                                        st.markdown(f"- **{txt}**: {ft:.1f} ft")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
    
    display_page_result_expander(st.session_state.fence_pages, col_f_res)
    display_page_result_expander(st.session_state.non_fence_pages, col_nf_res)


# ==============================================================================
# Unified Measurement Tool (Auto + Interactive)
# ==============================================================================

if st.session_state.processing_complete and st.session_state.fence_pages and enable_unified_measurement:
    st.markdown("---")
    st.markdown("<h2>📏 Unified Measurement Tool</h2>", unsafe_allow_html=True)
    st.caption("🤖 Auto-detected lines shown in cyan | 👆 Click to select manual lines | ✏️ Draw custom lines")
    
    # Auto-detect and verify scale PER PAGE using LLM
    from utils_vector import verify_scale_with_bar
    
    # Detect scale for each fence page (cached in session_state)
    if 'per_page_scale_info' not in st.session_state:
        st.session_state.per_page_scale_info = {}
    
    # User-drawn lines storage: {page_key: [{'start': (x,y), 'end': (x,y), 'category': cat}, ...]}
    if 'user_drawn_lines' not in st.session_state:
        st.session_state.user_drawn_lines = {}
    
    # Drawing mode per page
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = {}
    
    # Pending point for line drawing (first click of a two-click line)
    if 'pending_line_start' not in st.session_state:
        st.session_state.pending_line_start = {}
    
    # Detect scales for all pages if not already done
    with fitz.open(stream=BytesIO(st.session_state.original_pdf_bytes), filetype="pdf") as doc:
        for fence_page in st.session_state.fence_pages:
            page_num = fence_page['page_number']
            cache_key = f"page_{page_num}"
            if cache_key not in st.session_state.per_page_scale_info:
                try:
                    page_idx = fence_page['page_index_in_original_doc']
                    pdf_page = doc[page_idx]
                    # Use LLM for intelligent scale detection
                    scale_info = verify_scale_with_bar(pdf_page, llm=llm_analysis_instance)
                    st.session_state.per_page_scale_info[cache_key] = scale_info
                except Exception as e:
                    st.session_state.per_page_scale_info[cache_key] = {
                        'success': False, 'verified_scale': None, 'message': str(e)
                    }
    
    # Global settings (min line length only - scale is now per-page)
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        min_line_pts = st.number_input(
            "Min line length (pts)",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            help="Filter out short lines (hatching, text)",
            key="min_line_pts_input"
        )
    with col_g2:
        st.info("📐 Scale detected per page (see each tab)")
    
    # Zoom slider (higher default for better quality)
    zoom_level = st.slider("🔍 Zoom", min_value=600, max_value=2000, value=1200, step=100, 
                           help="Adjust image display width")
    
    # Create tabs for each fence page
    page_tabs = st.tabs([f"Page {p['page_number']}" for p in st.session_state.fence_pages])
    
    # Track line assignments per page: {page_key: {line_idx: category_name}}
    if 'line_assignments' not in st.session_state:
        st.session_state.line_assignments = {}
    
    # Track categories per page: {page_key: {cat_name: {indicator, keyword, color}}}
    if 'page_categories' not in st.session_state:
        st.session_state.page_categories = {}
    
    # Track active category per page
    if 'active_category_per_page' not in st.session_state:
        st.session_state.active_category_per_page = {}
    
    # Category colors for consistent assignment
    CATEGORY_COLORS = [
        (0, 255, 0),      # Green
        (255, 165, 0),    # Orange
        (0, 191, 255),    # Deep sky blue
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
        (255, 105, 180),  # Hot pink
        (173, 255, 47),   # Green yellow
    ]
    
    # Import PIL once outside loop
    from PIL import Image, ImageDraw
    
    # OPTIMIZATION 5: Use st.fragment for partial reruns (only rerun the page content, not entire app)
    @st.fragment
    def render_page_fragment(page_data, zoom_level, min_line_pts):
        """Fragment function for each page - only this reruns on interaction"""
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        page_idx = page_data['page_index_in_original_doc']
        pdf_width = page_data.get('pdf_width', 792)
        pdf_height = page_data.get('pdf_height', 612)
        
        # Extract lines from PDF page (cached)
        lines_cache_key = f"lines_{page_num}_{min_line_pts}"
        if lines_cache_key not in st.session_state:
            with fitz.open(stream=BytesIO(st.session_state.original_pdf_bytes), filetype="pdf") as doc:
                pdf_page = doc[page_idx]
                all_lines = extract_vector_lines(pdf_page)
                filtered_lines = [l for l in all_lines if l.length_pts >= min_line_pts]
                filtered_lines.sort(key=lambda l: l.length_pts, reverse=True)
                st.session_state[lines_cache_key] = filtered_lines
        
        lines = st.session_state.get(lines_cache_key, [])
        
        if not lines:
            st.warning(f"No lines found on this page (min length: {min_line_pts} pts)")
            return
        
        # Initialize line assignments for this page: {line_idx: category_name}
        if page_key not in st.session_state.line_assignments:
            st.session_state.line_assignments[page_key] = {}
        
        # Pre-populate from auto-detected lines by matching coordinates to vector lines
        # This runs once per analysis (keyed by auto_lines count to re-trigger on new analysis)
        unified_page = st.session_state.unified_measurements.get(page_key, {})
        auto_lines_data = unified_page.get('auto_lines', [])
        accepted_auto = unified_page.get('accepted_auto', set())
        auto_sync_key = f"auto_synced_{page_key}_{len(auto_lines_data)}"
        
        if auto_lines_data and accepted_auto and lines and auto_sync_key not in st.session_state:
            import math
            matched = 0
            matched_indices = set()
            for ai in accepted_auto:
                if ai >= len(auto_lines_data):
                    continue
                auto_line = auto_lines_data[ai]
                a_sx, a_sy = auto_line['start']
                a_ex, a_ey = auto_line['end']
                category = auto_line.get('category')
                if not category:
                    continue
                
                # Find closest vector line by endpoint distance
                best_idx = None
                best_dist = float('inf')
                for vi, vline in enumerate(lines):
                    v_sx, v_sy = vline.start
                    v_ex, v_ey = vline.end
                    # Try both orientations (line direction may differ)
                    d1 = math.hypot(a_sx - v_sx, a_sy - v_sy) + math.hypot(a_ex - v_ex, a_ey - v_ey)
                    d2 = math.hypot(a_sx - v_ex, a_sy - v_ey) + math.hypot(a_ex - v_sx, a_ey - v_sy)
                    d = min(d1, d2)
                    if d < best_dist:
                        best_dist = d
                        best_idx = vi
                
                # Coordinates come from same extract_vector_lines, should be near-exact
                if best_idx is not None and best_dist < 2.0:
                    st.session_state.line_assignments[page_key][best_idx] = category
                    matched_indices.add(best_idx)
                    matched += 1
                else:
                    print(f"[AUTO-PREPOP] No match for auto line {ai} (best_dist={best_dist:.2f})")
            
            # Track which vector line indices were auto-matched (for clear button)
            auto_matched_key = f"auto_matched_indices_{page_key}"
            st.session_state[auto_matched_key] = matched_indices
            st.session_state[auto_sync_key] = matched
            print(f"[AUTO-PREPOP] Page {page_num}: matched {matched}/{len(accepted_auto)} auto lines to {len(lines)} vector lines")
        
        # Initialize categories for this page from its definitions
        if page_key not in st.session_state.page_categories:
            categories = {}
            definitions = page_data.get('definitions', [])
            for d in definitions:
                indicator = d.get('indicator', '')
                keyword = d.get('keyword', '')
                if keyword:
                    cat_name = f"{indicator}: {keyword}" if indicator else keyword
                    if cat_name not in categories:
                        color_idx = len(categories)
                        categories[cat_name] = {
                            'indicator': indicator,
                            'keyword': keyword,
                            'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                        }
            st.session_state.page_categories[page_key] = categories
        
        page_categories = st.session_state.page_categories[page_key]
        
        # Initialize active category for this page
        if page_key not in st.session_state.active_category_per_page:
            cats = list(page_categories.keys())
            st.session_state.active_category_per_page[page_key] = cats[0] if cats else None
        
        # Show per-page scale info
        page_scale_info = st.session_state.per_page_scale_info.get(page_key, {})
        page_scale = page_scale_info.get('verified_scale') or page_scale_info.get('text_scale') or 360.0
        
        scale_col1, scale_col2 = st.columns([2, 1])
        with scale_col1:
            page_scale_input = st.number_input(
                f"Scale (Page {page_num})",
                min_value=1.0,
                max_value=1200.0,
                value=float(page_scale),
                step=12.0,
                help=f"1\" = {page_scale/12:.1f}' actual",
                key=f"scale_input_{page_num}"
            )
        with scale_col2:
            if page_scale_info.get('success'):
                confidence = page_scale_info.get('confidence', 'low')
                scale_text = page_scale_info.get('scale_text', '')
                display_text = f"✓ {scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'"
                if confidence == 'high':
                    st.success(display_text)
                elif confidence == 'medium':
                    st.warning(f"⚠ {scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'")
                else:
                    st.info(scale_text if scale_text else f"1\"={page_scale/12:.0f}'")
            else:
                st.warning("Not detected")
        
        # Show scale detection details
        with st.expander("🔍 Scale Detection Details", expanded=False):
            # Page size info
            page_size = page_scale_info.get('page_size', {})
            if page_size:
                size_str = f"{page_size.get('width_inches', 0):.1f}\" x {page_size.get('height_inches', 0):.1f}\""
                detected = page_size.get('detected_size', 'Unknown')
                st.markdown(f"**Page size:** {size_str} ({detected})")
            
            # Scale detection
            method = page_scale_info.get('method', 'unknown')
            st.markdown(f"**Detection method:** {method}")
            scale_text = page_scale_info.get('scale_text', '')
            st.markdown(f"**Detected scale text:** {scale_text if scale_text else 'None'}")
            st.markdown(f"**Confidence:** {page_scale_info.get('confidence', 'N/A')}")
            st.markdown(f"**Message:** {page_scale_info.get('message', 'N/A')}")
            if page_scale_info.get('verified_scale'):
                scale_val = page_scale_info['verified_scale']
                st.markdown(f"**Scale value:** 1\" = {scale_val/12:.0f}' ({scale_val} inches)")
            
            # Debug: show raw LLM response
            raw = page_scale_info.get('raw_response', '')
            if raw:
                st.markdown("**LLM Response:**")
                st.code(raw[:500], language=None)
            
            # Debug: show extracted text sample
            extracted = page_scale_info.get('extracted_text_sample', '')
            if extracted:
                st.markdown("**Extracted PDF Text (first 1500 chars):**")
                st.code(extracted, language=None)
        
        # =====================================================================
        # AUTO-DETECTED MEASUREMENTS SECTION
        # =====================================================================
        unified_page_data = st.session_state.unified_measurements.get(page_key, {})
        auto_lines = unified_page_data.get('auto_lines', [])
        
        if auto_lines:
            accepted_auto = unified_page_data.get('accepted_auto', set())
            accepted_count = len(accepted_auto)
            accepted_ft = sum(auto_lines[i].get('length_feet', 0) for i in accepted_auto if i < len(auto_lines))
            
            # Check how many auto lines are currently matched in line_assignments
            auto_matched_key = f"auto_matched_indices_{page_key}"
            auto_matched_indices = st.session_state.get(auto_matched_key, set())
            currently_assigned = sum(1 for idx in auto_matched_indices if idx in st.session_state.line_assignments.get(page_key, {}))
            
            st.markdown("#### 🤖 Auto-Detected Fence Lines (Pre-Selected)")
            
            auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
            with auto_col1:
                if currently_assigned > 0:
                    st.success(f"✓ {currently_assigned} lines matched & selected ({accepted_ft:.1f} ft)")
                elif accepted_count > 0:
                    st.warning(f"{accepted_count} auto lines detected but not yet synced to selections")
                else:
                    st.info("No auto-detected lines")
            with auto_col2:
                if st.button("🔄 Re-sync Auto", key=f"resync_auto_{page_num}", help="Re-match auto-detected lines to selectable vector lines"):
                    # Clear old sync keys to force re-matching
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"auto_synced_{page_key}")]
                    for k in keys_to_remove:
                        del st.session_state[k]
                    if auto_matched_key in st.session_state:
                        del st.session_state[auto_matched_key]
                    st.rerun(scope="fragment")
            with auto_col3:
                if st.button("❌ Clear Auto", key=f"clear_auto_{page_num}", help="Remove all auto-detected lines from selection"):
                    # Remove auto-matched assignments from line_assignments
                    page_assigns = st.session_state.line_assignments.get(page_key, {})
                    for idx in auto_matched_indices:
                        page_assigns.pop(idx, None)
                    st.session_state.unified_measurements[page_key]['accepted_auto'] = set()
                    st.session_state[auto_matched_key] = set()
                    # Clear sync keys
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"auto_synced_{page_key}")]
                    for k in keys_to_remove:
                        del st.session_state[k]
                    st.rerun(scope="fragment")
            
            # Show category breakdown for auto lines
            auto_by_cat = {}
            for i in accepted_auto:
                if i < len(auto_lines):
                    cat = auto_lines[i].get('category', 'Uncategorized')
                    if cat not in auto_by_cat:
                        auto_by_cat[cat] = {'count': 0, 'feet': 0}
                    auto_by_cat[cat]['count'] += 1
                    auto_by_cat[cat]['feet'] += auto_lines[i].get('length_feet', 0)
            
            if auto_by_cat:
                for cat, data in auto_by_cat.items():
                    cat_color = page_categories.get(cat, {}).get('color', (0, 255, 255))
                    st.markdown(f"<span style='color: rgb{cat_color};'>●</span> **{cat}**: {data['count']} lines, {data['feet']:.1f} ft", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category selector for this page
        st.markdown("#### 🏷️ Fence Categories (This Page)")
        cat_col1, cat_col2 = st.columns([3, 1])
        with cat_col1:
            category_options = list(page_categories.keys())
            if category_options:
                current_active = st.session_state.active_category_per_page.get(page_key)
                active_cat = st.selectbox(
                    "Assign lines to:",
                    options=category_options,
                    index=category_options.index(current_active) if current_active in category_options else 0,
                    key=f"category_selector_{page_num}"
                )
                st.session_state.active_category_per_page[page_key] = active_cat
                if active_cat:
                    color = page_categories[active_cat]['color']
                    st.markdown(f"<span style='color: rgb{color}; font-size: 20px;'>●</span> Click lines to assign", unsafe_allow_html=True)
            else:
                st.info("No fence categories detected on this page.")
        
        with cat_col2:
            with st.popover("➕ Add"):
                new_cat_name = st.text_input("Category name:", key=f"new_cat_{page_num}")
                if st.button("Add", key=f"add_cat_btn_{page_num}") and new_cat_name:
                    if new_cat_name not in st.session_state.page_categories[page_key]:
                        color_idx = len(st.session_state.page_categories[page_key])
                        st.session_state.page_categories[page_key][new_cat_name] = {
                            'indicator': '',
                            'keyword': new_cat_name,
                            'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                        }
                        st.session_state.active_category_per_page[page_key] = new_cat_name
                        st.rerun(scope="fragment")
        
        # Mode toggle: Select existing lines vs Draw custom lines
        mode_col1, mode_col2 = st.columns([1, 1])
        with mode_col1:
            if page_key not in st.session_state.drawing_mode:
                st.session_state.drawing_mode[page_key] = "select"
            
            drawing_mode = st.radio(
                "Mode:",
                options=["select", "draw"],
                format_func=lambda x: "📍 Select Lines" if x == "select" else "✏️ Draw Lines",
                horizontal=True,
                key=f"mode_{page_num}"
            )
            st.session_state.drawing_mode[page_key] = drawing_mode
        
        with mode_col2:
            if drawing_mode == "draw":
                st.caption("Draw lines on the image. They will be assigned to the active category.")
        
        # Cache line stats (keyed by page + min_line_pts + scale)
        # Use the page-specific scale input
        effective_scale = page_scale_input
        line_stats_key = f"line_stats_{page_num}_{min_line_pts}_{effective_scale}"
        if line_stats_key not in st.session_state:
            # Evict old line_stats for this page (accumulate on every scale change)
            for k in [k for k in list(st.session_state.keys())
                      if k.startswith(f"line_stats_{page_num}_") and k != line_stats_key]:
                del st.session_state[k]
            stats = []
            for i, line in enumerate(lines):
                length_inches = line.length_pts / 72.0
                length_feet = (length_inches * effective_scale) / 12.0
                stats.append({
                    'index': i,
                    'length_pts': line.length_pts,
                    'length_feet': length_feet,
                    'layer': line.layer or 'default',
                    'start': line.start,
                    'end': line.end
                })
            st.session_state[line_stats_key] = stats
        line_stats = st.session_state[line_stats_key]
        
        # Get base image on demand (regenerate instead of reading from session_state)
        base_img_bytes = page_data.get('highlighted_image_bytes') or page_data.get('original_image_bytes')
        if not base_img_bytes and st.session_state.original_pdf_bytes:
            # Regenerate on demand
            _defs = page_data.get('definitions', [])
            _insts = page_data.get('instances', [])
            _kws = page_data.get('keyword_matches', [])
            _defs_h = tuple(tuple(sorted(d.items())) for d in _defs) if _defs else ()
            _insts_h = tuple(tuple(sorted(i.items())) for i in _insts) if _insts else ()
            _kws_h = tuple(tuple(sorted(k.items())) for k in _kws if all(key in k for key in ['x0','y0','x1','y1'])) if _kws else ()
            _orig_img, _hl_img = get_page_image_on_demand(
                st.session_state.current_pdf_hash,
                st.session_state.original_pdf_bytes,
                page_idx, _defs_h, _insts_h, _kws_h,
                pdf_width, pdf_height,
                page_data.get('highlight_fence_text_app_setting', True),
            )
            base_img_bytes = _hl_img or _orig_img
        
        if base_img_bytes:
            
            # OPTIMIZATION 1: Cache resized base image (keyed by page + zoom)
            # Evict old zoom levels for this page to bound memory
            base_img_cache_key = f"base_img_{page_num}_{zoom_level}"
            if base_img_cache_key not in st.session_state:
                # Evict previous zoom level caches for this page
                for k in [k for k in list(st.session_state.keys())
                          if (k.startswith(f"base_img_{page_num}_") or
                              k.startswith(f"base_img_size_{page_num}_") or
                              k.startswith(f"drawn_img_{page_num}_"))
                          and k != base_img_cache_key]:
                    del st.session_state[k]
                
                base_img = Image.open(BytesIO(base_img_bytes)).convert('RGB')
                orig_width, orig_height = base_img.size
                ratio = zoom_level / orig_width
                new_width = zoom_level
                new_height = int(orig_height * ratio)
                # Use LANCZOS for high quality resize
                base_img = base_img.resize((new_width, new_height), Image.LANCZOS)
                # Store as compressed PNG bytes (~10x less memory than raw PIL)
                _buf = BytesIO()
                base_img.save(_buf, format='PNG', optimize=True)
                st.session_state[base_img_cache_key] = _buf.getvalue()
                st.session_state[f"base_img_size_{page_num}_{zoom_level}"] = (new_width, new_height)
                st.session_state[f"orig_img_size_{page_num}"] = (orig_width, orig_height)
                del base_img, _buf
            
            # Decompress on demand (~5ms, negligible vs rendering)
            base_img_cached = Image.open(BytesIO(st.session_state[base_img_cache_key]))
            img_width, img_height = st.session_state[f"base_img_size_{page_num}_{zoom_level}"]
            
            # Scale factors from PDF to image coordinates
            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            
            line_assignments = st.session_state.line_assignments.get(page_key, {})
            
            # OPTIMIZATION 2: Cache drawn image with assignments
            # Create a hashable key from assignment state
            assignment_tuple = tuple(sorted(line_assignments.items()))
            drawn_img_cache_key = f"drawn_img_{page_num}_{zoom_level}_{hash(assignment_tuple)}"
            
            # Auto lines are now part of line_assignments, no separate cache key needed
            
            if drawn_img_cache_key not in st.session_state:
                # Evict old drawn images for this page (they leak on every assignment change)
                for k in [k for k in list(st.session_state.keys())
                          if k.startswith(f"drawn_img_{page_num}_") and k != drawn_img_cache_key]:
                    del st.session_state[k]
                # Copy base image and draw assignments
                display_img = base_img_cached.copy()
                draw = ImageDraw.Draw(display_img)
                
                # First pass: Draw ALL selectable lines with subtle color (unassigned)
                for i, ls in enumerate(line_stats):
                    x0 = ls['start'][0] * scale_x
                    y0 = ls['start'][1] * scale_y
                    x1 = ls['end'][0] * scale_x
                    y1 = ls['end'][1] * scale_y
                    # Subtle gray-blue for unassigned lines
                    if i not in line_assignments:
                        draw.line([(x0, y0), (x1, y1)], fill=(150, 180, 200), width=1)
                
                # Second pass: Draw ASSIGNED lines with category colors (auto-matched + manually selected)
                for i, ls in enumerate(line_stats):
                    if i in line_assignments:
                        category = line_assignments[i]
                        cat_info = page_categories.get(category, {})
                        color = cat_info.get('color', (0, 255, 0))
                        
                        x0 = ls['start'][0] * scale_x
                        y0 = ls['start'][1] * scale_y
                        x1 = ls['end'][0] * scale_x
                        y1 = ls['end'][1] * scale_y
                        # Draw with category color
                        draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=6)  # White outline
                        draw.line([(x0, y0), (x1, y1)], fill=color, width=4)
                        draw.ellipse([(x0-5, y0-5), (x0+5, y0+5)], fill=color)
                        draw.ellipse([(x1-5, y1-5), (x1+5, y1+5)], fill=color)
                
                st.session_state[drawn_img_cache_key] = display_img
            
            display_img = st.session_state[drawn_img_cache_key]
            
            # Display clickable image and info side by side
            col_img, col_info = st.columns([3, 1])
            
            with col_img:
                # Initialize user-drawn lines for this page
                if page_key not in st.session_state.user_drawn_lines:
                    st.session_state.user_drawn_lines[page_key] = []
                
                if drawing_mode == "draw":
                    # DRAW MODE: Click two points to create a line
                    # Show pending start point if exists
                    pending_start = st.session_state.pending_line_start.get(page_key)
                    
                    # Draw user lines and pending point on image
                    draw_img = display_img.copy()
                    draw_overlay = ImageDraw.Draw(draw_img)
                    
                    # Draw existing user-drawn lines
                    user_lines = st.session_state.user_drawn_lines.get(page_key, [])
                    for ul in user_lines:
                        cat = ul.get('category')
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        x0 = ul['start'][0] * scale_x
                        y0 = ul['start'][1] * scale_y
                        x1 = ul['end'][0] * scale_x
                        y1 = ul['end'][1] * scale_y
                        draw_overlay.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=5)
                        draw_overlay.line([(x0, y0), (x1, y1)], fill=color, width=3)
                        draw_overlay.ellipse([(x0-4, y0-4), (x0+4, y0+4)], fill=color)
                        draw_overlay.ellipse([(x1-4, y1-4), (x1+4, y1+4)], fill=color)
                    
                    # Draw pending start point
                    if pending_start:
                        px, py = pending_start
                        img_px = px * scale_x
                        img_py = py * scale_y
                        draw_overlay.ellipse([(img_px-8, img_py-8), (img_px+8, img_py+8)], fill=(255, 255, 0), outline=(0, 0, 0))
                    
                    click_key = f"draw_click_{page_num}"
                    if click_key not in st.session_state:
                        st.session_state[click_key] = None
                    
                    click_result = streamlit_image_coordinates(
                        draw_img,
                        key=f"draw_img_{page_num}"
                    )
                    
                    if click_result is not None:
                        current_click = (click_result.get('x', 0), click_result.get('y', 0))
                        
                        if current_click != st.session_state[click_key]:
                            st.session_state[click_key] = current_click
                            click_x, click_y = current_click
                            pdf_click_x = click_x / scale_x
                            pdf_click_y = click_y / scale_y
                            
                            if pending_start is None:
                                # First click - set start point
                                st.session_state.pending_line_start[page_key] = (pdf_click_x, pdf_click_y)
                                st.rerun(scope="fragment")
                            else:
                                # Second click - create line
                                active_cat = st.session_state.active_category_per_page.get(page_key)
                                start_x, start_y = pending_start
                                end_x, end_y = pdf_click_x, pdf_click_y
                                
                                length_pts = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                                length_inches = length_pts / 72.0
                                length_feet = (length_inches * effective_scale) / 12.0
                                
                                new_line = {
                                    'start': (start_x, start_y),
                                    'end': (end_x, end_y),
                                    'category': active_cat,
                                    'length_pts': length_pts,
                                    'length_feet': length_feet
                                }
                                
                                if page_key not in st.session_state.user_drawn_lines:
                                    st.session_state.user_drawn_lines[page_key] = []
                                st.session_state.user_drawn_lines[page_key].append(new_line)
                                
                                # Clear pending start
                                st.session_state.pending_line_start[page_key] = None
                                st.rerun(scope="fragment")
                
                else:
                    # SELECT MODE: Use clickable image
                    click_key = f"last_click_{page_num}"
                    if click_key not in st.session_state:
                        st.session_state[click_key] = None
                    
                    click_result = streamlit_image_coordinates(
                        display_img,
                        key=f"click_img_{page_num}"
                    )
                    
                    # Handle click - find nearest line
                    if click_result is not None:
                        current_click = (click_result.get('x', 0), click_result.get('y', 0))
                        
                        if current_click != st.session_state[click_key]:
                            st.session_state[click_key] = current_click
                            click_x, click_y = current_click
                            pdf_click_x = click_x / scale_x
                            pdf_click_y = click_y / scale_y
                            
                            def point_to_line_distance(px, py, x0, y0, x1, y1):
                                dx = x1 - x0
                                dy = y1 - y0
                                if dx == 0 and dy == 0:
                                    return ((px - x0)**2 + (py - y0)**2)**0.5
                                t = max(0, min(1, ((px - x0)*dx + (py - y0)*dy) / (dx*dx + dy*dy)))
                                proj_x = x0 + t * dx
                                proj_y = y0 + t * dy
                                return ((px - proj_x)**2 + (py - proj_y)**2)**0.5
                            
                            min_dist = float('inf')
                            nearest_idx = -1
                            for i, ls in enumerate(line_stats):
                                dist = point_to_line_distance(
                                    pdf_click_x, pdf_click_y,
                                    ls['start'][0], ls['start'][1],
                                    ls['end'][0], ls['end'][1]
                                )
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_idx = i
                            
                            if nearest_idx >= 0 and min_dist < 30:
                                active_cat = st.session_state.active_category_per_page.get(page_key)
                                current_assignment = st.session_state.line_assignments[page_key].get(nearest_idx)
                                
                                if current_assignment == active_cat:
                                    # Only delete if key exists
                                    if nearest_idx in st.session_state.line_assignments[page_key]:
                                        del st.session_state.line_assignments[page_key][nearest_idx]
                                else:
                                    if active_cat:
                                        st.session_state.line_assignments[page_key][nearest_idx] = active_cat
                                st.rerun(scope="fragment")
            
            with col_info:
                st.markdown(f"**{len(lines)} detected lines**")
                if drawing_mode == "select":
                    st.caption("Click to assign to category")
                else:
                    pending = st.session_state.pending_line_start.get(page_key)
                    if pending:
                        st.warning("Click end point")
                        if st.button("Cancel", key=f"cancel_draw_{page_num}"):
                            st.session_state.pending_line_start[page_key] = None
                            st.rerun(scope="fragment")
                    else:
                        st.caption("Click start point")
                
                # Clear buttons
                clear_col1, clear_col2 = st.columns(2)
                with clear_col1:
                    if st.button("Clear Sel", key=f"clear_sel_{page_num}"):
                        st.session_state.line_assignments[page_key] = {}
                        st.rerun(scope="fragment")
                with clear_col2:
                    if st.button("Clear Drawn", key=f"clear_drawn_{page_num}"):
                        st.session_state.user_drawn_lines[page_key] = []
                        st.rerun(scope="fragment")
                
                # Show selected lines (from existing)
                line_assignments = st.session_state.line_assignments.get(page_key, {})
                if line_assignments:
                    by_category = {}
                    for idx, cat in line_assignments.items():
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(idx)
                    
                    st.markdown(f"**Selected: {len(line_assignments)}**")
                    for cat, indices in by_category.items():
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        cat_total = sum(line_stats[i]['length_feet'] for i in indices if i < len(line_stats))
                        st.markdown(f"<span style='color: rgb{color};'>●</span> **{cat}**: {len(indices)} lines, {cat_total:.1f} ft", unsafe_allow_html=True)
                
                # Show user-drawn lines
                user_lines = st.session_state.user_drawn_lines.get(page_key, [])
                if user_lines:
                    st.markdown("---")
                    st.markdown(f"**Drawn: {len(user_lines)}**")
                    # Group by category
                    drawn_by_cat = {}
                    for ul in user_lines:
                        cat = ul.get('category', 'Uncategorized')
                        if cat not in drawn_by_cat:
                            drawn_by_cat[cat] = []
                        drawn_by_cat[cat].append(ul)
                    
                    for cat, cat_lines in drawn_by_cat.items():
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        cat_total = sum(ul['length_feet'] for ul in cat_lines)
                        st.markdown(f"<span style='color: rgb{color};'>●</span> **{cat}**: {len(cat_lines)} drawn, {cat_total:.1f} ft", unsafe_allow_html=True)
        else:
            st.warning("Image not available")
    
    # Render each page tab using the fragment
    for tab_idx, (tab, page_data) in enumerate(zip(page_tabs, st.session_state.fence_pages)):
        with tab:
            render_page_fragment(page_data, zoom_level, min_line_pts)
    
    # Overall summary across all pages - grouped by category
    st.markdown("---")
    st.markdown("### 📊 Overall Summary")
    
    # Aggregate by category across all pages (auto + selected + drawn lines)
    category_totals = {}  # {category: {'auto': count, 'lines': count, 'feet': total, 'drawn': count}}
    grand_total_feet = 0
    grand_total_lines = 0
    
    for page_data in st.session_state.fence_pages:
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        lines_cache_key = f"lines_{page_num}_{min_line_pts}"
        
        # Get per-page scale
        page_scale_info = st.session_state.per_page_scale_info.get(page_key, {})
        page_scale = page_scale_info.get('verified_scale') or page_scale_info.get('text_scale') or 360.0
        
        # Selected lines from PDF (includes auto-matched + manually selected)
        # Use auto_matched_indices to distinguish auto vs manual
        auto_matched = st.session_state.get(f"auto_matched_indices_{page_key}", set())
        lines = st.session_state.get(lines_cache_key, [])
        line_assignments = st.session_state.line_assignments.get(page_key, {})
        for i, category in line_assignments.items():
            if i < len(lines):
                line = lines[i]
                length_inches = line.length_pts / 72.0
                length_feet = (length_inches * page_scale) / 12.0
                
                if category not in category_totals:
                    category_totals[category] = {'auto': 0, 'lines': 0, 'feet': 0, 'drawn': 0}
                if i in auto_matched:
                    category_totals[category]['auto'] += 1
                else:
                    category_totals[category]['lines'] += 1
                category_totals[category]['feet'] += length_feet
                
                grand_total_feet += length_feet
                grand_total_lines += 1
        
        # User-drawn lines
        user_lines = st.session_state.user_drawn_lines.get(page_key, [])
        for ul in user_lines:
            category = ul.get('category', 'Uncategorized')
            length_feet = ul.get('length_feet', 0)
            
            if category not in category_totals:
                category_totals[category] = {'auto': 0, 'lines': 0, 'feet': 0, 'drawn': 0}
            category_totals[category]['drawn'] += 1
            category_totals[category]['feet'] += length_feet
            
            grand_total_feet += length_feet
            grand_total_lines += 1
    
    if grand_total_lines > 0:
        # Show per-category breakdown
        st.markdown("#### By Category")
        for cat, totals in category_totals.items():
            # Find color from any page that has this category
            color = (0, 255, 0)  # default
            for pk, pc in st.session_state.page_categories.items():
                if cat in pc:
                    color = pc[cat].get('color', (0, 255, 0))
                    break
            col_cat, col_lines, col_feet = st.columns([3, 1, 1])
            with col_cat:
                st.markdown(f"<span style='color: rgb{color}; font-size: 18px;'>●</span> **{cat}**", unsafe_allow_html=True)
            with col_lines:
                auto = totals.get('auto', 0)
                selected = totals['lines']
                drawn = totals.get('drawn', 0)
                parts = []
                if auto:
                    parts.append(f"🤖{auto}")
                if selected:
                    parts.append(f"👆{selected}")
                if drawn:
                    parts.append(f"✏️{drawn}")
                st.markdown(", ".join(parts) if parts else "0")
            with col_feet:
                st.metric("Length", f"{totals['feet']:.1f} ft", label_visibility="collapsed")
        
        # Grand total
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Lines", grand_total_lines)
        with col_s2:
            st.metric("**Grand Total**", f"{grand_total_feet:.1f} ft")
        with col_s3:
            pages_with_assign = sum(1 for p in st.session_state.fence_pages 
                               if st.session_state.line_assignments.get(f"page_{p['page_number']}", {}))
            st.metric("Pages", pages_with_assign)
        
        if st.button("🗑️ Clear All Assignments", key="clear_all_selections"):
            st.session_state.line_assignments = {}
    else:
        st.info("Click lines in the page tabs above and assign them to categories to calculate totals.")
    
    # Element Specifications Summary (cross-page details)
    el_details = st.session_state.get('element_details', {})
    if el_details:
        st.markdown("---")
        st.markdown("#### 📋 Element Specifications (Cross-Page Details)")
        spec_rows = []
        for elem_name, details in el_details.items():
            if any(v for v in details.values() if v):
                spec_rows.append({
                    'Element': elem_name,
                    'Height': details.get('height', ''),
                    'Post Type': details.get('post_type', ''),
                    'Post Spacing': details.get('post_spacing', ''),
                    'Material': details.get('material', ''),
                    'Gauge': details.get('gauge', ''),
                    'Mesh Size': details.get('mesh_size', ''),
                    'Foundation': details.get('foundation', ''),
                    'Gate Info': details.get('gate_info', ''),
                    'Detail Page': details.get('detail_page', ''),
                })
        if spec_rows:
            st.dataframe(pd.DataFrame(spec_rows), hide_index=True, use_container_width=True)
            with st.expander("📝 Full Detail Text per Element", expanded=False):
                for elem_name, details in el_details.items():
                    full = details.get('full_details', '')
                    notes = details.get('notes', '')
                    if full or notes:
                        st.markdown(f"**{elem_name}:**")
                        if full:
                            st.markdown(f"  {full}")
                        if notes:
                            st.markdown(f"  *Notes: {notes}*")
    
    # Download section - always show when there are fence pages
    st.markdown("---")
    st.markdown("#### 📥 Downloads")
    
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        # Generate measurement PDF
        pdf_bytes, pdf_name = generate_measurement_pdf(
            st.session_state.original_pdf_bytes,
            st.session_state.fence_pages,
            st.session_state.line_assignments,
            st.session_state.user_drawn_lines,
            st.session_state.page_categories,
            st.session_state,
            min_line_pts,
            st.session_state.uploaded_pdf_name
        )
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF with Measurements",
                pdf_bytes,
                pdf_name,
                "application/pdf",
                key="dl_measurement_pdf"
            )
        else:
            st.error("Error generating PDF")
    
    with dl_col2:
        # Generate spreadsheet
        xlsx_data = generate_measurement_spreadsheet(
            st.session_state.fence_pages,
            st.session_state.line_assignments,
            st.session_state.user_drawn_lines,
            st.session_state.page_categories,
            st.session_state,
            st.session_state.per_page_scale_info,
            min_line_pts
        )
        base_name = os.path.splitext(st.session_state.uploaded_pdf_name)[0]
        st.download_button(
            "📊 Download Measurements Excel",
            xlsx_data,
            f"{base_name}_measurements.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_measurement_xlsx"
        )


# ==============================================================================
# Fallback Messages
# ==============================================================================

elif not st.session_state.original_pdf_bytes:
    st.info("Upload a PDF to begin analysis.")
elif not (openai_key and llm_analysis_instance):
    st.error("OpenAI models not initialized. Check API key.")
elif not ade_key:
    st.error("LandingAI API key required for ADE analysis.")
elif st.session_state.analysis_halted_due_to_error:
    st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>ADE Fence Detector App</p>", unsafe_allow_html=True)
