"""ADE Fence Detector — Streamlit Frontend.

Submits analysis jobs to the FastAPI backend (api_server.py) and renders
results. All heavy processing happens in background workers.
"""

import json
import os
import time
from io import BytesIO
from pathlib import Path

import fitz
import pandas as pd
import requests
import streamlit as st

from auth import get_session_id, get_user_id, get_user_email, require_auth, render_auth_widget
from config import cfg
from exports import generate_measurement_pdf, generate_measurement_spreadsheet
from state import init_session_state, get_view, set_view, load_job_results, ViewState
import job_registry

# Page config
st.set_page_config(page_title="ADE Fence Detector", layout="wide")

# Health probe (before auth)
try:
    if st.query_params.get("health"):
        st.write("ok")
        st.stop()
except Exception:
    pass

st.markdown("<h1 style='text-align:center;'>ADE Fence Detection in Engineering Drawings</h1>",
            unsafe_allow_html=True)

# Auth gate
require_auth()

# Session init
session_id = get_session_id()
user_id = get_user_id()
init_session_state()

API_URL = cfg.API_SERVER_URL
DISPLAY_IMAGE_DPI = 150


def _api_headers() -> dict:
    return {"X-User-Id": user_id}


def _api_get(path: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{path}", headers=_api_headers(), timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _api_post_file(path: str, pdf_bytes: bytes, filename: str, config: dict) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}{path}",
            headers=_api_headers(),
            files={"pdf": (filename, pdf_bytes, "application/pdf")},
            data={"config": json.dumps(config)},
            timeout=30,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _render_page_image(pdf_path: str, page_idx: int, dpi: int = 150) -> bytes | None:
    """Render a page from the PDF as a PNG image."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    render_auth_widget()
    st.markdown("---")

    # Toggles — matching original monolith sidebar exactly
    use_ade = st.toggle("Use ADE (LandingAI)", value=True, key="use_ade_toggle")
    enable_measurement = st.toggle("Unified Measurements", value=True, key="unified_measurement_toggle",
                                   help="Auto-detect fence lines and interactively select/draw additional lines")
    highlight_text = st.toggle("Highlight text & indicators", value=True, key="highlight_toggle")
    enable_nonlayer = st.toggle("Non-layer suggestions", value=False, key="nonlayer_suggestions_toggle",
                                help="Show auto-detected suggestions even when no fence layers found (less reliable)")
    low_dpi_mode = st.toggle("Low-DPI preview (faster)", value=False, key="low_dpi_toggle",
                             help="Render page previews at 110 dpi instead of 150. Exported PDFs unaffected.")
    if low_dpi_mode:
        DISPLAY_IMAGE_DPI = 110

    # Fence Keywords
    st.markdown("---")
    st.subheader("Fence Keywords")
    kw_str = st.text_area(
        "Custom keywords (one per line):",
        "\n".join(st.session_state.fence_keywords_app),
        height=120,
        key="kw_area",
    )
    if st.button("Update Keywords", key="update_kw"):
        st.session_state.fence_keywords_app = [
            k.strip().lower() for k in kw_str.split("\n") if k.strip()
        ]
        st.rerun()

    # My Jobs sidebar
    st.markdown("---")
    st.subheader("My Jobs")

    jobs_data = _api_get("/api/jobs")
    jobs = jobs_data.get("jobs", []) if jobs_data else []

    if not jobs:
        st.caption("No jobs yet. Upload a PDF to start.")

    _any_active = False
    for job in jobs[:10]:
        status = job.get("status", "?")
        fname = job.get("filename", "unknown.pdf")
        if len(fname) > 25:
            fname = fname[:22] + "..."

        age_s = max(0, int(time.time()) - job.get("created_at", 0))
        if age_s < 3600:
            age_str = f"{age_s // 60}m ago"
        elif age_s < 86400:
            age_str = f"{age_s // 3600}h ago"
        else:
            age_str = f"{age_s // 86400}d ago"

        icons = {"queued": "hourglass_flowing_sand", "running": "arrows_counterclockwise",
                 "completed": "white_check_mark", "failed": "x", "cancelled": "no_entry_sign"}
        icon = icons.get(status, "question")

        prog_text = ""
        if status in ("running", "queued"):
            _any_active = True
            prog = job.get("progress")
            if prog:
                prog_text = f" {prog.get('pct', 0)}%"
            elif status == "queued":
                qp = job.get("queue_position")
                if qp:
                    prog_text = f" #{qp} in queue"

        pages_text = ""
        if job.get("total_pages"):
            pages_text = f" | {job['total_pages']}p"
            if job.get("fence_count") is not None:
                pages_text += f", {job['fence_count']} fence"

        col1, col2 = st.columns([4, 1])
        with col1:
            label = f":{icon}: **{fname}**{pages_text}\n\n`{status}{prog_text}` | {age_str}"
            if st.button(label, key=f"job_{job['job_id'][:8]}", use_container_width=True):
                st.session_state._selected_job_id = job["job_id"]
                st.rerun()

        with col2:
            if status == "completed":
                hl_path = job_registry.get_highlighted_pdf_path(job["job_id"])
                if hl_path:
                    try:
                        st.download_button(
                            "PDF", hl_path.read_bytes(),
                            file_name=f"fence_{fname}",
                            mime="application/pdf",
                            key=f"dl_{job['job_id'][:8]}",
                        )
                    except Exception:
                        pass
            elif status in ("queued", "running"):
                if st.button("Cancel", key=f"cancel_{job['job_id'][:8]}"):
                    try:
                        requests.delete(f"{API_URL}/api/jobs/{job['job_id']}",
                                       headers=_api_headers(), timeout=5)
                    except Exception:
                        pass
                    st.rerun()
            elif status == "cancelled":
                if st.button("X", key=f"del_{job['job_id'][:8]}"):
                    try:
                        requests.delete(f"{API_URL}/api/jobs/{job['job_id']}",
                                       headers=_api_headers(), timeout=5)
                    except Exception:
                        pass
                    st.rerun()

    if _any_active:
        st.caption("_Refreshing in 5s..._")
        time.sleep(5)
        st.rerun()


# ==============================================================================
# Main Area
# ==============================================================================

# PDF Upload
st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload an Engineering PDF",
    type=["pdf"],
    key=f"pdf_upload_{st.session_state.get('uploader_counter', 0)}",
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_name = uploaded_file.name

    st.success(f"Uploaded: **{pdf_name}** ({len(pdf_bytes) / 1024 / 1024:.1f} MB)")

    if st.button("Analyze PDF", type="primary", key="analyze_btn"):
        with st.spinner("Submitting job..."):
            config_payload = {
                "analysis_model": cfg.ANALYSIS_MODEL,
                "classifier_model": cfg.CLASSIFIER_MODEL,
                "fence_keywords": st.session_state.fence_keywords_app,
                "use_ade": st.session_state.get("use_ade_toggle", True),
                "highlight_fence_text": st.session_state.get("highlight_toggle", True),
                "enable_unified_measurement": st.session_state.get("unified_measurement_toggle", True),
            }
            result = _api_post_file("/api/jobs", pdf_bytes, pdf_name, config_payload)
            if result and result.get("job_id"):
                st.session_state._selected_job_id = result["job_id"]
                st.session_state.uploader_counter = st.session_state.get("uploader_counter", 0) + 1
                st.success(f"Job submitted! ID: `{result['job_id'][:8]}`")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to submit job. Is the API server running?")


# ==============================================================================
# Results rendering
# ==============================================================================

def _show_results(results: dict, job: dict):
    """Render full analysis results for a completed job."""
    fence_pages = results.get("fence_pages", [])
    non_fence_pages = results.get("non_fence_pages", [])
    element_details = results.get("element_details", {})
    total_pages = results.get("total_pages", 0)
    timings = results.get("timings", {})
    per_page_scale = results.get("per_page_scale_info", {})
    per_page_measurements = results.get("unified_measurements", {})
    page_categories = results.get("page_categories", {})
    pdf_path = job.get("pdf_path")
    job_id = job.get("job_id", "")

    # Summary header
    summary_text = (
        f"### Final Summary (Completed)\n"
        f"- Processed: {total_pages}/{total_pages}\n"
        f"- Fence: {len(fence_pages)}\n"
        f"- Non-Fence: {len(non_fence_pages)}"
    )
    st.markdown(summary_text)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", total_pages)
    col2.metric("Fence Pages", len(fence_pages))
    col3.metric("Non-Fence Pages", len(non_fence_pages))
    if timings.get("total"):
        col4.metric("Analysis Time", f"{timings['total']:.0f}s")

    # Highlighted PDF download
    hl_path = job_registry.get_highlighted_pdf_path(job_id)
    if hl_path:
        st.download_button(
            "Download Highlighted Fence Pages (PDF)",
            hl_path.read_bytes(),
            file_name=f"fence_{job.get('filename', 'document')}.pdf",
            mime="application/pdf",
            key="dl_highlighted_main",
            type="primary",
        )

    # New Analysis button
    st.markdown("---")
    if st.button("New Analysis", type="primary", key="new_analysis_btn"):
        st.session_state._selected_job_id = None
        st.session_state.uploader_counter = st.session_state.get("uploader_counter", 0) + 1
        st.rerun()

    # Fence pages with tabs
    if fence_pages:
        sorted_pages = sorted(fence_pages, key=lambda x: x.get("page_idx", 0))
        page_labels = [f"Page {p.get('page_num', p.get('page_number', '?'))}" for p in sorted_pages]

        if len(sorted_pages) == 1:
            _render_fence_page(sorted_pages[0], pdf_path, per_page_scale, element_details, job_id)
        else:
            tabs = st.tabs(page_labels)
            for tab, page_data in zip(tabs, sorted_pages):
                with tab:
                    _render_fence_page(page_data, pdf_path, per_page_scale, element_details, job_id)

    # Element Specifications
    if element_details:
        st.markdown("---")
        st.markdown("#### Element Specifications (Cross-Page Details)")
        spec_rows = []
        for elem_name, details in element_details.items():
            if any(v for v in details.values() if v):
                spec_rows.append({
                    "Element": elem_name,
                    "Height": details.get("height", ""),
                    "Post Type": details.get("post_type", ""),
                    "Post Spacing": details.get("post_spacing", ""),
                    "Material": details.get("material", ""),
                    "Gauge": details.get("gauge", ""),
                    "Mesh Size": details.get("mesh_size", ""),
                    "Foundation": details.get("foundation", ""),
                    "Gate Info": details.get("gate_info", ""),
                    "Detail Page": details.get("detail_page", ""),
                })
        if spec_rows:
            st.dataframe(pd.DataFrame(spec_rows), hide_index=True, use_container_width=True)
            with st.expander("Full Detail Text per Element", expanded=False):
                for elem_name, details in element_details.items():
                    full = details.get("full_details", "")
                    notes = details.get("notes", "")
                    if full or notes:
                        st.markdown(f"**{elem_name}:**")
                        if full:
                            st.markdown(f"  {full}")
                        if notes:
                            st.markdown(f"  *Notes: {notes}*")

    # Non-fence pages
    if non_fence_pages:
        with st.expander(f"Non-Fence Pages ({len(non_fence_pages)})", expanded=False):
            page_nums = [p.get("page_num", p.get("page_number", "?")) for p in non_fence_pages]
            st.write(f"Pages: {', '.join(str(p) for p in page_nums)}")

    # Downloads section
    st.markdown("---")
    st.markdown("#### Downloads")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if pdf_path and os.path.exists(pdf_path) and fence_pages:
            try:
                m_pdf_bytes, m_pdf_name = generate_measurement_pdf(
                    pdf_path, fence_pages,
                    st.session_state.get("line_assignments", {}),
                    st.session_state.get("user_drawn_lines", {}),
                    page_categories,
                    uploaded_pdf_name=job.get("filename", "document.pdf"),
                )
                if m_pdf_bytes:
                    st.download_button(
                        "Download PDF with Measurements",
                        m_pdf_bytes, m_pdf_name,
                        "application/pdf",
                        key="dl_measurement_pdf",
                    )
            except Exception:
                pass

    with dl_col2:
        if fence_pages:
            try:
                xlsx_data = generate_measurement_spreadsheet(
                    fence_pages,
                    st.session_state.get("line_assignments", {}),
                    st.session_state.get("user_drawn_lines", {}),
                    page_categories,
                    per_page_scale,
                    element_details,
                )
                if xlsx_data:
                    base_name = os.path.splitext(job.get("filename", "document"))[0]
                    st.download_button(
                        "Download Measurements Excel",
                        xlsx_data,
                        f"{base_name}_measurements.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_measurement_xlsx",
                    )
            except Exception:
                pass

    # Timings
    if timings:
        with st.expander("Analysis Timings", expanded=False):
            for phase, secs in timings.items():
                if isinstance(secs, (int, float)):
                    st.markdown(f"- **{phase}**: {secs:.1f}s")


def _render_fence_page(page_data: dict, pdf_path: str | None,
                       per_page_scale: dict, element_details: dict, job_id: str):
    """Render a single fence page with image, details, and measurements."""
    page_idx = page_data.get("page_idx", 0)
    page_num = page_data.get("page_num", page_data.get("page_number", page_idx + 1))

    defns = page_data.get("definitions", [])
    insts = page_data.get("instances", [])
    kw_matches = page_data.get("keyword_matches", [])
    legend = page_data.get("legend_entries", [])
    scale = page_data.get("scale_info", {})
    measurements = page_data.get("measurements", {})

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Definitions", len(defns))
    mc2.metric("Instances", len(insts))
    mc3.metric("Keyword Matches", len(kw_matches))

    # Scale info
    if scale:
        scale_val = scale.get("verified_scale") or scale.get("text_scale")
        scale_text = scale.get("scale_text", "")
        confidence = scale.get("confidence", "")
        if scale_val:
            st.caption(f"Scale: {scale_text} ({scale_val:.1f} pts/ft, confidence: {confidence})")

        with st.expander("Scale Detection Details", expanded=False):
            st.json(scale)

    # Page image
    if pdf_path and os.path.exists(pdf_path):
        zoom_key = f"zoom_{job_id[:8]}_{page_num}"
        zoom_level = st.slider("Zoom", 50, 200, 100, 10, key=zoom_key)
        dpi = int(DISPLAY_IMAGE_DPI * zoom_level / 100)
        img_bytes = _render_page_image(pdf_path, page_idx, dpi=dpi)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)

    # Legend entries
    if legend:
        st.markdown("**Legend Entries:**")
        for entry in legend:
            name = entry.get("name", "?")
            desc = entry.get("description", "")
            st.markdown(f"- **{name}**: {desc}")

    # Definitions detail
    if defns:
        with st.expander(f"Definitions ({len(defns)})", expanded=False):
            for i, d in enumerate(defns):
                text = d.get("text", d.get("markdown", ""))[:200]
                bbox = d.get("bbox", "")
                st.markdown(f"**[{i+1}]** {text}")
                if bbox:
                    st.caption(f"bbox: {bbox}")

    # Instances detail
    if insts:
        with st.expander(f"Instances ({len(insts)})", expanded=False):
            for i, inst in enumerate(insts):
                text = inst.get("text", inst.get("markdown", ""))[:200]
                bbox = inst.get("bbox", "")
                st.markdown(f"**[{i+1}]** {text}")
                if bbox:
                    st.caption(f"bbox: {bbox}")

    # Keyword matches
    if kw_matches:
        with st.expander(f"Keyword Matches ({len(kw_matches)})", expanded=False):
            for m in kw_matches:
                st.markdown(f"- **{m.get('keyword', '?')}**: \"{m.get('text', '')}\"")

    # Measurements
    if measurements:
        m_info = measurements.get("page_info", {})
        m_lines = measurements.get("lines", [])
        m_total = measurements.get("total_length_ft", 0)

        with st.expander(f"Measurements ({len(m_lines)} lines, {m_total:.1f} ft)", expanded=False):
            if m_info.get("scale_factor"):
                st.caption(f"Scale: {m_info['scale_factor']:.1f} pts/ft")
            if m_lines:
                for ml in m_lines[:50]:
                    st.markdown(
                        f"- {ml.get('category', '?')}: {ml.get('length_ft', 0):.1f} ft "
                        f"({ml.get('length_pts', 0):.0f} pts)"
                    )
            if not m_lines:
                st.json(measurements)

    # Text preview
    text_preview = page_data.get("fence_text", "")
    if text_preview:
        with st.expander("Page Text Preview", expanded=False):
            st.text(text_preview[:3000])


# ==============================================================================
# Results View
# ==============================================================================

selected_job_id = st.session_state.get("_selected_job_id")

if selected_job_id:
    job_data = _api_get(f"/api/jobs/{selected_job_id}")
    if job_data is None:
        st.warning("Job not found.")
        st.session_state._selected_job_id = None
    else:
        status = job_data.get("status", "?")
        filename = job_data.get("filename", "unknown.pdf")

        st.markdown("---")
        st.subheader(f"Job: {filename}")

        if status == "queued":
            qp = job_data.get("queue_position", "?")
            st.info(f"Queued (position #{qp}). Waiting for worker...")
            if st.button("Cancel Job", key="cancel_queued"):
                requests.delete(f"{API_URL}/api/jobs/{selected_job_id}",
                               headers=_api_headers(), timeout=5)
                st.session_state._selected_job_id = None
                st.rerun()
            time.sleep(3)
            st.rerun()

        elif status == "running":
            prog = job_data.get("progress", {})
            pct = prog.get("pct", 0) if prog else 0
            phase = prog.get("phase", "") if prog else ""
            msg = prog.get("message", "Processing...") if prog else "Processing..."
            st.progress(pct / 100, text=f"[{phase}] {msg}")
            if st.button("Cancel Job", key="cancel_running"):
                requests.delete(f"{API_URL}/api/jobs/{selected_job_id}",
                               headers=_api_headers(), timeout=5)
                st.session_state._selected_job_id = None
                st.rerun()
            time.sleep(3)
            st.rerun()

        elif status == "failed":
            st.error(f"Analysis failed: {job_data.get('error_msg', 'Unknown error')}")
            if st.button("New Analysis", key="new_after_fail"):
                st.session_state._selected_job_id = None
                st.rerun()

        elif status == "completed":
            results = _api_get(f"/api/jobs/{selected_job_id}/results")
            if results is None:
                st.warning("Results not found on disk (may have expired).")
            else:
                _show_results(results, job_data)

        elif status == "cancelled":
            st.warning("This job was cancelled.")
            if st.button("New Analysis", key="new_after_cancel"):
                st.session_state._selected_job_id = None
                st.rerun()

elif not st.session_state.get("_selected_job_id"):
    st.info("Upload a PDF to begin analysis, or select a job from the sidebar.")


# Job registry cleanup
try:
    job_registry.cleanup_expired_jobs()
except Exception:
    pass
