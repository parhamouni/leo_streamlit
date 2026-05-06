"""ADE Fence Detector — Thin Streamlit Frontend.

Submits analysis jobs to the FastAPI backend (api_server.py) and renders
results. All heavy processing happens in background workers.
"""

import json
import os
import time
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st

from auth import get_session_id, get_user_id, get_user_email, require_auth, render_auth_widget
from config import cfg
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


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    render_auth_widget()
    st.markdown("---")

    # Configuration
    st.header("Configuration")

    # Model selection
    model_options = ["gpt-5.1", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini"]
    selected_model = st.selectbox(
        "Analysis Model",
        model_options,
        index=model_options.index(st.session_state.selected_model_for_analysis)
        if st.session_state.selected_model_for_analysis in model_options else 0,
        key="model_select",
    )
    st.session_state.selected_model_for_analysis = selected_model

    # Toggles
    st.markdown("---")
    use_ade = st.toggle("Use ADE (LandingAI)", value=True, key="use_ade_toggle")
    highlight_text = st.toggle("Highlight text & indicators", value=True, key="highlight_toggle")
    enable_measurement = st.toggle("Unified Measurements", value=True, key="measurement_toggle")

    # Keywords
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
            elif status in ("queued", "cancelled"):
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
                "analysis_model": st.session_state.selected_model_for_analysis,
                "classifier_model": cfg.CLASSIFIER_MODEL,
                "fence_keywords": st.session_state.fence_keywords_app,
                "use_ade": st.session_state.get("use_ade_toggle", True),
                "highlight_fence_text": st.session_state.get("highlight_toggle", True),
                "enable_unified_measurement": st.session_state.get("measurement_toggle", True),
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
    """Render analysis results for a completed job."""
    fence_pages = results.get("fence_pages", [])
    non_fence_pages = results.get("non_fence_pages", [])
    element_details = results.get("element_details", {})
    total_pages = results.get("total_pages", 0)
    timings = results.get("timings", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", total_pages)
    col2.metric("Fence Pages", len(fence_pages))
    col3.metric("Non-Fence Pages", len(non_fence_pages))
    if timings.get("total"):
        col4.metric("Analysis Time", f"{timings['total']:.0f}s")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        hl_path = job_registry.get_highlighted_pdf_path(job["job_id"])
        if hl_path:
            st.download_button(
                "Download Highlighted PDF",
                hl_path.read_bytes(),
                file_name=f"fence_{job.get('filename', 'document')}.pdf",
                mime="application/pdf",
                key="dl_highlighted",
                type="primary",
            )

    if element_details:
        with st.expander("Element Specifications", expanded=False):
            for name, details in element_details.items():
                st.markdown(f"**{name}**")
                for k, v in details.items():
                    if v:
                        st.markdown(f"  - {k}: {v}")

    if fence_pages:
        st.markdown("### Fence Pages")
        for page_data in sorted(fence_pages, key=lambda x: x.get("page_idx", 0)):
            page_num = page_data.get("page_num", page_data.get("page_number", "?"))

            with st.expander(f"Page {page_num}", expanded=False):
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

                if scale:
                    scale_val = scale.get("verified_scale") or scale.get("text_scale")
                    if scale_val:
                        st.caption(f"Scale: 1\" = {72 / scale_val:.1f}' ({scale_val:.1f} pts/ft)")

                if legend:
                    st.markdown("**Legend Entries:**")
                    for entry in legend:
                        st.markdown(f"- {entry.get('name', '?')}: {entry.get('description', '')}")

                if defns:
                    with st.popover("View Definitions"):
                        for d in defns:
                            st.json(d)

                if measurements:
                    st.markdown("**Measurements:**")
                    st.json(measurements)

                text_preview = page_data.get("fence_text", "")
                if text_preview:
                    with st.popover("Page Text Preview"):
                        st.text(text_preview[:2000])

    if non_fence_pages:
        with st.expander(f"Non-Fence Pages ({len(non_fence_pages)})", expanded=False):
            page_nums = [p.get("page_num", p.get("page_number", "?")) for p in non_fence_pages]
            st.write(f"Pages: {', '.join(str(p) for p in page_nums)}")

    if timings:
        with st.expander("Analysis Timings", expanded=False):
            for phase, secs in timings.items():
                if isinstance(secs, (int, float)):
                    st.markdown(f"- **{phase}**: {secs:.1f}s")


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
            time.sleep(3)
            st.rerun()

        elif status == "running":
            prog = job_data.get("progress", {})
            pct = prog.get("pct", 0) if prog else 0
            msg = prog.get("message", "Processing...") if prog else "Processing..."
            st.progress(pct / 100, text=msg)
            time.sleep(3)
            st.rerun()

        elif status == "failed":
            st.error(f"Analysis failed: {job_data.get('error_msg', 'Unknown error')}")

        elif status == "completed":
            results = _api_get(f"/api/jobs/{selected_job_id}/results")
            if results is None:
                st.warning("Results not found on disk (may have expired).")
            else:
                _show_results(results, job_data)

        elif status == "cancelled":
            st.warning("This job was cancelled.")


# Job registry cleanup
try:
    job_registry.cleanup_expired_jobs()
except Exception:
    pass
