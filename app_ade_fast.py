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
from umt import render_umt
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

# Auth gate (renders its own landing page for unauthenticated visitors)
require_auth()

st.markdown("<h1 style='text-align:center;'>ADE Fence Detection in Engineering Drawings</h1>",
            unsafe_allow_html=True)

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
            timeout=120,
        )
        if resp.ok:
            return resp.json()
        try:
            err = resp.json().get("detail") or resp.text
        except Exception:
            err = resp.text or f"HTTP {resp.status_code}"
        return {"error": str(err)[:200]}
    except Exception as e:
        return {"error": str(e)[:200]}


# Phase labels and weights for live progress UI (matches monolith)
_PHASE_LABELS = {
    "init": "Init",
    "phase1a": "Phase 1a — text extraction",
    "phase1b": "Phase 1b — OCR",
    "phase1c": "Phase 1c — page classification",
    "phase2": "Phase 2 — ADE detection",
    "phase3": "Phase 3 — legend / scale / measurement",
    "highlight": "Highlight PDF",
    "details": "Element details",
    "done": "Done",
    "error": "Error",
    "start": "Starting",
}


def _phase_window(phase: str) -> tuple[float, float] | None:
    """Map a phase to its (start, end) percent of overall progress."""
    table = {
        "init": (0, 5),
        "phase1a": (5, 15),
        "phase1b": (15, 30),
        "phase1c": (30, 45),
        "phase2": (45, 60),
        "phase3": (60, 93),
        "highlight": (93, 96),
        "details": (96, 99),
        "done": (99, 100),
        "start": (0, 5),
    }
    return table.get(phase)


def _format_dur(secs: float) -> str:
    secs = max(0.0, float(secs))
    if secs < 90:
        return f"{secs:.0f}s"
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}m{s:02d}s"


def _render_running(job_id: str, job_data: dict):
    """Live progress UI: dual bars + phase status + ETAs."""
    prog = job_data.get("progress") or {}
    phase = (prog.get("phase") or "start").lower()
    pct_overall = float(prog.get("pct") or 0.0)
    msg = prog.get("message") or "Working..."
    started_at = job_data.get("started_at") or int(time.time())
    elapsed = max(0, int(time.time()) - int(started_at))

    # Compute phase-local progress within its window
    win = _phase_window(phase)
    if win:
        lo, hi = win
        denom = max(0.001, hi - lo)
        phase_pct = max(0.0, min(1.0, (pct_overall - lo) / denom))
    else:
        phase_pct = 0.0

    # Estimate remaining time from overall % progress and elapsed
    if pct_overall >= 1.0:
        eta_total = max(0, int(elapsed * (100 - pct_overall) / max(0.1, pct_overall)))
    else:
        eta_total = None

    label = _PHASE_LABELS.get(phase, phase)

    total_line = f"⏱️ **Total** — {_format_dur(elapsed)} elapsed"
    if eta_total is not None:
        total_line += f" · ~{_format_dur(eta_total)} left"
    st.markdown(total_line)
    st.progress(min(1.0, pct_overall / 100.0))

    st.markdown(f"↳ **{label}**")
    st.progress(phase_pct)
    st.caption(msg)


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

    # ==========================================================================
    # My Jobs (top of sidebar — primary view)
    # ==========================================================================
    st.markdown("### 📂 My Jobs")

    jobs_data = _api_get("/api/jobs")
    all_jobs = jobs_data.get("jobs", []) if jobs_data else []

    # Hide cancelled jobs from the main list (they stay in DB until TTL or manual purge)
    visible_jobs = [j for j in all_jobs if j.get("status") != "cancelled"]
    cancelled_jobs = [j for j in all_jobs if j.get("status") == "cancelled"]

    selected_id = st.session_state.get("_selected_job_id")
    _any_active = False

    if not visible_jobs:
        st.caption("_No active jobs. Upload a PDF below._")

    STATUS_STYLE = {
        "queued":    {"icon": "⏳", "color": "#9CA3AF", "label": "Queued"},
        "running":   {"icon": "🔄", "color": "#3B82F6", "label": "Running"},
        "completed": {"icon": "✅", "color": "#10B981", "label": "Done"},
        "failed":    {"icon": "❌", "color": "#EF4444", "label": "Failed"},
    }

    for job in visible_jobs[:10]:
        status = job.get("status", "?")
        style = STATUS_STYLE.get(status, {"icon": "❔", "color": "#9CA3AF", "label": status})
        jid = job["job_id"]
        is_selected = jid == selected_id

        fname = job.get("filename", "unknown.pdf")
        fname_short = fname if len(fname) <= 32 else fname[:29] + "..."

        age_s = max(0, int(time.time()) - job.get("created_at", 0))
        if age_s < 60:
            age_str = "just now"
        elif age_s < 3600:
            age_str = f"{age_s // 60}m ago"
        elif age_s < 86400:
            age_str = f"{age_s // 3600}h ago"
        else:
            age_str = f"{age_s // 86400}d ago"

        # Status text + per-status detail
        if status == "running":
            _any_active = True
            prog = job.get("progress") or {}
            pct = prog.get("pct", 0)
            sub = f"{pct}%"
        elif status == "queued":
            _any_active = True
            qp = job.get("queue_position")
            sub = f"#{qp} in queue" if qp else "queued"
        elif status == "completed":
            tp = job.get("total_pages")
            fc = job.get("fence_count")
            if tp is not None and fc is not None:
                sub = f"{tp}p · {fc} fence"
            elif tp is not None:
                sub = f"{tp}p"
            else:
                sub = "ready"
        elif status == "failed":
            sub = "click to view error"
        else:
            sub = status

        with st.container(border=True):
            # Header row: status icon + filename + age
            head = (
                f"<div style='line-height:1.3;'>"
                f"<span style='color:{style['color']};font-weight:600;'>{style['icon']} {style['label']}</span>"
                f"<span style='color:#6B7280;float:right;font-size:0.8em;'>{age_str}</span>"
                f"<br><span style='font-size:0.95em;'>{fname_short}</span>"
                f"<br><span style='color:#6B7280;font-size:0.8em;'>{sub}</span>"
                f"</div>"
            )
            st.markdown(head, unsafe_allow_html=True)

            # Inline progress bar for running jobs
            if status == "running":
                pct = (job.get("progress") or {}).get("pct", 0)
                st.progress(min(1.0, pct / 100.0))

            # Action row
            if status == "completed":
                bcol1, bcol2 = st.columns([1, 1])
                with bcol1:
                    if st.button(
                        "Open" if not is_selected else "✓ Open",
                        key=f"open_{jid[:8]}",
                        use_container_width=True,
                        type=("primary" if is_selected else "secondary"),
                    ):
                        st.session_state._selected_job_id = jid
                        st.rerun()
                with bcol2:
                    hl_path = job_registry.get_highlighted_pdf_path(jid)
                    if hl_path:
                        try:
                            st.download_button(
                                "↓ PDF",
                                hl_path.read_bytes(),
                                file_name=f"fence_{fname}",
                                mime="application/pdf",
                                key=f"dl_{jid[:8]}",
                                use_container_width=True,
                            )
                        except Exception:
                            pass
            elif status in ("queued", "running"):
                bcol1, bcol2 = st.columns([1, 1])
                with bcol1:
                    if st.button(
                        "Open" if not is_selected else "✓ Open",
                        key=f"open_{jid[:8]}",
                        use_container_width=True,
                        type=("primary" if is_selected else "secondary"),
                    ):
                        st.session_state._selected_job_id = jid
                        st.rerun()
                with bcol2:
                    if st.button("✕ Cancel", key=f"cancel_{jid[:8]}",
                                 use_container_width=True):
                        try:
                            requests.delete(
                                f"{API_URL}/api/jobs/{jid}",
                                headers=_api_headers(), timeout=5,
                            )
                        except Exception:
                            pass
                        if jid == selected_id:
                            st.session_state._selected_job_id = None
                        st.rerun()
            elif status == "failed":
                bcol1, bcol2 = st.columns([1, 1])
                with bcol1:
                    if st.button(
                        "Open" if not is_selected else "✓ Open",
                        key=f"open_{jid[:8]}",
                        use_container_width=True,
                        type=("primary" if is_selected else "secondary"),
                    ):
                        st.session_state._selected_job_id = jid
                        st.rerun()
                with bcol2:
                    if st.button("🗑 Remove", key=f"del_{jid[:8]}",
                                 use_container_width=True):
                        try:
                            requests.delete(
                                f"{API_URL}/api/jobs/{jid}",
                                headers=_api_headers(), timeout=5,
                            )
                        except Exception:
                            pass
                        if jid == selected_id:
                            st.session_state._selected_job_id = None
                        st.rerun()

    # Hidden cancelled jobs — show a small expander with bulk-purge option
    if cancelled_jobs:
        with st.expander(f"Cancelled ({len(cancelled_jobs)})", expanded=False):
            for job in cancelled_jobs[:20]:
                jid = job["job_id"]
                fname = job.get("filename", "unknown.pdf")
                fname_short = fname if len(fname) <= 28 else fname[:25] + "..."
                col1, col2 = st.columns([4, 1])
                col1.caption(f"🚫 {fname_short}")
                with col2:
                    if st.button("✕", key=f"purge_{jid[:8]}",
                                 help="Permanently remove from DB"):
                        try:
                            requests.delete(
                                f"{API_URL}/api/jobs/{jid}",
                                headers=_api_headers(), timeout=5,
                            )
                            # Hard-delete (cancelled status already; need a hard purge endpoint)
                        except Exception:
                            pass
                        st.rerun()
            if st.button("Clear all cancelled", key="purge_all_cancelled",
                         use_container_width=True):
                for job in cancelled_jobs:
                    try:
                        requests.delete(
                            f"{API_URL}/api/jobs/{job['job_id']}",
                            headers=_api_headers(), timeout=5,
                        )
                    except Exception:
                        pass
                st.rerun()

    # ==========================================================================
    # Configuration
    # ==========================================================================
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
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
    st.markdown("### 🔍 Fence Keywords")
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

    # Cancel / Reset (only when a job is selected)
    if st.session_state.get("_selected_job_id"):
        st.markdown("---")
        if st.button("🛑 Cancel / Reset", use_container_width=True, key="sidebar_cancel_reset"):
            try:
                requests.delete(
                    f"{API_URL}/api/jobs/{st.session_state._selected_job_id}",
                    headers=_api_headers(), timeout=5,
                )
            except Exception:
                pass
            st.session_state._selected_job_id = None
            st.session_state.uploader_counter = st.session_state.get("uploader_counter", 0) + 1
            st.rerun()
        st.caption(
            "Cancels the selected job and returns to upload. Other jobs are unaffected."
        )

    if _any_active:
        st.caption("_Refreshing in 5s..._")
        time.sleep(5)
        st.rerun()


# ==============================================================================
# Main Area
# ==============================================================================

# PDF Upload (multi-file)
st.markdown("---")
uploaded_files = st.file_uploader(
    "Upload one or more Engineering PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_upload_{st.session_state.get('uploader_counter', 0)}",
    help="Drag in multiple files; each will queue as a separate job and process in order.",
)

if uploaded_files:
    # Validate sizes; collect (name, bytes, size_mb) for the submit step
    valid_files = []
    oversize_files = []
    for f in uploaded_files:
        b = f.read()
        smb = len(b) / 1024 / 1024
        if smb > cfg.MAX_PDF_MB:
            oversize_files.append((f.name, smb))
        else:
            valid_files.append((f.name, b, smb))

    for name, smb in oversize_files:
        st.error(
            f"`{name}` is too large ({smb:.0f} MB). Max is {cfg.MAX_PDF_MB} MB. Skipped."
        )

    if valid_files:
        total_mb = sum(s for _, _, s in valid_files)
        st.success(
            f"Ready to analyze **{len(valid_files)}** file(s) "
            f"({total_mb:.1f} MB total)"
        )
        with st.expander("Files to submit", expanded=(len(valid_files) <= 5)):
            for name, _b, smb in valid_files:
                st.markdown(f"- 📄 `{name}` — {smb:.1f} MB")

        btn_label = (
            "Analyze PDF" if len(valid_files) == 1
            else f"Analyze {len(valid_files)} PDFs"
        )
        if st.button(btn_label, type="primary", key="analyze_btn"):
            config_payload = {
                "analysis_model": cfg.ANALYSIS_MODEL,
                "classifier_model": cfg.CLASSIFIER_MODEL,
                "fence_keywords": st.session_state.fence_keywords_app,
                "use_ade": st.session_state.get("use_ade_toggle", True),
                "highlight_fence_text": st.session_state.get("highlight_toggle", True),
                "enable_unified_measurement": st.session_state.get("unified_measurement_toggle", True),
                "enable_nonlayer_suggestions": st.session_state.get("nonlayer_suggestions_toggle", False),
            }
            submitted_ids = []
            failed_files = []
            progress_bar = st.progress(0.0, text="Submitting jobs...")
            for i, (name, pdf_bytes, _smb) in enumerate(valid_files):
                progress_bar.progress(
                    (i + 1) / len(valid_files),
                    text=f"Submitting {i + 1}/{len(valid_files)}: {name}",
                )
                result = _api_post_file("/api/jobs", pdf_bytes, name, config_payload)
                if result and result.get("job_id"):
                    submitted_ids.append(result["job_id"])
                else:
                    err = (result or {}).get("error") or "unknown error"
                    failed_files.append((name, err))
            progress_bar.empty()

            for name, err in failed_files:
                st.error(f"`{name}`: {err}")

            if submitted_ids:
                # Auto-open the first submitted job so user sees its progress
                st.session_state._selected_job_id = submitted_ids[0]
                st.session_state.uploader_counter = (
                    st.session_state.get("uploader_counter", 0) + 1
                )
                st.success(
                    f"Submitted **{len(submitted_ids)}** job(s). "
                    f"Opening the first one — others are queued and visible in the sidebar."
                )
                time.sleep(1)
                st.rerun()
            elif not failed_files:
                st.error("Failed to submit. Is the API server running?")


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

    # Unified Measurement Tool
    if (st.session_state.get("unified_measurement_toggle", True)
            and fence_pages and pdf_path and os.path.exists(pdf_path)):
        st.markdown("---")
        try:
            render_umt(
                pdf_path=pdf_path,
                fence_pages=fence_pages,
                per_page_scale_info=per_page_scale,
                unified_measurements=per_page_measurements,
                element_details=element_details,
                enable_nonlayer=st.session_state.get("nonlayer_suggestions_toggle", False),
                low_dpi_mode=st.session_state.get("low_dpi_toggle", False),
                job_id=job_id,
            )
        except Exception as e:
            st.error(f"Measurement tool error: {e}")

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

    # Detection method badge
    if defns or insts:
        if defns and insts:
            badge = "🎯 ADE Detection"
        elif defns:
            badge = "🎯 ADE Definitions"
        elif insts:
            badge = "🎯 ADE Instances"
        else:
            badge = "❌ No Detection"
    elif kw_matches:
        badge = "🔤 Keyword-only"
    else:
        badge = "❌ No Detection"
    st.markdown(f"**Detection:** {badge}")

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

    # Analysis Log popover (LLM raw response if present)
    log_text = page_data.get("llm_raw_response") or page_data.get("analysis_log")
    if log_text:
        with st.popover("📋 Analysis Log"):
            st.text(str(log_text)[:8000])

    # Measurements
    if measurements:
        m_info = measurements.get("page_info", {})
        m_lines = measurements.get("lines", [])
        m_total = measurements.get("total_length_ft", 0)
        m_method = measurements.get("measurement_method", "")
        per_indicator = measurements.get("per_indicator", {})
        per_layer = measurements.get("per_layer", {})
        dimension_lines = measurements.get("dimension_lines", [])

        method_badges = {
            "layer": "🟢 Layer-based",
            "llm_guided": "🧠 LLM-guided",
            "skipped": "⏭️ Skipped",
            "": "",
        }
        method_badge = method_badges.get(m_method, m_method)

        header_label = f"Measurements ({len(m_lines)} lines, {m_total:.1f} ft)"
        if method_badge:
            header_label += f" — {method_badge}"

        with st.expander(header_label, expanded=False):
            if m_info.get("scale_factor"):
                st.caption(f"Scale: {m_info['scale_factor']:.1f} pts/ft")

            # Per-indicator breakdown
            if per_indicator:
                st.markdown("**Per-indicator breakdown:**")
                rows = []
                for ind, info in per_indicator.items():
                    if isinstance(info, dict):
                        rows.append({
                            "Indicator": ind,
                            "Lines": info.get("count", info.get("line_count", 0)),
                            "Length (ft)": round(info.get("length_ft", info.get("total_ft", 0)), 1),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Layer breakdown
            if per_layer:
                with st.expander(f"Layer breakdown ({len(per_layer)} layers)", expanded=False):
                    rows = []
                    for layer, info in per_layer.items():
                        if isinstance(info, dict):
                            rows.append({
                                "Layer": layer,
                                "Lines": info.get("count", 0),
                                "Length (ft)": round(info.get("length_ft", 0), 1),
                            })
                        else:
                            rows.append({"Layer": layer, "Value": str(info)[:100]})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Dimension lines
            if dimension_lines:
                with st.expander(f"Dimension lines ({len(dimension_lines)})", expanded=False):
                    for dl in dimension_lines[:30]:
                        text = dl.get("text", "")
                        length = dl.get("length_ft", dl.get("value", ""))
                        st.markdown(f"- {text} → {length}")

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
            _render_running(selected_job_id, job_data)
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
