"""Typed session state management for the Streamlit frontend.

Replaces the ~50 loose st.session_state keys with structured dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from config import cfg


@dataclass
class ViewState:
    """State for the currently-viewed job and its results."""
    current_job_id: str | None = None
    fence_pages: list[dict] = field(default_factory=list)
    non_fence_pages: list[dict] = field(default_factory=list)
    element_details: dict = field(default_factory=dict)
    per_page_scale_info: dict = field(default_factory=dict)
    unified_measurements: dict = field(default_factory=dict)
    page_categories: dict = field(default_factory=dict)
    total_pages: int = 0
    highlighted_pdf_bytes: bytes | None = None
    uploaded_pdf_name: str = ""


def init_session_state():
    """Initialize session state with defaults if not already set."""
    defaults = {
        "session_id": None,
        "fence_keywords_app": list(cfg.DEFAULT_FENCE_KEYWORDS),
        "selected_model_for_analysis": cfg.ANALYSIS_MODEL,
        "view": ViewState(),
        "uploader_counter": 0,
        "pdf_upload_bytes": None,
        "pdf_upload_name": None,
        # UMT state
        "line_assignments": {},
        "user_drawn_lines": {},
        "drawing_mode": {},
        "active_category_per_page": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_view() -> ViewState:
    """Get the current view state, creating if needed."""
    if "view" not in st.session_state:
        st.session_state.view = ViewState()
    return st.session_state.view


def set_view(view: ViewState):
    st.session_state.view = view


def load_job_results(results: dict, job: dict) -> ViewState:
    """Create a ViewState from API results and job metadata."""
    return ViewState(
        current_job_id=job.get("job_id"),
        fence_pages=results.get("fence_pages", []),
        non_fence_pages=results.get("non_fence_pages", []),
        element_details=results.get("element_details", {}),
        per_page_scale_info=results.get("per_page_scale_info", {}),
        unified_measurements=results.get("unified_measurements", {}),
        page_categories=results.get("page_categories", {}),
        total_pages=results.get("total_pages", 0),
        uploaded_pdf_name=job.get("filename", ""),
    )
