"""Shared loader for OpenAI / LandingAI / Google Cloud credentials.

Used by both the FastAPI worker (api_server.py) and the pipeline CLI
(`python -m pipeline`). Reads `.streamlit/secrets.toml` first; falls back
to environment variables. Returns the same shape PipelineConfig expects.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import toml

log = logging.getLogger("secrets_loader")


def load_api_keys(secrets_path: str | Path = ".streamlit/secrets.toml") -> dict:
    """Load OpenAI / LandingAI / Google Cloud credentials.

    Returns a dict with keys: openai_key, ade_key, google_cloud_config.
    Missing values are returned as empty string / None so PipelineConfig
    can be constructed without raising.
    """
    secrets: dict = {}
    path = Path(secrets_path)
    if path.exists():
        try:
            secrets = toml.load(str(path))
        except Exception as e:
            log.warning("Failed to load %s: %s", path, e)

    openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    ade_key = secrets.get("LANDINGAI_API_KEY", os.getenv("LANDINGAI_API_KEY", ""))

    google_cloud_config = None
    try:
        if "google_cloud" in secrets and "gcp_service_account" in secrets:
            google_cloud_config = {
                "project_number": secrets["google_cloud"]["project_number"],
                "location": secrets["google_cloud"]["location"],
                "processor_id": secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(secrets["gcp_service_account"]),
            }
    except Exception:
        pass

    return {
        "openai_key": openai_key,
        "ade_key": ade_key,
        "google_cloud_config": google_cloud_config,
    }
