"""Centralized configuration for the fence detection system.

Every env-var knob and hardcoded constant lives here. Import `cfg` and use
attribute access: `cfg.FENCE_WORKERS_PHASE1C`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int, lo: int = 1, hi: int = 9999) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default).lower()).lower() in ("true", "1", "yes")


def _str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    # --- Authentication ---
    AUTH_MODE: str = field(default_factory=lambda: _str_env("FENCE_AUTH_MODE", "streamlit_oidc").lower().strip())
    APP_PASSWORD: str = field(default_factory=lambda: _str_env("FENCE_APP_PASSWORD", ""))

    # --- Concurrency / Worker Pools ---
    LOW_MEMORY: bool = field(default_factory=lambda: _bool_env("FENCE_LOW_MEMORY", False))

    WORKERS_PHASE1A: int = field(default_factory=lambda: _int_env("FENCE_WORKERS_PHASE1A", 4, hi=8))
    WORKERS_PHASE1B: int = field(default_factory=lambda: _int_env("FENCE_WORKERS_PHASE1B", 6, hi=16))
    WORKERS_PHASE1C: int = field(default_factory=lambda: _int_env("FENCE_WORKERS_PHASE1C", 16, hi=16))
    WORKERS_PHASE2: int = field(default_factory=lambda: _int_env("FENCE_WORKERS_PHASE2", 5, hi=8))
    CLASSIFY_BATCH_SIZE: int = field(default_factory=lambda: _int_env("FENCE_CLASSIFY_BATCH_SIZE", 10, hi=25))
    OCR_BATCH_SIZE: int = field(default_factory=lambda: _int_env("FENCE_OCR_BATCH_SIZE", 15, hi=15))

    # --- Slot-level concurrency (analysis-wide) ---
    MAX_CONCURRENT_PER_USER: int = field(default_factory=lambda: _int_env("FENCE_MAX_CONCURRENT_PER_USER", 1, hi=5))
    RSS_REJECT_GB: float = field(default_factory=lambda: _float_env("FENCE_RSS_REJECT_GB", 9.0))

    # --- Memory / Image Cache ---
    RSS_CEILING_GB: float = field(default_factory=lambda: _float_env("FENCE_RSS_CEILING_GB", 6.0))
    IMG_CACHE_MAX: int = field(default_factory=lambda: _int_env("FENCE_IMG_CACHE_MAX", 10, hi=100))
    MAX_USER_BYTES: int = field(default_factory=lambda: _int_env("FENCE_MAX_USER_BYTES", 2 * 1024**3, hi=20 * 1024**3))

    # --- Disk / Cache ---
    CACHE_TTL_DAYS: float = field(default_factory=lambda: _float_env("FENCE_CACHE_TTL_DAYS", 1.0))
    PDF_TMP_DIR: str = "/tmp/fence_pdfs"

    # --- Analysis Pipeline ---
    ANALYSIS_MODEL: str = field(default_factory=lambda: _str_env("FENCE_ANALYSIS_MODEL", "gpt-5.1"))
    CLASSIFIER_MODEL: str = field(default_factory=lambda: _str_env("FENCE_CLASSIFIER_MODEL", "gpt-5-mini"))
    HIGHLIGHT_PDF_TIMEOUT: int = field(default_factory=lambda: _int_env("FENCE_HIGHLIGHT_PDF_TIMEOUT", 600, hi=1800))
    DETAILS_TIMEOUT: int = field(default_factory=lambda: _int_env("FENCE_DETAILS_TIMEOUT", 120, hi=600))

    # Phase 3
    PHASE3_USE_SUBPROCESS: bool = field(default_factory=lambda: _bool_env("FENCE_PHASE3_USE_SUBPROCESS", True))
    PHASE3_EAGER: bool = field(default=False)  # computed in __post_init__
    PHASE3_PREVIEW: int = field(default_factory=lambda: _int_env("FENCE_PHASE3_PREVIEW", 5, hi=40))
    RESET_KILL_STUBBORN_WORKERS: bool = field(default_factory=lambda: _bool_env("FENCE_RESET_KILL_STUBBORN_WORKERS", False))

    # OCR / ADE byte limits
    OCR_BATCH_TARGET_BYTES: int = field(default_factory=lambda: _int_env("FENCE_OCR_BATCH_TARGET_BYTES", 20 * 1024 * 1024, lo=1024 * 1024, hi=100 * 1024 * 1024))
    OCR_TOTAL_MAX_BYTES: int = field(default_factory=lambda: _int_env("FENCE_OCR_TOTAL_MAX_BYTES", 35 * 1024 * 1024, lo=1024 * 1024, hi=200 * 1024 * 1024))
    ADE_BATCH_MAX_BYTES: int = field(default_factory=lambda: _int_env("FENCE_ADE_BATCH_MAX_BYTES", 15 * 1024 * 1024, lo=1024 * 1024, hi=100 * 1024 * 1024))
    ADE_PAGE_MAX_BYTES: int = field(default_factory=lambda: _int_env("FENCE_ADE_PAGE_MAX_BYTES", 12 * 1024 * 1024, lo=1024 * 1024, hi=50 * 1024 * 1024))
    ADE_DEGRADE_THRESHOLD: int = field(default_factory=lambda: _int_env("FENCE_ADE_DEGRADE_THRESHOLD", 8 * 1024 * 1024, lo=1024 * 1024, hi=50 * 1024 * 1024))

    # Legend / batching
    SKIP_LEGEND_PREBATCH: bool = field(default_factory=lambda: _bool_env("FENCE_SKIP_LEGEND_PREBATCH", False))
    LEGEND_BATCH_SIZE: int = field(default_factory=lambda: _int_env("FENCE_LEGEND_BATCH_SIZE", 6, hi=20))

    # --- Display ---
    DISPLAY_IMAGE_DPI: int = 150
    HIGHLIGHT_COLOR_UI: tuple = (0, 0.9, 0)
    HIGHLIGHT_COLOR_INSTANCE: tuple = (0.9, 0, 0.9)
    HIGHLIGHT_WIDTH_UI: float = 2.0

    # --- Locking ---
    ANALYSIS_LOCK_PATH: str = "/tmp/fence_analysis.lock"
    ANALYSIS_LOCK_TTL_SECONDS: int = 4 * 60 * 60

    # --- API Server (new: FastAPI backend) ---
    API_SERVER_URL: str = field(default_factory=lambda: _str_env("FENCE_API_URL", "http://127.0.0.1:8503"))
    API_WORKER_COUNT: int = field(default_factory=lambda: _int_env("FENCE_API_WORKER_COUNT", 1, hi=8))
    RESULTS_TTL_HOURS: int = field(default_factory=lambda: _int_env("FENCE_RESULTS_TTL_HOURS", 24, hi=168))
    RESULTS_DIR: str = field(default_factory=lambda: os.path.expanduser(_str_env("FENCE_RESULTS_DIR", "~/.leo/results")))

    # --- Web-app migration: Supabase auth (orthogonal to streamlit AUTH_MODE) ---
    # API_AUTH_MODE values: "legacy_header" | "supabase" | "both"
    #   legacy_header — endpoints trust X-User-Id (current behavior)
    #   supabase      — endpoints require Authorization: Bearer <jwt>, verified via JWKS
    #   both          — accept either; prefer JWT when present
    API_AUTH_MODE: str = field(default_factory=lambda: _str_env("FENCE_API_AUTH_MODE", "legacy_header").lower().strip())
    SUPABASE_URL: str = field(default_factory=lambda: _str_env("SUPABASE_URL", ""))
    SUPABASE_ANON_KEY: str = field(default_factory=lambda: _str_env("SUPABASE_ANON_KEY", ""))
    SUPABASE_JWKS_URL: str = field(default_factory=lambda: _str_env("SUPABASE_JWKS_URL", ""))

    # --- Upload limits (mirror old monolith) ---
    MAX_PDF_MB: int = field(default_factory=lambda: _int_env("FENCE_MAX_PDF_MB", 500, lo=1, hi=2000))
    MAX_PAGES: int = field(default_factory=lambda: _int_env("FENCE_MAX_PAGES", 300, lo=1, hi=5000))

    # --- Spend tracking ---
    MAX_DAILY_SPEND_USD: float = field(default_factory=lambda: _float_env("FENCE_MAX_DAILY_SPEND_USD", 0.0))  # 0 = no limit

    # --- Default keywords ---
    DEFAULT_FENCE_KEYWORDS: list = field(default_factory=lambda: [
        'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
        'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
        'adot specifications', 'mag specifications', 'rail', 'railing',
        'bollards', 'handrails', 'wall', 'cmu',
        'operator', 'davis', 'bacon', 'davis-bacon', 'davis – bacon',
        'buy america', 'american', 'dug out',
    ])

    def __post_init__(self):
        # Phase 3 workers: conservative on low-memory hosts
        workers_phase3_default = 1 if self.LOW_MEMORY else 2
        object.__setattr__(self, 'WORKERS_PHASE3',
                           _int_env("FENCE_WORKERS_PHASE3", workers_phase3_default, hi=12))
        # Phase 3 eagerness: lazy on low-memory hosts
        eager_default = not self.LOW_MEMORY
        object.__setattr__(self, 'PHASE3_EAGER',
                           _bool_env("FENCE_PHASE3_EAGER", eager_default))
        # MAX_CONCURRENT: auto-detect from host memory if not set
        if os.environ.get("FENCE_MAX_CONCURRENT"):
            object.__setattr__(self, 'MAX_CONCURRENT',
                               _int_env("FENCE_MAX_CONCURRENT", 2, hi=10))
        else:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024**3)
            object.__setattr__(self, 'MAX_CONCURRENT', 1 if mem_gb < 24 else 2)


cfg = Config()
