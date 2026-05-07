"""Shared fixtures for the leo_streamlit test suite.

Tests must never touch the real ~/.leo or ~/.cache/fence_ade directories.
Each test gets its own temp directories via env-var overrides for fence_cache
and monkeypatched module paths for job_registry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable when pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fence_cache_root(tmp_path, monkeypatch):
    """Point fence_cache at a temp dir for the test.

    Resets the module-level memo so the next cache_root() call sees the env.
    """
    cache_dir = tmp_path / "fence_ade"
    monkeypatch.setenv("FENCE_CACHE_DIR", str(cache_dir))

    import fence_cache
    monkeypatch.setattr(fence_cache, "_CACHE_ROOT_CACHED", None, raising=False)
    return cache_dir


@pytest.fixture
def job_registry_temp(tmp_path, monkeypatch):
    """Point job_registry at a temp SQLite DB + results dir.

    job_registry pins paths at module load time, so we monkeypatch them
    AND reset the connection memo so the next _db() call rebuilds.
    """
    leo_dir = tmp_path / "leo"
    db_path = leo_dir / "jobs.db"
    results_root = leo_dir / "results"

    import job_registry
    monkeypatch.setattr(job_registry, "_LEO_DIR", leo_dir, raising=False)
    monkeypatch.setattr(job_registry, "_DB_PATH", db_path, raising=False)
    monkeypatch.setattr(job_registry, "_RESULTS_ROOT", results_root, raising=False)
    monkeypatch.setattr(job_registry, "_db_initialized", False, raising=False)
    monkeypatch.setattr(job_registry, "_conn", None, raising=False)

    yield job_registry

    # Close any open SQLite connection the test left open.
    if job_registry._conn is not None:
        try:
            job_registry._conn.close()
        except Exception:
            pass


@pytest.fixture
def fixture_pdf():
    """Path to a small fixture PDF in subset_gold/.

    Returns the smallest available; tests that need a full pipeline run
    against this path require API keys and are skipped without them.
    """
    candidates = [
        REPO_ROOT / "subset_gold" / "_3_ARCHITECTURAL_BASEBALL_fence_highlights_fence_highlights.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip("No fixture PDF available in subset_gold/")
