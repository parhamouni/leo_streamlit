"""Smoke tests for the pipeline CLI (`python -m pipeline`).

These verify the argparse surface and error handling, not full pipeline runs
(which require API keys and a real PDF — covered by the manual run via
`venv/bin/python -m pipeline subset_gold/<file>.pdf --out /tmp/out`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args, env=None):
    """Invoke `python -m pipeline ...` from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "pipeline", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_help_lists_main_options():
    res = _run_cli(["--help"])
    assert res.returncode == 0
    out = res.stdout
    assert "--out" in out
    assert "--secrets" in out
    assert "--no-ade" in out
    assert "--keywords" in out
    assert "--analysis-model" in out


def test_missing_pdf_path_exits_with_error(tmp_path):
    """Pointing at a nonexistent PDF returns 2 (file not found)."""
    res = _run_cli(["/no/such/file.pdf", "--out", str(tmp_path / "out")])
    assert res.returncode == 2
    assert "not found" in res.stderr.lower()


def test_missing_openai_key_exits_with_error(tmp_path):
    """No secrets and no OPENAI_API_KEY env → returncode 2 with helpful message."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    fake_secrets = tmp_path / "secrets.toml"
    fake_secrets.write_text("")  # empty; no keys

    import os
    env = {**os.environ}
    env.pop("OPENAI_API_KEY", None)
    env.pop("LANDINGAI_API_KEY", None)

    res = _run_cli(
        [str(fake_pdf), "--out", str(tmp_path / "out"), "--secrets", str(fake_secrets)],
        env=env,
    )
    assert res.returncode == 2
    assert "openai" in res.stderr.lower()
