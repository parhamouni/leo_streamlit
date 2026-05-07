"""Tests for the shared secrets loader (used by api_server + pipeline CLI)."""

from __future__ import annotations

import textwrap

from secrets_loader import load_api_keys


def test_loads_from_toml(tmp_path):
    p = tmp_path / "secrets.toml"
    p.write_text(textwrap.dedent('''
        OPENAI_API_KEY = "sk-from-toml"
        LANDINGAI_API_KEY = "land-from-toml"
    '''))
    keys = load_api_keys(p)
    assert keys["openai_key"] == "sk-from-toml"
    assert keys["ade_key"] == "land-from-toml"
    assert keys["google_cloud_config"] is None


def test_falls_back_to_env_when_toml_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("LANDINGAI_API_KEY", "land-from-env")

    keys = load_api_keys(tmp_path / "does-not-exist.toml")
    assert keys["openai_key"] == "sk-from-env"
    assert keys["ade_key"] == "land-from-env"


def test_toml_takes_precedence_over_env(tmp_path, monkeypatch):
    p = tmp_path / "secrets.toml"
    p.write_text('OPENAI_API_KEY = "sk-from-toml"\n')
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    keys = load_api_keys(p)
    assert keys["openai_key"] == "sk-from-toml"


def test_assembles_google_cloud_config(tmp_path):
    p = tmp_path / "secrets.toml"
    p.write_text(textwrap.dedent('''
        OPENAI_API_KEY = "sk-x"

        [google_cloud]
        project_number = "111"
        location = "us"
        processor_id = "p1"

        [gcp_service_account]
        type = "service_account"
        project_id = "leo-fences"
    '''))
    keys = load_api_keys(p)
    gcp = keys["google_cloud_config"]
    assert gcp is not None
    assert gcp["project_number"] == "111"
    assert gcp["location"] == "us"
    assert gcp["processor_id"] == "p1"
    assert gcp["service_account_info"]["project_id"] == "leo-fences"


def test_empty_toml_returns_empty_strings(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANDINGAI_API_KEY", raising=False)

    p = tmp_path / "secrets.toml"
    p.write_text("")
    keys = load_api_keys(p)
    assert keys["openai_key"] == ""
    assert keys["ade_key"] == ""
    assert keys["google_cloud_config"] is None
