"""Telemetry + spend dashboard for the fence stack.

Reads JSONL logs written by telemetry.py and spend_tracker.py and surfaces
recent runs, phase timings, peak RSS, and per-user activity. Read-only —
no writes back to the data files.

Run with:
    streamlit run tools/dashboard.py
or, with a custom port to keep it off prod's 8502:
    streamlit run tools/dashboard.py --server.port 8520

Looks at FENCE_TELEMETRY_DIR and FENCE_SPEND_DIR (same env vars as
telemetry.py / spend_tracker.py). Defaults: ~/.cache/fence_ade/_telemetry
and ~/.cache/fence_ade/_spend.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fence stack dashboard", layout="wide")
st.title("Fence stack dashboard")
st.caption("Telemetry + spend overview for app_ade_prod / app_ade_fast / api_server.")

TELEMETRY_DIR = Path(
    os.environ.get("FENCE_TELEMETRY_DIR", "~/.cache/fence_ade/_telemetry")
).expanduser()
SPEND_DIR = Path(
    os.environ.get("FENCE_SPEND_DIR", "~/.cache/fence_ade/_spend")
).expanduser()


@st.cache_data(ttl=30)
def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping any malformed lines."""
    out: list[dict] = []
    if not path.exists():
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


@st.cache_data(ttl=30)
def _load_recent_telemetry(days: int) -> pd.DataFrame:
    """Concatenate last N day-files from the telemetry dir."""
    if not TELEMETRY_DIR.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    files = sorted(TELEMETRY_DIR.glob("*.jsonl"))[-days:]
    for f in files:
        rows.extend(_load_jsonl(f))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


@st.cache_data(ttl=30)
def _load_all_spend() -> pd.DataFrame:
    """Concatenate every per-user spend file."""
    if not SPEND_DIR.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for p in SPEND_DIR.glob("*.jsonl"):
        rows.extend(_load_jsonl(p))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


# ── Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    days_back = st.slider("Telemetry: days to load", 1, 30, 7)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**Sources**\n\n"
        f"- Telemetry: `{TELEMETRY_DIR}`\n"
        f"- Spend: `{SPEND_DIR}`"
    )


tel_df = _load_recent_telemetry(days_back)
spend_df = _load_all_spend()


# ── Top-line metrics ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Telemetry records", f"{len(tel_df):,}")
col2.metric(
    "Distinct sessions",
    f"{tel_df.get('session_id', pd.Series(dtype=str)).dropna().nunique() if not tel_df.empty else 0}",
)
col3.metric("Spend records", f"{len(spend_df):,}")
col4.metric(
    "Distinct users",
    f"{spend_df.get('user_id', pd.Series(dtype=str)).nunique() if not spend_df.empty else 0}",
)

if tel_df.empty and spend_df.empty:
    st.info("No telemetry or spend data found yet. Run a job to generate some.")
    st.stop()


# ── Recent runs ───────────────────────────────────────────────────────────
st.subheader("Recent runs")
if not spend_df.empty:
    runs = spend_df[spend_df.get("provider") == "run_summary"].copy()
    if not runs.empty:
        runs = runs.sort_values("ts", ascending=False).head(20)
        # Pull phase timings out of nested dict
        phase_cols: dict[str, list] = {f"phase_{k}_s": [] for k in ("1a", "1b", "1c", "2", "3")}
        for _, row in runs.iterrows():
            pw = row.get("phase_wall_s") or {}
            if not isinstance(pw, dict):
                pw = {}
            for k in ("1a", "1b", "1c", "2", "3"):
                phase_cols[f"phase_{k}_s"].append(pw.get(k))
        for k, v in phase_cols.items():
            runs[k] = v

        cols_to_show = [
            "dt", "user_id", "pdf_sha8", "total_pages", "fence_pages",
            "total_wall_s",
            "phase_1a_s", "phase_1b_s", "phase_1c_s", "phase_2_s", "phase_3_s",
        ]
        cols_to_show = [c for c in cols_to_show if c in runs.columns]
        st.dataframe(runs[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.caption("No `run_summary` records yet (these are written at the end of each analysis).")
else:
    st.caption("No spend records yet.")


# ── Phase wall-time distribution ─────────────────────────────────────────
st.subheader("Phase wall-time (last 20 runs)")
if not spend_df.empty:
    runs = spend_df[spend_df.get("provider") == "run_summary"].copy()
    if not runs.empty:
        runs = runs.sort_values("ts", ascending=False).head(20)
        timings: list[dict] = []
        for _, row in runs.iterrows():
            pw = row.get("phase_wall_s") or {}
            if not isinstance(pw, dict):
                continue
            for phase, wall in pw.items():
                if wall is None:
                    continue
                timings.append({"phase": str(phase), "wall_s": float(wall)})
        if timings:
            tdf = pd.DataFrame(timings)
            agg = tdf.groupby("phase")["wall_s"].agg(["mean", "median", "max", "count"]).reset_index()
            agg = agg.sort_values("phase")
            st.dataframe(agg, use_container_width=True, hide_index=True)
            st.bar_chart(agg.set_index("phase")["mean"])


# ── Peak RSS ─────────────────────────────────────────────────────────────
st.subheader("Peak RSS per phase checkpoint")
if not tel_df.empty and "label" in tel_df.columns:
    cps = tel_df[tel_df["kind"] == "checkpoint"].copy()
    if not cps.empty and "rss_mb" in cps.columns:
        cps_grouped = (
            cps.groupby("label")["rss_mb"]
            .agg(["mean", "max", "count"])
            .reset_index()
            .sort_values("max", ascending=False)
        )
        st.dataframe(cps_grouped, use_container_width=True, hide_index=True)


# ── Event counts ─────────────────────────────────────────────────────────
st.subheader("Event counts")
if not tel_df.empty and "name" in tel_df.columns:
    events = tel_df[tel_df["kind"] == "event"].copy()
    if not events.empty:
        ec = events.groupby("name").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(ec, use_container_width=True, hide_index=True)


# ── Per-user run frequency ────────────────────────────────────────────────
st.subheader("Per-user activity (all time)")
if not spend_df.empty and "user_id" in spend_df.columns:
    runs = spend_df[spend_df.get("provider") == "run_summary"].copy()
    if not runs.empty:
        per_user = (
            runs.groupby("user_id")
            .agg(
                runs=("ts", "count"),
                total_pages=("total_pages", "sum"),
                last_seen=("ts", "max"),
            )
            .reset_index()
            .sort_values("runs", ascending=False)
            .head(50)
        )
        per_user["last_seen"] = pd.to_datetime(per_user["last_seen"], unit="s", utc=True)
        st.dataframe(per_user, use_container_width=True, hide_index=True)
