# app_ade_v2.py - ADE Fence Detector with app.py UI
import streamlit as st
import os
import sys
import toml
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import base64
from pathlib import Path
from io import BytesIO
import gc
import psutil
import pandas as pd
import fitz  # PyMuPDF
# Silence MuPDF's chatty stderr (e.g. repeated "object out of range" on damaged
# PDFs). The Python-level exceptions still fire; we just stop the C library from
# spamming stderr with thousands of duplicate messages that drown out real logs.
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass
from PIL import Image

# Import our consolidated ADE utilities
import utils_ade as ade

# Persistent disk cache for intermediate results (survives fence.service
# restarts; keyed by PDF hash + config). Imported only by the fast build.
import fence_cache

# Lightweight observability (JSONL appender). Near-zero overhead; see
# telemetry.py and tools/telemetry_report.py.
import telemetry

# Per-user spend tracking stub. Records are written but not enforced in v1.
import spend_tracker

# Register faulthandler so `kill -USR1 <pid>` dumps all thread stacks to
# stderr. Saves us ~11 min of guessing next time Phase 3 hangs. Writes to
# fast.err, which is already being tailed by our ops tooling.
import faulthandler, signal as _signal
try:
    faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
except Exception:
    pass  # SIGUSR1 may not be available on all platforms

# Interactive image click for measurement
from streamlit_image_coordinates import streamlit_image_coordinates

# Vector measurement utilities
from utils_vector import (
    measure_lines_in_selection,
    measure_at_click_point,
    infer_scale_from_page,
    extract_vector_lines,
    verify_scale_with_bar,
    verify_scale_with_bar_fast,
)

# Optional: LLM client
from langchain_openai import ChatOpenAI

# --- Highlight Appearance & Performance ---
HIGHLIGHT_COLOR_UI = (0, 0.9, 0)  # Green for definitions
HIGHLIGHT_COLOR_INSTANCE = (0.9, 0, 0.9)  # Purple for instances
HIGHLIGHT_WIDTH_UI = 2.0
DISPLAY_IMAGE_DPI = 150
ANALYSIS_LOCK_PATH = "/tmp/fence_analysis.lock"
ANALYSIS_LOCK_TTL_SECONDS = 4 * 60 * 60  # 4 hour safety ceiling

# --- Authentication ---
# FENCE_AUTH_MODE decides where user identity comes from:
#   "none"           — no auth; get_user_id() returns "dev_<session>". DEV ONLY.
#   "proxy_header"   — reverse proxy (NGINX / Cloudflare Access) injects
#                      the X-Forwarded-User header. Production-safe.
#   "streamlit_oidc" — Streamlit ≥1.42 native st.user (Google/Okta OIDC).
#                      Configure .streamlit/secrets.toml provider block.
#   "streamlit_password" — shared password in .streamlit/secrets.toml (weak).
AUTH_MODE = os.environ.get("FENCE_AUTH_MODE", "none").lower().strip()


# --- Concurrency (env-var configurable) ---
# Each phase has its own worker-count knob so ops can dial individually
# without a redeploy. Defaults were chosen to stay under typical tier-2
# API rate limits (OpenAI ~500 RPM, Landing AI tighter). If 429s show up,
# dial the relevant var DOWN.
#
# --- Production tuning presets ---
# Env var                            2 vCPU / 8 GB    8 vCPU / 32 GB
# FENCE_MAX_CONCURRENT               1                3
# FENCE_MAX_CONCURRENT_PER_USER      1                1
# FENCE_RSS_REJECT_GB                6.0              24.0
# FENCE_MAX_USER_BYTES               1073741824 (1G)  5368709120 (5G)
# FENCE_WORKERS_PHASE1A              2                4
# FENCE_WORKERS_PHASE1B              3                8
# FENCE_WORKERS_PHASE1C              8                16
# FENCE_WORKERS_PHASE2               3                5
# FENCE_WORKERS_PHASE3               6                14
# FENCE_CLASSIFY_BATCH_SIZE          10               15
# FENCE_OCR_BATCH_SIZE               15               15
# FENCE_CACHE_TTL_DAYS               7                30
# FENCE_PHASE3_EAGER                 false            false
# FENCE_PHASE3_PREVIEW               5                5
def _workers(name, default, cap=16):
    try:
        return max(1, min(cap, int(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default

# Defaults bumped after first-round profiling: gpt-5.1-mini + batching
# keeps tier-2 OpenAI accounts well under the rate ceiling at these
# widths. Dial DOWN via env if you see 429s.
FENCE_WORKERS_PHASE1A = _workers("FENCE_WORKERS_PHASE1A", 4, cap=8)   # subprocess pool (new in this round)
FENCE_WORKERS_PHASE1B = _workers("FENCE_WORKERS_PHASE1B", 6)          # was 4
FENCE_WORKERS_PHASE1C = _workers("FENCE_WORKERS_PHASE1C", 16)         # was 8
FENCE_WORKERS_PHASE2  = _workers("FENCE_WORKERS_PHASE2",  5, cap=8)   # was 3
FENCE_WORKERS_PHASE3  = _workers("FENCE_WORKERS_PHASE3",  12)         # was 6
FENCE_CLASSIFY_BATCH_SIZE = _workers("FENCE_CLASSIFY_BATCH_SIZE", 10, cap=25)
FENCE_OCR_BATCH_SIZE = _workers("FENCE_OCR_BATCH_SIZE", 15, cap=15)  # DocAI sync cap

# Phase 3 eagerness. Default TRUE preserves current throughput (all
# fence pages pre-computed in parallel before the sequential render
# loop turns them into cards). Flip to false in ops config only if
# telemetry (see Stage A) shows the pre-compute is dominating wall
# time AND users rarely review all pages. The sequential render loop
# runs regardless and populates session_state.fence_pages for every
# fence page, so exports are never partial in either mode — lazy
# mode just pushes the work from the pool to the serial loop.
FENCE_PHASE3_EAGER = os.environ.get("FENCE_PHASE3_EAGER", "true").lower() == "true"
FENCE_PHASE3_PREVIEW = _workers("FENCE_PHASE3_PREVIEW", 5, cap=40)


# --- Cache housekeeping ---
# Sweep aged cache entries at PROCESS startup. Must ONLY run on actual
# process boot, not on every Streamlit rerun — Streamlit re-executes the
# whole script on every widget interaction, and a rerun-wide wipe would
# nuke the cache dir the active session is reading from. Symptom was
# cache_hits=0 after every click + the analysis block re-entering from
# Phase 1a + the browser "freezing" on mid-run button clicks because
# Phase 1-2-3 all re-ran from scratch with no cache.
#
# We use a per-PID sentinel file under /tmp instead of a module-level
# variable because Streamlit's script runner re-executes this file in a
# fresh namespace every rerun — module globals from the previous rerun
# are NOT preserved. The PID is, though: same process → same sentinel
# file → wipe skipped.
_BOOT_SENTINEL = f"/tmp/.fence_ade_boot_done_{os.getpid()}"
if not os.path.exists(_BOOT_SENTINEL):
    try:
        # Per-run cache policy: on process start, wipe EVERY session_*
        # dir under the cache root. They can only belong to dead prior
        # sessions (no current process owns them yet). This is the
        # third tier of purge protection alongside the "new upload"
        # and "finally" purges that run during a live session.
        import shutil as _boot_sh
        _root = fence_cache.cache_root()
        _boot_wiped = 0
        for _d in _root.iterdir() if _root.exists() else []:
            if _d.is_dir() and _d.name.startswith("session_"):
                try:
                    _boot_sh.rmtree(_d, ignore_errors=True)
                    _boot_wiped += 1
                except Exception:
                    pass
        if _boot_wiped:
            print(f"[fence_cache] boot: wiped {_boot_wiped} orphan session dirs")
        # TTL sweep is legacy safety for anything that escapes our purges.
        _purged_dirs = fence_cache.sweep_old(ttl_days=float(os.environ.get("FENCE_CACHE_TTL_DAYS", "1")))
        if _purged_dirs:
            print(f"[fence_cache] Swept {_purged_dirs} expired PDF directories on startup.")
    except Exception as _e:
        print(f"[fence_cache] Startup sweep failed (non-fatal): {_e}")
    try:
        with open(_BOOT_SENTINEL, "w") as _f:
            _f.write("ok")
    except Exception:
        pass

st.set_page_config(page_title="ADE Fence Detector", layout="wide")

# Unauthenticated health probe for load balancers. Must run BEFORE any
# auth gate so the LB can check liveness without a session.
try:
    if st.query_params.get("health"):
        st.write("ok")
        st.stop()
except Exception:
    pass

st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>🔍 ADE Fence Detection in Engineering Drawings</h1>", unsafe_allow_html=True)


# ==============================================================================
# Session Management (matching app.py)
# ==============================================================================

def get_session_id():
    if 'session_id' not in st.session_state:
        # Full UUID (not sliced). 36-char keys eliminate collision risk
        # with many concurrent users; slot manager doesn't care about the
        # format, so existing short IDs in open tabs keep working.
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_user_email() -> str | None:
    """Return the raw logged-in email if known, else None. For UI display only."""
    try:
        if AUTH_MODE == "streamlit_oidc":
            u = getattr(st, "user", None)
            if u is not None and getattr(u, "is_logged_in", False):
                return getattr(u, "email", None)
        elif AUTH_MODE == "proxy_header":
            try:
                return st.context.headers.get("X-Forwarded-User")
            except Exception:
                return None
        elif AUTH_MODE == "streamlit_password":
            return st.session_state.get("_auth_email")
    except Exception:
        return None
    return None


def get_user_id() -> str:
    """Stable per-identity key usable as a filesystem path segment.

    Returns a 16-char hex hash of the authenticated identity. In 'none'
    mode, returns a dev-prefixed session-scoped key — callers that rely on
    user identity for isolation MUST check via require_auth() first.
    """
    email = get_user_email()
    if email:
        return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]
    if AUTH_MODE == "streamlit_password" and st.session_state.get("_auth_ok"):
        who = st.session_state.get("_auth_email", "")
        if who:
            return hashlib.sha256(who.strip().lower().encode()).hexdigest()[:16]
    return f"dev_{get_session_id()}"


def _streamlit_password_gate():
    """Minimal password auth for single-tenant deployments. Reads
    FENCE_APP_PASSWORD env var OR st.secrets['auth']['password']."""
    if st.session_state.get("_auth_ok"):
        return
    expected = os.environ.get("FENCE_APP_PASSWORD", "")
    if not expected:
        try:
            expected = st.secrets.get("auth", {}).get("password", "") if hasattr(st, "secrets") else ""
        except Exception:
            expected = ""
    if not expected:
        st.error("Auth misconfigured: FENCE_AUTH_MODE=streamlit_password requires "
                 "FENCE_APP_PASSWORD env var or [auth].password in secrets.toml.")
        st.stop()
    st.markdown("## Sign in")
    with st.form("_auth_form", clear_on_submit=False):
        email = st.text_input("Email", key="_auth_email_input")
        pw = st.text_input("Password", type="password", key="_auth_pw_input")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if pw == expected and email.strip():
            st.session_state["_auth_ok"] = True
            st.session_state["_auth_email"] = email.strip()
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()


def require_auth():
    """Block unauthenticated requests when AUTH_MODE != 'none'. Must run
    before any session/cache-scoped logic that depends on user identity."""
    if AUTH_MODE == "none":
        return
    if AUTH_MODE == "streamlit_password":
        _streamlit_password_gate()
        return
    # proxy_header / streamlit_oidc — just assert identity is present.
    if get_user_id().startswith("dev_"):
        st.error("Authentication required. Please sign in and retry.")
        st.stop()


def _render_auth_widget():
    """Small sidebar block showing who's logged in. Shown only when auth active."""
    if AUTH_MODE == "none":
        return
    email = get_user_email() or st.session_state.get("_auth_email")
    if not email:
        return
    with st.sidebar:
        st.markdown(f"**Signed in:** `{email}`")
        if AUTH_MODE == "streamlit_password":
            if st.button("Log out", key="_auth_logout"):
                for k in ("_auth_ok", "_auth_email"):
                    st.session_state.pop(k, None)
                st.rerun()


def initialize_session_state(session_id_val):
    print(f"SESSION {session_id_val} LOG: Initializing/checking session state.")
    default_state = {
        'session_id': session_id_val,
        'fence_pages': [],
        'non_fence_pages': [],
        'total_pages_processed_count': 0,
        'doc_total_pages': 0,
        'processing_complete': False,
        'analysis_halted_due_to_error': False,
        'fence_keywords_app': [
            'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 'mesh',
            'panel', 'chain link', 'masonry', 'fence details', 'canopy shading',
            'adot specifications', 'mag specifications', 'rail', 'railing',
            'bollards', 'handrails', 'wall', 'cmu',
            'operator', 'davis', 'bacon', 'davis-bacon', 'davis – bacon',
            'buy america', 'american', 'dug out',
        ],
        'run_analysis_triggered': False,
        'uploaded_pdf_name': None,
        'original_pdf_bytes': None,
        'current_pdf_hash': None,
        'highlighted_pdf_bytes_for_download': None,
        'last_uploaded_file_id': None,
        # Default to gpt-5.1 (same as app_ade.py). Earlier I'd set this
        # to gpt-5-mini to save tokens, but mini is too weak to pick
        # the specific fence layers out of a site-plan PDF's many
        # drawing layers in measure_fence_elements →
        # llm_identify_fence_layers. When LLM returns nothing the code
        # falls back to a keyword match like 'WALL'/'FENC'/'BARRIER'/
        # 'GATE' that on a typical site plan over-matches every wall
        # in the building, grabs >5000 lines, and triggers
        # measurement_method="skipped" — no per-indicator
        # categorisation, blank UMT canvas. gpt-5.1 picks the correct
        # specific layer (e.g. "FENCE-EXISTING") and the page lands
        # under the 5000-line cap with a proper "layer" result.
        # FENCE_ANALYSIS_MODEL env var still wins if set.
        'selected_model_for_analysis': os.environ.get("FENCE_ANALYSIS_MODEL", "gpt-5.1"),
        # Unified measurement storage
        'unified_measurements': {},  # {page_key: {'auto_lines': [...], 'manual_lines': [...], 'drawn_lines': [...]}}
        'per_page_scale_info': {},
        'page_categories': {},
        'active_category_per_page': {},
        'element_details': {},  # {element_name: {height, post_spacing, material, ...}}
        'fence_page_texts': {},  # {page_number: full_text} for cross-page detail extraction
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else \
                                    dict(value) if isinstance(value, dict) else \
                                    value
        elif key == 'session_id' and st.session_state.session_id != session_id_val:
            st.session_state.session_id = session_id_val


def _is_lock_holder_alive(lock_info: dict) -> bool:
    """Check if the process that created the lock is still running."""
    pid = lock_info.get("pid")
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        # Verify it's actually a streamlit process (not a recycled PID)
        return "streamlit" in " ".join(proc.cmdline()).lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def acquire_analysis_lock(session_id: str):
    """Ensure only one heavyweight analysis runs at a time."""
    now = int(time.time())
    owner = {}

    if os.path.exists(ANALYSIS_LOCK_PATH):
        try:
            with open(ANALYSIS_LOCK_PATH, "r", encoding="utf-8") as f:
                owner = json.load(f)
        except Exception:
            owner = {}

        started_at = int(owner.get("started_at", 0) or 0)
        # Process alive → lock is always valid (supports long-running analysis)
        if started_at and _is_lock_holder_alive(owner):
            return False, owner
        # Process dead or can't check — use TTL as fallback safety ceiling
        if started_at and (now - started_at) <= ANALYSIS_LOCK_TTL_SECONDS:
            print(f"LOG: Lock holder PID {owner.get('pid')} is dead, clearing stale lock.")
        # Stale, expired, or orphaned lock
        try:
            os.remove(ANALYSIS_LOCK_PATH)
        except Exception:
            pass

    lock_data = {
        "session_id": session_id,
        "pid": os.getpid(),
        "started_at": now,
    }
    with open(ANALYSIS_LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(lock_data, f)
    return True, lock_data


def release_analysis_lock(session_id: str):
    if not os.path.exists(ANALYSIS_LOCK_PATH):
        return
    try:
        with open(ANALYSIS_LOCK_PATH, "r", encoding="utf-8") as f:
            owner = json.load(f)
    except Exception:
        owner = {}

    if owner.get("session_id") == session_id:
        try:
            os.remove(ANALYSIS_LOCK_PATH)
        except Exception:
            pass


# ==============================================================================
# Concurrent-analyses slot manager (semaphore-of-N) + RSS capacity guard
# ==============================================================================
# Replaces the binary single-analysis lock. Up to FENCE_MAX_CONCURRENT
# analyses may run simultaneously in the same process. Each active
# analysis holds one slot in a shared JSON file; dead-PID slots are
# auto-reclaimed. A soft RSS guard blocks new slot acquisitions when
# the process is near the systemd memory ceiling, so the third user
# gets "server at capacity, try in 1 min" instead of an OOM kill.

ANALYSIS_SLOTS_PATH = "/tmp/fence_analysis.slots"
ANALYSIS_WAITERS_PATH = "/tmp/fence_analysis.waiters"
FENCE_MAX_CONCURRENT = _workers("FENCE_MAX_CONCURRENT", 2, cap=6)
# One user running one analysis at a time is the norm; cap=3 lets ops
# bump it for power users without opening the door to a single user
# hogging every slot.
FENCE_MAX_CONCURRENT_PER_USER = _workers("FENCE_MAX_CONCURRENT_PER_USER", 1, cap=3)
# Soft ceiling: reject new slot acquisitions when RSS is above this
# percentage of MemoryMax. At 12 GB total with 0.75 factor → reject
# above 9 GB. Existing analyses keep running unaffected.
FENCE_RSS_REJECT_GB = float(os.environ.get("FENCE_RSS_REJECT_GB", "9.0"))


def _read_slots():
    """Return list of slot dicts currently alive on disk. Auto-reaps
    slots whose owning PID is dead or whose TTL expired."""
    if not os.path.exists(ANALYSIS_SLOTS_PATH):
        return []
    try:
        with open(ANALYSIS_SLOTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    now = int(time.time())
    alive = []
    for s in raw:
        try:
            started = int(s.get("started_at", 0))
        except Exception:
            started = 0
        if started and (now - started) > ANALYSIS_LOCK_TTL_SECONDS:
            continue  # TTL expired — treat as dead
        if _is_lock_holder_alive(s):
            alive.append(s)
    return alive


def _write_slots(slots):
    tmp = ANALYSIS_SLOTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(slots, f)
    os.replace(tmp, ANALYSIS_SLOTS_PATH)


def _current_rss_gb():
    try:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def acquire_analysis_slot(session_id: str):
    """Take one of FENCE_MAX_CONCURRENT slots. Returns (ok, info).

    On failure:
      info['reason']   = 'busy' | 'capacity' | 'already_holding'
      info['active']   = list of current slot holders (for UX messaging)
      info['rss_gb']   = current process RSS

    On success:
      info is the slot record we wrote.
    """
    rss_gb = _current_rss_gb()
    slots = _read_slots()
    user_id = get_user_id()

    # Same session re-acquiring? return existing slot silently.
    for s in slots:
        if s.get("session_id") == session_id:
            return True, s

    # Per-user concurrent-analysis cap.
    user_slots = [s for s in slots if s.get("user_id") == user_id]
    if len(user_slots) >= FENCE_MAX_CONCURRENT_PER_USER:
        telemetry.event("slot_rejected", session_id=session_id,
                        reason="user_busy", user_id=user_id,
                        active_for_user=len(user_slots),
                        limit_per_user=FENCE_MAX_CONCURRENT_PER_USER)
        return False, {
            "reason": "user_busy",
            "active": slots,
            "active_for_user": user_slots,
            "limit_per_user": FENCE_MAX_CONCURRENT_PER_USER,
        }

    # RSS soft ceiling — reject only if at least one slot is already
    # held (don't block the first analysis just because the process is
    # already chunky from a prior run's leftovers).
    if slots and rss_gb >= FENCE_RSS_REJECT_GB:
        telemetry.event("slot_rejected", session_id=session_id,
                        reason="capacity", rss_gb=round(rss_gb, 2),
                        active_count=len(slots),
                        limit_gb=FENCE_RSS_REJECT_GB)
        return False, {
            "reason": "capacity",
            "active": slots,
            "rss_gb": round(rss_gb, 2),
            "limit_gb": FENCE_RSS_REJECT_GB,
        }

    if len(slots) >= FENCE_MAX_CONCURRENT:
        telemetry.event("slot_rejected", session_id=session_id,
                        reason="busy", rss_gb=round(rss_gb, 2),
                        active_count=len(slots),
                        limit=FENCE_MAX_CONCURRENT)
        return False, {
            "reason": "busy",
            "active": slots,
            "rss_gb": round(rss_gb, 2),
            "limit": FENCE_MAX_CONCURRENT,
        }

    new_slot = {
        "session_id": session_id,
        "user_id": get_user_id(),
        "pid": os.getpid(),
        "started_at": int(time.time()),
    }
    slots.append(new_slot)
    try:
        _write_slots(slots)
    except Exception as e:
        print(f"[slots] write failed: {e}")
        telemetry.event("slot_rejected", session_id=session_id,
                        reason="io_error", error=str(e))
        return False, {"reason": "io_error", "error": str(e)}
    telemetry.event("slot_acquired", session_id=session_id,
                    rss_gb=round(rss_gb, 2),
                    active_count=len(slots),
                    limit=FENCE_MAX_CONCURRENT)
    return True, new_slot


def release_analysis_slot(session_id: str):
    # Diagnostic: every slot release now logs the top user-code frame that
    # called it. Helps untangle "which path fired?" when a run dies
    # unexpectedly — button click, finally, file-missing bail-out, etc.
    try:
        import traceback as _tb
        _stack = _tb.extract_stack(limit=8)
        # Drop this frame itself; pick the nearest caller inside app_ade_fast.
        _caller = "unknown"
        for f in reversed(_stack[:-1]):
            if "app_ade_fast.py" in (f.filename or ""):
                _caller = f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                break
    except Exception:
        _caller = "trace_failed"

    slots = _read_slots()
    held = any(s.get("session_id") == session_id for s in slots)
    new_slots = [s for s in slots if s.get("session_id") != session_id]
    try:
        _write_slots(new_slots)
    except Exception as e:
        print(f"[slots] release write failed: {e}")
    if held:
        telemetry.event("slot_released", session_id=session_id,
                        active_count=len(new_slots),
                        caller=_caller)


# --- FIFO waiter queue (Stage E11) ---
# Persistent JSON list of sessions waiting for a slot. Rejected
# acquisitions enqueue; the UI polls queue_position() on rerun and
# shows "You are #N in queue". First-in-first-out is enforced by
# comparing enqueued_at when deciding who's allowed to acquire next.

def _read_waiters() -> list:
    if not os.path.exists(ANALYSIS_WAITERS_PATH):
        return []
    try:
        with open(ANALYSIS_WAITERS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    now = int(time.time())
    alive = []
    for w in raw:
        # Drop waiters whose PID is dead or whose TTL expired — same
        # safety rules as _read_slots.
        try:
            started = int(w.get("enqueued_at", 0))
        except Exception:
            started = 0
        if started and (now - started) > ANALYSIS_LOCK_TTL_SECONDS:
            continue
        if _is_lock_holder_alive(w):
            alive.append(w)
    return alive


def _write_waiters(waiters: list) -> None:
    tmp = ANALYSIS_WAITERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(waiters, f)
    os.replace(tmp, ANALYSIS_WAITERS_PATH)


def enqueue_waiter(session_id: str) -> int:
    """Add session to the waiters list if not already there; return its
    1-based FIFO position."""
    waiters = _read_waiters()
    for i, w in enumerate(waiters):
        if w.get("session_id") == session_id:
            return i + 1
    waiters.append({
        "session_id": session_id,
        "user_id": get_user_id(),
        "pid": os.getpid(),
        "enqueued_at": int(time.time()),
    })
    # Sort to preserve FIFO across re-reads.
    waiters.sort(key=lambda w: int(w.get("enqueued_at", 0)))
    try:
        _write_waiters(waiters)
    except Exception as e:
        print(f"[waiters] write failed: {e}")
    try:
        telemetry.event("waiter_enqueued", session_id=session_id,
                        position=len(waiters))
    except Exception:
        pass
    return len(waiters)


def dequeue_waiter(session_id: str) -> None:
    waiters = _read_waiters()
    new = [w for w in waiters if w.get("session_id") != session_id]
    if len(new) == len(waiters):
        return
    try:
        _write_waiters(new)
    except Exception as e:
        print(f"[waiters] dequeue write failed: {e}")


def queue_position(session_id: str) -> int:
    """1-based position for this session, or 0 if not queued."""
    for i, w in enumerate(_read_waiters()):
        if w.get("session_id") == session_id:
            return i + 1
    return 0


current_session_id = get_session_id()
# Gate everything that follows behind auth (no-op when AUTH_MODE=none).
require_auth()
_render_auth_widget()
initialize_session_state(current_session_id)


# --- fence_cache wrappers (per-session ephemeral) ---
# Cache lives for exactly one analysis run. Scope key = session UUID, so
# each browser session / analysis gets its own dir under the cache root.
# The dir is purged on analysis end (finally block) AND when the user
# uploads a new PDF. Nothing persists across runs.
def _cache_scope() -> str:
    """Per-session cache dir name. Never shared, never reused."""
    return f"session_{get_session_id()}"

def _cache_get(phase, pdf_sha, params, page_idx=None, user_scope=None):
    # user_scope= lets a caller pin the scope explicitly. Phase 2 / Phase 3
    # worker threads MUST pass this — calling _cache_scope() from a
    # ThreadPoolExecutor worker gives you a bogus ghost session because
    # `st.session_state` isn't bound to the thread's script run context,
    # so the fallback creates a fresh uuid. Data then lives under one
    # user scope and gets looked up under another.
    return fence_cache.get(phase, pdf_sha, params, page_idx=page_idx,
                           user_scope=user_scope or _cache_scope())

def _cache_put(phase, pdf_sha, params, value, page_idx=None, user_scope=None):
    return fence_cache.put(phase, pdf_sha, params, value, page_idx=page_idx,
                           user_scope=user_scope or _cache_scope())

def _cache_probe(phase, pdf_sha, params, page_indices=None, user_scope=None):
    return fence_cache.probe(phase, pdf_sha, params, page_indices=page_indices,
                             user_scope=user_scope or _cache_scope())

def _reset_analysis_state(purge_cache: bool = True, preserve_uploader: bool = False):
    """Tear down all state related to the current analysis so the user
    can upload/start fresh. Called by the "New Analysis" / "Cancel" buttons.

    - Releases the analysis slot (so another user isn't blocked)
    - Clears the uploaded PDF from disk
    - Resets analysis-related session_state keys (fence/non_fence_pages,
      measurement dicts, etc.) while preserving user preferences
      (classifier choice, keyword list, fence keywords)
    - Optionally purges the per-session cache dir

    Does NOT kill the Python process. Next script rerun starts clean.
    """
    try:
        release_analysis_slot(current_session_id)
    except Exception:
        pass
    try:
        _cleanup_pdf_on_disk(current_session_id)
    except Exception:
        pass
    # Preserve user preferences
    _keep = {
        'session_id': st.session_state.get('session_id'),
        'selected_model_for_analysis': st.session_state.get('selected_model_for_analysis'),
        'fence_keywords_app': st.session_state.get('fence_keywords_app'),
    }
    # Clear all analysis / measurement / per-page state. Wholesale wipe
    # of st.session_state keys we control; user interaction widgets keep
    # their own Streamlit-managed state.
    _clear_prefixes = (
        'base_img_', 'drawn_img_', 'line_stats_', 'lines_',
        'auto_synced_', 'auto_matched_indices_',
        'orig_img_size_', 'base_img_size_', 'click_key_',
        '_phase3_', '_img_cache', '_page_img_loaded_',
        '_umt_pg_loaded_', '_umt_tool_loaded',
    )
    _clear_exact = {
        'fence_pages', 'non_fence_pages', 'processing_complete',
        'analysis_halted_due_to_error', 'run_analysis_triggered',
        'uploaded_pdf_name', 'original_pdf_bytes', 'current_pdf_hash',
        'highlighted_pdf_bytes_for_download', 'last_uploaded_file_id',
        'unified_measurements', 'per_page_scale_info', 'page_categories',
        'active_category_per_page', 'element_details', 'fence_page_texts',
        'total_pages_processed_count', 'doc_total_pages',
        'user_drawn_lines', 'line_assignments', 'drawing_mode',
        'pending_line_start', 'pdf_disk_path', 'broken_pages',
        'highlighted_pdf_filename_for_download', 'last_run_timings',
        'pdf_uploader_main',
    }
    if preserve_uploader:
        # When _reset_analysis_state is invoked from the new-file
        # upload path, the file widget is the one that triggered us —
        # deleting 'pdf_uploader_main' from session_state would drop
        # the just-uploaded file, and the script's next rerun would
        # see file_uploader return None. Keep it in that case.
        _clear_exact.discard('pdf_uploader_main')
    for k in list(st.session_state.keys()):
        if k in _clear_exact or any(k.startswith(p) for p in _clear_prefixes):
            try:
                del st.session_state[k]
            except Exception:
                pass
    # Reinstall preserved preferences so the user doesn't have to reset them.
    for k, v in _keep.items():
        if v is not None:
            st.session_state[k] = v
    if purge_cache:
        try:
            _purge_session_cache()
        except Exception:
            pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # Also drop @st.cache_resource entries (cached LLM / fitz clients).
    # These objects hold httpx connection pools and pymupdf doc handles
    # that can pin 100+ MB of RSS between runs otherwise.
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    gc.collect()
    # Return freed arenas to the OS where we can; without this the
    # glibc heap retains the high-water mark of the previous run.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    print(f"SESSION {current_session_id} LOG: _reset_analysis_state done "
          "(slot released, state cleared)")


def _purge_session_cache():
    """Delete this session's entire cache dir. Safe to call repeatedly;
    swallows errors (cache is best-effort)."""
    try:
        import shutil as _sh
        _dir = fence_cache.cache_root() / _cache_scope()
        if _dir.exists():
            _sh.rmtree(_dir, ignore_errors=True)
            print(f"SESSION {current_session_id} LOG: Purged session cache dir {_dir}")
    except Exception as _pe:
        print(f"SESSION {current_session_id} LOG: purge session cache failed (non-fatal): {_pe}")

# --- Disk-backed PDF storage ------------------------------------------------
# Store uploaded PDFs on disk (/tmp) instead of in session_state RAM.
# Saves up to 500 MB per concurrent session.
_PDF_TMP_DIR = "/tmp/fence_pdfs"
os.makedirs(_PDF_TMP_DIR, exist_ok=True)
MAX_CONCURRENT_UPLOADS = 6  # total sessions with PDFs loaded (viewing is cheap)

# On process start, clean only truly old temp files (>1 hour).
# Walks both the legacy flat layout and the per-user nested layout.
_boot_now = time.time()
def _boot_clean(path):
    try:
        if _boot_now - os.path.getmtime(path) > 3600:
            os.remove(path)
    except Exception:
        pass
for _f in os.listdir(_PDF_TMP_DIR):
    _fp = os.path.join(_PDF_TMP_DIR, _f)
    try:
        if os.path.isdir(_fp):
            for _sub in os.listdir(_fp):
                _boot_clean(os.path.join(_fp, _sub))
        else:
            _boot_clean(_fp)
    except Exception:
        pass


def _user_pdf_dir(user_id: str) -> str:
    """Per-user subdir under _PDF_TMP_DIR so one user's uploads can't
    starve another's disk quota or leak into another's listing."""
    d = os.path.join(_PDF_TMP_DIR, user_id)
    os.makedirs(d, exist_ok=True)
    return d


def _user_disk_usage(user_id: str) -> int:
    """Sum of PDF bytes currently on disk for one user."""
    d = _user_pdf_dir(user_id)
    total = 0
    try:
        for f in os.listdir(d):
            try:
                total += os.path.getsize(os.path.join(d, f))
            except Exception:
                continue
    except Exception:
        pass
    return total


# Per-user disk quota (Stage E10). Uploads that would push a user over this
# size are rejected with a clear UI error. Default 2 GB; ops can tune per
# deployment. Existing on-disk files are counted toward the quota so a
# user can't sidestep by re-uploading to a new session.
FENCE_MAX_USER_BYTES = int(os.environ.get("FENCE_MAX_USER_BYTES", str(2 * 1024**3)))


def _save_pdf_to_disk(pdf_bytes: bytes, session_id: str, pdf_hash: str) -> str:
    """Write PDF bytes to the per-user temp dir and return the path.

    Enforces FENCE_MAX_USER_BYTES. Raises RuntimeError if the quota
    would be exceeded; callers should surface this as a UI error."""
    user_id = get_user_id()
    used = _user_disk_usage(user_id)
    if used + len(pdf_bytes) > FENCE_MAX_USER_BYTES:
        raise RuntimeError(
            f"Per-user disk quota exceeded: {(used + len(pdf_bytes)) / (1024**3):.1f} GB "
            f"> {FENCE_MAX_USER_BYTES / (1024**3):.1f} GB cap. "
            f"Delete unused uploads or contact an admin to raise the quota."
        )
    d = _user_pdf_dir(user_id)
    path = os.path.join(d, f"{session_id}_{pdf_hash}.pdf")
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    print(f"SESSION {session_id} LOG: PDF saved to disk ({len(pdf_bytes)/(1024*1024):.1f} MB) -> {path}")
    return path

def _get_pdf_bytes() -> bytes | None:
    """Read PDF bytes from disk (.pdf or .done). Returns None if missing.
    Touches the file to signal the session is still active."""
    path = st.session_state.get('pdf_disk_path')
    if not path or not os.path.exists(path):
        return None
    try:
        os.utime(path)  # bump mtime → prevents stale cleanup
    except Exception:
        pass
    with open(path, "rb") as f:
        return f.read()

def _cleanup_pdf_on_disk(session_id: str):
    """Remove any temp PDFs for this session (checks all user subdirs
    plus the legacy flat layout for one release cycle)."""
    try:
        # New per-user layout
        for sub in os.listdir(_PDF_TMP_DIR):
            sub_path = os.path.join(_PDF_TMP_DIR, sub)
            if os.path.isdir(sub_path):
                try:
                    for f in os.listdir(sub_path):
                        if f.startswith(f"{session_id}_"):
                            os.remove(os.path.join(sub_path, f))
                except Exception:
                    continue
            elif os.path.isfile(sub_path) and sub.startswith(f"{session_id}_"):
                # Legacy flat layout — still cleaned up for compatibility.
                try:
                    os.remove(sub_path)
                except Exception:
                    pass
    except Exception:
        pass

_STALE_ACTIVE_SECONDS = 120     # 2 min — active upload with no activity → stale
_STALE_DONE_SECONDS = 3600      # 1 hour — finished session temp file cleanup

def _mark_session_done():
    """Rename .pdf → .done so it no longer counts as 'active' but stays
    available for on-demand reads (exports, zoom, measurements)."""
    path = st.session_state.get('pdf_disk_path')
    if path and path.endswith('.pdf') and os.path.exists(path):
        done_path = path.replace('.pdf', '.done')
        try:
            os.rename(path, done_path)
            st.session_state.pdf_disk_path = done_path
            print(f"SESSION LOG: Marked session done: {done_path}")
        except Exception:
            pass

def _count_active_sessions() -> int:
    """Count sessions with uploaded PDFs across all per-user subdirs,
    auto-cleaning stale ones. Only .pdf files count as 'active'
    (pre-analysis or mid-analysis); .done files are finished sessions
    that are cheap to keep around for result viewing."""
    now = time.time()
    active = 0

    def _inspect(fpath: str, fname: str):
        nonlocal active
        try:
            age = now - os.path.getmtime(fpath)
        except Exception:
            return
        if fname.endswith('.pdf'):
            if age > _STALE_ACTIVE_SECONDS:
                try:
                    os.remove(fpath)
                    print(f"LOG: Removed stale active PDF {fname} (age {age/60:.0f} min)")
                except Exception:
                    pass
            else:
                active += 1
        elif fname.endswith('.done'):
            if age > _STALE_DONE_SECONDS:
                try:
                    os.remove(fpath)
                    print(f"LOG: Removed stale done PDF {fname} (age {age/60:.0f} min)")
                except Exception:
                    pass

    try:
        for entry in os.listdir(_PDF_TMP_DIR):
            entry_path = os.path.join(_PDF_TMP_DIR, entry)
            if os.path.isdir(entry_path):
                # Per-user subdir
                try:
                    for f in os.listdir(entry_path):
                        _inspect(os.path.join(entry_path, f), f)
                except Exception:
                    continue
            elif os.path.isfile(entry_path):
                # Legacy flat layout
                _inspect(entry_path, entry)
    except Exception:
        pass
    return active

# --- Memory pressure relief ------------------------------------------------
_DYNAMIC_CACHE_PREFIXES = (
    'base_img_', 'drawn_img_', 'line_stats_', 'lines_',
    'auto_synced_', 'auto_matched_indices_',
    'orig_img_size_', 'base_img_size_', 'click_key_',
)
_RSS_CEILING_GB = float(os.environ.get("FENCE_RSS_CEILING_GB", "6.0"))  # env-tunable; lowered from 8 so the relief kicks earlier

def _check_memory_pressure(label: str = ""):
    """Purge dynamic image/line caches if process RSS exceeds threshold.
    `label` is optional and only used in the telemetry event for context."""
    rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    if rss_gb > _RSS_CEILING_GB:
        purged = 0
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in _DYNAMIC_CACHE_PREFIXES):
                del st.session_state[k]
                purged += 1
        # Also evict the per-session LRU we introduced in Stage B1 — it
        # holds rendered page bytes that are cheap to recompute.
        try:
            if _IMG_CACHE_KEY in st.session_state:
                st.session_state[_IMG_CACHE_KEY].clear()
        except Exception:
            pass
        st.cache_data.clear()
        gc.collect()
        rss_after = psutil.Process().memory_info().rss / (1024 ** 3)
        print(f"SESSION {current_session_id} LOG: Memory pressure relief "
              f"({rss_gb:.1f} -> {rss_after:.1f} GB), purged {purged} keys")
        try:
            telemetry.event("memory_pressure_relief",
                            label=label, rss_before_gb=round(rss_gb, 2),
                            rss_after_gb=round(rss_after, 2),
                            purged=purged)
        except Exception:
            pass

_check_memory_pressure()

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_image_download_link_html(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-button">{text}</a>'


@telemetry.timed("generate_page_images")
def generate_page_images(page_idx, pdf_bytes, definitions, instances, pdf_width, pdf_height):
    """Generate original and highlighted images for a page."""
    try:
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as doc:
            page = doc.load_page(page_idx)
            
            # Original image
            pix_orig = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI)
            original_bytes = pix_orig.tobytes("png")
            
            # Highlighted image
            highlighted_bytes = ade.highlight_page_image(
                original_bytes,
                definitions,
                instances,
                pdf_width,
                pdf_height
            )
            
            return original_bytes, highlighted_bytes
    except Exception as e:
        print(f"SESSION {current_session_id} ERROR: Image generation failed: {e}")
        return None, None


# Single shared fitz Document per session, keyed by PDF hash. Avoids the
# per-call fitz.open() deserialization cost. Main thread only — worker
# threads (e.g. Phase 3) must still open their own Document because
# PyMuPDF docs are NOT thread-safe.
@st.cache_resource(show_spinner=False, max_entries=1)
@telemetry.timed("shared_fitz_doc_open")
def _get_shared_fitz_doc(pdf_bytes_hash):
    """Open the session's PDF once and keep it in memory. Invalidated by
    hash change — when the user uploads a new PDF, a fresh entry is
    created and the old one is evicted (max_entries=2 keeps one in flight
    during the swap)."""
    _path = st.session_state.get('pdf_disk_path')
    if _path and os.path.exists(_path):
        return fitz.open(_path)
    return None


@telemetry.timed("get_page_image_render")
def _render_page_image(_pdf_bytes_hash, pdf_bytes, page_idx, definitions, instances, keyword_matches,
                       pdf_width, pdf_height, highlight, measurement_lines=None, dpi=None):
    """Raw renderer — no caching. Call through get_page_image_on_demand."""
    effective_dpi = dpi if dpi is not None else 150
    try:
        # Prefer the shared doc (opened once per session) to avoid the
        # fitz.open deserialize cost on every cache miss. Fall back to
        # per-call open if the shared doc is unavailable.
        doc = _get_shared_fitz_doc(_pdf_bytes_hash)
        if doc is not None:
            try:
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=effective_dpi)
                original_bytes = pix.tobytes("png")
                del pix
            except Exception as _de:
                # Shared doc went bad — fall through to the per-call open below.
                print(f"[img] shared-doc read failed for page {page_idx}: {_de}; falling back")
                doc = None
        if doc is None:
            with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as _d:
                page = _d.load_page(page_idx)
                pix = page.get_pixmap(dpi=effective_dpi)
                original_bytes = pix.tobytes("png")
                del pix

        highlighted_bytes = None
        if highlight:
            if definitions or instances:
                highlighted_bytes = ade.highlight_page_image(
                    original_bytes, definitions, instances, pdf_width, pdf_height
                )
            elif keyword_matches:
                highlighted_bytes = ade.highlight_keyword_matches(
                    original_bytes, keyword_matches, pdf_width, pdf_height
                )

        return original_bytes, highlighted_bytes
    except Exception as e:
        print(f"ERROR: On-demand image generation failed for page {page_idx}: {e}")
        return None, None


# Per-session LRU of rendered page images (10 entries max).
# Replaces the previous @st.cache_data which was process-global across all
# browser sessions — that both bloated memory and leaked rendered pages
# between users. Session-scoped state gives isolation for free.
_IMG_CACHE_KEY = "_img_cache"      # OrderedDict[tuple, (orig, highlighted)]
_IMG_CACHE_MAX = int(os.environ.get("FENCE_IMG_CACHE_MAX", "10"))

def _img_cache_key(_pdf_bytes_hash, page_idx, definitions, instances, keyword_matches,
                   pdf_width, pdf_height, highlight, measurement_lines, dpi):
    """Deterministic hashable key from render inputs. Hashes list inputs by their
    JSON form so list order stays significant (same args → same key)."""
    def _h(v):
        try:
            return json.dumps(v, sort_keys=True, default=str)
        except Exception:
            return repr(v)
    return (
        _pdf_bytes_hash, int(page_idx),
        _h(definitions), _h(instances), _h(keyword_matches),
        round(float(pdf_width), 3), round(float(pdf_height), 3),
        bool(highlight), _h(measurement_lines),
        int(dpi) if dpi is not None else None,
    )

def get_page_image_on_demand(_pdf_bytes_hash, pdf_bytes, page_idx, definitions, instances, keyword_matches,
                              pdf_width, pdf_height, highlight, measurement_lines=None, dpi=None):
    """Session-scoped LRU wrapper around _render_page_image. Caches up to
    FENCE_IMG_CACHE_MAX=10 entries per session; oldest evicted on insert.

    Rationale: previous @st.cache_data(max_entries=32) was GLOBAL across
    Streamlit sessions, which both bloated RAM under multi-user load and
    risked leaking rendered page bytes between users. A per-session
    OrderedDict in session_state is smaller, isolated, and cheap.
    """
    from collections import OrderedDict
    try:
        cache = st.session_state.get(_IMG_CACHE_KEY)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            st.session_state[_IMG_CACHE_KEY] = cache
    except Exception:
        # Running outside a Streamlit run context (worker thread) — skip cache.
        return _render_page_image(_pdf_bytes_hash, pdf_bytes, page_idx, definitions, instances,
                                  keyword_matches, pdf_width, pdf_height, highlight,
                                  measurement_lines, dpi)

    key = _img_cache_key(_pdf_bytes_hash, page_idx, definitions, instances, keyword_matches,
                         pdf_width, pdf_height, highlight, measurement_lines, dpi)
    if key in cache:
        cache.move_to_end(key)     # mark as recently used
        return cache[key]

    result = _render_page_image(_pdf_bytes_hash, pdf_bytes, page_idx, definitions, instances,
                                keyword_matches, pdf_width, pdf_height, highlight,
                                measurement_lines, dpi)
    cache[key] = result
    cache.move_to_end(key)
    while len(cache) > _IMG_CACHE_MAX:
        cache.popitem(last=False)  # evict oldest
    return result


def generate_combined_highlighted_pdf(original_pdf_bytes, fence_pages_results_list, uploaded_pdf_name_base, session_id):
    """Generate a combined PDF with only fence-related pages highlighted."""
    print(f"SESSION {session_id} LOG: Generating combined highlighted PDF.")
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No data for PDF."
    
    output_doc = fitz.open()
    input_doc = None
    
    try:
        input_doc = fitz.open(stream=BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        print(f"SESSION {session_id} ERROR: Opening original PDF for combined: {e}")
        if output_doc:
            output_doc.close()
        return None, f"Error opening original PDF: {e}"
    
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        if page_idx is None:
            continue
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            
            # Get page rotation and MediaBox dimensions for coordinate transform
            # Coordinates in definitions/instances are in DISPLAY space (after rotation)
            # but draw_rect expects MediaBox space, so we need to reverse the transform
            rotation = page_out.rotation
            mediabox_w = page_out.mediabox.width
            mediabox_h = page_out.mediabox.height
            
            def reverse_rotation_transform(x0, y0, x1, y1):
                """Transform display coords back to MediaBox coords for PDF annotation."""
                if rotation == 0:
                    return x0, y0, x1, y1
                elif rotation == 90:
                    # Display->MediaBox: (x,y) -> (y, mediabox_h - x)
                    return y0, mediabox_h - x1, y1, mediabox_h - x0
                elif rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                elif rotation == 270:
                    # Display->MediaBox: (x,y) -> (mediabox_w - y, x)
                    return mediabox_w - y1, x0, mediabox_w - y0, x1
                return x0, y0, x1, y1
            
            # Draw definition boxes (green)
            definitions = res_data.get('definitions', [])
            for d in definitions:
                mx0, my0, mx1, my1 = reverse_rotation_transform(d['x0'], d['y0'], d['x1'], d['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)
            
            # Draw instance boxes (purple)
            instances = res_data.get('instances', [])
            for inst in instances:
                mx0, my0, mx1, my1 = reverse_rotation_transform(inst['x0'], inst['y0'], inst['x1'], inst['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)
            
            # Draw keyword match boxes (orange) - for fallback detection
            keyword_matches = res_data.get('keyword_matches', [])
            for kw in keyword_matches:
                if all(k in kw for k in ['x0', 'y0', 'x1', 'y1']):
                    mx0, my0, mx1, my1 = reverse_rotation_transform(kw['x0'], kw['y0'], kw['x1'], kw['y1'])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
                    
        except Exception as e_pi:
            print(f"SESSION {session_id} Err process pg {page_idx} for PDF: {e_pi}")
    
    pdf_bytes, fname = None, "error.pdf"
    if len(output_doc) > 0:
        try:
            pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
            base, ext = os.path.splitext(uploaded_pdf_name_base)
            fname = f"{base}_fence_highlights{ext}"
        except Exception as e_s:
            print(f"SESSION {session_id} Err PDF tobytes: {e_s}")
            fname = f"err_save_{uploaded_pdf_name_base}.pdf"
    
    if input_doc:
        input_doc.close()
    if output_doc:
        output_doc.close()
    
    print(f"SESSION {session_id} LOG: Finished generating combined PDF. Success: {pdf_bytes is not None}")
    return (pdf_bytes, fname) if pdf_bytes else (None, fname)


def generate_measurement_pdf(original_pdf_bytes, fence_pages_results_list, line_assignments, user_drawn_lines, 
                             page_categories, session_state, min_line_pts, uploaded_pdf_name_base):
    """Generate PDF with measurement lines highlighted by category."""
    if not fence_pages_results_list or not original_pdf_bytes:
        return None, "No data for PDF."
    
    output_doc = fitz.open()
    input_doc = None
    
    try:
        input_doc = fitz.open(stream=BytesIO(original_pdf_bytes), filetype="pdf")
    except Exception as e:
        if output_doc:
            output_doc.close()
        return None, f"Error opening original PDF: {e}"
    
    sorted_pages = sorted(fence_pages_results_list, key=lambda x: x.get('page_index_in_original_doc', float('inf')))
    
    for res_data in sorted_pages:
        page_idx = res_data.get('page_index_in_original_doc')
        page_num = res_data.get('page_number')
        page_key = f"page_{page_num}"
        
        if page_idx is None:
            continue
        
        try:
            output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
            page_out = output_doc.load_page(len(output_doc) - 1)
            
            rotation = page_out.rotation
            mediabox_w = page_out.mediabox.width
            mediabox_h = page_out.mediabox.height
            
            def reverse_rotation_transform(x0, y0, x1, y1):
                if rotation == 0:
                    return x0, y0, x1, y1
                elif rotation == 90:
                    return y0, mediabox_h - x1, y1, mediabox_h - x0
                elif rotation == 180:
                    return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                elif rotation == 270:
                    return mediabox_w - y1, x0, mediabox_w - y0, x1
                return x0, y0, x1, y1
            
            # Draw definition boxes (green)
            definitions = res_data.get('definitions', [])
            for d in definitions:
                mx0, my0, mx1, my1 = reverse_rotation_transform(d['x0'], d['y0'], d['x1'], d['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)
            
            # Draw instance boxes (purple)
            instances = res_data.get('instances', [])
            for inst in instances:
                mx0, my0, mx1, my1 = reverse_rotation_transform(inst['x0'], inst['y0'], inst['x1'], inst['y1'])
                r = fitz.Rect(mx0, my0, mx1, my1)
                r.normalize()
                if not r.is_empty and r.is_valid:
                    page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)
            
            # Draw keyword match boxes (orange)
            keyword_matches = res_data.get('keyword_matches', [])
            for kw in keyword_matches:
                if all(k in kw for k in ['x0', 'y0', 'x1', 'y1']):
                    mx0, my0, mx1, my1 = reverse_rotation_transform(kw['x0'], kw['y0'], kw['x1'], kw['y1'])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
            
            # Get categories for this page
            categories = page_categories.get(page_key, {})
            
            # Auto-detected lines are now included in line_assignments via coordinate matching,
            # so they'll be drawn with category colors below (no separate cyan pass needed)
            
            # Get lines from session state - try multiple key formats
            lines = []
            for key in session_state.keys():
                if key.startswith(f"lines_{page_num}_"):
                    lines = session_state[key]
                    break
            
            page_assignments = line_assignments.get(page_key, {})
            
            for line_idx, category in page_assignments.items():
                idx = int(line_idx) if isinstance(line_idx, str) else line_idx
                if idx < len(lines):
                    line = lines[idx]
                    cat_info = categories.get(category, {})
                    color_rgb = cat_info.get('color', (0, 255, 0))
                    # Convert 0-255 to 0-1
                    color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)

                    if isinstance(line, dict):
                        sx, sy = line['start']
                        ex, ey = line['end']
                    else:
                        sx, sy = line.start
                        ex, ey = line.end
                    mx0, my0, mx1, my1 = reverse_rotation_transform(sx, sy, ex, ey)
                    
                    page_out.draw_line((mx0, my0), (mx1, my1), color=color, width=3.0, overlay=True)
            
            # Draw user-drawn lines
            user_lines = user_drawn_lines.get(page_key, [])
            for ul in user_lines:
                category = ul.get('category')
                cat_info = categories.get(category, {})
                color_rgb = cat_info.get('color', (0, 255, 0))
                color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)
                
                sx, sy = ul['start']
                ex, ey = ul['end']
                mx0, my0, mx1, my1 = reverse_rotation_transform(sx, sy, ex, ey)
                
                page_out.draw_line((mx0, my0), (mx1, my1), color=color, width=3.0, overlay=True)
                # Draw endpoints
                page_out.draw_circle((mx0, my0), 3, color=color, fill=color, overlay=True)
                page_out.draw_circle((mx1, my1), 3, color=color, fill=color, overlay=True)
                
        except Exception as e:
            print(f"Error processing page {page_idx} for measurement PDF: {e}")
    
    base, ext = os.path.splitext(uploaded_pdf_name_base)
    fname = f"{base}_measurements{ext}"
    
    try:
        pdf_bytes = output_doc.tobytes(garbage=2, deflate=True)
    except Exception as e:
        print(f"Error generating PDF bytes: {e}")
        pdf_bytes = None
    
    if input_doc:
        input_doc.close()
    if output_doc:
        output_doc.close()
    
    return (pdf_bytes, fname)


def _lookup_element_details(category: str, element_details: dict) -> dict:
    """Look up element details for a category name, trying exact and fuzzy matching."""
    if not element_details:
        return {}
    # Exact match
    if category in element_details:
        return element_details[category]
    # Case-insensitive match
    cat_lower = category.lower()
    for name, details in element_details.items():
        if name.lower() == cat_lower:
            return details
    # Partial match (category contains element name or vice versa)
    for name, details in element_details.items():
        if name.lower() in cat_lower or cat_lower in name.lower():
            return details
    return {}


def generate_measurement_spreadsheet(fence_pages, line_assignments, user_drawn_lines, page_categories, 
                                     session_state, per_page_scale_info, min_line_pts):
    """Generate Excel workbook (bytes) with multiple sheets: Measurements, Summary, Element Specifications."""
    import io
    rows = []
    element_details = session_state.get('element_details', {})
    
    # Debug: print what we're working with
    print(f"CSV Debug: line_assignments = {line_assignments}")
    print(f"CSV Debug: user_drawn_lines = {user_drawn_lines}")
    print(f"CSV Debug: element_details keys = {list(element_details.keys())}")
    
    # Detail columns to include
    DETAIL_COLS = ['Height', 'Post Type', 'Post Spacing', 'Material', 'Gauge', 'Mesh Size', 'Detail Page', 'Full Details']
    
    def _build_row(page_num, category, row_type, length_feet, length_pts, page_scale):
        """Build a row dict with measurement + detail columns."""
        row = {
            'Page': page_num,
            'Category': category,
            'Type': row_type,
            'Length (ft)': round(length_feet, 2),
            'Length (pts)': round(length_pts, 2),
            'Scale': page_scale,
        }
        # Look up details for this category
        details = _lookup_element_details(category, element_details)
        row['Height'] = details.get('height', '')
        row['Post Type'] = details.get('post_type', '')
        row['Post Spacing'] = details.get('post_spacing', '')
        row['Material'] = details.get('material', '')
        row['Gauge'] = details.get('gauge', '')
        row['Mesh Size'] = details.get('mesh_size', '')
        row['Detail Page'] = details.get('detail_page', '')
        row['Full Details'] = details.get('full_details', '')
        return row
    
    for page_data in fence_pages:
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        
        # Get scale
        scale_info = per_page_scale_info.get(page_key, {})
        page_scale = scale_info.get('verified_scale') or scale_info.get('text_scale') or 360.0
        
        # Get lines from session state - try multiple key formats
        lines = []
        lines_key_found = None
        for key in list(session_state.keys()):
            if key.startswith(f"lines_{page_num}_"):
                lines = session_state[key]
                lines_key_found = key
                break
        
        print(f"CSV Debug: page {page_num}, lines_key={lines_key_found}, num_lines={len(lines)}")
        
        categories = page_categories.get(page_key, {})
        
        # Selected lines (includes auto-matched + manually selected)
        auto_matched = session_state.get(f"auto_matched_indices_{page_key}", set())
        page_assignments = line_assignments.get(page_key, {})
        for line_idx, category in page_assignments.items():
            idx = int(line_idx) if isinstance(line_idx, str) else line_idx
            if idx < len(lines):
                line = lines[idx]
                length_pts = line['length_pts'] if isinstance(line, dict) else line.length_pts
                length_inches = length_pts / 72.0
                length_feet = (length_inches * page_scale) / 12.0
                rows.append(_build_row(
                    page_num, category,
                    'Auto' if idx in auto_matched else 'Selected',
                    length_feet, length_pts, page_scale
                ))
        
        # User-drawn lines
        user_lines = user_drawn_lines.get(page_key, [])
        for ul in user_lines:
            rows.append(_build_row(
                page_num, ul.get('category', 'Uncategorized'),
                'Drawn',
                ul.get('length_feet', 0), ul.get('length_pts', 0), page_scale
            ))
    
    # Define all columns (measurement + detail)
    all_columns = ['Page', 'Category', 'Type', 'Length (ft)', 'Length (pts)', 'Scale'] + DETAIL_COLS
    
    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows)
        
        # Add summary rows
        summary_rows = []
        for cat in df['Category'].unique():
            cat_df = df[df['Category'] == cat]
            summary_row = {
                'Page': 'TOTAL',
                'Category': cat,
                'Type': 'Summary',
                'Length (ft)': round(cat_df['Length (ft)'].sum(), 2),
                'Length (pts)': '',
                'Scale': ''
            }
            # Include details in summary row too
            details = _lookup_element_details(cat, element_details)
            summary_row['Height'] = details.get('height', '')
            summary_row['Post Type'] = details.get('post_type', '')
            summary_row['Post Spacing'] = details.get('post_spacing', '')
            summary_row['Material'] = details.get('material', '')
            summary_row['Gauge'] = details.get('gauge', '')
            summary_row['Mesh Size'] = details.get('mesh_size', '')
            summary_row['Detail Page'] = details.get('detail_page', '')
            summary_row['Full Details'] = details.get('full_details', '')
            summary_rows.append(summary_row)
        
        # Grand total
        grand_row = {
            'Page': 'GRAND',
            'Category': 'TOTAL',
            'Type': 'Summary',
            'Length (ft)': round(df['Length (ft)'].sum(), 2),
            'Length (pts)': '',
            'Scale': ''
        }
        for col in DETAIL_COLS:
            grand_row[col] = ''
        summary_rows.append(grand_row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Ensure column order for measurements sheet (no detail cols cluttering it)
        meas_cols = ['Page', 'Category', 'Type', 'Length (ft)', 'Length (pts)', 'Scale']
        meas_final = [c for c in meas_cols if c in df.columns]
        df_meas = df[meas_final]
        
        # Summary sheet: totals per category + grand total
        summ_final = [c for c in all_columns if c in summary_df.columns]
        df_summ = summary_df[summ_final]
        
        # Element Specifications sheet
        el_details = element_details or {}
        spec_rows = []
        for elem_name, details in el_details.items():
            if any(v for v in details.values() if v):
                spec_rows.append({
                    'Element': elem_name,
                    'Height': details.get('height', ''),
                    'Post Type': details.get('post_type', ''),
                    'Post Spacing': details.get('post_spacing', ''),
                    'Material': details.get('material', ''),
                    'Gauge': details.get('gauge', ''),
                    'Mesh Size': details.get('mesh_size', ''),
                    'Foundation': details.get('foundation', ''),
                    'Gate Info': details.get('gate_info', ''),
                    'Detail Page': details.get('detail_page', ''),
                    'Full Details': details.get('full_details', ''),
                    'Notes': details.get('notes', ''),
                })
        df_specs = pd.DataFrame(spec_rows) if spec_rows else pd.DataFrame()
        
        # Write to Excel with multiple sheets
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_meas.to_excel(writer, sheet_name='Measurements', index=False)
            df_summ.to_excel(writer, sheet_name='Summary', index=False)
            if not df_specs.empty:
                df_specs.to_excel(writer, sheet_name='Element Specifications', index=False)
        return buf.getvalue()
    
    # Return empty Excel with headers if no data
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pd.DataFrame(columns=all_columns).to_excel(writer, sheet_name='Measurements', index=False)
    return buf.getvalue()


# ==============================================================================
# Sidebar (matching app.py structure)
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load secrets if available
    secrets = {}
    if os.path.exists(".streamlit/secrets.toml"):
        secrets = toml.load(".streamlit/secrets.toml")
    
    # 1. OpenAI Key
    openai_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        openai_key = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input_sidebar")
    
    # 2. LandingAI (ADE) Key
    ade_key = secrets.get("LANDINGAI_API_KEY", os.getenv("LANDINGAI_API_KEY"))
    if not ade_key:
        ade_key = st.text_input("Enter LandingAI API Key", type="password", key="ade_key_input_sidebar")
    
    # 3. Google Cloud Config (JSON) - load silently without UI messages
    google_cloud_config = None
    try:
        if "google_cloud" in secrets and "gcp_service_account" in secrets:
            google_cloud_config = {
                "project_number": secrets["google_cloud"]["project_number"],
                "location": secrets["google_cloud"]["location"],
                "processor_id": secrets["google_cloud"]["processor_id"],
                "service_account_info": dict(secrets["gcp_service_account"])
            }
            print(f"SESSION {current_session_id} LOG: Google Cloud config loaded from secrets")
    except Exception as e:
        print(f"SESSION {current_session_id} WARNING: Could not load Google Cloud config: {e}")
    
    # Highlight toggle
    st.markdown("---")
    highlight_fence_text_app = st.toggle("🔍 Highlight text & indicators", value=True, key="highlight_toggle")
    
    # ADE usage toggle
    use_ade = st.toggle("🧠 Use ADE (LandingAI)", value=True, key="use_ade_toggle")
    
    # Unified Measurement toggle (auto-detection + interactive editing)
    enable_unified_measurement = st.toggle("📏 Unified Measurements", value=True, key="unified_measurement_toggle",
                                           help="Auto-detect fence lines and interactively select/draw additional lines")
    enable_nonlayer_suggestions = st.toggle("🔬 Non-layer suggestions", value=False, key="nonlayer_suggestions_toggle",
                                            help="Show auto-detected suggestions even when no fence layers found (less reliable)")

    # Low-DPI display mode — renders preview images at 110 dpi instead of
    # 150 dpi. ~50% smaller images = less RAM, faster render, snappier
    # scroll. Exported PDFs stay at 150 dpi. Off by default so existing
    # users see no change.
    low_dpi_mode = st.toggle(
        "⚡ Low-DPI preview (faster)", value=False, key="low_dpi_toggle",
        help="Render page previews at 110 dpi instead of 150. Exported PDFs unaffected."
    )
    DISPLAY_IMAGE_DPI = 110 if low_dpi_mode else 150

    # Debug mode (disabled in UI)
    DEBUG_MODE = False
    
    # Fence Keywords
    st.markdown("---")
    st.subheader("Fence Keywords")
    if 'fence_keywords_app' not in st.session_state:
        st.session_state.fence_keywords_app = ['fence']
    custom_keywords_str = st.text_area(
        "Custom keywords (one per line):",
        "\n".join(st.session_state.fence_keywords_app),
        height=150,
        key="kw_text_area"
    )
    if st.button("Update Keywords", key="update_kw_btn"):
        st.session_state.fence_keywords_app = [k.strip().lower() for k in custom_keywords_str.split("\n") if k.strip()]
        st.rerun()
    
    FENCE_KEYWORDS_APP = st.session_state.fence_keywords_app


# ==============================================================================
# Initialize LLM (cached to avoid re-init on every rerun)
# ==============================================================================

@st.cache_resource
def get_llm_instance(api_key: str, model: str):
    """Cache LLM instance to avoid slow re-initialization on every page load.
    Tightened timeouts (was 180s × 2 retries = 360s worst case) after a
    Phase 3 run stalled 11 min on a single hung API call: 60s × 1 retry
    caps worst-case per call at ~120s so one bad response can't park a
    worker thread indefinitely."""
    print(f"LOG: Creating cached LLM instance for model {model}")
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
        timeout=60,
        max_retries=1,
    )


@st.cache_resource
def get_scale_llm_instance(api_key: str, model: str):
    """Dedicated LLM client for scale vision/text detection. Vision calls
    are slower than text but we still cap at 90s × 1 retry so a hang
    can't stall the whole Phase 3 pool."""
    print(f"LOG: Creating cached SCALE LLM instance for model {model}")
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
        timeout=90,
        max_retries=1,
    )


# Phase 1c fence classification is a simple yes/no task over short
# excerpts. A smaller, faster model (gpt-5.1-mini) is typically 2-3×
# faster per call and ~5× cheaper than the default with no accuracy
# loss on this task. Overridable via env FENCE_CLASSIFIER_MODEL.
FENCE_CLASSIFIER_MODEL = os.environ.get("FENCE_CLASSIFIER_MODEL", "gpt-5-mini")

@st.cache_resource
def get_classifier_llm_instance(api_key: str, model: str):
    """Lean LLM client for Phase 1c page classification. Same interface as
    get_llm_instance, different defaults (shorter timeout, lighter model)."""
    print(f"LOG: Creating cached CLASSIFIER LLM instance for model {model}")
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
        timeout=60,
        max_retries=2,
    )


llm_analysis_instance = None
scale_llm_instance = None
classifier_llm_instance = None
if openai_key:
    try:
        llm_analysis_instance = get_llm_instance(openai_key, st.session_state.selected_model_for_analysis)
        scale_llm_instance = get_scale_llm_instance(openai_key, st.session_state.selected_model_for_analysis)
        classifier_llm_instance = get_classifier_llm_instance(openai_key, FENCE_CLASSIFIER_MODEL)
    except Exception as e:
        st.error(f"LLM Init Error: {e}")
        openai_key = None
        print(f"SESSION {current_session_id} ERROR: LLM Init Error: {e}")


# LLM client warm-up. The first .invoke() on a fresh httpx connection
# pays ~1-3s for TCP + TLS + HTTP/2 handshake to OpenAI. Firing tiny
# async probes at app-start parallelizes that cost with the user's
# upload → click-analyze motion, so their first real call starts warm.
#
# @st.cache_resource on the helper means this runs ONCE per session per
# (key, model) tuple — not every rerun. Probes are fire-and-forget via
# a daemon thread so they never block the UI.
@st.cache_resource(show_spinner=False)
def _warm_llm_connections(key_fingerprint: str, models_csv: str):
    import threading
    def _ping(inst, label):
        try:
            t0 = time.perf_counter()
            r = inst.invoke("ok") if hasattr(inst, "invoke") else None
            dt = time.perf_counter() - t0
            print(f"[warmup] {label} ready in {dt:.2f}s")
        except Exception as _we:
            print(f"[warmup] {label} failed (non-fatal): {_we}")
    if llm_analysis_instance is not None:
        threading.Thread(
            target=_ping, args=(llm_analysis_instance, "analysis-llm"),
            daemon=True, name="warmup-analysis",
        ).start()
    if classifier_llm_instance is not None and classifier_llm_instance is not llm_analysis_instance:
        threading.Thread(
            target=_ping, args=(classifier_llm_instance, "classifier-llm"),
            daemon=True, name="warmup-classifier",
        ).start()
    # Scale LLM is often a distinct model; probe it too if separate.
    if scale_llm_instance is not None \
            and scale_llm_instance is not llm_analysis_instance \
            and scale_llm_instance is not classifier_llm_instance:
        threading.Thread(
            target=_ping, args=(scale_llm_instance, "scale-llm"),
            daemon=True, name="warmup-scale",
        ).start()
    return {"warmed_at": time.time()}

if openai_key and llm_analysis_instance is not None:
    _warm_llm_connections(
        hashlib.sha256(openai_key.encode()).hexdigest()[:8],
        f"{st.session_state.selected_model_for_analysis}|{FENCE_CLASSIFIER_MODEL}",
    )


# ==============================================================================
# Main App Flow
# ==============================================================================

# Sidebar: always-visible "reset" control. When an analysis is running,
# this doubles as a Cancel button (Streamlit re-runs the script on button
# click, which our _reset_analysis_state() handles by releasing the slot
# and clearing session state before the analysis block re-evaluates its
# re-entry condition).
with st.sidebar:
    _reset_label = "🛑 Cancel / Reset"
    _has_active = (
        st.session_state.get('run_analysis_triggered')
        or st.session_state.get('processing_complete')
        or st.session_state.get('uploaded_pdf_name')
    )
    if _has_active:
        if st.button(_reset_label, key="sidebar_reset_btn"):
            _reset_analysis_state(purge_cache=True)
            st.rerun()
        st.caption("Frees the analysis slot and clears uploaded PDF + results.")

st.markdown("<div class='section-header'><h2>📄 Upload Engineering Drawings</h2></div>", unsafe_allow_html=True)
uploaded_pdf_file_obj = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader_main")

if uploaded_pdf_file_obj:
    print(f"SESSION {current_session_id} LOG: PDF uploaded: {uploaded_pdf_file_obj.name}")
    current_file_id = f"{uploaded_pdf_file_obj.name}_{uploaded_pdf_file_obj.size}"
    
    # Guard: reject excessively large files to prevent OOM
    MAX_FILE_SIZE_MB = 500
    file_size_mb = uploaded_pdf_file_obj.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({file_size_mb:.0f} MB). Maximum is {MAX_FILE_SIZE_MB} MB to prevent memory issues.")
        st.stop()
    
    # Guard: cap total loaded PDFs to prevent disk/memory exhaustion
    _active = _count_active_sessions()
    if _active >= MAX_CONCURRENT_UPLOADS and not st.session_state.get('pdf_disk_path'):
        st.warning(f"Server is busy ({_active} active sessions). Please try again in a few minutes.")
        st.stop()

    # Detect if temp PDF was lost (e.g. server restart) — re-save silently
    _existing_path = st.session_state.get('pdf_disk_path')
    if _existing_path and not os.path.exists(_existing_path):
        print(f"SESSION {current_session_id} LOG: Temp PDF missing ({_existing_path}), re-saving.")
        _raw_bytes = uploaded_pdf_file_obj.getvalue()
        _hash = hashlib.sha256(_raw_bytes).hexdigest()
        try:
            st.session_state.pdf_disk_path = _save_pdf_to_disk(_raw_bytes, current_session_id, _hash)
        except RuntimeError as _qe:
            st.error(str(_qe))
            st.stop()
        del _raw_bytes
        # Don't reset state or clear caches — session results are still valid

    if st.session_state.get('last_uploaded_file_id') != current_file_id:
        print(f"SESSION {current_session_id} LOG: New file detected. Full reset for {current_file_id}.")

        # Full tear-down via the same helper the explicit Cancel / New
        # Analysis buttons use. Previously this branch only purged a
        # subset of state (dynamic image-cache prefixes, measurement
        # dicts) and left behind the big-ticket items between runs:
        #   - highlighted_pdf_bytes_for_download (~100-200 MB for a
        #     ~50-fence-page deck)
        #   - _img_cache LRU of rendered PNGs
        #   - @st.cache_resource LLM / DocAI clients with their httpx
        #     connection pools
        # Each re-upload on the same Streamlit session therefore bled
        # ~500-700 MB of RSS the OS could never reclaim. Route through
        # _reset_analysis_state so every known source of retained state
        # gets wiped in one place — if we miss something new, we add it
        # there and every reset path picks it up. preserve_uploader=True
        # keeps the file-uploader widget's state so the file we just
        # accepted from the user isn't dropped before we can save it.
        _reset_analysis_state(purge_cache=True, preserve_uploader=True)

        # Rebuild the default keys (_reset_analysis_state preserves only
        # session_id + user preferences; everything else needs a fresh
        # default before we repopulate with this upload).
        initialize_session_state(current_session_id)

        st.session_state.uploaded_pdf_name = uploaded_pdf_file_obj.name
        _raw_bytes = uploaded_pdf_file_obj.getvalue()
        st.session_state.current_pdf_hash = hashlib.sha256(_raw_bytes).hexdigest()
        # Store PDF on disk instead of RAM (saves up to 500 MB per session)
        _cleanup_pdf_on_disk(current_session_id)  # remove old file first
        try:
            st.session_state.pdf_disk_path = _save_pdf_to_disk(
                _raw_bytes, current_session_id, st.session_state.current_pdf_hash
            )
        except RuntimeError as _qe:
            st.error(str(_qe))
            st.stop()
        st.session_state.original_pdf_bytes = None  # no longer kept in RAM
        del _raw_bytes
        st.session_state.last_uploaded_file_id = current_file_id

        print(f"SESSION {current_session_id} LOG: New-file reset complete; rerunning.")
        st.rerun()
    
    if openai_key and llm_analysis_instance and \
       (ade_key or not use_ade) and \
       not st.session_state.run_analysis_triggered and \
       not st.session_state.processing_complete and \
       not st.session_state.analysis_halted_due_to_error:
        if st.button("▶ Start Analysis", type="primary", key="start_analysis_btn"):
            print(f"SESSION {current_session_id} LOG: Triggering analysis.")
            # User explicitly started a new analysis — wipe any stale
            # cache from prior runs of this same file in this same
            # session so the run is truly fresh. Mid-analysis reruns
            # (browser sleep, widget clicks) do NOT reach this branch;
            # they skip straight through the gate or the post-complete
            # display, so their caches are preserved.
            _purge_session_cache()
            st.session_state.run_analysis_triggered = True
            st.rerun()
        else:
            st.info("Ready to run. Click Start Analysis to begin processing.")


# ==============================================================================
# Analysis Execution Block
# ==============================================================================

if st.session_state.run_analysis_triggered and \
   st.session_state.get('pdf_disk_path') and \
   llm_analysis_instance and \
   (ade_key or not use_ade) and \
   not st.session_state.analysis_halted_due_to_error and \
   not st.session_state.processing_complete:

    slot_acquired, slot_info = acquire_analysis_slot(current_session_id)
    if not slot_acquired:
        reason = slot_info.get("reason", "busy")
        active = slot_info.get("active", []) or []
        if reason == "capacity":
            st.warning(
                f"⚠ Server near memory limit ({slot_info.get('rss_gb', 0)} GB used, "
                f"cap {slot_info.get('limit_gb', 0)} GB). "
                f"{len(active)} analysis(es) already running — please try again in a minute."
            )
        elif reason == "user_busy":
            st.warning(
                f"You already have {slot_info.get('limit_per_user', 1)} analysis running. "
                "Wait for it to finish (or cancel it in the other tab) before starting another."
            )
        elif reason == "busy":
            holders = ", ".join(s.get("session_id", "?")[:8] for s in active[:3])
            oldest = min((int(s.get("started_at", 0)) for s in active), default=0)
            age_min = (int(time.time()) - oldest) // 60 if oldest else 0
            # Enqueue this session so users see consistent FIFO position
            # across browser reruns.
            position = enqueue_waiter(current_session_id)
            st.warning(
                f"All {slot_info.get('limit', FENCE_MAX_CONCURRENT)} analysis slots are "
                f"in use ({holders}; oldest running {age_min} min). "
                f"**You are #{position} in queue.** The page will auto-refresh every 5s."
            )
            # Auto-rerun so the user's rank updates without them clicking.
            time.sleep(5)
            st.rerun()
        else:
            st.warning(f"Could not start analysis: {reason}")

        # Stale-slot recovery: show a force-clear for any slot ≥ 15 min old
        stale = [s for s in active
                 if (int(time.time()) - int(s.get("started_at", 0))) > 900]
        if stale:
            if st.button("🔓 Force clear stale slots (≥15 min)", key="force_clear_slots"):
                for s in stale:
                    release_analysis_slot(s.get("session_id", ""))
                st.success(f"Cleared {len(stale)} stale slot(s). Click **Start Analysis** again.")
                st.rerun()
        st.stop()
    # We got a slot — drop out of the waiter queue if we were in it.
    dequeue_waiter(current_session_id)

    print(f"SESSION {current_session_id} LOG: Starting ADE-based PDF processing "
          f"(slot acquired; {_current_rss_gb():.1f} GB RSS).")
    file_bytes = _get_pdf_bytes()
    if not file_bytes:
        st.error("PDF file is no longer available (server was restarted). Please re-upload the file.")
        st.session_state.run_analysis_triggered = False
        st.session_state.last_uploaded_file_id = None  # force re-upload
        release_analysis_slot(current_session_id)
        st.stop()
    def _keep_session_alive():
        """Touch temp PDF to prevent stale-file cleanup during long analysis runs."""
        _path = st.session_state.get('pdf_disk_path')
        if _path and os.path.exists(_path):
            try:
                os.utime(_path)
            except Exception:
                pass

    # Per-phase memory logging. psutil is already imported for
    # _check_memory_pressure; this just adds a checkpoint helper.
    def _mem_snapshot(label):
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"SESSION {current_session_id} LOG: RSS @ {label}: {rss_mb:.0f} MB")
            return rss_mb
        except Exception:
            return 0.0

    _mem_snapshot("start of analysis")

    # --- ETA helper -----------------------------------------------------
    # Used by every phase's progress hook to show "X left" alongside the
    # current count. Formula is simple linear extrapolation from items
    # completed so far; first-item ETA is suppressed because one data
    # point isn't enough for a stable estimate.
    def _fmt_eta(secs):
        """Human-readable time-remaining. '45s', '2m 30s', '1h 5m'."""
        if secs is None or secs < 0:
            return "…"
        s = int(round(secs))
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"

    # Per-phase ETAs used to show "(12/50, 1m 03s left)" next to the
    # status text, which confused users because the big "Total … Xm
    # left" line above the progress bar used a different formula and
    # almost never agreed. The only ETA the user sees now is the Total
    # line (fed by _update_total_eta); phase status shows only the
    # progress fraction.
    def _eta_suffix(done, total, phase_t0, phase_key=None):
        """Progress suffix — just '(done/total)', no embedded time.

        Also pushes live phase-rate data into _current_phase_tracker so
        the Total ETA line can use it as a more accurate signal.  If
        callers don't pass phase_key we best-effort infer it from the
        _phase_t0 identity (not required — just avoids churn)."""
        if total <= 0:
            return f"(0/0)"
        # Infer phase_key from the current values of _phase*_t0 if not given.
        if phase_key is None:
            for k, v in _PHASE_T0_MAP.items():
                if v is not None and v == phase_t0:
                    phase_key = k
                    break
        if phase_key:
            _track_phase(phase_key, done, total, phase_t0)
        return f"({done}/{total})"

    # Populated as each phase starts so _eta_suffix can tag ETAs with
    # the right phase key. Values are set in the blocks below.
    _PHASE_T0_MAP = {'1a': None, '1b': None, '1c': None,
                     '2': None, '3a': None, '3b': None}

    # Internal tracker so _update_total_eta can use the current phase's
    # live rate (when we're inside a phase's progress loop) as a much
    # more accurate signal than fraction-of-total extrapolation.
    _current_phase_tracker = {"key": None, "done": 0, "total": 0, "t0": 0.0}
    def _track_phase(phase_key, done, total, phase_t0):
        _current_phase_tracker["key"] = phase_key
        _current_phase_tracker["done"] = done
        _current_phase_tracker["total"] = total
        _current_phase_tracker["t0"] = phase_t0

    # Open PDF to get page count. Use the DISK PATH (not BytesIO) so fitz
    # demand-loads pages from the file instead of holding the whole
    # document in memory. This is the single biggest per-analysis memory
    # win — saves roughly as many MB as the PDF is in size.
    _pdf_disk_path_analysis = st.session_state.get('pdf_disk_path')
    doc_proc = None
    try:
        if _pdf_disk_path_analysis and os.path.exists(_pdf_disk_path_analysis):
            doc_proc = fitz.open(_pdf_disk_path_analysis)
        else:
            # Fallback: stream-based (shouldn't happen in normal flow).
            doc_proc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
        st.session_state.doc_total_pages = len(doc_proc)
        print(f"SESSION {current_session_id} LOG: PDF opened, {st.session_state.doc_total_pages} pages.")

        MAX_PAGES = 300
        if st.session_state.doc_total_pages > MAX_PAGES:
            st.error(f"PDF has {st.session_state.doc_total_pages} pages (max {MAX_PAGES}). Please split the document.")
            st.session_state.processing_complete = True
            st.session_state.analysis_halted_due_to_error = True
            _mark_session_done()
            doc_proc.close()
            st.stop()

        # Integrity scan — detect pages damaged by truncated/corrupt xref.
        # Common when a PDF was split by file-size for email and recombined.
        # We probe each page cheaply (just the page dict — fast even on large PDFs);
        # deeper failures (e.g. content-stream objects missing) are caught again
        # by per-phase try/except below and added to broken_pages on the fly.
        _broken_pages = set()
        for _pi in range(st.session_state.doc_total_pages):
            try:
                _p = doc_proc[_pi]
                _ = _p.rect.width  # forces page-dict resolution; fails if page xref missing
            except Exception as _pe:
                _broken_pages.add(_pi)
                print(f"SESSION {current_session_id} WARNING: page {_pi + 1} is damaged ({_pe}) — will be skipped")
        st.session_state.broken_pages = _broken_pages
        if _broken_pages:
            _n_bad = len(_broken_pages)
            _n_total = st.session_state.doc_total_pages
            if _n_bad == _n_total:
                st.error(
                    f"This PDF is damaged — none of its {_n_total} pages could be read. "
                    "This typically happens when a file was split by size (e.g. for email) "
                    "and the pieces were not recombined correctly. Please re-export the original "
                    "PDF using page-range split and re-upload."
                )
                st.session_state.processing_complete = True
                st.session_state.analysis_halted_due_to_error = True
                _mark_session_done()
                doc_proc.close()
                st.stop()
            _bad_sorted = sorted(p + 1 for p in _broken_pages)
            _bad_preview = ", ".join(str(p) for p in _bad_sorted[:15])
            if len(_bad_sorted) > 15:
                _bad_preview += f", … (+{len(_bad_sorted) - 15} more)"
            st.warning(
                f"⚠ This PDF is partially damaged: **{_n_bad} of {_n_total} pages** are unreadable "
                f"(pages: {_bad_preview}). Analysis will continue on the {_n_total - _n_bad} "
                "readable pages; the damaged pages will be marked as skipped. "
                "Tip: if you split the PDF for email, split by *page range* rather than by file size."
            )
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        st.session_state.processing_complete = True
        st.session_state.analysis_halted_due_to_error = True
        _mark_session_done()
        if doc_proc:
            doc_proc.close()
        print(f"SESSION {current_session_id} ERROR: Failed to open PDF for processing: {e}")
        st.stop()
    
    # UI Setup (matching app.py)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results (Live)</h2>", unsafe_allow_html=True)
    summary_placeholder = st.empty()
    col_f, col_nf = st.columns(2)
    with col_f:
        st.subheader("✅ Fence-Related Pages")
    with col_nf:
        st.subheader("❌ Non-Fence Pages")
    total_eta_placeholder = st.empty()  # big "Total — Xm elapsed · ~Ym left"
    prog_bar = st.progress(0)           # overall progress bar (all phases)
    phase_eta_placeholder = st.empty()  # secondary "↳ Phase <N> — Xm elapsed · ~Ym left"
    phase_prog_bar = st.progress(0)     # current-phase-only progress bar
    status_txt_area = st.empty()

    # --- Running total-ETA state ---
    # Two estimators feed the big "Total" line:
    #   (a) History median — if this PDF has run before, use the median
    #       wall-clock as a fixed baseline. Accurate from second 0.
    #   (b) Progress extrapolation — elapsed ÷ fraction_done. Kicks in
    #       after we cross ~5% progress. Noisy at the start, tightens up.
    # We show whichever is available; if both, prefer history until
    # the live extrapolation disagrees by more than 30%, then switch.
    _analysis_start_ts = time.perf_counter()
    # Mutable cell (list) so the helper below can update it without needing
    # `nonlocal` — Streamlit runs this script at module scope, where
    # `nonlocal` isn't valid. [current_overall_progress]
    _overall_cell = [0.0]
    _history_median_total_s = None
    try:
        # Load up to 5 prior runs for this exact PDF hash.
        _tdir_peek = fence_cache.cache_root() / "_timings"
        if _tdir_peek.exists():
            _peek_candidates = sorted(
                _tdir_peek.glob(f"*_{_pdf_sha[:12]}.json"),
                key=lambda p: p.stat().st_mtime, reverse=True
            )[:5]
            _peek_totals = []
            for _pp in _peek_candidates:
                try:
                    with open(_pp, "r", encoding="utf-8") as _pf:
                        _pjd = json.load(_pf)
                        _ts = _pjd.get('total_seconds')
                        if _ts and _ts > 0:
                            _peek_totals.append(float(_ts))
                except Exception:
                    continue
            if _peek_totals:
                _history_median_total_s = sorted(_peek_totals)[len(_peek_totals) // 2]
    except Exception:
        pass

    def _update_total_eta(overall_progress=None):
        """Render the big 'N min elapsed, ~M min left' line above the
        progress bar.

        Uses whichever signal is most accurate, in priority order:
          1. **Live phase rate** (if we're inside a tracked phase):
             remaining = this-phase's page-rate-remaining + estimated
             remaining for later phases based on fixed phase weights.
             This matches the implicit ETA the user reads from the
             phase status line, so the two numbers can't disagree.
          2. **History median** from prior runs on this PDF.
          3. **Fraction extrapolation** (old behavior, fallback).
        """
        if overall_progress is not None:
            # Never go backwards (protects against race-y updates).
            _overall_cell[0] = max(_overall_cell[0], min(1.0, overall_progress))
        elapsed = time.perf_counter() - _analysis_start_ts

        remaining = None
        source = None

        # Estimator A (preferred when available): live phase rate
        _ph = _current_phase_tracker
        if (_ph["key"] is not None and _ph["done"] >= 2
                and _ph["total"] > 0 and _ph["t0"] > 0):
            phase_elapsed = time.perf_counter() - _ph["t0"]
            if phase_elapsed >= 1.0:
                rate = _ph["done"] / phase_elapsed
                if rate > 0:
                    phase_remaining = max(0.0, (_ph["total"] - _ph["done"]) / rate)
                    # How much of the OVERALL wall-time budget lives in
                    # phases later than the one we're in?  Use the same
                    # weights as the progress bar, but as an estimate of
                    # the remaining phases' wall time relative to the
                    # current one.  Crude but more stable than the
                    # fraction-based ETA at the phase boundaries.
                    _cur_lo, _cur_hi = _PHASE_SLICE.get(_ph["key"], (0.0, 1.0))
                    _cur_w = max(1e-6, _cur_hi - _cur_lo)
                    _later_w = max(0.0, 1.0 - _cur_hi)
                    later_remaining = phase_remaining * (_later_w / _cur_w)
                    remaining = phase_remaining + later_remaining
                    source = "live phase rate"

        # Estimator B: history median
        if remaining is None and _history_median_total_s and _history_median_total_s > 0:
            remaining = max(0.0, _history_median_total_s - elapsed)
            source = "history"

        # Estimator C: fraction-of-total extrapolation (legacy fallback)
        if remaining is None and _overall_cell[0] >= 0.05:
            total_s_est = elapsed / _overall_cell[0]
            remaining = max(0.0, total_s_est - elapsed)
            source = "overall fraction"

        elapsed_s = _fmt_eta(elapsed)
        if remaining is None:
            line = f"⏱️ **Total** — {elapsed_s} elapsed · estimating total…"
        else:
            tag = ""
            if source == "history":
                tag = " (from last run)"
            line = f"⏱️ **Total** — {elapsed_s} elapsed · **~{_fmt_eta(remaining)} left**{tag}"
        total_eta_placeholder.markdown(line)

        # Secondary line: current phase only. Same source data as the
        # phase status text, so there's no math mismatch possible.
        _PHASE_LABEL = {
            '1a': 'Phase 1a — text',
            '1b': 'Phase 1b — OCR',
            '1c': 'Phase 1c — classify',
            '2':  'Phase 2 — ADE',
            '3a': 'Phase 3 — pre-compute',
            '3b': 'Phase 3 — render',
        }
        if _ph["key"] is not None and _ph["t0"] > 0:
            phase_elapsed = max(0.0, time.perf_counter() - _ph["t0"])
            phase_elapsed_s = _fmt_eta(phase_elapsed)
            label = _PHASE_LABEL.get(_ph["key"], f"Phase {_ph['key']}")
            # Phase-local progress bar value (fraction of this phase).
            phase_frac = 0.0
            if _ph["total"] > 0:
                phase_frac = min(1.0, max(0.0, _ph["done"] / _ph["total"]))
            try:
                phase_prog_bar.progress(phase_frac)
            except Exception:
                pass
            if _ph["done"] >= 2 and phase_elapsed >= 1.0 and _ph["total"] > 0:
                rate = _ph["done"] / phase_elapsed
                phase_rem = max(0.0, (_ph["total"] - _ph["done"]) / rate) if rate > 0 else 0
                phase_line = (f"↳ {label} — {phase_elapsed_s} elapsed · "
                              f"**~{_fmt_eta(phase_rem)} left** "
                              f"({_ph['done']}/{_ph['total']})")
            else:
                phase_line = f"↳ {label} — {phase_elapsed_s} elapsed · starting…"
            phase_eta_placeholder.markdown(phase_line)
        else:
            phase_eta_placeholder.empty()

    # Seed the line so it's visible from t=0 (before any phase completes).
    _update_total_eta(0.0)

    # Phase-weight map for converting per-phase progress into overall
    # progress. Re-weighted from observed wall-time on a 58-fence-page
    # PDF:
    #   1a ~2m, 1b ~1m, 1c ~30s, 2 ~1m, 3a ~8-12m, 3b ~1m
    # Total ≈ 15-18m. Phase 3 pre-compute is the dominant cost, so
    # giving it 55%+ of the bar keeps the overall-fraction ETA and the
    # phase-rate ETA from diverging as much at the 2→3 boundary.
    _PHASE_SLICE = {
        '1a': (0.00, 0.15),
        '1b': (0.15, 0.22),
        '1c': (0.22, 0.26),
        '2':  (0.26, 0.35),
        '3a': (0.35, 0.90),
        '3b': (0.90, 1.00),
    }
    def _phase_overall(phase_key, frac):
        """Map phase-local progress (0..1) to overall progress (0..1)
        using the phase-slice layout above."""
        lo, hi = _PHASE_SLICE.get(phase_key, (0.0, 1.0))
        return lo + max(0.0, min(1.0, frac)) * (hi - lo)
    
    try:
        total_pages = st.session_state.doc_total_pages

        # --- Disk cache setup: key by PDF hash + config fingerprint ---
        # Any flag that affects per-page output must be in params_hash; if it
        # changes between runs, cache entries from the old run are ignored.
        _pdf_sha = st.session_state.get('current_pdf_hash') or hashlib.sha256(file_bytes).hexdigest()
        _cache_params = fence_cache.params_hash(
            model=st.session_state.get('selected_model_for_analysis', ''),
            keywords=tuple(sorted(FENCE_KEYWORDS_APP)),
            use_ade=bool(use_ade and ade_key),
            highlight_fence_text=bool(highlight_fence_text_app),
            unified_measurement=bool(enable_unified_measurement),
            dpi=DISPLAY_IMAGE_DPI,
        )
        # Persist so post-analysis code paths (UMT tab sync, etc.) can
        # read back from phase3_measure cache using the SAME keys the
        # analysis wrote with. Without this, render_page_fragment had
        # no way to reach the cache and raised NameError.
        st.session_state['_pdf_sha_cached'] = _pdf_sha
        st.session_state['_cache_params_cached'] = _cache_params
        print(f"SESSION {current_session_id} LOG: cache key pdf={_pdf_sha[:8]} params={_cache_params}")

        # Per-phase timing accumulators (logged at end of each phase).
        _phase_timings = {}
        _phase_t0 = time.perf_counter()
        telemetry.phase_checkpoint("analysis_start",
                                   session_id=current_session_id,
                                   pdf_sha8=_pdf_sha[:8],
                                   total_pages=total_pages)

        # =================================================================
        # PHASE 1: Pre-filter ALL pages to identify fence pages
        # Text extraction + keyword/LLM scan (no ADE yet)
        #
        # OPTIMIZATION: Split into 3 sub-steps:
        #   1a. Extract native PDF text (CPU, fast, sequential — needs fitz page)
        #   1b. Run Google OCR in parallel (I/O-bound network calls)
        #   1c. Run fence detection (may need LLM, sequential)
        # =================================================================
        _page_cache = {}
        _fence_page_indices = []
        
        # --- Step 1a: Extract native PDF text + prepare OCR inputs (fast, sequential) ---
        _pdf_lines_by_page = {}
        _page_dims = {}
        _single_page_pdfs = {}  # Cache for reuse in Phase 2 fallback
        
        _broken = st.session_state.get('broken_pages', set())
        status_txt_area.text(f"extracting native text from {total_pages} pages…")

        # Per-page hard timeout for Phase 1a text extraction, via SUBPROCESS.
        #
        # Some pages in subtly-damaged PDFs cause fitz.get_text("dict") to
        # enter MuPDF's internal recovery loop — 100% CPU, no progress, never
        # returns. The loop happens in C code that holds the GIL without
        # yielding, so no Python-level timeout (signal.alarm, threading
        # timeout, asyncio) can interrupt it — we tried; the main thread
        # also blocks trying to reacquire the GIL.
        #
        # The only reliable escape hatch is to run extraction in a separate
        # OS process and SIGKILL it on timeout. ops/page_extractor.py reads
        # the PDF from disk and prints one JSON line; subprocess.run with
        # timeout gives us a hard time bound.
        import subprocess, json as _json
        PHASE_1A_PAGE_TIMEOUT = 20  # seconds — healthy pages finish in < 1s
        _extractor_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ops", "page_extractor.py"
        )
        _pdf_disk_path_for_extract = st.session_state.get('pdf_disk_path')

        def _extract_via_subprocess(page_idx, timeout):
            """Run get_native_pdf_lines in a subprocess; SIGKILL on timeout.
            Returns (lines, error_or_None). Always returns within ~timeout
            seconds, regardless of MuPDF's internal behavior."""
            if not _pdf_disk_path_for_extract or not os.path.exists(_pdf_disk_path_for_extract):
                return None, RuntimeError("PDF disk file missing — cannot extract out-of-process")
            try:
                result = subprocess.run(
                    [sys.executable, _extractor_script, _pdf_disk_path_for_extract, str(page_idx)],
                    capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return None, TimeoutError(
                    f"page extraction exceeded {timeout}s — MuPDF recovery loop on malformed content"
                )
            except Exception as _se:
                return None, RuntimeError(f"subprocess launch failed: {_se}")
            if not result.stdout:
                return None, RuntimeError(
                    f"worker produced no output (rc={result.returncode}): {result.stderr[:200]}"
                )
            try:
                resp = _json.loads(result.stdout.strip().splitlines()[-1])
            except Exception as _je:
                return None, RuntimeError(f"could not parse worker output: {_je}; raw={result.stdout[:200]}")
            if resp.get("ok"):
                return resp.get("lines", []), None
            return None, RuntimeError(resp.get("error", "unknown worker failure"))

        # --- Batched subprocess extraction (Phase 1a speedup) ------------------
        # One subprocess launch per page is ~300ms of pure startup overhead
        # (Python import + fitz.open). For a 132-page PDF that's ~40s just
        # to pay the launch cost. Batching amortises this — a single
        # subprocess opens the PDF once and extracts N pages before
        # exiting. Timeout scales linearly with batch size plus a small
        # process-startup budget. On batch timeout/parse failure we
        # transparently fall back to single-page extraction so one bad
        # page doesn't poison the whole batch.
        FENCE_PHASE1A_BATCH_SIZE = _workers("FENCE_PHASE1A_BATCH_SIZE", 5, cap=25)

        def _extract_batch_via_subprocess(batch_indices, per_page_timeout):
            """Batch variant. Returns {page_idx: (lines, err)} for pages
            that finished inside the subprocess before timeout.

            The subprocess writes one JSON line per page so partial
            progress survives even when MuPDF hangs on a later page and
            we SIGKILL the subprocess. Pages the subprocess never
            reported on stay out of the returned dict — the caller
            retries those single-page so we still bound their time."""
            if not _pdf_disk_path_for_extract or not os.path.exists(_pdf_disk_path_for_extract):
                return None
            pages_csv = ",".join(str(pi) for pi in batch_indices)
            # 3s startup budget + per-page slack. Healthy pages finish
            # in a handful of ms so in practice we rarely hit this cap.
            batch_timeout = 3 + per_page_timeout * max(1, len(batch_indices))
            import subprocess as _sp
            import threading as _th
            proc = _sp.Popen(
                [sys.executable, _extractor_script, _pdf_disk_path_for_extract,
                 "--pages", pages_csv],
                stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
            )
            collected = {}
            # Read stdout in a thread so we can enforce a hard timeout.
            def _drain():
                try:
                    for line in proc.stdout:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        pr = obj.get("page_result")
                        if pr and pr.get("page_idx") is not None:
                            pi = pr["page_idx"]
                            if pr.get("ok"):
                                collected[pi] = (pr.get("lines", []), None)
                            else:
                                collected[pi] = (None, RuntimeError(pr.get("error", "extract failed")))
                except Exception:
                    pass
            t = _th.Thread(target=_drain, daemon=True)
            t.start()
            try:
                proc.wait(timeout=batch_timeout)
            except _sp.TimeoutExpired:
                # Kill the hung subprocess; keep whatever we've already drained.
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            t.join(timeout=1)
            # Return whatever we got; caller fills in the rest.
            return collected

        _phase1a_cache_hits = 0
        _phase1a_t0 = _phase_t0  # local so ETA is pinned to start of 1a
        # Flip the phase tracker immediately so the UI Phase line shows
        # "Phase 1a" from t=0 rather than lagging behind the status text
        # until the first page completes.
        _track_phase('1a', 0, total_pages, _phase1a_t0)
        _update_total_eta()
        _phase1a_done = 0

        # --- Phase 1a: page-dims pass (fast, sequential — reads doc_proc) ---
        # Collect page dimensions + cache hits on the main thread first,
        # then dispatch actual extraction to a subprocess pool. doc_proc
        # is NOT thread-safe, so all doc_proc access stays serial.
        _pages_needing_extract = []
        for page_idx in range(total_pages):
            _keep_session_alive()
            if page_idx in _broken:
                _page_dims[page_idx] = (0, 0)
                _pdf_lines_by_page[page_idx] = []
                continue
            try:
                page = doc_proc[page_idx]
                _page_dims[page_idx] = (page.rect.width, page.rect.height)
            except Exception as _e:
                print(f"SESSION {current_session_id} WARNING: Phase 1a page {page_idx + 1} open failed: {_e} — skipping")
                _broken.add(page_idx)
                _page_dims[page_idx] = (0, 0)
                _pdf_lines_by_page[page_idx] = []
                continue

            _cached_1a = _cache_get("phase1a", _pdf_sha, _cache_params, page_idx=page_idx)
            if _cached_1a is not None:
                _pdf_lines_by_page[page_idx] = _cached_1a
                _phase1a_cache_hits += 1
                continue
            _pages_needing_extract.append(page_idx)

        # --- Phase 1a: extraction pass (batched parallel subprocesses) ---
        # Batching amortises the ~300ms-per-launch Python+fitz startup
        # cost. FENCE_WORKERS_PHASE1A batches run in parallel; each
        # batch contains FENCE_PHASE1A_BATCH_SIZE pages. If a batch
        # times out (one damaged page can hang MuPDF for the entire
        # batch), we transparently fall back to per-page extraction
        # for just that batch so we still isolate the bad page.
        if _pages_needing_extract:
            # Drives the tracker for the Phase ETA line (no UI text — it
            # duplicated what the ETA line already shows).
            _eta_suffix(0, len(_pages_needing_extract), _phase1a_t0, phase_key='1a')
            status_txt_area.empty()

            _batches_1a = [
                _pages_needing_extract[i:i + FENCE_PHASE1A_BATCH_SIZE]
                for i in range(0, len(_pages_needing_extract), FENCE_PHASE1A_BATCH_SIZE)
            ]
            print(f"SESSION {current_session_id} LOG: Phase 1a — {len(_batches_1a)} batches "
                  f"of up to {FENCE_PHASE1A_BATCH_SIZE} pages, {FENCE_WORKERS_PHASE1A} workers")

            def _run_1a_batch(batch):
                """Try batch extract, then single-page fallback for any
                page the batch subprocess didn't report on (hang SIGKILL
                or probe-skip). Returns {page_idx: (lines, err)} for the
                entire batch."""
                partial = _extract_batch_via_subprocess(batch, PHASE_1A_PAGE_TIMEOUT) or {}
                missing = [pi for pi in batch if pi not in partial]
                for pi in missing:
                    partial[pi] = _extract_via_subprocess(pi, PHASE_1A_PAGE_TIMEOUT)
                return partial

            with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1A) as _p1a_pool:
                _p1a_futs = {_p1a_pool.submit(_run_1a_batch, b): b for b in _batches_1a}
                _p1a_complete = 0
                for _fut in as_completed(_p1a_futs):
                    _keep_session_alive()
                    batch = _p1a_futs[_fut]
                    try:
                        batch_results = _fut.result()
                    except Exception as _fe:
                        batch_results = {pi: (None, _fe) for pi in batch}
                    for pi in batch:
                        lines, err = batch_results.get(pi, (None, RuntimeError("no result returned")))
                        _p1a_complete += 1
                        _phase1a_done += 1
                        if err is not None:
                            print(f"SESSION {current_session_id} WARNING: Phase 1a page {pi + 1}: {err} — marking damaged and skipping")
                            _broken.add(pi)
                            _pdf_lines_by_page[pi] = []
                        else:
                            _pdf_lines_by_page[pi] = lines
                            _cache_put("phase1a", _pdf_sha, _cache_params, lines, page_idx=pi)
                    _ov = _phase_overall('1a', _p1a_complete / max(len(_pages_needing_extract), 1))
                    prog_bar.progress(_ov)
                    # side-effect: updates tracker for Phase ETA line
                    _eta_suffix(_p1a_complete, len(_pages_needing_extract), _phase1a_t0, phase_key='1a')
                    _update_total_eta(_ov)
        st.session_state.broken_pages = _broken
        _phase_timings['1a'] = time.perf_counter() - _phase_t0
        print(f"SESSION {current_session_id} LOG: Phase 1a done in {_phase_timings['1a']:.1f}s "
              f"(cache hits: {_phase1a_cache_hits}/{total_pages - len(_broken)})")
        telemetry.phase_checkpoint("phase1a_end",
                                   session_id=current_session_id,
                                   pdf_sha8=_pdf_sha[:8],
                                   wall_s=round(_phase_timings['1a'], 3),
                                   cache_hits=_phase1a_cache_hits,
                                   total_pages=total_pages - len(_broken))
        _check_memory_pressure("after phase1a")
        # Make sure the total-ETA line reflects Phase 1a completion even
        # when everything was cache-hit (the extraction pool may never
        # have run).
        _update_total_eta(_phase_overall('1a', 1.0))
        _phase_t0 = time.perf_counter()

        # Batch-create single-page PDFs for OCR (reused in Phase 2 fallback)
        # Uses one fitz.open() instead of N separate open/close cycles
        if google_cloud_config:
            for page_idx in range(total_pages):
                if page_idx in _broken:
                    continue
                try:
                    _tmp = fitz.open()
                    _tmp.insert_pdf(doc_proc, from_page=page_idx, to_page=page_idx)
                    _single_page_pdfs[page_idx] = _tmp.tobytes()
                    _tmp.close()
                except Exception as _e:
                    print(f"SESSION {current_session_id} WARNING: single-page PDF prep failed for page {page_idx + 1}: {_e} — skipping")
                    _broken.add(page_idx)
        st.session_state.broken_pages = _broken

        # doc_proc is no longer needed on the main thread:
        #  - Phase 1a populated _page_dims for every page
        #  - Phase 2 workers open their own fitz handle from the disk path
        #  - Phase 3 workers open their own fitz handle
        #  - Phase 3 sequential render step uses the @st.cache_resource
        #    shared doc (also disk-backed)
        # Close it now to reclaim 100-300 MB of fitz internal state.
        _mem_snapshot("before closing doc_proc")
        try:
            doc_proc.close()
        except Exception:
            pass
        doc_proc = None
        gc.collect()
        _mem_snapshot("after closing doc_proc")
        
        # --- Step 1b: Run Google OCR in parallel across all pages ---
        _ocr_lines_by_page = {i: [] for i in range(total_pages)}
        
        if google_cloud_config:
            status_txt_area.text(f"running OCR on {total_pages} pages (batched)…")

            # --- Batched OCR (up to FENCE_OCR_BATCH_SIZE pages per request) ---
            # Google Document AI's sync endpoint accepts a multi-page PDF
            # and returns per-page OCR results in one round trip. That
            # replaces N×1-page calls with ~N/15 calls, cutting network
            # overhead dramatically on larger documents.
            _ocr_page_indices = []
            _phase1b_cache_hits = 0
            for pi in range(total_pages):
                if pi in _broken:
                    continue
                _cached_1b = _cache_get("phase1b", _pdf_sha, _cache_params, page_idx=pi)
                if _cached_1b is not None:
                    _ocr_lines_by_page[pi] = _cached_1b
                    _phase1b_cache_hits += 1
                else:
                    _ocr_page_indices.append(pi)

            _phase1b_t0 = time.perf_counter()
            _track_phase('1b', 0, max(1, len(locals().get('_ocr_page_indices', []) or [])),
                         _phase1b_t0)
            _update_total_eta()
            _ocr_batches = [
                _ocr_page_indices[i:i + FENCE_OCR_BATCH_SIZE]
                for i in range(0, len(_ocr_page_indices), FENCE_OCR_BATCH_SIZE)
            ]
            _pdf_path_for_ocr = st.session_state.get('pdf_disk_path')

            # DocAI hard limit is 40 MB per request. Leave margin for PDF
            # structure overhead and a small safety factor.
            _OCR_TOTAL_MAX_BYTES = int(os.environ.get("FENCE_OCR_TOTAL_MAX_BYTES", str(35 * 1024 * 1024)))

            def _run_ocr_batch(batch_indices):
                """Worker: OCR a batch of pages in one API call.
                Returns {orig_idx: ocr_lines}. On batch failure, falls
                back to per-page single-page OCR for this batch.

                Uses safe_multi_page_pdf_from_path with a generous 10 MB
                per-page cap first. If the total batch still exceeds the
                DocAI 40 MB limit (e.g. 15 pages × 10 MB would blow it),
                rebuilds with a stricter per-page cap so everything fits.
                """
                try:
                    # Build a multi-page PDF containing only this batch
                    if _pdf_path_for_ocr and os.path.exists(_pdf_path_for_ocr):
                        batch_pdf = ade.safe_multi_page_pdf_from_path(
                            _pdf_path_for_ocr, batch_indices,
                            per_page_max_bytes=10 * 1024 * 1024,
                        )
                        # Total-size safety: if combined size still over
                        # DocAI's cap, rebuild with a tighter per-page
                        # budget so every page fits proportionally.
                        if len(batch_pdf) > _OCR_TOTAL_MAX_BYTES:
                            tight_cap = max(
                                256 * 1024,  # absolute minimum 256 KB/page
                                _OCR_TOTAL_MAX_BYTES // max(len(batch_indices), 1),
                            )
                            print(f"SESSION {current_session_id} LOG: OCR batch "
                                  f"({len(batch_indices)}p) rebuilt at tighter cap "
                                  f"{tight_cap/1024/1024:.1f}MB/page "
                                  f"(was {len(batch_pdf)/1024/1024:.1f}MB > "
                                  f"{_OCR_TOTAL_MAX_BYTES/1024/1024:.0f}MB limit)")
                            batch_pdf = ade.safe_multi_page_pdf_from_path(
                                _pdf_path_for_ocr, batch_indices,
                                per_page_max_bytes=tight_cap,
                            )
                    else:
                        # Fallback: stitch from single-page cache
                        _tmpdoc = fitz.open()
                        for _pi in batch_indices:
                            if _pi in _single_page_pdfs:
                                _sp = fitz.open(stream=_single_page_pdfs[_pi], filetype="pdf")
                                _tmpdoc.insert_pdf(_sp, from_page=0, to_page=0)
                                _sp.close()
                        batch_pdf = _tmpdoc.tobytes()
                        _tmpdoc.close()
                    page_dims_by_local = {
                        local: _page_dims[orig] for local, orig in enumerate(batch_indices)
                    }
                    result_by_local = ade.run_google_ocr_blocks_multipage(
                        batch_pdf, google_cloud_config, page_dims_by_local,
                    )
                    out = {}
                    for local, orig in enumerate(batch_indices):
                        out[orig] = result_by_local.get(local, [])
                    return out
                except Exception as _be:
                    print(f"SESSION {current_session_id} WARNING: OCR batch ({len(batch_indices)} pages) "
                          f"failed ({_be}); falling back to per-page")
                    out = {}
                    for pi in batch_indices:
                        try:
                            pdf_w, pdf_h = _page_dims[pi]
                            out[pi] = ade.run_google_ocr_blocks(
                                _single_page_pdfs[pi], google_cloud_config, pdf_w, pdf_h
                            )
                        except Exception as _pe:
                            print(f"SESSION {current_session_id} WARNING: OCR fallback page {pi + 1} failed: {_pe}")
                            out[pi] = []
                    return out

            with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1B) as executor:
                futures = {executor.submit(_run_ocr_batch, b): bi
                           for bi, b in enumerate(_ocr_batches)}
                pages_done = 0
                batches_done = 0
                for future in as_completed(futures):
                    _keep_session_alive()
                    batches_done += 1
                    try:
                        _batch_result = future.result()
                    except Exception as e:
                        bi = futures[future]
                        print(f"SESSION {current_session_id} WARNING: OCR batch {bi} worker crashed: {e}")
                        _batch_result = {pi: [] for pi in _ocr_batches[bi]}
                    for pi, ocr_result in _batch_result.items():
                        _ocr_lines_by_page[pi] = ocr_result
                        _cache_put("phase1b", _pdf_sha, _cache_params, ocr_result, page_idx=pi)
                        pages_done += 1
                    _ov = pages_done / max(len(_ocr_page_indices), 1) * 0.15
                    prog_bar.progress(_ov)
                    _eta_suffix(batches_done, len(_ocr_batches), _phase1b_t0, phase_key='1b')
                    _update_total_eta(_ov)

            _phase_timings['1b'] = time.perf_counter() - _phase_t0
            print(f"SESSION {current_session_id} LOG: Phase 1b done in {_phase_timings['1b']:.1f}s "
                  f"— {len(_ocr_batches)} batches, "
                  f"{len(_ocr_page_indices)} pages, cache hits: {_phase1b_cache_hits}/{total_pages - len(_broken)}")
            telemetry.phase_checkpoint("phase1b_end",
                                       session_id=current_session_id,
                                       pdf_sha8=_pdf_sha[:8],
                                       wall_s=round(_phase_timings['1b'], 3),
                                       cache_hits=_phase1b_cache_hits,
                                       batches=len(_ocr_batches),
                                       pages=len(_ocr_page_indices))
            _check_memory_pressure("after phase1b")
            _phase_t0 = time.perf_counter()
            
            # Free single-page PDFs after OCR — they can be recreated on-demand for ADE fallback
            _single_page_pdfs.clear()
            gc.collect()
            print(f"SESSION {current_session_id} LOG: Freed single-page PDF cache after OCR")
        else:
            # OCR disabled entirely — still mark timing bracket closed.
            _phase_timings['1b'] = time.perf_counter() - _phase_t0
            telemetry.phase_checkpoint("phase1b_end",
                                       session_id=current_session_id,
                                       pdf_sha8=_pdf_sha[:8],
                                       wall_s=round(_phase_timings['1b'], 3),
                                       ocr_disabled=True)
            _phase_t0 = time.perf_counter()
        
        # --- Step 1c: Run fence detection (keyword scan + batched LLM) ---
        #
        # Rewritten to batch LLM calls instead of one-per-page.
        #
        # Flow:
        #   1. Keyword-scan every non-broken, non-cached page (fast, no network).
        #      Precompiled regex — see scan_page_for_keywords_fast.
        #   2. Classify each scanned page as:
        #        a) no_keywords   -> fence_found=False, no LLM needed
        #        b) high_signal   -> fence_found=True,  no LLM needed
        #        c) needs_llm     -> ambiguous; queue for batched LLM confirmation
        #   3. Dispatch (c) in batches of FENCE_CLASSIFY_BATCH_SIZE, run
        #      FENCE_WORKERS_PHASE1C batches concurrently. At 136 pages with
        #      batch=10, worst case becomes ~14 LLM calls instead of 136.
        #
        # Disk cache: phase1c is per-page; any page with a cache hit skips
        # the whole pipeline.

        _pending_indices = []
        _phase1c_cache_hits = 0
        for page_idx in range(total_pages):
            if page_idx in _broken:
                print(f"SESSION {current_session_id} LOG: skipping damaged page {page_idx + 1} in Phase 1c.")
                _page_cache[page_idx] = {
                    'pdf_lines': [],
                    'ocr_lines': [],
                    'prefilter_result': {
                        "fence_found": False,
                        "method": "skipped_damaged",
                        "matched_lines": [],
                    },
                    'skipped_damaged': True,
                }
                continue
            _cached_1c = _cache_get("phase1c", _pdf_sha, _cache_params, page_idx=page_idx)
            if _cached_1c is not None:
                _page_cache[page_idx] = {
                    'pdf_lines': _pdf_lines_by_page[page_idx],
                    'ocr_lines': _ocr_lines_by_page[page_idx],
                    'prefilter_result': _cached_1c,
                }
                if _cached_1c.get("fence_found"):
                    _fence_page_indices.append(page_idx)
                _phase1c_cache_hits += 1
                continue
            _pending_indices.append(page_idx)

        # Mirror the HIGH_SIGNAL set used by fallback_fence_detection_fast;
        # kept locally so Phase 1c's classification decisions are transparent.
        _HIGH_SIGNAL_KW = {
            'fence', 'fencing', 'gate', 'gates', 'chain link', 'guardrail',
            'railing', 'handrail', 'bollard', 'barrier',
        }

        # Step 1 of new flow: run the keyword scan for all pending pages.
        # Collect those that need LLM confirmation into a single queue.
        _needs_llm = []   # list of (page_idx, page_text, matched_keywords)
        _scan_results = {}  # page_idx -> (keyword_result, decision_method or None)
        for page_idx in _pending_indices:
            _kres = ade.scan_page_for_keywords_fast(
                _pdf_lines_by_page[page_idx],
                _ocr_lines_by_page[page_idx],
                FENCE_KEYWORDS_APP,
            )
            _scan_results[page_idx] = _kres
            if not _kres["has_keywords"]:
                continue  # decided: not fence
            matched_lower = {kw.lower() for kw in _kres["matched_keywords"]}
            if matched_lower & _HIGH_SIGNAL_KW:
                continue  # decided: fence via high signal
            # Ambiguous — needs LLM confirmation.
            _all_lines = _pdf_lines_by_page[page_idx] + _ocr_lines_by_page[page_idx]
            _page_text = " ".join(line.get("text", "") for line in _all_lines)
            _needs_llm.append((page_idx, _page_text, _kres["matched_keywords"]))

        # Step 2: batched LLM calls. Split into batches, run in parallel.
        # Use the cheaper/faster classifier model (gpt-5.1-mini by default)
        # — the task is yes/no on short excerpts, doesn't need the heavy
        # extraction model. Falls back to analysis model if classifier
        # init failed.
        _llm_by_page = {}
        _classifier = classifier_llm_instance or llm_analysis_instance
        if _needs_llm and _classifier:
            _batches = [
                _needs_llm[i:i + FENCE_CLASSIFY_BATCH_SIZE]
                for i in range(0, len(_needs_llm), FENCE_CLASSIFY_BATCH_SIZE)
            ]
            _classifier_label = getattr(_classifier, 'model_name', '') or getattr(_classifier, 'model', '') or '?'
            status_txt_area.text(
                f"{len(_needs_llm)} ambiguous pages · {len(_batches)} LLM batch(es) · "
                f"{FENCE_WORKERS_PHASE1C} workers · {_classifier_label}"
            )

            def _run_batch(batch):
                return ade.llm_classify_pages_batch(
                    _classifier, batch, FENCE_KEYWORDS_APP,
                    batch_size=FENCE_CLASSIFY_BATCH_SIZE,
                )

            _phase1c_llm_t0 = time.perf_counter()
            _track_phase('1c', 0, max(1, len(locals().get('_needs_llm', []) or [])),
                         _phase1c_llm_t0)
            _update_total_eta()
            with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE1C) as _cls_pool:
                _futs = {_cls_pool.submit(_run_batch, b): bi for bi, b in enumerate(_batches)}
                _done = 0
                for _fut in as_completed(_futs):
                    _keep_session_alive()
                    _done += 1
                    _ov = 0.15 + _done / max(len(_batches), 1) * 0.15
                    prog_bar.progress(_ov); _update_total_eta(_ov)
                    try:
                        _batch_result = _fut.result()
                        _llm_by_page.update(_batch_result)
                    except Exception as _be:
                        _bi = _futs[_fut]
                        print(f"SESSION {current_session_id} WARNING: LLM batch {_bi} failed: {_be}")
                    _eta_suffix(_done, len(_batches), _phase1c_llm_t0, phase_key='1c')
                    _update_total_eta()

        # Step 3: merge keyword + LLM decisions into prefilter_result per page.
        for page_idx in _pending_indices:
            _kres = _scan_results[page_idx]
            if not _kres["has_keywords"]:
                _prefilter = {
                    "fence_found": False,
                    "method": "keyword_scan",
                    "matched_keywords": [],
                    "matched_lines": [],
                    "llm_result": None,
                }
            elif {kw.lower() for kw in _kres["matched_keywords"]} & _HIGH_SIGNAL_KW:
                _prefilter = {
                    "fence_found": True,
                    "method": "keyword_high_signal",
                    "matched_keywords": _kres["matched_keywords"],
                    "matched_lines": _kres["matched_lines"],
                    "llm_result": None,
                }
            else:
                _llm_r = _llm_by_page.get(page_idx)
                if _llm_r and _llm_r.get("confidence", 0.0) >= 0.5:
                    _prefilter = {
                        "fence_found": bool(_llm_r.get("is_fence_related", False)),
                        "method": "llm_confirmed",
                        "matched_keywords": _kres["matched_keywords"],
                        "matched_lines": _kres["matched_lines"],
                        "llm_result": _llm_r,
                    }
                else:
                    # LLM unavailable or low confidence — trust the keywords.
                    _prefilter = {
                        "fence_found": True,
                        "method": "keyword_only",
                        "matched_keywords": _kres["matched_keywords"],
                        "matched_lines": _kres["matched_lines"],
                        "llm_result": _llm_r,
                    }

            _page_cache[page_idx] = {
                'pdf_lines': _pdf_lines_by_page[page_idx],
                'ocr_lines': _ocr_lines_by_page[page_idx],
                'prefilter_result': _prefilter,
            }
            if _prefilter.get("fence_found"):
                _fence_page_indices.append(page_idx)
            _cache_put("phase1c", _pdf_sha, _cache_params, _prefilter, page_idx=page_idx)

        status_txt_area.text(
            f"classified {len(_pending_indices)} pages "
            f"({len(_needs_llm)} via LLM, {len(_pending_indices) - len(_needs_llm)} by keyword)"
        )

        # Keep original page order for deterministic batching and display.
        _fence_page_indices.sort()
        _phase_timings['1c'] = time.perf_counter() - _phase_t0
        print(f"SESSION {current_session_id} LOG: Phase 1c done in {_phase_timings['1c']:.1f}s "
              f"(cache hits: {_phase1c_cache_hits}/{total_pages - len(_broken)})")
        print(f"SESSION {current_session_id} LOG: Phase 1 complete — "
              f"{len(_fence_page_indices)}/{total_pages} fence pages detected")
        telemetry.phase_checkpoint("phase1c_end",
                                   session_id=current_session_id,
                                   pdf_sha8=_pdf_sha[:8],
                                   wall_s=round(_phase_timings['1c'], 3),
                                   cache_hits=_phase1c_cache_hits,
                                   fence_pages=len(_fence_page_indices),
                                   total_pages=total_pages)
        _check_memory_pressure("after phase1c")

        # Compact non-fence pages: Phase 3 only reads the pdf_lines/ocr_lines
        # for fence pages. Everything else holds onto text blocks that are
        # never consulted again — drop them. Shared list objects mean we
        # need to clear BOTH _page_cache[i]['pdf_lines'] and _pdf_lines_by_page[i].
        _fence_set = set(_fence_page_indices)
        _kept_bytes = 0
        for _pi in range(total_pages):
            if _pi in _fence_set or _pi in _broken:
                continue
            if _pi in _page_cache:
                _page_cache[_pi]['pdf_lines'] = []
                _page_cache[_pi]['ocr_lines'] = []
            _pdf_lines_by_page[_pi] = []
            _ocr_lines_by_page[_pi] = []
            _kept_bytes += 1
        if _kept_bytes:
            gc.collect()
            print(f"SESSION {current_session_id} LOG: compacted text for "
                  f"{_kept_bytes} non-fence pages")
        _mem_snapshot("after Phase 1c compaction")
        _phase_t0 = time.perf_counter()
        
        # =================================================================
        # PHASE 2: Batch ADE for fence pages (smart batching by size)
        # =================================================================
        _ade_chunks_by_page = {}

        # Capture the authoritative cache scope ONCE, in the main thread.
        # Phase 2 worker threads must use this scope — calling
        # _cache_scope() from inside a ThreadPoolExecutor worker does
        # not see st.session_state and would land writes under a ghost
        # session dir the main thread can never read back.
        _worker_scope = _cache_scope()

        if use_ade and ade_key and _fence_page_indices:
            # Belt-and-suspenders: damaged pages should never be fence pages
            # (Phase 1c skipped them) but filter again in case something slipped through.
            _fence_page_indices = [i for i in _fence_page_indices if i not in _broken]

            # Populate per-page cache hits FIRST; only uncached pages need to
            # be sent to ADE. On a re-run this can zero out the ADE work.
            _phase2_cache_hits = 0
            _fence_pages_to_fetch = []
            for _pi in _fence_page_indices:
                _cached_2 = _cache_get("phase2", _pdf_sha, _cache_params, page_idx=_pi)
                if _cached_2 is not None:
                    _ade_chunks_by_page[_pi] = _cached_2
                    _phase2_cache_hits += 1
                else:
                    _fence_pages_to_fetch.append(_pi)
            print(f"SESSION {current_session_id} LOG: Phase 2 cache hits: "
                  f"{_phase2_cache_hits}/{len(_fence_page_indices)}; "
                  f"{len(_fence_pages_to_fetch)} pages need ADE")

            # LandingAI documented sync-endpoint caps: 50 MB / 50 pages
            # per request. Our defaults are conservative (15 MB / 10 pages)
            # because empirically ADE starts rejecting with 422 well
            # before the 50 MB cap. All three knobs env-configurable.
            _ADE_BATCH_MAX_BYTES = int(os.environ.get("FENCE_ADE_BATCH_MAX_BYTES", str(15 * 1024 * 1024)))
            _ADE_PAGE_MAX_BYTES  = int(os.environ.get("FENCE_ADE_PAGE_MAX_BYTES",  str(12 * 1024 * 1024)))
            # Set FENCE_ADE_BATCH_PAGES=1 to disable multi-page batching
            # — each fence page gets its own ADE request. Slower overall
            # (50 round-trips instead of 5-6) but sometimes yields
            # different chunking on dense engineering detail sheets
            # where ADE's per-document context may merge or drop small
            # regions in a crowded multi-page batch.
            _ADE_BATCH_PAGES = int(os.environ.get("FENCE_ADE_BATCH_PAGES", "10"))

            # Path-based batching: avoids keeping the whole PDF in RAM for
            # size-estimation / batch-PDF construction. Workers read from
            # disk on demand (fitz demand-loads pages from path).
            _pdf_path_for_phase2 = st.session_state.get('pdf_disk_path') or ''
            if _pdf_path_for_phase2 and os.path.exists(_pdf_path_for_phase2):
                _batches = ade.create_page_batches_from_path(
                    _pdf_path_for_phase2, _fence_pages_to_fetch,
                    max_batch_bytes=_ADE_BATCH_MAX_BYTES,
                    max_pages_per_batch=_ADE_BATCH_PAGES,
                ) if _fence_pages_to_fetch else []
            else:
                _batches = ade.create_page_batches(
                    file_bytes, _fence_pages_to_fetch,
                    max_batch_bytes=_ADE_BATCH_MAX_BYTES,
                    max_pages_per_batch=_ADE_BATCH_PAGES,
                ) if _fence_pages_to_fetch else []

            # Parallelize batches across FENCE_WORKERS_PHASE2 workers. Each
            # worker is self-contained (builds its own single/multi-page PDF,
            # hits the ADE endpoint, aligns chunks). doc_proc is NOT thread-
            # safe, so we use _page_dims (populated in Phase 1a) for page
            # dimensions instead of reaching into doc_proc from workers.

            def _run_phase2_batch(batch_idx, batch):
                """Worker: process one batch. Returns {page_idx: chunks_or_None}.

                Builds the batch PDF via safe_multi_page_pdf_from_path so
                any oversized page gets raster-downsampled BEFORE hitting
                LandingAI, avoiding the 147 MB → 422 failure we used to
                see on drawings with embedded high-res imagery.
                """
                result = {}
                try:
                    if _pdf_path_for_phase2 and os.path.exists(_pdf_path_for_phase2):
                        batch_pdf = ade.safe_multi_page_pdf_from_path(
                            _pdf_path_for_phase2, batch,
                            per_page_max_bytes=_ADE_PAGE_MAX_BYTES,
                        )
                    else:
                        # Byte-path fallback. No downsampling here (no path
                        # to reopen per page); oversized pages will still
                        # fail and fall through to individual retry below.
                        batch_pdf = ade.create_multi_page_pdf(file_bytes, batch)
                except Exception as _e:
                    print(f"[APP] ADE batch {batch_idx + 1} PDF-build failed ({_e}); marking all pages for fallback")
                    for _orig_idx in batch:
                        result[_orig_idx] = None
                    return result
                print(f"SESSION {current_session_id} LOG: ADE batch {batch_idx + 1}: "
                      f"{len(batch)} pages, {len(batch_pdf) / 1024:.0f}KB")

                # Helper — re-run ADE on ONE page and store its result.
                # Used both for whole-batch failures and for single-page
                # "zero chunks returned in a multi-page batch" recovery.
                # Returns True if the single-page call produced a non-empty
                # chunk list (caller can use that to count retries).
                def _retry_single(orig_idx):
                    try:
                        if _pdf_path_for_phase2 and os.path.exists(_pdf_path_for_phase2):
                            single_pdf = ade.safe_single_page_pdf_from_path(
                                _pdf_path_for_phase2, orig_idx,
                                max_bytes=_ADE_PAGE_MAX_BYTES,
                            )
                        else:
                            single_pdf = ade.create_single_page_pdf(file_bytes, orig_idx)
                        single_resp = ade.ade_parse_document(single_pdf, ade_key)
                        del single_pdf
                        if single_resp["success"]:
                            pdf_w, pdf_h = _page_dims.get(orig_idx, (0.0, 0.0))
                            chunks = ade.align_ade_chunks_to_page(single_resp, 0, pdf_w, pdf_h)
                            result[orig_idx] = chunks
                            _cache_put("phase2", _pdf_sha, _cache_params, chunks,
                                       page_idx=orig_idx, user_scope=_worker_scope)
                            return bool(chunks)
                        result[orig_idx] = None
                        print(f"[APP] Page {orig_idx + 1}: individual ADE also failed")
                        return False
                    except Exception as _e:
                        result[orig_idx] = None
                        print(f"[APP] Page {orig_idx + 1}: individual ADE error: {_e}")
                        return False

                resp = ade.ade_parse_document(batch_pdf, ade_key)
                del batch_pdf  # release the transient batch-PDF bytes ASAP
                if resp["success"]:
                    # Track pages that came back with zero chunks so we can
                    # retry them individually. ADE's per-document context
                    # occasionally merges/drops small regions when many
                    # dense engineering detail pages share one request —
                    # the same page alone usually returns a proper chunk
                    # list. Only retry inside MULTI-page batches; a legit
                    # empty page in a single-page batch shouldn't be
                    # double-billed.
                    _zero_chunk_retries = []
                    for local_idx, orig_idx in enumerate(batch):
                        try:
                            pdf_w, pdf_h = _page_dims.get(orig_idx, (0.0, 0.0))
                            chunks = ade.align_ade_chunks_to_page(resp, local_idx, pdf_w, pdf_h)
                            result[orig_idx] = chunks
                            _cache_put("phase2", _pdf_sha, _cache_params, chunks,
                                       page_idx=orig_idx, user_scope=_worker_scope)
                            if len(batch) > 1 and not chunks:
                                _zero_chunk_retries.append(orig_idx)
                        except Exception as _e:
                            result[orig_idx] = None
                            print(f"[APP] Page {orig_idx + 1}: align failed ({_e}) — marking as no ADE")
                    if _zero_chunk_retries:
                        print(f"[APP] ADE batch {batch_idx + 1}: "
                              f"{len(_zero_chunk_retries)} page(s) returned 0 chunks "
                              "— retrying individually: "
                              + ", ".join(str(i + 1) for i in _zero_chunk_retries))
                        for orig_idx in _zero_chunk_retries:
                            _retry_single(orig_idx)
                else:
                    # Batch failed — retry every page individually
                    # (sequential within this worker; other batches keep
                    # running in parallel).
                    print(f"[APP] ADE batch {batch_idx + 1} failed: {resp.get('error')} — retrying individually")
                    for orig_idx in batch:
                        _retry_single(orig_idx)
                return result

            status_txt_area.text(
                f"{len(_batches)} ADE batch(es) across {FENCE_WORKERS_PHASE2} workers…"
            )
            with ThreadPoolExecutor(max_workers=FENCE_WORKERS_PHASE2) as _ade_pool:
                _ade_futs = {
                    _ade_pool.submit(_run_phase2_batch, bi, b): bi
                    for bi, b in enumerate(_batches)
                }
                _batch_done = 0
                _phase2_t0 = time.perf_counter()
                _track_phase('2', 0, max(1, len(locals().get('_batches', []) or [])),
                             _phase2_t0)
                _update_total_eta()
                for _fut in as_completed(_ade_futs):
                    _keep_session_alive()
                    _batch_done += 1
                    _ov = 0.3 + _batch_done / max(len(_batches), 1) * 0.3
                    prog_bar.progress(_ov); _update_total_eta(_ov)
                    try:
                        _ade_chunks_by_page.update(_fut.result())
                    except Exception as _e:
                        _bi = _ade_futs[_fut]
                        print(f"SESSION {current_session_id} WARNING: Phase 2 batch {_bi} worker crashed: {_e}")
                        for _oi in _batches[_bi]:
                            _ade_chunks_by_page.setdefault(_oi, None)
                    _eta_suffix(_batch_done, len(_batches), _phase2_t0, phase_key='2')
                    _update_total_eta()

            _ok = sum(1 for v in _ade_chunks_by_page.values() if v is not None)
            _phase_timings['2'] = time.perf_counter() - _phase_t0
            print(f"SESSION {current_session_id} LOG: Phase 2 done in {_phase_timings['2']:.1f}s "
                  f"— ADE results for {_ok}/{len(_fence_page_indices)} fence pages "
                  f"(cache hits: {_phase2_cache_hits})")
            telemetry.phase_checkpoint("phase2_end",
                                       session_id=current_session_id,
                                       pdf_sha8=_pdf_sha[:8],
                                       wall_s=round(_phase_timings['2'], 3),
                                       cache_hits=_phase2_cache_hits,
                                       ade_ok=_ok,
                                       fence_pages=len(_fence_page_indices))
            _check_memory_pressure("after phase2")
            _phase_t0 = time.perf_counter()

        # Phase 2 was the last place that needed the whole PDF as bytes
        # (for batch PDF construction). Phase 3 workers open from the disk
        # path directly. Drop file_bytes now to reclaim 50-200 MB.
        _mem_snapshot("before dropping file_bytes")
        try:
            del file_bytes
        except NameError:
            pass
        gc.collect()
        _mem_snapshot("after dropping file_bytes")
        
        # =================================================================
        # PHASE 3 PRE-COMPUTE: parallel pass over fence pages
        # -----------------------------------------------------------------
        # The sequential Phase 3 loop below does a vision-LLM scale call,
        # legend extraction (LLM), and measurement (LLM) per page. Even
        # with per-phase disk caching, a cold first run used to serialize
        # all of this. Here we pre-compute those three in parallel across
        # FENCE_WORKERS_PHASE3 workers and just write the results to
        # fence_cache. The sequential loop after this hits the cache and
        # skips the expensive calls.
        #
        # Thread-safety:
        #   - PyMuPDF Document is NOT thread-safe, so each worker opens
        #     its own fitz.open(pdf_disk_path) and closes it.
        #   - LangChain ChatOpenAI clients ARE safe for concurrent
        #     independent calls — sharing llm_analysis_instance is fine.
        #   - No Streamlit calls, no st.session_state writes inside workers.
        #   - _cache_put() is atomic (tmp + replace); concurrent
        #     writers to the same key are idempotent.
        # =================================================================
        _pdf_disk_path_for_phase3 = st.session_state.get('pdf_disk_path')
        _fence_pages_for_phase3 = [i for i in _fence_page_indices if i not in _broken]

        # --- Batched legend extraction (Task D) ---------------------------
        # Before dispatching the precompute pool, do a single up-front
        # batched LLM pass that pulls fence-related items out of every
        # legend chunk across every fence page. Workers then read from
        # these dicts (keyed by (page_idx, chunk_idx)) instead of firing
        # their own per-chunk LLM calls.
        #
        # When the phase3_legend cache is already populated for a page,
        # we skip collecting its chunks entirely — the worker will read
        # directly from disk cache.
        _prefill_legend = {}   # {(page_idx, chunk_idx): [items]}
        _prefill_figure = {}   # {(page_idx, chunk_idx): [items]}
        if _fence_pages_for_phase3 and llm_analysis_instance:
            _to_batch_legend = {}   # {(pi, ci): text}
            _segmented_by_page = {} # {pi: (legend_chunks, figure_chunks)}  cached so worker doesn't resegment
            for _pi in _fence_pages_for_phase3:
                if _cache_get("phase3_legend", _pdf_sha, _cache_params, page_idx=_pi) is not None:
                    continue  # worker will hit cache; no need to extract
                _ade = _ade_chunks_by_page.get(_pi) or []
                if not _ade:
                    continue
                _leg, _fig = ade.segment_chunks(_ade)
                _segmented_by_page[_pi] = (_leg, _fig)
                for _ci, _c in enumerate(_leg):
                    _txt = _c.get("text", "")
                    if _txt:
                        _to_batch_legend[(_pi, _ci)] = _txt

            # Opt-out: when FENCE_SKIP_LEGEND_PREBATCH=true, skip the
            # pre-batch entirely and let each Phase 3 worker extract
            # legends for its own page. Slower overall (loses prompt
            # cache across pages) but eliminates the single-point-of-
            # hang that has parked runs for 10+ min when an OpenAI call
            # blocked on ssl.recv past its advertised timeout.
            if os.environ.get("FENCE_SKIP_LEGEND_PREBATCH", "false").lower() == "true":
                print(f"SESSION {current_session_id} LOG: Phase 3 legend pre-batch SKIPPED "
                      "(FENCE_SKIP_LEGEND_PREBATCH=true); workers will self-extract")
                _to_batch_legend = {}  # clear so the block below is a no-op

            if _to_batch_legend:
                _batch_t0 = time.perf_counter()
                # Parallelise the legend pre-batch. The old path made one
                # sequential LLM call for every batch_size chunks, which
                # on a 50-page deck meant ~80 calls × 5-10s = 7-15 min
                # of main-thread blocking between Phase 2 and Phase 3.
                # UI appeared "stuck" because no progress message was
                # updated while this ran.
                #
                # Shard the chunk dict into N workers so the pre-batch
                # runs in parallel. Keeps the prompt-cache benefit of
                # batching (static system prompt) while cutting total
                # wall time by ~Nx.
                _legend_batch_size = int(os.environ.get("FENCE_LEGEND_BATCH_SIZE", "6"))
                _legend_workers = _workers("FENCE_LEGEND_PREBATCH_WORKERS", 4, cap=8)
                _legend_keys = list(_to_batch_legend.keys())
                # Slice keys into _legend_workers shards while keeping
                # each shard's size a multiple of batch_size so the
                # underlying batcher packs prompts efficiently.
                def _shards(keys, n):
                    if n <= 1 or len(keys) <= _legend_batch_size:
                        return [keys]
                    # ceil divide to produce at most n shards.
                    chunk = max(_legend_batch_size,
                                (len(keys) + n - 1) // n)
                    return [keys[i:i + chunk] for i in range(0, len(keys), chunk)]

                _k_shards = _shards(_legend_keys, _legend_workers)
                print(f"SESSION {current_session_id} LOG: Phase 3 legend pre-batch: "
                      f"{len(_to_batch_legend)} chunks across {len(_segmented_by_page)} pages, "
                      f"{len(_k_shards)} parallel shards")
                status_txt_area.text(
                    f"Phase 2→3 — legend pre-batch ({len(_to_batch_legend)} chunks, "
                    f"{len(_k_shards)} parallel shards)…"
                )

                def _run_legend_shard(shard_keys):
                    shard_dict = {k: _to_batch_legend[k] for k in shard_keys}
                    return ade.llm_extract_fence_elements_batch(
                        llm_analysis_instance,
                        shard_dict,
                        FENCE_KEYWORDS_APP,
                        batch_size=_legend_batch_size,
                    )

                _items_by_id = {}
                _shards_done = 0
                # Per-shard HARD timeout. Previously I computed this as
                # batches_per_shard × per_call_cap which came out to
                # ~40+ min on medium PDFs — effectively no timeout. Now
                # a single fixed cap: if a shard isn't done in this
                # many seconds, abandon it and let Phase 3 workers
                # extract those legends themselves. 120s is enough for
                # a typical shard when OpenAI is responsive.
                _SHARD_HARD_TIMEOUT = _workers("FENCE_LEGEND_SHARD_TIMEOUT", 120, cap=600)
                try:
                    with ThreadPoolExecutor(max_workers=len(_k_shards)) as _lp:
                        _lfuts = {_lp.submit(_run_legend_shard, s): i
                                  for i, s in enumerate(_k_shards)}
                        import concurrent.futures as _cf_legend
                        for _lf in list(_lfuts.keys()):
                            try:
                                _part = _lf.result(timeout=_SHARD_HARD_TIMEOUT)
                                _items_by_id.update(_part or {})
                            except _cf_legend.TimeoutError:
                                print(f"[legend-prebatch] shard {_lfuts[_lf]} "
                                      f"exceeded {_SHARD_HARD_TIMEOUT}s — skipping; "
                                      "workers will extract those legends themselves")
                                telemetry.event("legend_prebatch_shard_timeout",
                                                session_id=current_session_id,
                                                shard=_lfuts[_lf],
                                                timeout_s=_SHARD_HARD_TIMEOUT)
                            except Exception as _se:
                                print(f"[legend-prebatch] shard {_lfuts[_lf]} failed: {_se}")
                            _shards_done += 1
                            status_txt_area.text(
                                f"Phase 2→3 — legend pre-batch "
                                f"({_shards_done}/{len(_k_shards)} shards done)"
                            )
                        _lp.shutdown(wait=False, cancel_futures=True)
                    for _key, _items in _items_by_id.items():
                        _prefill_legend[_key] = _items
                    _batch_dt = time.perf_counter() - _batch_t0
                    print(f"SESSION {current_session_id} LOG: Phase 3 legend pre-batch done in {_batch_dt:.1f}s "
                          f"({_shards_done}/{len(_k_shards)} shards)")
                    telemetry.event("legend_prebatch_done",
                                    session_id=current_session_id,
                                    total_chunks=len(_to_batch_legend),
                                    shards=len(_k_shards),
                                    wall_s=round(_batch_dt, 3))
                except Exception as _be:
                    print(f"SESSION {current_session_id} WARNING: Phase 3 legend pre-batch failed ({_be}); "
                          "workers will fall back to per-chunk LLM calls")

        if _fence_pages_for_phase3 and _pdf_disk_path_for_phase3 and os.path.exists(_pdf_disk_path_for_phase3):

            def _phase3_precompute(page_idx):
                """Worker: populate phase3_legend, phase3_scale, phase3_measure
                caches for one fence page. Returns None; all results go
                through fence_cache."""
                worker_doc = None
                try:
                    worker_doc = fitz.open(_pdf_disk_path_for_phase3)
                    try:
                        worker_page = worker_doc[page_idx]
                    except Exception:
                        return
                    pdf_lines = _pdf_lines_by_page.get(page_idx, [])
                    ocr_lines = _ocr_lines_by_page.get(page_idx, [])
                    ade_chunks = _ade_chunks_by_page.get(page_idx) or []
                    # Reuse segmentation done during the pre-batch pass if available.
                    if page_idx in _segmented_by_page:
                        legend_chunks, figure_chunks = _segmented_by_page[page_idx]
                    else:
                        legend_chunks, figure_chunks = ade.segment_chunks(ade_chunks) if ade_chunks else ([], [])

                    # Assemble per-page prefill dict {chunk_idx: [items]}
                    # from the global prefill map populated above.
                    _page_legend_prefill = {}
                    for (_ppi, _pci), _items in _prefill_legend.items():
                        if _ppi == page_idx:
                            _page_legend_prefill[_pci] = _items

                    # 1. Legend entries
                    definitions = []
                    if highlight_fence_text_app and legend_chunks:
                        cached = _cache_get("phase3_legend", _pdf_sha, _cache_params,
                                            page_idx=page_idx, user_scope=_worker_scope)
                        if cached is not None:
                            definitions = cached
                        else:
                            try:
                                definitions = ade.extract_legend_entries(
                                    legend_chunks=legend_chunks,
                                    pdf_lines=pdf_lines,
                                    ocr_lines=ocr_lines,
                                    fence_keywords=FENCE_KEYWORDS_APP,
                                    llm=llm_analysis_instance,
                                    figure_chunks=figure_chunks,
                                    prefilled_legend_items=_page_legend_prefill or None,
                                )
                                _cache_put("phase3_legend", _pdf_sha, _cache_params, definitions,
                                           page_idx=page_idx, user_scope=_worker_scope)
                            except Exception as _le:
                                print(f"[phase3_precompute] page {page_idx + 1} legend error: {_le}")

                    # 2. Page tokens (needed for instance finding, then for measurement)
                    try:
                        rotation = worker_page.rotation
                        mediabox_w = worker_page.mediabox.width
                        mediabox_h = worker_page.mediabox.height
                        native_words = worker_page.get_text("words")
                    except Exception:
                        native_words, rotation, mediabox_w, mediabox_h = [], 0, 0, 0

                    def _xform(x0, y0, x1, y1):
                        if rotation == 90:
                            return mediabox_h - y1, x0, mediabox_h - y0, x1
                        if rotation == 180:
                            return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                        if rotation == 270:
                            return y0, mediabox_w - x1, y1, mediabox_w - x0
                        return x0, y0, x1, y1

                    all_page_tokens = []
                    for w in native_words:
                        nx0, ny0, nx1, ny1 = _xform(w[0], w[1], w[2], w[3])
                        all_page_tokens.append({
                            "text": w[4], "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1,
                        })

                    # 3. Instances — fast (numpy), no cache needed
                    instances = []
                    if definitions and figure_chunks:
                        try:
                            instances = ade.find_instances_in_figures_fast(
                                definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines,
                            )
                        except Exception as _ie:
                            print(f"[phase3_precompute] page {page_idx + 1} instances error: {_ie}")

                    # 4. Scale detection (vision LLM) — cache-aware
                    detected_scale = None
                    scale_cached = _cache_get("phase3_scale", _pdf_sha, _cache_params,
                                              page_idx=page_idx, user_scope=_worker_scope)
                    if scale_cached is not None:
                        detected_scale = scale_cached.get('verified_scale')
                    else:
                        try:
                            # Fast chain: regex on title-block text first,
                            # vision LLM only if regex doesn't find a
                            # confident pattern. Saves ~2-5s on most pages.
                            scale_info = verify_scale_with_bar_fast(
                                worker_page, llm=scale_llm_instance or llm_analysis_instance
                            )
                            if scale_info.get('success') is not False or scale_info.get('verified_scale'):
                                _cache_put("phase3_scale", _pdf_sha, _cache_params, scale_info,
                                           page_idx=page_idx, user_scope=_worker_scope)
                            if scale_info.get('success') and scale_info.get('verified_scale'):
                                detected_scale = scale_info['verified_scale']
                        except Exception as _se:
                            print(f"[phase3_precompute] page {page_idx + 1} scale error: {_se}")

                    # 5. Measurement (geometry + LLM layer match) — cache-aware
                    if enable_unified_measurement and (definitions or instances):
                        if _cache_get("phase3_measure", _pdf_sha, _cache_params,
                                      page_idx=page_idx, user_scope=_worker_scope) is None:
                            try:
                                ocr_full_text = "\n".join(line.get('text', '') for line in ocr_lines) if ocr_lines else None
                                measurement_result = ade.measure_fence_elements(
                                    worker_page, definitions, instances,
                                    figure_chunks=figure_chunks,
                                    llm=llm_analysis_instance,
                                    light_llm=classifier_llm_instance,  # layer-name match → mini
                                    scale_factor=detected_scale or 1.0,
                                    ocr_text=ocr_full_text,
                                )
                                if measurement_result:
                                    _cache_put("phase3_measure", _pdf_sha, _cache_params,
                                               measurement_result, page_idx=page_idx,
                                               user_scope=_worker_scope)
                            except Exception as _me:
                                print(f"[phase3_precompute] page {page_idx + 1} measure error: {_me}")
                finally:
                    if worker_doc is not None:
                        try:
                            worker_doc.close()
                        except Exception:
                            pass

            # Hard per-page wall-clock cap. Even though LLM clients have
            # their own request timeouts, we've seen Phase 3 stalls where
            # a single worker blocks the whole pool for >10 min. This
            # is a belt-and-braces timeout: if a page hasn't completed
            # in FENCE_PHASE3_PAGE_TIMEOUT seconds, we stop waiting on
            # its future and move on. The worker thread itself may keep
            # running in the background (Python can't cancel threads)
            # but the main loop is no longer blocked by it, and the
            # fence_cache writes from that worker (if any) still land.
            # Bumped default from 180s to 300s. Each subprocess runs 3
            # serial LLM calls (legend → scale → measure); on dense
            # title pages with 8-12 legend items, the legend step alone
            # can take 90s + scale 30s + measure 60s = 180s at the edge,
            # and a slow OpenAI response on any one call pushed clean
            # pages into timeouts. 300s gives real headroom; we still
            # cap at 600s so a genuinely hung subprocess eventually
            # dies instead of blocking the whole analysis forever.
            FENCE_PHASE3_PAGE_TIMEOUT = _workers("FENCE_PHASE3_PAGE_TIMEOUT", 300, cap=600)

            # Opt-in subprocess isolation. When FENCE_PHASE3_USE_SUBPROCESS=true
            # each per-page worker runs in a short-lived child process
            # (ops/phase3_worker.py) instead of a thread. Trades a small
            # per-page startup cost for dramatic memory hygiene: each
            # subprocess peaks around 150-200 MB and releases every byte
            # to the OS on exit. Main process RSS stays flat. Hung LLM
            # calls are actually killable (SIGKILL on timeout, which
            # Python threads don't support).
            FENCE_PHASE3_USE_SUBPROCESS = os.environ.get(
                "FENCE_PHASE3_USE_SUBPROCESS", "false"
            ).lower() == "true"
            _phase3_worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "ops", "phase3_worker.py"
            )

            def _phase3_precompute_via_subprocess(page_idx):
                """Subprocess-isolated per-page worker. Blocks until the
                child exits or hits FENCE_PHASE3_PAGE_TIMEOUT + a small
                IPC budget. All results go through fence_cache on disk,
                so there's nothing to return to the parent."""
                import subprocess as _sp
                import json as _json
                # Build the task JSON. The heavy fields (pdf_lines,
                # ocr_lines, ade_chunks) are typically 10-200 KB per
                # page — well within stdin buffer limits.
                _page_prefill = {
                    _pci: _items
                    for (_ppi, _pci), _items in _prefill_legend.items()
                    if _ppi == page_idx
                }
                _task = {
                    "pdf_path":                 _pdf_disk_path_for_phase3,
                    "page_idx":                 int(page_idx),
                    "pdf_sha":                  _pdf_sha,
                    "cache_params":             _cache_params,
                    "user_scope":               _worker_scope,
                    "fence_cache_dir":          os.environ.get("FENCE_CACHE_DIR", ""),
                    "openai_api_key":           openai_key or "",
                    "analysis_model":           st.session_state.get("selected_model_for_analysis", "gpt-5.1"),
                    "classifier_model":         FENCE_CLASSIFIER_MODEL,
                    "scale_model":              st.session_state.get("selected_model_for_analysis", "gpt-5.1"),
                    "fence_keywords":           list(FENCE_KEYWORDS_APP),
                    "pdf_lines":                _pdf_lines_by_page.get(page_idx, []),
                    "ocr_lines":                _ocr_lines_by_page.get(page_idx, []),
                    "ade_chunks":               _ade_chunks_by_page.get(page_idx) or [],
                    "legend_prefill":           _page_prefill,
                    "highlight_fence_text_app": bool(highlight_fence_text_app),
                    "enable_unified_measurement": bool(enable_unified_measurement),
                }
                task_json = _json.dumps(_task, default=str)
                # 15s IPC budget (task write + startup imports) on top
                # of the per-page LLM budget. Keeps the SIGKILL bounded.
                _wallclock_cap = FENCE_PHASE3_PAGE_TIMEOUT + 15
                proc = _sp.Popen(
                    [sys.executable, _phase3_worker_script],
                    stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
                )
                try:
                    out, err = proc.communicate(task_json, timeout=_wallclock_cap)
                    rc = proc.returncode
                except _sp.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        out, err = proc.communicate(timeout=5)
                    except Exception:
                        out, err = "", ""
                    # Dump the drained stderr so we can see WHICH step the
                    # subprocess was executing when the wallclock cap hit
                    # — previously we just raised TimeoutError and lost
                    # the stderr breadcrumbs, leaving "why did this page
                    # time out?" unanswerable.
                    if err:
                        _err_tail = err.strip().splitlines()[-20:]
                        for _line in _err_tail:
                            print(f"[phase3_worker page {page_idx + 1} stderr on timeout] {_line}")
                    raise TimeoutError(
                        f"phase3 worker for page {page_idx + 1} exceeded "
                        f"{_wallclock_cap}s — SIGKILLed"
                    )
                if rc != 0:
                    # Worker reported failure; parse the last JSON line
                    # for an error description (still useful for logs).
                    try:
                        _last = out.strip().splitlines()[-1]
                        _rec = _json.loads(_last)
                        raise RuntimeError(
                            f"subprocess exit={rc}: {_rec.get('error', 'unknown')}"
                        )
                    except Exception:
                        raise RuntimeError(
                            f"phase3 worker exit={rc}; stderr={err[:200]}"
                        )

            def _phase3_precompute_timed(page_idx):
                """Wrap the chosen worker impl with per-page timing +
                telemetry emission so we see progress per-page in the
                log stream rather than only at phase3_end."""
                _t0 = time.perf_counter()
                try:
                    if FENCE_PHASE3_USE_SUBPROCESS:
                        _phase3_precompute_via_subprocess(page_idx)
                    else:
                        _phase3_precompute(page_idx)
                    _status = "ok"
                except Exception as _pe:
                    _status = f"error:{type(_pe).__name__}"
                    print(f"[phase3_precompute] page {page_idx + 1} worker raised: {_pe}")
                _dt = time.perf_counter() - _t0
                try:
                    telemetry.event(
                        "phase3_page_done",
                        session_id=current_session_id,
                        pdf_sha8=_pdf_sha[:8],
                        page_idx=page_idx,
                        wall_s=round(_dt, 3),
                        status=_status,
                        worker=("subprocess" if FENCE_PHASE3_USE_SUBPROCESS else "thread"),
                    )
                except Exception:
                    pass

            # Lazy mode: only pre-compute the first FENCE_PHASE3_PREVIEW
            # pages. The sequential render loop below populates the rest
            # on visit via the same cache-aware code paths. Exports call
            # _ensure_phase3_complete() so nothing is ever partial when
            # the spreadsheet gets written.
            if FENCE_PHASE3_EAGER:
                _p3_pages = list(_fence_pages_for_phase3)
                _p3_mode = "eager"
            else:
                _p3_pages = list(_fence_pages_for_phase3[:FENCE_PHASE3_PREVIEW])
                _p3_mode = f"lazy (preview {len(_p3_pages)}/{len(_fence_pages_for_phase3)})"
            status_txt_area.text(
                f"pre-compute ({_p3_mode}) — {len(_p3_pages)} fence pages "
                f"across {FENCE_WORKERS_PHASE3} workers…"
            )
            _p3pre_t0 = time.perf_counter()
            _track_phase('3a', 0, max(1, len(_p3_pages)), _p3pre_t0)
            _update_total_eta()
            _p3timeouts = 0
            if _p3_pages:
                with ThreadPoolExecutor(max_workers=max(1, min(FENCE_WORKERS_PHASE3, len(_p3_pages)))) as _p3pool:
                    _p3futs = {_p3pool.submit(_phase3_precompute_timed, pi): pi for pi in _p3_pages}
                    _p3done = 0
                    # Iterate with per-future timeout rather than
                    # as_completed(...) which would block forever on a
                    # stuck future. We walk futures in a fixed order and
                    # bail on each one at FENCE_PHASE3_PAGE_TIMEOUT.
                    import concurrent.futures as _cf
                    for _pf, _pi in list(_p3futs.items()):
                        _keep_session_alive()
                        try:
                            _pf.result(timeout=FENCE_PHASE3_PAGE_TIMEOUT)
                        except _cf.TimeoutError:
                            _p3timeouts += 1
                            print(f"[phase3_precompute] page {_pi + 1} exceeded "
                                  f"{FENCE_PHASE3_PAGE_TIMEOUT}s — abandoning worker, moving on")
                            try:
                                telemetry.event(
                                    "phase3_page_timeout",
                                    session_id=current_session_id,
                                    page_idx=_pi,
                                    timeout_s=FENCE_PHASE3_PAGE_TIMEOUT,
                                )
                            except Exception:
                                pass
                        except Exception as _re:
                            print(f"[phase3_precompute] page {_pi + 1} raised: {_re}")
                        _p3done += 1
                        _ov = 0.6 + _p3done / max(len(_p3_pages), 1) * 0.25
                        prog_bar.progress(_ov)
                        _eta_suffix(_p3done, len(_p3_pages), _p3pre_t0, phase_key='3a')
                        _update_total_eta(_ov)
                        # Only show status text when there's something unique
                        # to say (timeouts). Otherwise the Phase ETA line above
                        # already covers progress.
                        if _p3timeouts:
                            status_txt_area.text(f"Phase 3 (pre-compute) — {_p3timeouts} timeout(s)")
                    _p3pool.shutdown(wait=False, cancel_futures=True)
            _p3pre_secs = time.perf_counter() - _p3pre_t0
            _p3timeouts_summary = f", {_p3timeouts} timeouts" if _p3timeouts else ""
            print(f"SESSION {current_session_id} LOG: Phase 3 pre-compute done in {_p3pre_secs:.1f}s "
                  f"({_p3_mode}, {FENCE_WORKERS_PHASE3} workers{_p3timeouts_summary})")
            # Remember which pages still need Phase 3 finalization before export.
            st.session_state['_phase3_pending'] = [i for i in _fence_pages_for_phase3 if i not in _p3_pages]
            st.session_state['_phase3_precompute_fn'] = "_ensure_phase3_complete"

            # Memory reclaim after pre-compute: the ADE chunks and
            # segmented legend/figure chunks were only consumed by the
            # workers above. Nothing reads them in the Phase 3
            # sequential render loop (it re-reads from fence_cache on
            # disk), so we can drop them now. On a 48-fence-page run
            # this frees ~100-200 MB before the sequential loop starts
            # accumulating render state.
            _freed = 0
            try:
                _freed += sum(len(v) if v else 0 for v in _ade_chunks_by_page.values())
                _ade_chunks_by_page.clear()
            except Exception:
                pass
            try:
                _freed += len(_segmented_by_page)
                _segmented_by_page.clear()
            except Exception:
                pass
            # Drop the legend-pre-batch intermediate too (if it ran)
            try:
                _to_batch_legend.clear()
                _prefill_legend.clear()
                _prefill_figure.clear()
            except Exception:
                pass
            gc.collect()
            _mem_snapshot("after phase3 precompute cleanup")
            _check_memory_pressure("after phase3 precompute cleanup")

        # =================================================================
        # PHASE 3: Sequential render loop (fast — every expensive call is
        # now a cache hit thanks to the pre-compute pass above).
        # =================================================================
        _phase3b_t0 = time.perf_counter()
        _track_phase('3b', 0, max(1, total_pages), _phase3b_t0)
        _update_total_eta()
        for page_idx in range(total_pages):
            _keep_session_alive()
            page_num = page_idx + 1
            st.session_state.total_pages_processed_count = page_num
            _ov = 0.85 + page_num / total_pages * 0.15
            prog_bar.progress(_ov); _update_total_eta(_ov)
            _eta_render = _eta_suffix(page_num, total_pages, _phase3b_t0, phase_key='3b')

            # Skip damaged pages entirely — render a clear "skipped" card and move on.
            if page_idx in _broken:
                status_txt_area.text(f"skipping damaged page {page_num}")
                print(f"SESSION {current_session_id} LOG: Phase 3 skip damaged page {page_num}.")
                st.session_state.non_fence_pages.append({
                    'page_number': page_num,
                    'page_index_in_original_doc': page_idx,
                    'fence_found': False,
                    'skipped_damaged': True,
                })
                with col_nf:
                    with st.expander(f"Page {page_num} — ⚠ skipped (damaged)", expanded=False):
                        st.warning(
                            "This page could not be read because the PDF is damaged "
                            "(missing content objects). It was skipped so the rest of "
                            "the document could still be analyzed."
                        )
                summary_placeholder.markdown(
                    f"### Summary (Processed: {page_num}/{total_pages})\n"
                    f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                    f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages) - sum(1 for p in st.session_state.non_fence_pages if p.get('skipped_damaged'))}\n"
                    f"- ⚠ Skipped (damaged): {sum(1 for p in st.session_state.non_fence_pages if p.get('skipped_damaged'))}"
                )
                continue

            # Phase ETA line above shows the page count; status_txt_area only
            # flips to sub-step messages ("detecting scale…" etc.) during the
            # inner steps. At the top of a render iteration it stays quiet.
            status_txt_area.empty()
            print(f"SESSION {current_session_id} LOG: Processing page {page_num}.")

            # Get page dimensions (cheap) — defer image rendering to fence pages only.
            # doc_proc was closed after Phase 1a; use the @st.cache_resource
            # shared fitz doc (disk-backed, lazy-loaded pages) instead.
            try:
                _shared_doc = _get_shared_fitz_doc(_pdf_sha)
                page = _shared_doc[page_idx]
                pdf_width, pdf_height = page.rect.width, page.rect.height
            except Exception as _e:
                # Page became unreadable between phases — treat as damaged.
                print(f"SESSION {current_session_id} WARNING: Phase 3 page {page_num} failed to open ({_e}); marking damaged")
                _broken.add(page_idx)
                st.session_state.broken_pages = _broken
                st.session_state.non_fence_pages.append({
                    'page_number': page_num,
                    'page_index_in_original_doc': page_idx,
                    'fence_found': False,
                    'skipped_damaged': True,
                })
                with col_nf:
                    with st.expander(f"Page {page_num} — ⚠ skipped (damaged)", expanded=False):
                        st.warning("Page could not be opened — PDF is damaged. Skipped.")
                continue
            page_img_bytes = None  # Rendered lazily below only for fence pages

            # Load cached pre-filter results
            _cached = _page_cache[page_idx]
            pdf_lines = _cached['pdf_lines']
            ocr_lines = _cached['ocr_lines']
            prefilter_result = _cached['prefilter_result']
            
            # Initialize variables
            chunks = []
            legend_chunks = []
            figure_chunks = []
            definitions = []
            instances = []
            fallback_result = None
            keyword_matches = []
            measurement_result = {}  # Initialize to empty dict to prevent undefined errors
            # Default detection method based on prefilter; can be overridden later
            detection_method = prefilter_result.get("method", "none")
            fence_found = prefilter_result["fence_found"]
            
            if not fence_found:
                # =====================================================================
                # NON-FENCE PAGE: Store reasoning + page dims; the image
                # itself is rendered on demand by display_page_result_expander
                # via the per-session LRU cache (same code path as fence
                # pages) so memory stays bounded.
                # =====================================================================
                print(f"[APP] Page {page_num}: Not fence-related, skipping.")
                _nf_reason = (
                    prefilter_result.get("reason")
                    or ("Keyword match found but context classified non-fence"
                        if prefilter_result.get("matched_keywords") else
                        "No fence-related keywords or content detected")
                )
                st.session_state.non_fence_pages.append({
                    'page_number': page_num,
                    'page_index_in_original_doc': page_idx,
                    'fence_found': False,
                    'reason': _nf_reason,
                    'method': prefilter_result.get("method", ""),
                    'confidence': prefilter_result.get("confidence"),
                    'matched_keywords': prefilter_result.get("matched_keywords", []),
                    'pdf_width': pdf_width,
                    'pdf_height': pdf_height,
                })
                with col_nf:
                    # Keep the expander open across the st.rerun() that
                    # fires when the user clicks "Load page image" —
                    # otherwise Streamlit snaps it back to expanded=False
                    # on the next run and the user sees their click as
                    # "the expander closed itself".
                    _nf_expanded = bool(st.session_state.get(f'_page_img_loaded_{page_idx}'))
                    with st.expander(f"Page {page_num}", expanded=_nf_expanded):
                        _bits = []
                        if prefilter_result.get("method"):
                            _bits.append(f"method `{prefilter_result['method']}`")
                        _conf = prefilter_result.get("confidence")
                        if _conf is not None:
                            try:
                                _bits.append(f"confidence {float(_conf):.0%}")
                            except Exception:
                                _bits.append(f"confidence {_conf}")
                        if _bits:
                            st.caption(" · ".join(_bits))
                        st.markdown(f"**Reasoning:** {_nf_reason}")
                        if prefilter_result.get("matched_keywords"):
                            st.caption(
                                "Keywords present but rejected: "
                                + ", ".join(prefilter_result["matched_keywords"][:6])
                            )
                        # Lazy-load the actual page render. Button gate
                        # keeps memory flat when the user just scrolls past
                        # this panel — rendering a 150 DPI image of a
                        # 36x24 engineering sheet is ~2-5 MB per page.
                        _nf_img_flag = f'_page_img_loaded_{page_idx}'
                        if not st.session_state.get(_nf_img_flag):
                            # on_click pattern: the callback runs BEFORE
                            # the script reruns, so the flag is True by
                            # the time the if/else here re-evaluates and
                            # we drop into the else branch that renders
                            # the image. No explicit st.rerun() needed.
                            def _on_load_nf(flag=_nf_img_flag):
                                st.session_state[flag] = True

                            st.button("🖼️ Load page image",
                                      key=f'_btn_{_nf_img_flag}',
                                      use_container_width=True,
                                      on_click=_on_load_nf)
                            st.caption("Click to render this page from the PDF.")
                        else:
                            _pdf_bytes_nf = _get_pdf_bytes()
                            _nf_orig, _nf_hl = None, None
                            if _pdf_bytes_nf:
                                try:
                                    _kw_for_nf = [
                                        k for k in (prefilter_result.get("matched_lines") or [])
                                        if all(key in k for key in ['x0', 'y0', 'x1', 'y1'])
                                    ]
                                    _nf_orig, _nf_hl = get_page_image_on_demand(
                                        _pdf_sha, _pdf_bytes_nf, page_idx,
                                        [], [], _kw_for_nf,
                                        pdf_width, pdf_height,
                                        bool(highlight_fence_text_app) and bool(_kw_for_nf),
                                        dpi=DISPLAY_IMAGE_DPI,
                                    )
                                except Exception as _nf_e:
                                    print(f"[APP] non-fence image render for page {page_num} failed: {_nf_e}")
                            _disp_nf = _nf_hl or _nf_orig
                            if _disp_nf:
                                st.image(_disp_nf, caption=f"Page {page_num}")
                            else:
                                st.caption("📷 Image unavailable.")
                # Update summary
                summary_placeholder.markdown(
                    f"### Summary (Processed: {page_num}/{total_pages})\n"
                    f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                    f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
                )
                continue
            else:
                # =====================================================================
                # FENCE PAGE: Use pre-computed ADE chunks from batch
                # =====================================================================
                if use_ade and ade_key:
                    # In-memory dict was cleared after Phase 3 pre-compute
                    # to free RAM. The chunks are still on disk in the
                    # fence_cache — load them from there on miss.
                    _ade_chunks = _ade_chunks_by_page.get(page_idx)
                    if _ade_chunks is None:
                        _ade_chunks = _cache_get(
                            "phase2", _pdf_sha, _cache_params, page_idx=page_idx,
                        )
                    if _ade_chunks is None:
                        print(f"[APP] Page {page_num}: ADE failed for this page, using pre-filter fallback.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                        detection_method = prefilter_result["method"]
                    else:
                        status_txt_area.text(f"page {page_num}: extracting definitions…")
                        chunks = _ade_chunks
                        legend_chunks, figure_chunks = ade.segment_chunks(chunks)
                else:
                    # ADE disabled or no key: rely solely on pre-filter result
                    print(f"[APP] Page {page_num}: ADE is disabled or missing key; using pre-filter result only.")
                    fallback_result = prefilter_result
                    keyword_matches = prefilter_result.get("matched_lines", [])
                
                # =====================================================================
                # STEP 5: Process chunks (runs for ALL fence pages)
                # =====================================================================
                # Debug visualization — DEBUG_MODE only, renders its own
                # copy of the page image on demand (live analysis loop no
                # longer keeps page_img_bytes around).
                if DEBUG_MODE and (legend_chunks or pdf_lines or ocr_lines):
                    _dbg_img = page.get_pixmap(dpi=DISPLAY_IMAGE_DPI).tobytes("png")
                    debug_bytes = ade.debug_visualize_coordinates(
                        _dbg_img, legend_chunks, pdf_lines, ocr_lines, pdf_width, pdf_height
                    )
                    st.image(debug_bytes, caption=f"DEBUG: Layers Page {page_num}", use_container_width=True)
                    del _dbg_img
                
                # Extract fence-related definitions from legend chunks
                if highlight_fence_text_app and legend_chunks:
                    _cached_leg = _cache_get("phase3_legend", _pdf_sha, _cache_params, page_idx=page_idx)
                    if _cached_leg is not None:
                        definitions = _cached_leg
                    else:
                        definitions = ade.extract_legend_entries(
                            legend_chunks=legend_chunks,
                            pdf_lines=pdf_lines,
                            ocr_lines=ocr_lines,
                            fence_keywords=FENCE_KEYWORDS_APP,
                            llm=llm_analysis_instance,
                            figure_chunks=figure_chunks
                        )
                        _cache_put("phase3_legend", _pdf_sha, _cache_params, definitions, page_idx=page_idx)
                
                # Get all page tokens for instance finding
                # IMPORTANT: Transform MediaBox coords to display coords for rotated pages
                native_words = page.get_text("words")
                rotation = page.rotation
                mediabox_w = page.mediabox.width
                mediabox_h = page.mediabox.height
                print(f"[DEBUG] Page {page_num} rotation: {rotation}°, MediaBox: {mediabox_w:.0f}x{mediabox_h:.0f}")
                
                def transform_for_rotation(x0, y0, x1, y1):
                    """Transform MediaBox coords to display coords based on page rotation"""
                    if rotation == 0:
                        return x0, y0, x1, y1
                    elif rotation == 90:
                        return mediabox_h - y1, x0, mediabox_h - y0, x1
                    elif rotation == 180:
                        return mediabox_w - x1, mediabox_h - y1, mediabox_w - x0, mediabox_h - y0
                    elif rotation == 270:
                        return y0, mediabox_w - x1, y1, mediabox_w - x0
                    return x0, y0, x1, y1
                
                all_page_tokens = []
                for w in native_words:
                    nx0, ny0, nx1, ny1 = transform_for_rotation(w[0], w[1], w[2], w[3])
                    all_page_tokens.append({
                        "text": w[4], 
                        "x0": nx0, "y0": ny0, 
                        "x1": nx1, "y1": ny1
                    })
                
                if all_page_tokens:
                    sample = all_page_tokens[0]
                    print(f"[DEBUG] Sample token after transform: '{sample['text']}' at ({sample['x0']:.1f}, {sample['y0']:.1f})")
                
                # Find instances in figures (numpy-prefiltered variant — same output)
                if definitions and figure_chunks:
                    instances = ade.find_instances_in_figures_fast(definitions, figure_chunks, all_page_tokens, ocr_lines=ocr_lines)
                
                # =====================================================================
                # STEP 6: Scale Detection + Smart Fence Measurement
                # =====================================================================
                page_key = f"page_{page_num}"
                
                # Detect scale ONCE using the full chain (vision GPT -> text LLM -> regex).
                # Disk-cached: the vision-LLM call here is 2-5s per page and survives
                # fence.service restart via fence_cache, so a re-run costs ~0ms per page.
                detected_scale = None
                if page_key not in st.session_state.per_page_scale_info:
                    _cached_scale = _cache_get("phase3_scale", _pdf_sha, _cache_params, page_idx=page_idx)
                    if _cached_scale is not None:
                        scale_info = _cached_scale
                        st.session_state.per_page_scale_info[page_key] = scale_info
                        detected_scale = scale_info.get('verified_scale')
                        print(f"[APP] Page {page_num}: Scale from cache = {detected_scale}")
                    else:
                        # Pre-compute already ran for all fence pages. If cache
                        # is missing here, that page's pre-compute worker timed
                        # out or errored. DON'T retry the LLM in the sequential
                        # loop — that's where the 120s+ stalls came from. Treat
                        # scale as "unknown" for this page; measurement will use
                        # scale_factor=1.0 and downstream stats will flag it.
                        print(f"[APP] Page {page_num}: scale cache miss — "
                              "pre-compute didn't finalize this page; skipping LLM retry")
                        scale_info = {
                            'success': False,
                            'verified_scale': None,
                            'message': 'pre-compute did not complete for this page',
                        }
                        st.session_state.per_page_scale_info[page_key] = scale_info
                        # detected_scale stays None → measurement uses 1.0 fallback
                else:
                    scale_info = st.session_state.per_page_scale_info[page_key]
                    detected_scale = scale_info.get('verified_scale')
                
                # Measure fence elements (pass detected scale so it skips its own detection)
                # OPTIMIZATION: Always pass a value (default 1.0) to prevent
                # measure_fence_elements from re-running scale detection (redundant vision LLM call)
                measurement_result = {}
                if enable_unified_measurement and (definitions or instances):
                    _cached_meas = _cache_get("phase3_measure", _pdf_sha, _cache_params, page_idx=page_idx)
                    if _cached_meas is not None:
                        measurement_result = _cached_meas
                    else:
                        # Pre-compute owns measurement. Cache miss here means
                        # pre-compute didn't complete for this page (timeout,
                        # crash, or skipped). Don't re-run the LLM — that was
                        # the single biggest cause of 10+ min stalls in the
                        # sequential loop. The page's expander card will just
                        # show zero measured line-feet.
                        print(f"[APP] Page {page_num}: measurement cache miss — "
                              "pre-compute didn't finalize; skipping LLM retry")
                        measurement_result = {}
                
                # Store auto-detected lines in unified measurement structure.
                # Match app_ade.py's gating: 'layer' is the reliable path
                # (explicit fence layers in the PDF) and is ALWAYS shown.
                # 'llm_guided' / 'skipped' are noisier fallbacks that only
                # render when the user explicitly opts in via the
                # "🔬 Non-layer suggestions" sidebar toggle.
                measurement_method = measurement_result.get('measurement_method', 'none') if measurement_result else 'none'
                _accept = (measurement_method == 'layer') or (
                    enable_nonlayer_suggestions and measurement_method in ('llm_guided', 'skipped')
                )

                if measurement_result and measurement_result.get('all_fence_lines') and _accept:
                    auto_lines = []
                    all_fence_lines = measurement_result.get('all_fence_lines', [])
                    scale_factor = measurement_result.get('page_info', {}).get('scale_factor', 1.0)
                    layer_to_category = measurement_result.get('layer_to_category', {})
                    
                    # Map each line to its category using layer→category mapping.
                    # For llm_guided results the PDF has no explicit fence
                    # layers and layer_to_category is empty; in that case
                    # assign a single fallback category so the lines still
                    # appear on the canvas (user can re-categorize later).
                    # llm_guided pages have no layer → category mapping;
                    # skipped pages bailed out of categorisation to
                    # avoid a 5000-line stall. Both need a fallback
                    # category so their lines render on the canvas.
                    _needs_fallback = measurement_method in ('llm_guided', 'skipped')
                    _fallback_cat = "🔍 Auto-detected (LLM-guided)" if measurement_method == 'llm_guided' else (
                        "🔍 Auto-detected (high-density, layer-skipped)" if measurement_method == 'skipped' else None
                    )
                    for line in all_fence_lines:
                        length_pts = line.length_pts
                        length_inches = length_pts / 72.0
                        length_feet = (length_inches * scale_factor) / 12.0

                        line_layer = getattr(line, 'layer', None) or ''
                        # Use LLM-matched layer→category mapping
                        category = layer_to_category.get(line_layer)

                        # Fallback: if layer not in mapping, try partial match
                        if not category and line_layer:
                            for mapped_layer, cat in layer_to_category.items():
                                if mapped_layer in line_layer or line_layer in mapped_layer:
                                    category = cat
                                    break

                        # Final fallback: llm_guided site plans have no layers
                        # at all. Tag every candidate line as the fallback
                        # category so the UI renders them instead of silently
                        # dropping them.
                        if not category:
                            category = _fallback_cat

                        if category:
                            auto_lines.append({
                                'start': line.start,
                                'end': line.end,
                                'length_pts': length_pts,
                                'length_feet': length_feet,
                                'layer': line_layer,
                                'category': category,
                                'source': 'auto',
                                'method': measurement_method,
                            })
                    
                    if auto_lines:
                        if page_key not in st.session_state.unified_measurements:
                            st.session_state.unified_measurements[page_key] = {
                                'auto_lines': [], 'manual_lines': [], 'drawn_lines': [], 'accepted_auto': set()
                            }
                        st.session_state.unified_measurements[page_key]['auto_lines'] = auto_lines
                        # Auto-accept all detected lines by default
                        st.session_state.unified_measurements[page_key]['accepted_auto'] = set(range(len(auto_lines)))
                        print(f"[AUTO] Page {page_num}: {len(auto_lines)} layer-based lines stored with categories from {len(layer_to_category)} layer mappings")
                    else:
                        print(f"[AUTO] Page {page_num}: layer-based but no lines matched to categories")
                elif measurement_method != 'layer':
                    print(f"[AUTO] Page {page_num}: skipping suggestions (method={measurement_method}, not layer-based)")
                
                # Scale info already stored above (before measurement)
                
                # Initialize categories from definitions (for all pages)
                if page_key not in st.session_state.page_categories:
                    categories = {}
                    CATEGORY_COLORS = [
                        (0, 255, 0), (255, 165, 0), (0, 191, 255), (255, 0, 255),
                        (255, 255, 0), (0, 255, 255), (255, 105, 180), (173, 255, 47),
                    ]
                    for d in definitions:
                        indicator = d.get('indicator', '')
                        keyword = d.get('keyword', '')
                        if keyword:
                            cat_name = f"{indicator}: {keyword}" if indicator else keyword
                            if cat_name not in categories:
                                color_idx = len(categories)
                                categories[cat_name] = {
                                    'indicator': indicator,
                                    'keyword': keyword,
                                    'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                                }
                    st.session_state.page_categories[page_key] = categories
                
                # DEBUG: Show coordinate info if enabled
                if DEBUG_MODE:
                    with st.expander(f"🔧 DEBUG Page {page_num}", expanded=True):
                        st.markdown(f"**PDF size:** {pdf_width:.1f} x {pdf_height:.1f}")
                        st.markdown(f"**All ADE Chunks:** {len(chunks)}")
                        for i, c in enumerate(chunks[:10]):
                            st.markdown(f"  - `{c.get('type')}`: ({c.get('x0'):.1f}, {c.get('y0'):.1f}) - ({c.get('x1'):.1f}, {c.get('y1'):.1f})")
                        st.markdown(f"**Figure/Architectural Chunks:** {len(figure_chunks)}")
                        for i, fc in enumerate(figure_chunks):
                            st.markdown(f"  - `{fc.get('type')}`: ({fc.get('x0'):.1f}, {fc.get('y0'):.1f}) - ({fc.get('x1'):.1f}, {fc.get('y1'):.1f})")
                        st.markdown(f"**Legend-like Chunks:** {len(legend_chunks)}")
                        st.markdown(f"**Definitions found:** {len(definitions)}")
                        for i, d in enumerate(definitions[:5]):
                            kw = d.get('keyword', '')[:30]
                            st.markdown(f"  - `{d.get('indicator')}`: {kw}... @ ({d.get('x0'):.1f}, {d.get('y0'):.1f})")
                        st.markdown(f"**Instances found:** {len(instances)}")
                        for i, inst in enumerate(instances[:10]):
                            st.markdown(f"  - `{inst.get('indicator')}` @ ({inst.get('x0'):.1f}, {inst.get('y0'):.1f})")
                        st.markdown(f"**Total page tokens:** {len(all_page_tokens)}")
                        if all_page_tokens:
                            st.markdown(f"**Sample tokens (first 20):**")
                            for t in all_page_tokens[:20]:
                                st.markdown(f"  - `{t.get('text')}` @ ({t.get('x0'):.1f}, {t.get('y0'):.1f})")
                
                # Determine detection method
                if definitions or instances:
                    detection_method = "ade"
                    fence_found = True
                else:
                    # No structured data found, fall back to pre-filter results
                    if not fallback_result:
                        print(f"[APP] Page {page_num}: No definitions found, using pre-filter results.")
                        fallback_result = prefilter_result
                        keyword_matches = prefilter_result.get("matched_lines", [])
                    detection_method = prefilter_result.get("method", "none")
            
            # Collect full page text for cross-page detail extraction
            if fence_found:
                page_text_parts = []
                for pl in pdf_lines:
                    t = pl.get('text', '').strip()
                    if t:
                        page_text_parts.append(t)
                for ol in ocr_lines:
                    t = ol.get('text', '').strip()
                    if t:
                        page_text_parts.append(t)
                full_page_text = "\n".join(page_text_parts)
                if full_page_text.strip():
                    st.session_state.fence_page_texts[page_num] = full_page_text
            
            # Build text snippet from definitions or fallback keywords
            text_snippet = None
            if definitions:
                snippets = [f"{d.get('indicator', '')} - {d.get('keyword', '')}" for d in definitions[:3]]
                text_snippet = "; ".join(snippets)
            elif fallback_result and fallback_result.get("matched_keywords"):
                text_snippet = "Keywords: " + ", ".join(fallback_result["matched_keywords"][:5])
            
            # Images are NOT rendered here anymore (see #4 in the perf
            # plan). Previously we paid ~300-500 ms CPU per fence page
            # rendering page_img_bytes + highlight overlays + cyan fence
            # lines, then sent 2-6 MB of PNG bytes over WS per card. On
            # a 68-fence-page PDF that was ~25 s of CPU + ~200-400 MB
            # of transient image data. The post-analysis
            # display_page_result_expander re-renders images on demand
            # via the per-session LRU (get_page_image_on_demand), so
            # moving the render there is net-zero for UX once results
            # settle, and a big memory/CPU win during the live loop.
            page_img_bytes = None
            original_img_bytes = None
            highlighted_img_bytes = None
            
            # Build result structure (matching app.py format)
            # detection_method already set above based on flow
            llm_result = fallback_result.get("llm_result") if fallback_result else None
            
            # Strip heavy VectorLine objects from measurement_result before storing
            # Keep only summary data (totals, indicator_measurements, layer info)
            measurement_result_light = {}
            if measurement_result:
                measurement_result_light = {
                    k: v for k, v in measurement_result.items()
                    if k != 'all_fence_lines'
                }
            
            analysis_result = {
                'page_number': page_num,
                'page_index_in_original_doc': page_idx,
                'fence_found': fence_found,
                'text_found': fence_found,
                'text_response': json.dumps({
                    "answer": "yes" if fence_found else "no",
                    "confidence": 0.9 if definitions else (llm_result["confidence"] if llm_result else 0.6),
                    "signals": [d.get('keyword', '') for d in definitions[:5]] if definitions else (fallback_result.get("matched_keywords", []) if fallback_result else []),
                    "reason": f"Found {len(definitions)} definitions, {len(instances)} instances" if definitions else (llm_result["reason"] if llm_result else f"Keyword match: {fallback_result.get('matched_keywords', [])}" if fallback_result else "No fence content")
                }),
                'text_snippet': text_snippet,
                'definitions': definitions,
                'instances': instances,
                'keyword_matches': keyword_matches,
                'fallback_result': fallback_result,
                'measurements': measurement_result_light,
                'detection_method': detection_method,
                'highlight_fence_text_app_setting': highlight_fence_text_app,
                'original_image_bytes': None,
                'highlighted_image_bytes': None,
                'pdf_width': pdf_width,
                'pdf_height': pdf_height,
                'chunk_count': len(chunks),
                'legend_count': len(legend_chunks),
                'figure_count': len(figure_chunks),
            }
            
            # Add to fence pages list (non-fence pages handled via early continue above)
            st.session_state.fence_pages.append(analysis_result)
            
            # Display in fence column (non-fence pages handled via early continue above)
            with col_f:
                exp_title = f"Page {page_num}"
                if True:
                    reasons = []
                    if definitions:
                        reasons.append("Definitions")
                    if instances:
                        reasons.append("Instances")
                    if keyword_matches and not definitions:
                        reasons.append("Keywords")
                    if highlight_fence_text_app and (definitions or instances or keyword_matches):
                        reasons.append("Highlights")
                    if reasons:
                        exp_title += f" ({' & '.join(reasons)})"
                
                # Fence pages start collapsed on purpose. Expanding is
                # what triggers get_page_image_on_demand to render + cache
                # the page image — leaving 60 expanders open would pin
                # 60 rendered PNGs in session_state (blowing past the
                # LRU cap), and the user doesn't want to see all of them
                # at once anyway.
                # Keep the expander open across the Load-image rerun.
                _live_expanded = bool(st.session_state.get(f'_page_img_loaded_{page_idx}'))
                with st.expander(exp_title, expanded=_live_expanded):
                    img_col, det_col = st.columns([2, 1])

                    with img_col:
                        # Lazy page-image loading. Streamlit's expander
                        # still executes its body even when collapsed, so
                        # we can't rely on expander state alone to defer
                        # the (2-5 MB PNG) render. Instead we gate behind
                        # a session_state flag set by an explicit button;
                        # before the user clicks, we only show the button
                        # and nothing gets rendered / cached. After click,
                        # the flag stays set so the image is shown on
                        # every rerun of this expander.
                        _img_flag = f'_page_img_loaded_{page_idx}'
                        if not st.session_state.get(_img_flag):
                            # on_click pattern: callback runs before the
                            # rerun so the flag is True when the if/else
                            # re-evaluates next run.
                            def _on_load_fence(flag=_img_flag):
                                st.session_state[flag] = True

                            st.button("🖼️ Load page image",
                                      key=f'_btn_{_img_flag}',
                                      use_container_width=True,
                                      on_click=_on_load_fence)
                            st.caption("Image not rendered yet (saves memory). "
                                       "Click above to read from disk.")
                        else:
                            _pdf_bytes_live = _get_pdf_bytes()
                            _orig_live, _hl_live = None, None
                            if _pdf_bytes_live:
                                try:
                                    _kw_filtered = [
                                        k for k in (keyword_matches or [])
                                        if all(key in k for key in ['x0', 'y0', 'x1', 'y1'])
                                    ]
                                    _orig_live, _hl_live = get_page_image_on_demand(
                                        _pdf_sha, _pdf_bytes_live, page_idx,
                                        definitions or [], instances or [], _kw_filtered,
                                        pdf_width, pdf_height,
                                        bool(highlight_fence_text_app) and bool(fence_found),
                                        dpi=DISPLAY_IMAGE_DPI,
                                    )
                                except Exception as _img_e:
                                    print(f"[APP] live image render for page {page_num} failed: {_img_e}")
                            _disp_live = _hl_live or _orig_live
                            if _disp_live:
                                st.image(
                                    _disp_live,
                                    caption=f"Page {page_num}{' (Highlighted)' if _hl_live else ''}",
                                )
                            else:
                                st.caption("📷 Image unavailable (page or PDF read error)")
                    
                    with det_col:
                        # Detection method badge
                        # Badge logic: if ADE got structured chunks, show ADE
                        # badge. Otherwise, if the page has ANY keyword evidence
                        # (from any prefilter method) OR the LLM confirmed it,
                        # show the keyword-based detection badge — previous
                        # code only recognised three method names and wrongly
                        # fell through to "No Detection" for pages flagged by
                        # keyword_scan / keyword_high_signal, which confused
                        # users when ADE returned 0 chunks.
                        _has_keywords = bool(keyword_matches) or detection_method in (
                            "llm_confirmed", "keyword_only", "keyword_scan", "keyword_high_signal",
                        )
                        if detection_method == "ade":
                            st.success("🎯 ADE Detection")
                        elif detection_method == "llm_confirmed":
                            st.warning("🔍 Keyword + LLM")
                        elif _has_keywords:
                            st.warning("🔤 Keyword-based Detection")
                        else:
                            st.info("❌ No Detection")
                        
                        # ADE Stats (compact)
                        st.metric("ADE Chunks", len(chunks))
                        col_leg, col_fig = st.columns(2)
                        with col_leg:
                            st.metric("Legend", len(legend_chunks))
                        with col_fig:
                            st.metric("Figure", len(figure_chunks))
                        
                        # Text response popover
                        if analysis_result.get('text_response'):
                            with st.popover("Analysis Log"):
                                st.markdown(f"_{analysis_result['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # NEW: Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (no ADE structures found on this page)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                    
                    # Show Measurements (for ALL fence pages, not just keyword matches)
                    meas_method_stored = measurement_result.get('measurement_method', 'none') if measurement_result else 'none'
                    if measurement_result and (meas_method_stored == 'layer' or enable_nonlayer_suggestions) and (measurement_result.get('indicator_measurements') or measurement_result.get('proximity_totals', {}).get('total_segments', 0) > 0):
                        st.markdown("---")
                        st.markdown("### 📏 Fence Measurements")
                        
                        page_info = measurement_result.get('page_info', {})
                        scale_factor = page_info.get('scale_factor', 1.0)
                        method = measurement_result.get('measurement_method', 'unknown')
                        
                        # Show method badge
                        if method == "layer":
                            st.info("📂 Method: Layer-based (fence layers detected)")
                        elif method == "proximity":
                            st.info("🎯 Method: Proximity-based (fallback)")
                        elif method == "llm_guided":
                            st.info("🤖 Method: LLM-guided (adaptive filtering)")
                        elif method == "length_filter":
                            st.info("📏 Method: Length-filtered (no layers, using segment length)")
                        elif method == "no_layers":
                            st.error("❌ No fence layers found in PDF - measurement not available")
                        
                        # Show scale info
                        if page_info.get('scale_detected'):
                            st.success(f"✅ Scale: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                        else:
                            st.warning("⚠️ Scale not detected - raw measurements")
                        
                        # Show totals
                        prox_totals = measurement_result.get('proximity_totals', {})
                        if prox_totals.get('total_segments', 0) > 0:
                            col_pts, col_ft = st.columns(2)
                            with col_pts:
                                st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                            with col_ft:
                                st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                            
                            # Per-indicator breakdown
                            indicator_meas = measurement_result.get('indicator_measurements', {})
                            if indicator_meas:
                                st.markdown("**Per-Indicator:**")
                                for ind, stats in indicator_meas.items():
                                    pts = stats.get('run_length_pts', 0)
                                    ft = stats.get('run_length_feet', 0)
                                    segs = stats.get('run_segment_count', 0)
                                    count = stats.get('instance_count', 0)
                                    st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                        
                        # Layer breakdown (secondary)
                        if measurement_result.get('fence_layers'):
                            with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                totals = measurement_result.get('totals', {})
                                st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                for layer in measurement_result['fence_layers']:
                                    l_stats = measurement_result['layer_measurements'].get(layer, {})
                                    segs = l_stats.get('total_segments', 0)
                                    ft = l_stats.get('total_length_feet', 0)
                                    runs = l_stats.get('connected_runs', 0)
                                    st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                        
                        # Dimension line measurements
                        dim_measurements = measurement_result.get('dimension_measurements', [])
                        if dim_measurements:
                            with st.expander("📐 Dimension Line Measurements", expanded=False):
                                st.caption(f"Found {len(dim_measurements)} dimension annotations")
                                for dm in dim_measurements[:10]:
                                    ft = dm.get('actual_ft', 0)
                                    txt = dm.get('measurement_text', '')
                                    st.markdown(f"- **{txt}**: {ft:.1f} ft")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
            
            # Update summary
            summary_placeholder.markdown(
                f"### Summary (Processed: {st.session_state.total_pages_processed_count}/{total_pages})\n"
                f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
                f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
            )
            
            time.sleep(0.05)

        # Mark processing_complete as soon as fence_pages / non_fence_pages
        # are fully populated by the render loop. Everything that follows
        # (element detail LLM, timings persist, combined-highlighted-PDF
        # generation) is polish that updates session_state in place — it
        # doesn't need the gate open. With the flag set here, any
        # Streamlit rerun triggered by a button click (e.g. "🖼️ Load page
        # image" inside an expander) now enters the post-analysis display
        # branch immediately instead of re-entering the analysis block
        # and running the render loop again. Previously the user would
        # see progress bars cycle through Phase 1-3 every click because
        # processing_complete only flipped at the very end of the
        # analysis branch, after the highlighted-PDF combine.
        st.session_state.processing_complete = True

        # Free Phase 1/2 caches to reduce memory (especially for large PDFs)
        del _page_cache, _pdf_lines_by_page, _ocr_lines_by_page, _ade_chunks_by_page
        _single_page_pdfs.clear()
        gc.collect()
        print(f"SESSION {current_session_id} LOG: Freed processing caches, gc.collect() done")
        
        # =====================================================================
        # CROSS-PAGE DETAIL EXTRACTION
        # After all pages processed, extract detailed specs for each element
        # =====================================================================
        if st.session_state.fence_pages and st.session_state.fence_page_texts:
            status_txt_area.text("Extracting element details across pages...")
            # Collect unique element names from all definitions
            all_element_names = []
            seen_elements = set()
            for fp in st.session_state.fence_pages:
                for d in fp.get('definitions', []):
                    kw = d.get('keyword', '').strip()
                    desc_val = d.get('description', '').strip()
                    if kw and kw not in seen_elements and desc_val != "Indicator Code":
                        ind = d.get('indicator', '').strip()
                        element_label = f"{ind}: {kw}" if ind else kw
                        all_element_names.append(element_label)
                        seen_elements.add(kw)
            
            if all_element_names:
                print(f"[APP] Extracting details for {len(all_element_names)} elements: {all_element_names}")
                try:
                    element_details = ade.extract_element_details(
                        llm=llm_analysis_instance,
                        element_names=all_element_names,
                        page_texts=st.session_state.fence_page_texts,
                    )
                    st.session_state.element_details = element_details
                    print(f"[APP] Element details extracted: {len(element_details)} elements")
                except Exception as e:
                    print(f"[APP] Element detail extraction error: {e}")
                    st.session_state.element_details = {}

        # Free fence_page_texts (only needed for cross-page extraction above)
        st.session_state.fence_page_texts = {}
        gc.collect()

        # Phase 3 timing close + per-phase summary
        _phase_timings['3'] = time.perf_counter() - _phase_t0
        _total_analysis_s = sum(_phase_timings.values())
        telemetry.phase_checkpoint("phase3_end",
                                   session_id=current_session_id,
                                   pdf_sha8=_pdf_sha[:8],
                                   wall_s=round(_phase_timings['3'], 3),
                                   total_wall_s=round(_total_analysis_s, 3),
                                   fence_pages=len(_fence_page_indices))
        _check_memory_pressure("after phase3")
        # Spend record — v1 tracks run-level stats without provider totals.
        # When LLM callbacks exist, individual per-call records will join
        # this one in the user's JSONL.
        spend_tracker.record_spend(
            user_id=get_user_id(),
            provider="run_summary",
            pdf_sha8=_pdf_sha[:8],
            total_pages=total_pages,
            fence_pages=len(_fence_page_indices),
            total_wall_s=round(_total_analysis_s, 3),
            phase_wall_s={k: round(v, 3) for k, v in _phase_timings.items()},
        )
        print(f"SESSION {current_session_id} LOG: ========== PHASE TIMINGS ==========")
        for _ph, _s in _phase_timings.items():
            print(f"SESSION {current_session_id} LOG:   Phase {_ph}: {_s:.1f}s "
                  f"({100.0 * _s / max(_total_analysis_s, 1e-9):.0f}%)")
        print(f"SESSION {current_session_id} LOG:   TOTAL:   {_total_analysis_s:.1f}s")
        print(f"SESSION {current_session_id} LOG: ====================================")

        # Persist timings to session_state + disk so the user can compare
        # runs before/after. The disk history accumulates rows keyed by
        # (pdf_hash, timestamp) under ~/.cache/fence_ade/_timings.
        _phase_name_map = {
            '1a': 'Phase 1a — text extraction',
            '1b': 'Phase 1b — OCR',
            '1c': 'Phase 1c — classification',
            '2':  'Phase 2  — ADE',
            '3':  'Phase 3  — render + precompute',
        }
        _timings_row = {
            'pdf_hash':      _pdf_sha[:12],
            'pdf_name':      st.session_state.get('uploaded_pdf_name', ''),
            'total_pages':   total_pages,
            'fence_pages':   len(_fence_page_indices),
            'timestamp':     int(time.time()),
            'classifier':    FENCE_CLASSIFIER_MODEL,
            'workers_1a':    FENCE_WORKERS_PHASE1A,
            'workers_1b':    FENCE_WORKERS_PHASE1B,
            'workers_1c':    FENCE_WORKERS_PHASE1C,
            'workers_2':     FENCE_WORKERS_PHASE2,
            'workers_3':     FENCE_WORKERS_PHASE3,
            'phase_seconds': {k: round(v, 2) for k, v in _phase_timings.items()},
            'total_seconds': round(_total_analysis_s, 2),
        }
        st.session_state.last_run_timings = _timings_row
        try:
            _tdir = fence_cache.cache_root() / "_timings"
            _tdir.mkdir(parents=True, exist_ok=True)
            _tpath = _tdir / f"{_timings_row['timestamp']}_{_timings_row['pdf_hash']}.json"
            with open(_tpath, "w", encoding="utf-8") as _tf:
                json.dump(_timings_row, _tf, default=str)
        except Exception as _te:
            print(f"[timings] persist failed (non-fatal): {_te}")

        # Processing complete — mark session as done so it doesn't block new users
        st.session_state.processing_complete = True
        _mark_session_done()
        prog_bar.empty()
        status_txt_area.success("All pages processed!")

        # Run timings + per-PDF run history are still persisted to disk
        # (telemetry + ~/.cache/fence_ade/_timings/*.json) and still
        # printed to server stdout — see the phase-timings print block
        # earlier in this function. The user doesn't need the pandas
        # tables in the UI, so we just log them server-side.
        print(f"SESSION {current_session_id} LOG: === run timings ===")
        for _ph_key, _ph_label in _phase_name_map.items():
            _s = _phase_timings.get(_ph_key, 0.0)
            _pct = 100.0 * _s / max(_total_analysis_s, 1e-9)
            print(f"SESSION {current_session_id} LOG:   {_ph_label}: {_s:.1f}s ({_pct:.0f}%)")
        print(f"SESSION {current_session_id} LOG:   TOTAL: {_total_analysis_s:.1f}s")
        
        # Generate combined PDF
        _pdf_for_highlight = _get_pdf_bytes()
        if st.session_state.fence_pages and _pdf_for_highlight:
            pdf_b, pdf_n = generate_combined_highlighted_pdf(
                _pdf_for_highlight,
                st.session_state.fence_pages,
                st.session_state.uploaded_pdf_name,
                current_session_id
            )
            if pdf_b:
                st.session_state.highlighted_pdf_bytes_for_download = pdf_b
                st.session_state.highlighted_pdf_filename_for_download = pdf_n
            else:
                st.warning(f"Could not generate PDF: {pdf_n}")
        
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.session_state.analysis_halted_due_to_error = True
        _mark_session_done()
        print(f"SESSION {current_session_id} ERROR: {e}")
    finally:
        if doc_proc:
            doc_proc.close()
            print(f"SESSION {current_session_id} LOG: Closed main processing PDF document.")
        # Only purge the session cache when the run truly finished (reached
        # processing_complete=True OR explicitly halted). Without this guard,
        # a Streamlit script rerun mid-analysis — triggered by something like
        # a WebSocket disconnect/reconnect when the browser sleeps — would
        # wipe the cache while the original execution's Phase 3 pre-compute
        # was still the authoritative result, forcing the rerun to redo
        # everything from scratch. Keep the cache if we didn't reach a
        # terminal state; the NEXT rerun's cache_get calls will then hit
        # the previously-computed pages.
        _run_terminated = bool(
            st.session_state.get('processing_complete')
            or st.session_state.get('analysis_halted_due_to_error')
        )
        if _run_terminated:
            _purge_session_cache()
            release_analysis_slot(current_session_id)
        else:
            # Same reason we keep the cache: the slot also has to stay. If
            # we release it here, a Streamlit rerun mid-analysis (browser
            # sleep / widget re-render) drops the slot, and no other code
            # path re-acquires it — the run is dead. Leaving the slot in
            # place means the NEXT rerun hits acquire_analysis_slot's
            # same-session dedup and silently takes back its own slot,
            # then resumes phases from the disk cache.
            print(f"SESSION {current_session_id} LOG: analysis block exited "
                  "without terminal state (likely Streamlit rerun) — "
                  "keeping session cache AND slot so next run can resume")
    
    # Final summary
    final_summary_text = (
        f"### Final Summary ({'Halted' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    summary_placeholder.markdown(final_summary_text)
    
    # Download button
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_main"
        )

    # "New Analysis" button — wipes state + frees the slot so the
    # next user (or this user with a different file) can proceed
    # immediately without refreshing the browser.
    st.markdown("---")
    _nac1, _nac2 = st.columns([1, 3])
    with _nac1:
        if st.button("🆕 New Analysis", type="primary", key="new_analysis_btn_main"):
            _reset_analysis_state(purge_cache=True)
            st.rerun()
    with _nac2:
        st.caption("Starts fresh — clears cached results, releases the analysis slot, lets you upload a new file.")


# ==============================================================================
# Display Previously Processed Results (on rerun)
# ==============================================================================

elif st.session_state.processing_complete:
    print(f"SESSION {current_session_id} LOG: Displaying previously processed results (rerun).")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    
    final_summary_text_rerun = (
        f"### Final Summary ({'Halted Previously' if st.session_state.analysis_halted_due_to_error else 'Completed'})\n"
        f"- Processed: {st.session_state.total_pages_processed_count}/{st.session_state.doc_total_pages}\n"
        f"- ✅ Fence: {len(st.session_state.fence_pages)}\n"
        f"- ❌ Non-Fence: {len(st.session_state.non_fence_pages)}"
    )
    st.markdown(final_summary_text_rerun)
    
    if st.session_state.get('highlighted_pdf_bytes_for_download') and not st.session_state.analysis_halted_due_to_error:
        st.download_button(
            "⬇️ Download Highlighted Fence Pages (PDF)",
            st.session_state.highlighted_pdf_bytes_for_download,
            st.session_state.highlighted_pdf_filename_for_download,
            "application/pdf",
            key="dl_combined_pdf_rerun"
        )

    # Re-shown here so the user sees the same option when coming back to a
    # previously-completed session (without this, the only way to reset was
    # to refresh the browser).
    _nrc1, _nrc2 = st.columns([1, 3])
    with _nrc1:
        if st.button("🆕 New Analysis", type="primary", key="new_analysis_btn_rerun"):
            _reset_analysis_state(purge_cache=True)
            st.rerun()
    with _nrc2:
        st.caption("Clears these results and frees the analysis slot.")

    col_f_res, col_nf_res = st.columns(2)
    with col_f_res:
        st.subheader(f"✅ Fence-Related Pages ({len(st.session_state.fence_pages)})")
    with col_nf_res:
        st.subheader(f"❌ Non-Fence Pages ({len(st.session_state.non_fence_pages)})")
    
    def display_page_result_expander(res_data_list, target_column_res):
        for res_data_item in res_data_list:
            with target_column_res:
                exp_title_res = f"Page {res_data_item['page_number']}"
                definitions = res_data_item.get('definitions', [])
                instances = res_data_item.get('instances', [])
                keyword_matches = res_data_item.get('keyword_matches', [])
                detection_method = res_data_item.get('detection_method', 'none')
                fallback_result = res_data_item.get('fallback_result')
                
                if res_data_item.get('fence_found'):
                    reasons_res = []
                    if definitions:
                        reasons_res.append("Definitions")
                    if instances:
                        reasons_res.append("Instances")
                    if keyword_matches and not definitions:
                        reasons_res.append("Keywords")
                    if res_data_item.get('highlight_fence_text_app_setting', True) and \
                       (definitions or instances or keyword_matches or res_data_item.get('measurements')):
                        reasons_res.append("Highlights")
                    if reasons_res:
                        exp_title_res += f" ({' & '.join(reasons_res)})"
                
                _pidx_for_exp = res_data_item.get('page_index_in_original_doc', 0)
                _flag_for_exp = f'_page_img_loaded_{_pidx_for_exp}'
                _res_expanded = bool(st.session_state.get(_flag_for_exp))
                # We're not using st.expander here — Streamlit expanders
                # don't expose their open/closed state to Python, so we
                # couldn't hook a "free this page's memory" callback into
                # a close-click. Instead use a single full-width button
                # that toggles between two states. When the user clicks
                # to OPEN, on_click flips _flag_for_exp = True and the
                # body renders below. When they click again to CLOSE,
                # the callback flips it back to False AND evicts the
                # rendered image bytes / unified_measurements lines for
                # this page from session_state, so the memory actually
                # comes back.
                def _on_toggle_page(flag=_flag_for_exp,
                                    pidx=_pidx_for_exp,
                                    pkey=f"page_{res_data_item['page_number']}"):
                    new = not bool(st.session_state.get(flag))
                    st.session_state[flag] = new
                    if not new:
                        # Closing: free this page's heavy state.
                        # 1) drop image bytes from the LRU
                        try:
                            from collections import OrderedDict as _OD
                            _ic = st.session_state.get(_IMG_CACHE_KEY)
                            if isinstance(_ic, _OD):
                                _drop = [k for k in list(_ic.keys()) if k[1] == pidx]
                                for _k in _drop:
                                    _ic.pop(_k, None)
                        except Exception:
                            pass
                        # 2) drop unified_measurements row for this page
                        try:
                            st.session_state.unified_measurements.pop(pkey, None)
                        except Exception:
                            pass
                        # 3) drop UMT-side per-page state (line stats,
                        #    resized base/drawn image WEBPs, etc.)
                        try:
                            for _k in list(st.session_state.keys()):
                                if any(_k.startswith(p) for p in (
                                    'base_img_', 'base_img_size_', 'drawn_img_',
                                    'orig_img_size_', 'line_stats_', 'lines_',
                                    'auto_synced_', 'auto_matched_indices_',
                                    'click_key_',
                                )):
                                    if f"_{res_data_item['page_number']}_" in _k or _k.endswith(f"_{res_data_item['page_number']}"):
                                        del st.session_state[_k]
                        except Exception:
                            pass
                        try:
                            import gc as _gc; _gc.collect()
                            import ctypes as _ct
                            _ct.CDLL("libc.so.6").malloc_trim(0)
                        except Exception:
                            pass

                _arrow = "▼" if _res_expanded else "▶"
                st.button(
                    f"{_arrow}  {exp_title_res}",
                    key=f'_btn_expand_{_pidx_for_exp}',
                    use_container_width=True,
                    on_click=_on_toggle_page,
                )

                if _res_expanded:

                    img_col_r, det_col_r = st.columns([2, 1])

                    with img_col_r:
                        # Lazy-load gate: Streamlit runs the expander body
                        # on every script rerun regardless of whether the
                        # user has visually opened it, so without an
                        # explicit flag every "Load page image" click on
                        # ONE page would fire a rerun and render images
                        # for ALL pages. The flag is scoped per-page so
                        # only pages the user has clicked actually render.
                        #
                        # Initialise _orig_r / _hl_r up here so the
                        # download-link block below always has a defined
                        # binding — even for pages the user hasn't
                        # expanded. Without this the rerun path into the
                        # download links raised UnboundLocalError on the
                        # first "Load page image" click.
                        _orig_r, _hl_r = None, None
                        _pidx_r = res_data_item.get('page_index_in_original_doc', 0)
                        _img_flag_r = f'_page_img_loaded_{_pidx_r}'
                        if not st.session_state.get(_img_flag_r):
                            # on_click pattern — same as the other Load
                            # buttons. Note: this inner image button
                            # uses the SAME _page_img_loaded_<idx> flag
                            # as the outer Load-page-details button, so
                            # in practice it's only seen when the user
                            # took some other path; either click flips
                            # the same flag.
                            def _on_load_inner(flag=_img_flag_r):
                                st.session_state[flag] = True

                            st.button("🖼️ Load page image",
                                      key=f'_btn_{_img_flag_r}',
                                      use_container_width=True,
                                      on_click=_on_load_inner)
                            st.caption("Click above to render this page from the PDF.")
                        else:
                            # Pass raw dict lists (NOT hash-flattened
                            # tuples). highlight_page_image expects
                            # dict.get('x0') etc. — tuples silently crash
                            # inside its try/except and fall back to the
                            # un-highlighted PNG. The LRU cache key is
                            # computed from json.dumps inside
                            # _img_cache_key, so raw structures are fine.
                            _pdf_bytes_r = _get_pdf_bytes()
                            if _pdf_bytes_r and not res_data_item.get('skipped_damaged'):
                                _kws_for_r = [
                                    k for k in (keyword_matches or [])
                                    if all(key in k for key in ['x0', 'y0', 'x1', 'y1'])
                                ]
                                _orig_r, _hl_r = get_page_image_on_demand(
                                    st.session_state.current_pdf_hash,
                                    _pdf_bytes_r,
                                    _pidx_r,
                                    definitions or [], instances or [], _kws_for_r,
                                    res_data_item.get('pdf_width', 792),
                                    res_data_item.get('pdf_height', 612),
                                    res_data_item.get('highlight_fence_text_app_setting', True)
                                        and res_data_item.get('fence_found', False),
                                    dpi=DISPLAY_IMAGE_DPI,
                                )
                            disp_img_r = _hl_r or _orig_r
                            if disp_img_r:
                                st.image(disp_img_r, caption=f"Page {res_data_item['page_number']}")
                            elif res_data_item.get('skipped_damaged'):
                                st.warning("⚠ Page could not be rendered — PDF damage on this page.")
                        
                        dl_links_rerun = []
                        if _hl_r:
                            dl_links_rerun.append(get_image_download_link_html(
                                _hl_r,
                                f"page_{res_data_item['page_number']}_hl.png",
                                "DL HL Img"
                            ))
                        if _orig_r:
                            dl_links_rerun.append(get_image_download_link_html(
                                _orig_r,
                                f"page_{res_data_item['page_number']}_orig.png",
                                "DL Orig Img"
                            ))
                        if dl_links_rerun:
                            st.markdown(" ".join(dl_links_rerun), unsafe_allow_html=True)
                    
                    with det_col_r:
                        # Non-fence pages get a reasoning panel instead of
                        # the fence-specific detection metrics.
                        if not res_data_item.get('fence_found'):
                            if res_data_item.get('skipped_damaged'):
                                st.warning("⚠ Skipped — damaged page")
                            else:
                                st.info("❌ Non-fence")
                            _reason = res_data_item.get('reason')
                            if _reason:
                                st.markdown(f"**Why:** {_reason}")
                            _method = res_data_item.get('method')
                            if _method:
                                st.caption(f"method: `{_method}`")
                            _conf = res_data_item.get('confidence')
                            if _conf is not None:
                                try:
                                    st.caption(f"confidence: {float(_conf):.0%}")
                                except Exception:
                                    st.caption(f"confidence: {_conf}")
                            _kws = res_data_item.get('matched_keywords') or []
                            if _kws:
                                st.caption(
                                    "keywords present but rejected: "
                                    + ", ".join(_kws[:6])
                                )
                        else:
                            # Detection method badge — same logic as the live
                            # analysis loop (any keyword-prefilter method OR
                            # matched keywords → Keyword-based Detection).
                            _has_kw = bool(keyword_matches) or detection_method in (
                                "llm_confirmed", "keyword_only", "keyword_scan", "keyword_high_signal",
                            )
                            if detection_method == "ade":
                                st.success("🎯 ADE Detection")
                            elif detection_method == "llm_confirmed":
                                st.warning("🔍 Keyword + LLM")
                            elif _has_kw:
                                st.warning("🔤 Keyword-based Detection")
                            else:
                                st.info("❌ No Detection")

                            # ADE Stats (compact)
                            st.metric("ADE Chunks", res_data_item.get('chunk_count', 0))
                            col_leg_r, col_fig_r = st.columns(2)
                            with col_leg_r:
                                st.metric("Legend", res_data_item.get('legend_count', 0))
                            with col_fig_r:
                                st.metric("Figure", res_data_item.get('figure_count', 0))

                            if res_data_item.get('text_response'):
                                with st.popover("Analysis Log"):
                                    st.markdown(f"_{res_data_item['text_response']}_")
                    
                    # Found Items Section (below the image/details row)
                    st.subheader("Found Items")
                    
                    if definitions:
                        st.markdown("### 🟢 Definitions (Legend)")
                        df_def = pd.DataFrame(definitions)
                        # Filter out "Indicator Code" helper rows
                        if "description" in df_def.columns:
                            df_display = df_def[df_def["description"] != "Indicator Code"]
                            if not df_display.empty:
                                display_cols = ["indicator", "keyword", "description"]
                                available_cols = [c for c in display_cols if c in df_display.columns]
                                st.dataframe(df_display[available_cols], hide_index=True)
                            else:
                                st.info("No definition details available.")
                        else:
                            st.dataframe(df_def, hide_index=True)
                        
                        # Show element details if available
                        el_details = st.session_state.get('element_details', {})
                        if el_details:
                            st.markdown("### 📋 Element Specifications")
                            detail_rows = []
                            seen_kw = set()
                            for d in definitions:
                                kw = d.get('keyword', '').strip()
                                if not kw or kw in seen_kw or d.get('description', '') == 'Indicator Code':
                                    continue
                                seen_kw.add(kw)
                                ind = d.get('indicator', '').strip()
                                cat_label = f"{ind}: {kw}" if ind else kw
                                details = _lookup_element_details(cat_label, el_details)
                                if details and any(v for v in details.values() if v):
                                    detail_rows.append({
                                        'Element': cat_label,
                                        'Height': details.get('height', ''),
                                        'Post Type': details.get('post_type', ''),
                                        'Post Spacing': details.get('post_spacing', ''),
                                        'Material': details.get('material', ''),
                                        'Gauge': details.get('gauge', ''),
                                        'Mesh Size': details.get('mesh_size', ''),
                                        'Detail Page': details.get('detail_page', ''),
                                    })
                            if detail_rows:
                                st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
                                # Full details in expandable section
                                with st.expander("📝 Full Detail Text", expanded=False):
                                    for dr in detail_rows:
                                        elem = dr['Element']
                                        full = _lookup_element_details(elem, el_details).get('full_details', '')
                                        if full:
                                            st.markdown(f"**{elem}:** {full}")
                    
                    if instances:
                        st.markdown("### 🟣 Instances (Drawings)")
                        df_inst = pd.DataFrame(instances)
                        if "indicator" in df_inst.columns:
                            st.dataframe(df_inst[["indicator"]], hide_index=True)
                        else:
                            st.dataframe(df_inst, hide_index=True)
                    
                    # Show keyword matches from fallback detection
                    if keyword_matches and not definitions:
                        st.markdown("### 🟠 Keyword Matches (no ADE structures found on this page)")
                        df_kw = pd.DataFrame(keyword_matches)
                        if not df_kw.empty:
                            display_cols = ["keyword", "text"]
                            available_cols = [c for c in display_cols if c in df_kw.columns]
                            if available_cols:
                                # Deduplicate by text
                                df_kw_unique = df_kw.drop_duplicates(subset=["text"])
                                st.dataframe(df_kw_unique[available_cols], hide_index=True)
                        
                        # Show LLM reasoning if available
                        if fallback_result and fallback_result.get("llm_result"):
                            llm_res = fallback_result["llm_result"]
                            st.markdown("**LLM Analysis:**")
                            st.markdown(f"- Confidence: {llm_res.get('confidence', 0):.0%}")
                            st.markdown(f"- Reason: {llm_res.get('reason', 'N/A')}")
                        
                        # Show Measurements
                        measurements = res_data_item.get('measurements')
                        if measurements and (measurements.get('indicator_measurements') or measurements.get('totals', {}).get('total_length_feet', 0) > 0):
                            st.markdown("---")
                            st.markdown("### 📏 Fence Measurements")
                            
                            page_info = measurements.get('page_info', {})
                            scale_factor = page_info.get('scale_factor', 1.0)
                            
                            # Show scale info prominently
                            if page_info.get('scale_detected'):
                                st.success(f"✅ Scale Auto-Detected: 1\" = {scale_factor/12:.0f}' (factor: {scale_factor})")
                            else:
                                st.warning("⚠️ Scale not detected - showing raw measurements")
                            
                            # Show proximity-based measurements (primary)
                            prox_totals = measurements.get('proximity_totals', {})
                            if prox_totals.get('total_segments', 0) > 0:
                                st.markdown("#### 🎯 Near Detected Indicators:")
                                col_pts, col_ft = st.columns(2)
                                with col_pts:
                                    st.metric("Total (Points)", f"{prox_totals.get('total_length_pts', 0):,.0f} pts")
                                with col_ft:
                                    st.metric("Total (Scaled)", f"{prox_totals.get('total_length_feet', 0):.1f} ft")
                                
                                # Per-indicator breakdown
                                indicator_meas = measurements.get('indicator_measurements', {})
                                if indicator_meas:
                                    st.markdown("**Per-Indicator:**")
                                    for ind, stats in indicator_meas.items():
                                        pts = stats.get('run_length_pts', 0)
                                        ft = stats.get('run_length_feet', 0)
                                        segs = stats.get('run_segment_count', 0)
                                        count = stats.get('instance_count', 0)
                                        st.markdown(f"- **{ind}**: {pts:,.0f} pts | **{ft:.1f} ft** ({segs} segs, {count} instances)")
                            
                            # Layer breakdown (secondary)
                            if measurements.get('fence_layers'):
                                with st.expander("📂 Layer-Based Breakdown", expanded=False):
                                    totals = measurements.get('totals', {})
                                    st.caption(f"Total from layers: {totals.get('total_segments', 0)} segs, {totals.get('total_length_feet', 0):.1f} ft")
                                    for layer in measurements['fence_layers']:
                                        l_stats = measurements['layer_measurements'].get(layer, {})
                                        segs = l_stats.get('total_segments', 0)
                                        ft = l_stats.get('total_length_feet', 0)
                                        runs = l_stats.get('connected_runs', 0)
                                        st.markdown(f"- `{layer}`: {segs} segs | {ft:.1f} ft ({runs} runs)")
                            
                            # Dimension line measurements
                            dim_measurements = measurements.get('dimension_measurements', [])
                            if dim_measurements:
                                with st.expander("📐 Dimension Line Measurements", expanded=False):
                                    st.caption(f"Found {len(dim_measurements)} dimension annotations")
                                    for dm in dim_measurements[:10]:
                                        ft = dm.get('actual_ft', 0)
                                        txt = dm.get('measurement_text', '')
                                        st.markdown(f"- **{txt}**: {ft:.1f} ft")
                    
                    # Show message if nothing found
                    if not definitions and not instances and not keyword_matches:
                        st.info("No fence-related items found on this page.")
    
    display_page_result_expander(st.session_state.fence_pages, col_f_res)
    display_page_result_expander(st.session_state.non_fence_pages, col_nf_res)


# ==============================================================================
# Unified Measurement Tool (Auto + Interactive)
# ==============================================================================

if st.session_state.processing_complete and st.session_state.fence_pages and enable_unified_measurement:
    st.markdown("---")
    st.markdown("<h2>📏 Unified Measurement Tool</h2>", unsafe_allow_html=True)
    st.caption("🤖 Auto-detected lines shown in cyan | 👆 Click to select manual lines | ✏️ Draw custom lines")

    # Whole-tool lazy gate. Rendering the tool for every fence page
    # costs ~1-4 MB of session_state per page plus an LLM scale-detect
    # call per page — a 68-fence-page deck paid that on every rerun.
    # Gate the entire section behind an explicit button; only users who
    # actually need the interactive measurement canvas load it. The
    # user's modifications (line assignments, drawn lines) are
    # persisted in session_state, so re-opening the tool restores them.
    _UMT_LOADED_KEY = "_umt_tool_loaded"
    if not st.session_state.get(_UMT_LOADED_KEY):
        _umt_col1, _umt_col2 = st.columns([1, 3])
        with _umt_col1:
            # on_click pattern — same as the analysis-side Load buttons.
            def _on_open_umt():
                st.session_state[_UMT_LOADED_KEY] = True

            st.button("📏 Open Measurement Tool",
                      key="_umt_open_btn",
                      type="primary",
                      use_container_width=True,
                      on_click=_on_open_umt)
        with _umt_col2:
            st.caption(
                "Interactive canvas for auto-detected + manually drawn fence lines. "
                "Heavy (scale-detect LLM + vector line detection per page) — "
                "only loads when requested. Your assignments are saved."
            )
        st.stop()  # skip the rest of the tool this run

    # Auto-detect and verify scale PER PAGE using LLM
    from utils_vector import verify_scale_with_bar
    
    # Detect scale for each fence page (cached in session_state)
    if 'per_page_scale_info' not in st.session_state:
        st.session_state.per_page_scale_info = {}
    
    # User-drawn lines storage: {page_key: [{'start': (x,y), 'end': (x,y), 'category': cat}, ...]}
    if 'user_drawn_lines' not in st.session_state:
        st.session_state.user_drawn_lines = {}
    
    # Drawing mode per page
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = {}
    
    # Pending point for line drawing (first click of a two-click line)
    if 'pending_line_start' not in st.session_state:
        st.session_state.pending_line_start = {}
    
    # Detect scales for all pages if not already done
    _pdf_bytes_for_scale = _get_pdf_bytes()
    with fitz.open(stream=BytesIO(_pdf_bytes_for_scale), filetype="pdf") as doc:
        for fence_page in st.session_state.fence_pages:
            page_num = fence_page['page_number']
            cache_key = f"page_{page_num}"
            if cache_key not in st.session_state.per_page_scale_info:
                try:
                    page_idx = fence_page['page_index_in_original_doc']
                    pdf_page = doc[page_idx]
                    # Use LLM for intelligent scale detection
                    scale_info = verify_scale_with_bar(pdf_page, llm=scale_llm_instance or llm_analysis_instance)
                    st.session_state.per_page_scale_info[cache_key] = scale_info
                except Exception as e:
                    st.session_state.per_page_scale_info[cache_key] = {
                        'success': False, 'verified_scale': None, 'message': str(e)
                    }
    
    # Global settings (min line length only - scale is now per-page)
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        min_line_pts = st.number_input(
            "Min line length (pts)",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            help="Filter out short lines (hatching, text)",
            key="min_line_pts_input"
        )
    with col_g2:
        st.info("📐 Scale detected per page (see each tab)")
    
    # Zoom slider (higher default for better quality)
    zoom_level = st.slider("🔍 Zoom", min_value=600, max_value=2000, value=1200, step=100, 
                           help="Adjust image display width")
    
    # Create tabs for each fence page
    page_tabs = st.tabs([f"Page {p['page_number']}" for p in st.session_state.fence_pages])
    
    # Track line assignments per page: {page_key: {line_idx: category_name}}
    if 'line_assignments' not in st.session_state:
        st.session_state.line_assignments = {}
    
    # Track categories per page: {page_key: {cat_name: {indicator, keyword, color}}}
    if 'page_categories' not in st.session_state:
        st.session_state.page_categories = {}
    
    # Track active category per page
    if 'active_category_per_page' not in st.session_state:
        st.session_state.active_category_per_page = {}
    
    # Category colors for consistent assignment
    CATEGORY_COLORS = [
        (0, 255, 0),      # Green
        (255, 165, 0),    # Orange
        (0, 191, 255),    # Deep sky blue
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
        (255, 105, 180),  # Hot pink
        (173, 255, 47),   # Green yellow
    ]
    
    # Import PIL once outside loop
    from PIL import Image, ImageDraw
    
    # OPTIMIZATION 5: Use st.fragment for partial reruns (only rerun the page content, not entire app)
    @st.fragment
    def render_page_fragment(page_data, zoom_level, min_line_pts):
        """Fragment function for each page - only this reruns on interaction"""
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        page_idx = page_data['page_index_in_original_doc']
        pdf_width = page_data.get('pdf_width', 792)
        pdf_height = page_data.get('pdf_height', 612)

        # On-demand re-sync of auto_lines from the phase3_measure disk
        # cache. When a page's auto_lines slot is empty but the cache
        # has a real measurement_result (either 'layer' or 'llm_guided'),
        # build auto_lines here so the canvas shows them. This lets the
        # user open the measurement tool AFTER a run finished and still
        # see layer-less (site-plan-style) pages populated — without
        # having to re-run analysis or flip the Non-layer suggestions
        # toggle. Idempotent: skipped if auto_lines already populated.
        _sync_meas = st.session_state.unified_measurements.get(page_key, {})
        # _pdf_sha / _cache_params are closure vars of the analysis
        # block — not visible from here. The analysis saves them to
        # session_state right after computing so this post-analysis
        # code can pull them back out. If they're missing (analysis
        # hasn't run yet this session) just skip the sync.
        _umt_pdf_sha = st.session_state.get('_pdf_sha_cached')
        _umt_cache_params = st.session_state.get('_cache_params_cached')
        if not _sync_meas.get('auto_lines') and _umt_pdf_sha and _umt_cache_params:
            _cached_meas = _cache_get("phase3_measure", _umt_pdf_sha, _umt_cache_params,
                                      page_idx=page_idx)
            if _cached_meas and _cached_meas.get('all_fence_lines'):
                _mm = _cached_meas.get('measurement_method', 'none')
                # Same gating as the render loop: 'layer' always renders,
                # 'llm_guided' / 'skipped' only when the user has flipped
                # the Non-layer suggestions sidebar toggle on.
                _sync_accept = (_mm == 'layer') or (
                    enable_nonlayer_suggestions and _mm in ('llm_guided', 'skipped')
                )
                if _sync_accept:
                    _l2c = _cached_meas.get('layer_to_category', {}) or {}
                    _sf = _cached_meas.get('page_info', {}).get('scale_factor', 1.0)
                    _fallback_cat = (
                        "🔍 Auto-detected (LLM-guided)" if _mm == 'llm_guided'
                        else "🔍 Auto-detected (high-density, layer-skipped)" if _mm == 'skipped'
                        else None
                    )
                    _auto_now = []
                    for _ln in _cached_meas['all_fence_lines']:
                        _lyr = getattr(_ln, 'layer', None) or ''
                        _cat = _l2c.get(_lyr)
                        if not _cat and _lyr:
                            for _ml, _c in _l2c.items():
                                if _ml in _lyr or _lyr in _ml:
                                    _cat = _c
                                    break
                        if not _cat:
                            _cat = _fallback_cat
                        if not _cat:
                            continue
                        _lpts = _ln.length_pts
                        _auto_now.append({
                            'start': _ln.start,
                            'end': _ln.end,
                            'length_pts': _lpts,
                            'length_feet': ((_lpts / 72.0) * _sf) / 12.0,
                            'layer': _lyr,
                            'category': _cat,
                            'source': 'auto',
                            'method': _mm,
                        })
                    if _auto_now:
                        if page_key not in st.session_state.unified_measurements:
                            st.session_state.unified_measurements[page_key] = {
                                'auto_lines': [], 'manual_lines': [],
                                'drawn_lines': [], 'accepted_auto': set(),
                            }
                        st.session_state.unified_measurements[page_key]['auto_lines'] = _auto_now
                        st.session_state.unified_measurements[page_key]['accepted_auto'] = set(range(len(_auto_now)))
                        print(f"[UMT] page {page_num}: synced {len(_auto_now)} auto lines "
                              f"from phase3_measure cache (method={_mm})")
        
        # Extract lines from PDF page (cached)
        lines_cache_key = f"lines_{page_num}_{min_line_pts}"
        if lines_cache_key not in st.session_state:
            # Evict previous line caches for this page when min length changes.
            for k in [k for k in list(st.session_state.keys())
                      if k.startswith(f"lines_{page_num}_") and k != lines_cache_key]:
                del st.session_state[k]
            with fitz.open(stream=BytesIO(_get_pdf_bytes()), filetype="pdf") as doc:
                pdf_page = doc[page_idx]
                all_lines = extract_vector_lines(pdf_page)
                filtered_lines = [l for l in all_lines if l.length_pts >= min_line_pts]
                filtered_lines.sort(key=lambda l: l.length_pts, reverse=True)
                # Store only compact, serializable fields instead of full VectorLine objects.
                st.session_state[lines_cache_key] = [{
                    'start': (float(l.start[0]), float(l.start[1])),
                    'end': (float(l.end[0]), float(l.end[1])),
                    'length_pts': float(l.length_pts),
                    'layer': l.layer or 'default',
                } for l in filtered_lines]
        
        lines = st.session_state.get(lines_cache_key, [])
        
        if not lines:
            st.warning(f"No lines found on this page (min length: {min_line_pts} pts)")
            return
        
        # Initialize line assignments for this page: {line_idx: category_name}
        if page_key not in st.session_state.line_assignments:
            st.session_state.line_assignments[page_key] = {}
        
        # Pre-populate from auto-detected lines by matching coordinates to vector lines
        # This runs once per analysis (keyed by auto_lines count to re-trigger on new analysis)
        unified_page = st.session_state.unified_measurements.get(page_key, {})
        auto_lines_data = unified_page.get('auto_lines', [])
        accepted_auto = unified_page.get('accepted_auto', set())
        auto_sync_key = f"auto_synced_{page_key}_{len(auto_lines_data)}"
        
        if auto_lines_data and accepted_auto and lines and auto_sync_key not in st.session_state:
            import math
            matched = 0
            matched_indices = set()
            for ai in accepted_auto:
                if ai >= len(auto_lines_data):
                    continue
                auto_line = auto_lines_data[ai]
                a_sx, a_sy = auto_line['start']
                a_ex, a_ey = auto_line['end']
                category = auto_line.get('category')
                if not category:
                    continue
                
                # Find closest vector line by endpoint distance
                best_idx = None
                best_dist = float('inf')
                for vi, vline in enumerate(lines):
                    v_sx, v_sy = vline['start']
                    v_ex, v_ey = vline['end']
                    # Try both orientations (line direction may differ)
                    d1 = math.hypot(a_sx - v_sx, a_sy - v_sy) + math.hypot(a_ex - v_ex, a_ey - v_ey)
                    d2 = math.hypot(a_sx - v_ex, a_sy - v_ey) + math.hypot(a_ex - v_sx, a_ey - v_sy)
                    d = min(d1, d2)
                    if d < best_dist:
                        best_dist = d
                        best_idx = vi
                
                # Coordinates come from same extract_vector_lines, should be near-exact
                if best_idx is not None and best_dist < 2.0:
                    st.session_state.line_assignments[page_key][best_idx] = category
                    matched_indices.add(best_idx)
                    matched += 1
                else:
                    print(f"[AUTO-PREPOP] No match for auto line {ai} (best_dist={best_dist:.2f})")
            
            # Track which vector line indices were auto-matched (for clear button)
            auto_matched_key = f"auto_matched_indices_{page_key}"
            st.session_state[auto_matched_key] = matched_indices
            st.session_state[auto_sync_key] = matched
            print(f"[AUTO-PREPOP] Page {page_num}: matched {matched}/{len(accepted_auto)} auto lines to {len(lines)} vector lines")
        
        # Initialize categories for this page from its definitions
        if page_key not in st.session_state.page_categories:
            categories = {}
            definitions = page_data.get('definitions', [])
            for d in definitions:
                indicator = d.get('indicator', '')
                keyword = d.get('keyword', '')
                if keyword:
                    cat_name = f"{indicator}: {keyword}" if indicator else keyword
                    if cat_name not in categories:
                        color_idx = len(categories)
                        categories[cat_name] = {
                            'indicator': indicator,
                            'keyword': keyword,
                            'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                        }
            st.session_state.page_categories[page_key] = categories
        
        page_categories = st.session_state.page_categories[page_key]
        
        # Initialize active category for this page
        if page_key not in st.session_state.active_category_per_page:
            cats = list(page_categories.keys())
            st.session_state.active_category_per_page[page_key] = cats[0] if cats else None
        
        # Show per-page scale info
        page_scale_info = st.session_state.per_page_scale_info.get(page_key, {})
        page_scale = page_scale_info.get('verified_scale') or page_scale_info.get('text_scale') or 360.0
        scale_min = 1.0
        scale_max = 1200.0
        page_scale_clamped = max(scale_min, min(float(page_scale), scale_max))
        
        scale_col1, scale_col2 = st.columns([2, 1])
        with scale_col1:
            page_scale_input = st.number_input(
                f"Scale (Page {page_num})",
                min_value=scale_min,
                max_value=scale_max,
                value=page_scale_clamped,
                step=12.0,
                help=f"1\" = {page_scale_clamped/12:.1f}' actual",
                key=f"scale_input_{page_num}"
            )
            if float(page_scale) != page_scale_clamped:
                st.caption(
                    f"Detected scale {float(page_scale):.1f} was outside allowed range "
                    f"({scale_min:.0f}-{scale_max:.0f}) and was clamped."
                )
        with scale_col2:
            if page_scale_info.get('success'):
                confidence = page_scale_info.get('confidence', 'low')
                scale_text = page_scale_info.get('scale_text', '')
                display_text = f"✓ {scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'"
                if confidence == 'high':
                    st.success(display_text)
                elif confidence == 'medium':
                    st.warning(f"⚠ {scale_text}" if scale_text else f"1\"={page_scale/12:.0f}'")
                else:
                    st.info(scale_text if scale_text else f"1\"={page_scale/12:.0f}'")
            else:
                st.warning("Not detected")
        
        # Show scale detection details
        with st.expander("🔍 Scale Detection Details", expanded=False):
            # Page size info
            page_size = page_scale_info.get('page_size', {})
            if page_size:
                size_str = f"{page_size.get('width_inches', 0):.1f}\" x {page_size.get('height_inches', 0):.1f}\""
                detected = page_size.get('detected_size', 'Unknown')
                st.markdown(f"**Page size:** {size_str} ({detected})")
            
            # Scale detection
            method = page_scale_info.get('method', 'unknown')
            st.markdown(f"**Detection method:** {method}")
            scale_text = page_scale_info.get('scale_text', '')
            st.markdown(f"**Detected scale text:** {scale_text if scale_text else 'None'}")
            st.markdown(f"**Confidence:** {page_scale_info.get('confidence', 'N/A')}")
            st.markdown(f"**Message:** {page_scale_info.get('message', 'N/A')}")
            if page_scale_info.get('verified_scale'):
                scale_val = page_scale_info['verified_scale']
                st.markdown(f"**Scale value:** 1\" = {scale_val/12:.0f}' ({scale_val} inches)")
            
            # Debug: show raw LLM response
            raw = page_scale_info.get('raw_response', '')
            if raw:
                st.markdown("**LLM Response:**")
                st.code(raw[:500], language=None)
            
            # Debug: show extracted text sample
            extracted = page_scale_info.get('extracted_text_sample', '')
            if extracted:
                st.markdown("**Extracted PDF Text (first 1500 chars):**")
                st.code(extracted, language=None)
        
        # =====================================================================
        # AUTO-DETECTED MEASUREMENTS SECTION
        # =====================================================================
        unified_page_data = st.session_state.unified_measurements.get(page_key, {})
        auto_lines = unified_page_data.get('auto_lines', [])
        
        if auto_lines:
            accepted_auto = unified_page_data.get('accepted_auto', set())
            accepted_count = len(accepted_auto)
            accepted_ft = sum(auto_lines[i].get('length_feet', 0) for i in accepted_auto if i < len(auto_lines))
            
            # Check how many auto lines are currently matched in line_assignments
            auto_matched_key = f"auto_matched_indices_{page_key}"
            auto_matched_indices = st.session_state.get(auto_matched_key, set())
            currently_assigned = sum(1 for idx in auto_matched_indices if idx in st.session_state.line_assignments.get(page_key, {}))
            
            st.markdown("#### 🤖 Auto-Detected Fence Lines (Pre-Selected)")
            
            auto_col1, auto_col2, auto_col3 = st.columns([2, 1, 1])
            with auto_col1:
                if currently_assigned > 0:
                    st.success(f"✓ {currently_assigned} lines matched & selected ({accepted_ft:.1f} ft)")
                elif accepted_count > 0:
                    st.warning(f"{accepted_count} auto lines detected but not yet synced to selections")
                else:
                    st.info("No auto-detected lines")
            with auto_col2:
                if st.button("🔄 Re-sync Auto", key=f"resync_auto_{page_num}", help="Re-match auto-detected lines to selectable vector lines"):
                    # Clear old sync keys to force re-matching
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"auto_synced_{page_key}")]
                    for k in keys_to_remove:
                        del st.session_state[k]
                    if auto_matched_key in st.session_state:
                        del st.session_state[auto_matched_key]
                    st.rerun(scope="fragment")
            with auto_col3:
                if st.button("❌ Clear Auto", key=f"clear_auto_{page_num}", help="Remove all auto-detected lines from selection"):
                    # Remove auto-matched assignments from line_assignments
                    page_assigns = st.session_state.line_assignments.get(page_key, {})
                    for idx in auto_matched_indices:
                        page_assigns.pop(idx, None)
                    st.session_state.unified_measurements[page_key]['accepted_auto'] = set()
                    st.session_state[auto_matched_key] = set()
                    # Clear sync keys
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(f"auto_synced_{page_key}")]
                    for k in keys_to_remove:
                        del st.session_state[k]
                    st.rerun(scope="fragment")
            
            # Show category breakdown for auto lines
            auto_by_cat = {}
            for i in accepted_auto:
                if i < len(auto_lines):
                    cat = auto_lines[i].get('category', 'Uncategorized')
                    if cat not in auto_by_cat:
                        auto_by_cat[cat] = {'count': 0, 'feet': 0}
                    auto_by_cat[cat]['count'] += 1
                    auto_by_cat[cat]['feet'] += auto_lines[i].get('length_feet', 0)
            
            if auto_by_cat:
                for cat, data in auto_by_cat.items():
                    cat_color = page_categories.get(cat, {}).get('color', (0, 255, 255))
                    st.markdown(f"<span style='color: rgb{cat_color};'>●</span> **{cat}**: {data['count']} lines, {data['feet']:.1f} ft", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category selector for this page
        st.markdown("#### 🏷️ Fence Categories (This Page)")
        cat_col1, cat_col2 = st.columns([3, 1])
        with cat_col1:
            category_options = list(page_categories.keys())
            if category_options:
                current_active = st.session_state.active_category_per_page.get(page_key)
                active_cat = st.selectbox(
                    "Assign lines to:",
                    options=category_options,
                    index=category_options.index(current_active) if current_active in category_options else 0,
                    key=f"category_selector_{page_num}"
                )
                st.session_state.active_category_per_page[page_key] = active_cat
                if active_cat:
                    color = page_categories[active_cat]['color']
                    st.markdown(f"<span style='color: rgb{color}; font-size: 20px;'>●</span> Click lines to assign", unsafe_allow_html=True)
            else:
                st.info("No fence categories detected on this page.")
        
        with cat_col2:
            with st.popover("➕ Add"):
                new_cat_name = st.text_input("Category name:", key=f"new_cat_{page_num}")
                if st.button("Add", key=f"add_cat_btn_{page_num}") and new_cat_name:
                    if new_cat_name not in st.session_state.page_categories[page_key]:
                        color_idx = len(st.session_state.page_categories[page_key])
                        st.session_state.page_categories[page_key][new_cat_name] = {
                            'indicator': '',
                            'keyword': new_cat_name,
                            'color': CATEGORY_COLORS[color_idx % len(CATEGORY_COLORS)]
                        }
                        st.session_state.active_category_per_page[page_key] = new_cat_name
                        st.rerun(scope="fragment")
        
        # Mode toggle: Select existing lines vs Draw custom lines
        mode_col1, mode_col2 = st.columns([1, 1])
        with mode_col1:
            if page_key not in st.session_state.drawing_mode:
                st.session_state.drawing_mode[page_key] = "select"
            
            drawing_mode = st.radio(
                "Mode:",
                options=["select", "draw"],
                format_func=lambda x: "📍 Select Lines" if x == "select" else "✏️ Draw Lines",
                horizontal=True,
                key=f"mode_{page_num}"
            )
            st.session_state.drawing_mode[page_key] = drawing_mode
        
        with mode_col2:
            if drawing_mode == "draw":
                st.caption("Draw lines on the image. They will be assigned to the active category.")
        
        # Cache line stats (keyed by page + min_line_pts + scale)
        # Use the page-specific scale input
        effective_scale = page_scale_input
        line_stats_key = f"line_stats_{page_num}_{min_line_pts}_{effective_scale}"
        if line_stats_key not in st.session_state:
            # Evict old line_stats for this page (accumulate on every scale change)
            for k in [k for k in list(st.session_state.keys())
                      if k.startswith(f"line_stats_{page_num}_") and k != line_stats_key]:
                del st.session_state[k]
            stats = []
            for i, line in enumerate(lines):
                length_inches = line['length_pts'] / 72.0
                length_feet = (length_inches * effective_scale) / 12.0
                stats.append({
                    'index': i,
                    'length_pts': line['length_pts'],
                    'length_feet': length_feet,
                    'layer': line.get('layer') or 'default',
                    'start': line['start'],
                    'end': line['end']
                })
            st.session_state[line_stats_key] = stats
        line_stats = st.session_state[line_stats_key]
        
        # Get base image on demand (regenerate instead of reading from session_state)
        base_img_bytes = page_data.get('highlighted_image_bytes') or page_data.get('original_image_bytes')
        _pdf_bytes_img = _get_pdf_bytes() if not base_img_bytes else None
        if not base_img_bytes and _pdf_bytes_img:
            # Regenerate on demand. Pass raw dict lists — NOT the
            # hash-flattened tuples the earlier version built. The hash
            # is computed inside _img_cache_key; the renderer needs
            # .get('x0') access on each item for the highlight overlay
            # to draw. Flattened tuples silently broke the overlay and
            # returned the un-highlighted original PNG, which is what
            # the line-selection canvas was showing.
            _defs = page_data.get('definitions', []) or []
            _insts = page_data.get('instances', []) or []
            _kws = [
                k for k in (page_data.get('keyword_matches', []) or [])
                if all(key in k for key in ['x0', 'y0', 'x1', 'y1'])
            ]
            _orig_img, _hl_img = get_page_image_on_demand(
                st.session_state.current_pdf_hash,
                _pdf_bytes_img,
                page_idx, _defs, _insts, _kws,
                pdf_width, pdf_height,
                page_data.get('highlight_fence_text_app_setting', True),
                dpi=DISPLAY_IMAGE_DPI,
            )
            # Prefer the highlighted base for the line-selection canvas
            # — users expect to see the green-definition / magenta-instance
            # rectangles while picking lines, not the raw page.
            base_img_bytes = _hl_img or _orig_img
        
        if base_img_bytes:
            
            # OPTIMIZATION 1: Cache resized base image (keyed by page + zoom)
            # Evict old zoom levels for this page to bound memory
            base_img_cache_key = f"base_img_{page_num}_{zoom_level}"
            if base_img_cache_key not in st.session_state:
                # Evict previous zoom level caches for this page
                for k in [k for k in list(st.session_state.keys())
                          if (k.startswith(f"base_img_{page_num}_") or
                              k.startswith(f"base_img_size_{page_num}_") or
                              k.startswith(f"drawn_img_{page_num}_"))
                          and k != base_img_cache_key]:
                    del st.session_state[k]
                
                base_img = Image.open(BytesIO(base_img_bytes)).convert('RGB')
                orig_width, orig_height = base_img.size
                ratio = zoom_level / orig_width
                new_width = zoom_level
                new_height = int(orig_height * ratio)
                # Use LANCZOS for high quality resize
                base_img = base_img.resize((new_width, new_height), Image.LANCZOS)
                # Store as compressed WEBP bytes to reduce session memory footprint.
                _buf = BytesIO()
                base_img.save(_buf, format='WEBP', quality=88, method=6)
                st.session_state[base_img_cache_key] = _buf.getvalue()
                st.session_state[f"base_img_size_{page_num}_{zoom_level}"] = (new_width, new_height)
                st.session_state[f"orig_img_size_{page_num}"] = (orig_width, orig_height)
                del base_img, _buf
            
            # Decompress on demand (~5ms, negligible vs rendering)
            base_img_cached = Image.open(BytesIO(st.session_state[base_img_cache_key]))
            img_width, img_height = st.session_state[f"base_img_size_{page_num}_{zoom_level}"]
            
            # Scale factors from PDF to image coordinates
            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            
            line_assignments = st.session_state.line_assignments.get(page_key, {})
            
            # OPTIMIZATION 2: Cache drawn image with assignments
            # Create a hashable key from assignment state
            assignment_tuple = tuple(sorted(line_assignments.items()))
            drawn_img_cache_key = f"drawn_img_{page_num}_{zoom_level}_{hash(assignment_tuple)}"
            
            # Auto lines are now part of line_assignments, no separate cache key needed
            
            if drawn_img_cache_key not in st.session_state:
                # Evict old drawn images for this page (they leak on every assignment change)
                for k in [k for k in list(st.session_state.keys())
                          if k.startswith(f"drawn_img_{page_num}_") and k != drawn_img_cache_key]:
                    del st.session_state[k]
                # Copy base image and draw assignments
                display_img = base_img_cached.copy()
                draw = ImageDraw.Draw(display_img)
                
                # First pass: Draw ALL selectable lines with subtle color (unassigned)
                for i, ls in enumerate(line_stats):
                    x0 = ls['start'][0] * scale_x
                    y0 = ls['start'][1] * scale_y
                    x1 = ls['end'][0] * scale_x
                    y1 = ls['end'][1] * scale_y
                    # Subtle gray-blue for unassigned lines
                    if i not in line_assignments:
                        draw.line([(x0, y0), (x1, y1)], fill=(150, 180, 200), width=1)
                
                # Second pass: Draw ASSIGNED lines with category colors (auto-matched + manually selected)
                for i, ls in enumerate(line_stats):
                    if i in line_assignments:
                        category = line_assignments[i]
                        cat_info = page_categories.get(category, {})
                        color = cat_info.get('color', (0, 255, 0))
                        
                        x0 = ls['start'][0] * scale_x
                        y0 = ls['start'][1] * scale_y
                        x1 = ls['end'][0] * scale_x
                        y1 = ls['end'][1] * scale_y
                        # Draw with category color
                        draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=6)  # White outline
                        draw.line([(x0, y0), (x1, y1)], fill=color, width=4)
                        draw.ellipse([(x0-5, y0-5), (x0+5, y0+5)], fill=color)
                        draw.ellipse([(x1-5, y1-5), (x1+5, y1+5)], fill=color)
                
                _drawn_buf = BytesIO()
                display_img.save(_drawn_buf, format='WEBP', quality=90, method=6)
                st.session_state[drawn_img_cache_key] = _drawn_buf.getvalue()
                del _drawn_buf
            
            display_img = Image.open(BytesIO(st.session_state[drawn_img_cache_key])).convert("RGB")
            
            # Display clickable image and info side by side
            col_img, col_info = st.columns([3, 1])
            
            with col_img:
                # Initialize user-drawn lines for this page
                if page_key not in st.session_state.user_drawn_lines:
                    st.session_state.user_drawn_lines[page_key] = []
                
                if drawing_mode == "draw":
                    # DRAW MODE: Click two points to create a line
                    # Show pending start point if exists
                    pending_start = st.session_state.pending_line_start.get(page_key)
                    
                    # Draw user lines and pending point on image
                    draw_img = display_img.copy()
                    draw_overlay = ImageDraw.Draw(draw_img)
                    
                    # Draw existing user-drawn lines
                    user_lines = st.session_state.user_drawn_lines.get(page_key, [])
                    for ul in user_lines:
                        cat = ul.get('category')
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        x0 = ul['start'][0] * scale_x
                        y0 = ul['start'][1] * scale_y
                        x1 = ul['end'][0] * scale_x
                        y1 = ul['end'][1] * scale_y
                        draw_overlay.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=5)
                        draw_overlay.line([(x0, y0), (x1, y1)], fill=color, width=3)
                        draw_overlay.ellipse([(x0-4, y0-4), (x0+4, y0+4)], fill=color)
                        draw_overlay.ellipse([(x1-4, y1-4), (x1+4, y1+4)], fill=color)
                    
                    # Draw pending start point
                    if pending_start:
                        px, py = pending_start
                        img_px = px * scale_x
                        img_py = py * scale_y
                        draw_overlay.ellipse([(img_px-8, img_py-8), (img_px+8, img_py+8)], fill=(255, 255, 0), outline=(0, 0, 0))
                    
                    click_key = f"draw_click_{page_num}"
                    if click_key not in st.session_state:
                        st.session_state[click_key] = None
                    
                    click_result = streamlit_image_coordinates(
                        draw_img,
                        key=f"draw_img_{page_num}"
                    )
                    
                    if click_result is not None:
                        current_click = (click_result.get('x', 0), click_result.get('y', 0))
                        
                        if current_click != st.session_state[click_key]:
                            st.session_state[click_key] = current_click
                            click_x, click_y = current_click
                            pdf_click_x = click_x / scale_x
                            pdf_click_y = click_y / scale_y
                            
                            if pending_start is None:
                                # First click - set start point
                                st.session_state.pending_line_start[page_key] = (pdf_click_x, pdf_click_y)
                                st.rerun(scope="fragment")
                            else:
                                # Second click - create line
                                active_cat = st.session_state.active_category_per_page.get(page_key)
                                start_x, start_y = pending_start
                                end_x, end_y = pdf_click_x, pdf_click_y
                                
                                length_pts = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                                length_inches = length_pts / 72.0
                                length_feet = (length_inches * effective_scale) / 12.0
                                
                                new_line = {
                                    'start': (start_x, start_y),
                                    'end': (end_x, end_y),
                                    'category': active_cat,
                                    'length_pts': length_pts,
                                    'length_feet': length_feet
                                }
                                
                                if page_key not in st.session_state.user_drawn_lines:
                                    st.session_state.user_drawn_lines[page_key] = []
                                st.session_state.user_drawn_lines[page_key].append(new_line)
                                
                                # Clear pending start
                                st.session_state.pending_line_start[page_key] = None
                                st.rerun(scope="fragment")
                
                else:
                    # SELECT MODE: Use clickable image
                    click_key = f"last_click_{page_num}"
                    if click_key not in st.session_state:
                        st.session_state[click_key] = None
                    
                    click_result = streamlit_image_coordinates(
                        display_img,
                        key=f"click_img_{page_num}"
                    )
                    
                    # Handle click - find nearest line
                    if click_result is not None:
                        current_click = (click_result.get('x', 0), click_result.get('y', 0))
                        
                        if current_click != st.session_state[click_key]:
                            st.session_state[click_key] = current_click
                            click_x, click_y = current_click
                            pdf_click_x = click_x / scale_x
                            pdf_click_y = click_y / scale_y
                            
                            def point_to_line_distance(px, py, x0, y0, x1, y1):
                                dx = x1 - x0
                                dy = y1 - y0
                                if dx == 0 and dy == 0:
                                    return ((px - x0)**2 + (py - y0)**2)**0.5
                                t = max(0, min(1, ((px - x0)*dx + (py - y0)*dy) / (dx*dx + dy*dy)))
                                proj_x = x0 + t * dx
                                proj_y = y0 + t * dy
                                return ((px - proj_x)**2 + (py - proj_y)**2)**0.5
                            
                            min_dist = float('inf')
                            nearest_idx = -1
                            for i, ls in enumerate(line_stats):
                                dist = point_to_line_distance(
                                    pdf_click_x, pdf_click_y,
                                    ls['start'][0], ls['start'][1],
                                    ls['end'][0], ls['end'][1]
                                )
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_idx = i
                            
                            if nearest_idx >= 0 and min_dist < 30:
                                active_cat = st.session_state.active_category_per_page.get(page_key)
                                current_assignment = st.session_state.line_assignments[page_key].get(nearest_idx)
                                
                                if current_assignment == active_cat:
                                    # Only delete if key exists
                                    if nearest_idx in st.session_state.line_assignments[page_key]:
                                        del st.session_state.line_assignments[page_key][nearest_idx]
                                else:
                                    if active_cat:
                                        st.session_state.line_assignments[page_key][nearest_idx] = active_cat
                                st.rerun(scope="fragment")
            
            with col_info:
                st.markdown(f"**{len(lines)} detected lines**")
                if drawing_mode == "select":
                    st.caption("Click to assign to category")
                else:
                    pending = st.session_state.pending_line_start.get(page_key)
                    if pending:
                        st.warning("Click end point")
                        if st.button("Cancel", key=f"cancel_draw_{page_num}"):
                            st.session_state.pending_line_start[page_key] = None
                            st.rerun(scope="fragment")
                    else:
                        st.caption("Click start point")
                
                # Clear buttons
                clear_col1, clear_col2 = st.columns(2)
                with clear_col1:
                    if st.button("Clear Sel", key=f"clear_sel_{page_num}"):
                        st.session_state.line_assignments[page_key] = {}
                        st.rerun(scope="fragment")
                with clear_col2:
                    if st.button("Clear Drawn", key=f"clear_drawn_{page_num}"):
                        st.session_state.user_drawn_lines[page_key] = []
                        st.rerun(scope="fragment")
                
                # Show selected lines (from existing)
                line_assignments = st.session_state.line_assignments.get(page_key, {})
                if line_assignments:
                    by_category = {}
                    for idx, cat in line_assignments.items():
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(idx)
                    
                    st.markdown(f"**Selected: {len(line_assignments)}**")
                    for cat, indices in by_category.items():
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        cat_total = sum(line_stats[i]['length_feet'] for i in indices if i < len(line_stats))
                        st.markdown(f"<span style='color: rgb{color};'>●</span> **{cat}**: {len(indices)} lines, {cat_total:.1f} ft", unsafe_allow_html=True)
                
                # Show user-drawn lines
                user_lines = st.session_state.user_drawn_lines.get(page_key, [])
                if user_lines:
                    st.markdown("---")
                    st.markdown(f"**Drawn: {len(user_lines)}**")
                    # Group by category
                    drawn_by_cat = {}
                    for ul in user_lines:
                        cat = ul.get('category', 'Uncategorized')
                        if cat not in drawn_by_cat:
                            drawn_by_cat[cat] = []
                        drawn_by_cat[cat].append(ul)
                    
                    for cat, cat_lines in drawn_by_cat.items():
                        cat_info = page_categories.get(cat, {})
                        color = cat_info.get('color', (0, 255, 0))
                        cat_total = sum(ul['length_feet'] for ul in cat_lines)
                        st.markdown(f"<span style='color: rgb{color};'>●</span> **{cat}**: {len(cat_lines)} drawn, {cat_total:.1f} ft", unsafe_allow_html=True)
        else:
            st.warning("Image not available")
    
    # Render each page tab using the fragment. BUT: Streamlit tabs
    # run every tab's body on every rerun, and render_page_fragment
    # kicks off vector line detection + image resize + canvas build —
    # that's ~1-4 MB session_state per page. Gate each tab behind a
    # "Load this page" button so a user who only wants Page 3 doesn't
    # pay the cost for pages 1, 2, 4, 5 they never touched. Edits they
    # make on any loaded page persist in session_state regardless.
    def _evict_other_umt_pages(active_pg_num: int):
        """Purge heavy per-page measurement state for EVERY page except
        active_pg_num. Keeps memory to one page's worth of rendered
        image + line stats at a time. The user's actual EDITS
        (line_assignments, user_drawn_lines, page_categories) live in
        per-page dicts and are NOT touched — they persist across page
        switches so the user can come back to a page and see their
        assignments.
        """
        _heavy_prefixes = (
            "base_img_", "base_img_size_", "drawn_img_", "orig_img_size_",
            "line_stats_", "lines_", "auto_synced_", "auto_matched_indices_",
            "click_key_",
        )
        _active_suffix = f"_{active_pg_num}_"
        _active_exact = f"_{active_pg_num}"
        purged = 0
        for k in list(st.session_state.keys()):
            if not any(k.startswith(p) for p in _heavy_prefixes):
                continue
            # These keys embed the page number in their name, e.g.
            # base_img_3_1200, drawn_img_5_1200_<hash>, line_stats_2_30_360.0.
            # Keep only entries tagged with the active page; drop the rest.
            _rest = k.split("_", 2)[-1] if "_" in k else ""
            if f"_{active_pg_num}_" in k or k.endswith(_active_exact):
                continue
            try:
                del st.session_state[k]
                purged += 1
            except Exception:
                pass
        if purged:
            import gc as _gc
            _gc.collect()
            try:
                import ctypes as _ct
                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            print(f"SESSION {current_session_id} LOG: UMT evicted "
                  f"{purged} heavy state keys for non-active pages")

    for tab_idx, (tab, page_data) in enumerate(zip(page_tabs, st.session_state.fence_pages)):
        with tab:
            _umt_pg_num = page_data['page_number']
            _umt_pg_flag = f"_umt_pg_loaded_{_umt_pg_num}"
            if not st.session_state.get(_umt_pg_flag):
                _btn_col, _cap_col = st.columns([1, 3])
                with _btn_col:
                    # on_click callback runs BEFORE the script re-runs,
                    # so by the time `if not st.session_state.get(flag)`
                    # is evaluated again, the flag is already True and
                    # we drop straight into render_page_fragment. That
                    # avoids the previous double-click pattern (where
                    # the body's branch was decided BEFORE the click
                    # handler set the flag, so the click only "showed"
                    # in the next interaction). And because we're still
                    # using a button widget, Streamlit's natural rerun
                    # preserves the active tab — no st.rerun() needed,
                    # so we don't lose the tab the user just clicked
                    # on like the explicit-rerun version did.
                    def _on_load_page(active=_umt_pg_num,
                                      flag=_umt_pg_flag):
                        _evict_other_umt_pages(active)
                        for _k in list(st.session_state.keys()):
                            if _k.startswith("_umt_pg_loaded_") and _k != flag:
                                del st.session_state[_k]
                        st.session_state[flag] = True

                    st.button(f"📏 Load page {_umt_pg_num}",
                              key=f"_umt_pg_btn_{_umt_pg_num}",
                              type="primary",
                              use_container_width=True,
                              on_click=_on_load_page)
                with _cap_col:
                    st.caption(
                        "Click to detect vector lines + render the measurement "
                        "canvas for this page. Opens this page and offloads "
                        "any other currently-loaded page's images from memory. "
                        "Your line assignments persist across page switches."
                    )
            else:
                render_page_fragment(page_data, zoom_level, min_line_pts)
    
    # Overall summary across all pages - grouped by category
    st.markdown("---")
    st.markdown("### 📊 Overall Summary")
    
    # Aggregate by category across all pages (auto + selected + drawn lines)
    category_totals = {}  # {category: {'auto': count, 'lines': count, 'feet': total, 'drawn': count}}
    grand_total_feet = 0
    grand_total_lines = 0
    
    for page_data in st.session_state.fence_pages:
        page_num = page_data['page_number']
        page_key = f"page_{page_num}"
        lines_cache_key = f"lines_{page_num}_{min_line_pts}"
        
        # Get per-page scale
        page_scale_info = st.session_state.per_page_scale_info.get(page_key, {})
        page_scale = page_scale_info.get('verified_scale') or page_scale_info.get('text_scale') or 360.0
        
        # Selected lines from PDF (includes auto-matched + manually selected)
        # Use auto_matched_indices to distinguish auto vs manual
        auto_matched = st.session_state.get(f"auto_matched_indices_{page_key}", set())
        lines = st.session_state.get(lines_cache_key, [])
        line_assignments = st.session_state.line_assignments.get(page_key, {})
        for i, category in line_assignments.items():
            if i < len(lines):
                line = lines[i]
                length_inches = line['length_pts'] / 72.0
                length_feet = (length_inches * page_scale) / 12.0
                
                if category not in category_totals:
                    category_totals[category] = {'auto': 0, 'lines': 0, 'feet': 0, 'drawn': 0}
                if i in auto_matched:
                    category_totals[category]['auto'] += 1
                else:
                    category_totals[category]['lines'] += 1
                category_totals[category]['feet'] += length_feet
                
                grand_total_feet += length_feet
                grand_total_lines += 1
        
        # User-drawn lines
        user_lines = st.session_state.user_drawn_lines.get(page_key, [])
        for ul in user_lines:
            category = ul.get('category', 'Uncategorized')
            length_feet = ul.get('length_feet', 0)
            
            if category not in category_totals:
                category_totals[category] = {'auto': 0, 'lines': 0, 'feet': 0, 'drawn': 0}
            category_totals[category]['drawn'] += 1
            category_totals[category]['feet'] += length_feet
            
            grand_total_feet += length_feet
            grand_total_lines += 1
    
    if grand_total_lines > 0:
        # Show per-category breakdown
        st.markdown("#### By Category")
        for cat, totals in category_totals.items():
            # Find color from any page that has this category
            color = (0, 255, 0)  # default
            for pk, pc in st.session_state.page_categories.items():
                if cat in pc:
                    color = pc[cat].get('color', (0, 255, 0))
                    break
            col_cat, col_lines, col_feet = st.columns([3, 1, 1])
            with col_cat:
                st.markdown(f"<span style='color: rgb{color}; font-size: 18px;'>●</span> **{cat}**", unsafe_allow_html=True)
            with col_lines:
                auto = totals.get('auto', 0)
                selected = totals['lines']
                drawn = totals.get('drawn', 0)
                parts = []
                if auto:
                    parts.append(f"🤖{auto}")
                if selected:
                    parts.append(f"👆{selected}")
                if drawn:
                    parts.append(f"✏️{drawn}")
                st.markdown(", ".join(parts) if parts else "0")
            with col_feet:
                st.metric("Length", f"{totals['feet']:.1f} ft", label_visibility="collapsed")
        
        # Grand total
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Lines", grand_total_lines)
        with col_s2:
            st.metric("**Grand Total**", f"{grand_total_feet:.1f} ft")
        with col_s3:
            pages_with_assign = sum(1 for p in st.session_state.fence_pages 
                               if st.session_state.line_assignments.get(f"page_{p['page_number']}", {}))
            st.metric("Pages", pages_with_assign)
        
        if st.button("🗑️ Clear All Assignments", key="clear_all_selections"):
            st.session_state.line_assignments = {}
    else:
        st.info("Click lines in the page tabs above and assign them to categories to calculate totals.")
    
    # Element Specifications Summary (cross-page details)
    el_details = st.session_state.get('element_details', {})
    if el_details:
        st.markdown("---")
        st.markdown("#### 📋 Element Specifications (Cross-Page Details)")
        spec_rows = []
        for elem_name, details in el_details.items():
            if any(v for v in details.values() if v):
                spec_rows.append({
                    'Element': elem_name,
                    'Height': details.get('height', ''),
                    'Post Type': details.get('post_type', ''),
                    'Post Spacing': details.get('post_spacing', ''),
                    'Material': details.get('material', ''),
                    'Gauge': details.get('gauge', ''),
                    'Mesh Size': details.get('mesh_size', ''),
                    'Foundation': details.get('foundation', ''),
                    'Gate Info': details.get('gate_info', ''),
                    'Detail Page': details.get('detail_page', ''),
                })
        if spec_rows:
            st.dataframe(pd.DataFrame(spec_rows), hide_index=True, use_container_width=True)
            with st.expander("📝 Full Detail Text per Element", expanded=False):
                for elem_name, details in el_details.items():
                    full = details.get('full_details', '')
                    notes = details.get('notes', '')
                    if full or notes:
                        st.markdown(f"**{elem_name}:**")
                        if full:
                            st.markdown(f"  {full}")
                        if notes:
                            st.markdown(f"  *Notes: {notes}*")
    
    # Download section - always show when there are fence pages
    st.markdown("---")
    st.markdown("#### 📥 Downloads")
    
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        # Generate measurement PDF
        pdf_bytes, pdf_name = generate_measurement_pdf(
            _get_pdf_bytes(),
            st.session_state.fence_pages,
            st.session_state.line_assignments,
            st.session_state.user_drawn_lines,
            st.session_state.page_categories,
            st.session_state,
            min_line_pts,
            st.session_state.uploaded_pdf_name
        )
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF with Measurements",
                pdf_bytes,
                pdf_name,
                "application/pdf",
                key="dl_measurement_pdf"
            )
        else:
            st.error("Error generating PDF")
    
    with dl_col2:
        # Generate spreadsheet
        xlsx_data = generate_measurement_spreadsheet(
            st.session_state.fence_pages,
            st.session_state.line_assignments,
            st.session_state.user_drawn_lines,
            st.session_state.page_categories,
            st.session_state,
            st.session_state.per_page_scale_info,
            min_line_pts
        )
        base_name = os.path.splitext(st.session_state.uploaded_pdf_name)[0]
        st.download_button(
            "📊 Download Measurements Excel",
            xlsx_data,
            f"{base_name}_measurements.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_measurement_xlsx"
        )


# ==============================================================================
# Fallback Messages
# ==============================================================================

elif not st.session_state.get('pdf_disk_path'):
    st.info("Upload a PDF to begin analysis.")
elif not (openai_key and llm_analysis_instance):
    st.error("OpenAI models not initialized. Check API key.")
elif not ade_key:
    st.error("LandingAI API key required for ADE analysis.")
elif st.session_state.analysis_halted_due_to_error:
    st.error("Analysis was halted. Upload file again or try a different one.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>ADE Fence Detector App</p>", unsafe_allow_html=True)
