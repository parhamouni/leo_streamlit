"""Lightweight observability for app_ade_fast.py.

Emits JSONL records into FENCE_TELEMETRY_DIR (default ~/.cache/fence_ade/_telemetry,
one file per day). Three kinds of records:

  - "timed"      — per-call wall / cpu / rss_delta / peak_rss for @timed functions
  - "checkpoint" — phase-boundary snapshot of rss / cpu_percent
  - "event"      — arbitrary structured event (e.g. slot_acquired, slot_rejected)

Designed to be near-zero overhead: one psutil read + one append per call.
Safe to import unconditionally; writes are best-effort (swallowed on error).

Read with `tools/telemetry_report.py` or any `jq`-style tool.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # psutil should always be available in this app, but guard anyway
    psutil = None  # type: ignore
    _PROC = None


def _log_dir() -> Path:
    """Resolve the telemetry directory; create on first call."""
    p = Path(os.environ.get("FENCE_TELEMETRY_DIR",
                            "~/.cache/fence_ade/_telemetry")).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback to /tmp if home isn't writable
        p = Path("/tmp/fence_telemetry")
        p.mkdir(parents=True, exist_ok=True)
    return p


def _today_path() -> Path:
    return _log_dir() / f"{time.strftime('%Y-%m-%d')}.jsonl"


def _emit(kind: str, **fields: Any) -> None:
    """Append one record. Swallows every error — telemetry must not break the app."""
    try:
        rec = {"ts": round(time.time(), 3), "kind": kind, "pid": os.getpid()}
        rec.update(fields)
        line = json.dumps(rec, default=str)
        with open(_today_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _rss_mb() -> float:
    if _PROC is None:
        return 0.0
    try:
        return round(_PROC.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def _cpu_seconds() -> float:
    if _PROC is None:
        return 0.0
    try:
        t = _PROC.cpu_times()
        return t.user + t.system
    except Exception:
        return 0.0


def timed(label: str) -> Callable:
    """Decorator: log wall_s, cpu_s, rss_delta_mb, peak_rss_mb for every call.

    Usage:
        @telemetry.timed("phase3.render_page")
        def render_page(...): ...

    Overhead: ~2 psutil reads + 1 file append per call (microseconds).
    """
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            c0 = _cpu_seconds()
            r0 = _rss_mb()
            peak = r0
            err: str | None = None
            try:
                result = fn(*a, **kw)
                return result
            except Exception as e:
                err = type(e).__name__
                raise
            finally:
                r1 = _rss_mb()
                peak = max(peak, r1)
                _emit("timed",
                      label=label,
                      wall_s=round(time.perf_counter() - t0, 3),
                      cpu_s=round(_cpu_seconds() - c0, 3),
                      rss_delta_mb=round(r1 - r0, 1),
                      peak_rss_mb=peak,
                      err=err)
        return wrap
    return deco


def phase_checkpoint(label: str, session_id: str = "", pdf_sha8: str = "",
                     **extra: Any) -> None:
    """Emit a phase-boundary snapshot. Call at the start/end of each phase."""
    _emit("checkpoint",
          label=label,
          session_id=session_id,
          pdf_sha8=pdf_sha8,
          rss_mb=_rss_mb(),
          cpu_s_total=round(_cpu_seconds(), 3),
          **extra)


def event(name: str, **fields: Any) -> None:
    """Emit an arbitrary structured event (e.g. slot_acquired, slot_rejected)."""
    _emit("event", name=name, rss_mb=_rss_mb(), **fields)
