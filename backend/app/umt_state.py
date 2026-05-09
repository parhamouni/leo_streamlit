"""Per-job UMT (Unified Measurement Tool) state persistence.

Stores the user's manual measurement edits — category palette, per-line
assignments, hand-drawn lines, scale overrides — co-located with the
job's results.json so it lifecycles with the job (TTL cleanup, deletion).

Single-editor assumption: no concurrent-write coordination. Atomic
write-then-rename keeps a half-written file from being read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import job_registry

_FILENAME = "umt_state.json"

MAX_CATEGORIES_PER_PAGE = 50
MAX_USER_LINES_PER_PAGE = 2000


def _state_path(job_id: str) -> Path | None:
    job = job_registry.get_job(job_id)
    if job is None:
        return None
    results_dir = job.get("results_dir")
    if not results_dir:
        return None
    return Path(results_dir) / _FILENAME


def empty_state() -> dict[str, Any]:
    return {"version": 1, "pages": {}}


def load(job_id: str) -> dict[str, Any]:
    """Return the persisted state, or an empty skeleton if none exists."""
    path = _state_path(job_id)
    if path is None or not path.exists():
        return empty_state()
    try:
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict) or "pages" not in data:
            return empty_state()
        return data
    except Exception:
        return empty_state()


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    os.replace(str(tmp), str(path))


def save(job_id: str, state: dict[str, Any]) -> None:
    path = _state_path(job_id)
    if path is None:
        raise FileNotFoundError(f"job {job_id} has no results_dir")
    _atomic_write(path, state)


def validate_page_state(page_state: Any) -> dict[str, Any]:
    """Validate and normalise an inbound page-state payload.

    Raises ValueError on invalid input. Returns a sanitised dict.
    """
    if not isinstance(page_state, dict):
        raise ValueError("page_state must be an object")

    out: dict[str, Any] = {}

    cats = page_state.get("categories") or {}
    if not isinstance(cats, dict):
        raise ValueError("categories must be an object")
    if len(cats) > MAX_CATEGORIES_PER_PAGE:
        raise ValueError(
            f"too many categories (max {MAX_CATEGORIES_PER_PAGE})"
        )
    norm_cats: dict[str, dict[str, Any]] = {}
    for name, info in cats.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(info, dict):
            continue
        color = info.get("color")
        if (not isinstance(color, (list, tuple))
                or len(color) != 3
                or not all(isinstance(c, (int, float)) and 0 <= c <= 255 for c in color)):
            color = [0, 255, 0]
        norm_cats[name.strip()] = {
            "indicator": str(info.get("indicator") or "").strip(),
            "keyword": str(info.get("keyword") or name).strip(),
            "color": [int(color[0]), int(color[1]), int(color[2])],
        }
    out["categories"] = norm_cats

    assignments = page_state.get("line_assignments") or {}
    if not isinstance(assignments, dict):
        raise ValueError("line_assignments must be an object")
    norm_assignments: dict[str, str] = {}
    for k, v in assignments.items():
        try:
            int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        norm_assignments[str(int(k))] = v.strip()
    out["line_assignments"] = norm_assignments

    drawn = page_state.get("user_drawn_lines") or []
    if not isinstance(drawn, list):
        raise ValueError("user_drawn_lines must be an array")
    if len(drawn) > MAX_USER_LINES_PER_PAGE:
        raise ValueError(
            f"too many user-drawn lines (max {MAX_USER_LINES_PER_PAGE})"
        )
    norm_drawn = []
    for ln in drawn:
        if not isinstance(ln, dict):
            continue
        start = ln.get("start")
        end = ln.get("end")
        if not (isinstance(start, (list, tuple)) and len(start) == 2
                and isinstance(end, (list, tuple)) and len(end) == 2):
            continue
        try:
            sx, sy = float(start[0]), float(start[1])
            ex, ey = float(end[0]), float(end[1])
        except (TypeError, ValueError):
            continue
        cat = ln.get("category")
        if not isinstance(cat, str) or not cat.strip():
            continue
        norm_drawn.append({
            "start": [sx, sy],
            "end": [ex, ey],
            "category": cat.strip(),
        })
    out["user_drawn_lines"] = norm_drawn

    scale = page_state.get("scale_override")
    if scale is not None:
        try:
            scale_f = float(scale)
            if scale_f > 0:
                out["scale_override"] = scale_f
        except (TypeError, ValueError):
            pass

    min_len = page_state.get("min_line_pts")
    if min_len is not None:
        try:
            min_len_f = float(min_len)
            if min_len_f >= 0:
                out["min_line_pts"] = min_len_f
        except (TypeError, ValueError):
            pass

    return out


def update_page(job_id: str, page_num: int, page_state: dict[str, Any]) -> dict[str, Any]:
    """Validate + upsert one page's state. Returns the updated full state."""
    page_key = f"page_{int(page_num)}"
    sanitised = validate_page_state(page_state)
    state = load(job_id)
    state.setdefault("pages", {})[page_key] = sanitised
    save(job_id, state)
    return state


def delete_page(job_id: str, page_num: int) -> dict[str, Any]:
    """Remove edits for a single page. Returns the updated full state."""
    page_key = f"page_{int(page_num)}"
    state = load(job_id)
    pages = state.setdefault("pages", {})
    if page_key in pages:
        del pages[page_key]
        save(job_id, state)
    return state
