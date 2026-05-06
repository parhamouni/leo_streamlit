"""SQLite-backed job registry for the Leo Streamlit analysis app.

DB path: ~/.leo/jobs.db  (WAL mode, check_same_thread=False)
Results: ~/.leo/results/<job_id>/

Thread-safety strategy:
  - SQLite opened with WAL journal mode (concurrent reads, serialised writes
    at the DB level).
  - A module-level threading.Lock() wraps every write (belt-and-suspenders).
  - Reads hold the lock too so callers never see a half-committed row.

Multiple imports are safe: the schema is only created once, guarded by
`_db_initialized`.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FENCE_JOB_TTL_SECONDS: int = int(os.environ.get("FENCE_JOB_TTL_SECONDS", 172800))  # 48 h

_LEO_DIR = Path("~/.leo").expanduser()
_DB_PATH = _LEO_DIR / "jobs.db"
_RESULTS_ROOT = _LEO_DIR / "results"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_db_initialized: bool = False
_conn: sqlite3.Connection | None = None  # shared, thread-safe via WAL + _lock

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    filename        TEXT NOT NULL,
    pdf_hash        TEXT,
    pdf_path        TEXT,
    status          TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    started_at      INTEGER,
    completed_at    INTEGER,
    expires_at      INTEGER,
    total_pages     INTEGER,
    fence_count     INTEGER,
    non_fence_count INTEGER,
    results_dir     TEXT,
    error_msg       TEXT,
    config_json     TEXT,
    queue_position  INTEGER
);
"""

_MIGRATE_COLUMNS = [
    ("config_json", "TEXT"),
    ("queue_position", "INTEGER"),
]

# Statuses considered "active" (never expire from get_user_jobs).
_ACTIVE_STATUSES = ("queued", "running", "phases_ready")


def _db() -> sqlite3.Connection:
    """Return the shared connection, initialising it on first call."""
    global _db_initialized, _conn

    if _db_initialized and _conn is not None:
        return _conn

    with _lock:
        # Re-check inside the lock (another thread may have raced us here).
        if _db_initialized and _conn is not None:
            return _conn

        _LEO_DIR.mkdir(parents=True, exist_ok=True)
        _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(_DB_PATH),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        conn.row_factory = sqlite3.Row

        # WAL mode: readers don't block writers; writers don't block readers.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(_CREATE_TABLE_SQL)

        for col_name, col_type in _MIGRATE_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

        _conn = conn
        _db_initialized = True
        return _conn


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Convert a sqlite3.Row to a plain dict, or return None."""
    if row is None:
        return None
    return dict(row)


def _placeholders(fields: dict) -> tuple[str, list]:
    """Build 'col = ?, col = ?, ...' fragment and the values list."""
    assignments = ", ".join(f"{col} = ?" for col in fields)
    return assignments, list(fields.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_job(
    user_id: str,
    filename: str,
    pdf_hash: str | None = None,
    pdf_path: str | None = None,
    config_json: str | None = None,
) -> str:
    """Create a new job row and its results directory.

    Returns the new job_id (UUID4 string).
    """
    job_id = str(uuid.uuid4())
    now = int(time.time())
    expires_at = now + FENCE_JOB_TTL_SECONDS

    results_dir = _RESULTS_ROOT / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    db = _db()
    with _lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            max_pos = db.execute(
                "SELECT COALESCE(MAX(queue_position), 0) FROM jobs WHERE status = 'queued'"
            ).fetchone()[0]
            queue_position = max_pos + 1

            row = {
                "job_id": job_id,
                "user_id": user_id,
                "filename": filename,
                "pdf_hash": pdf_hash,
                "pdf_path": pdf_path,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "expires_at": expires_at,
                "total_pages": None,
                "fence_count": None,
                "non_fence_count": None,
                "results_dir": str(results_dir),
                "error_msg": None,
                "config_json": config_json,
                "queue_position": queue_position,
            }

            cols = ", ".join(row.keys())
            placeholders = ", ".join("?" for _ in row)
            sql = f"INSERT INTO jobs ({cols}) VALUES ({placeholders})"
            db.execute(sql, list(row.values()))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise

    return job_id


def delete_job(job_id: str) -> None:
    """Hard-delete a job row from the registry. Silent if not found."""
    db = _db()
    with _lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise


def update_job(job_id: str, **fields: Any) -> None:
    """Update arbitrary columns on an existing job.

    Silently does nothing if the job_id is not found.
    Fields that map to column names are accepted; unknown keys raise ValueError
    so callers catch typos early.
    """
    if not fields:
        return

    _KNOWN_COLUMNS = {
        "user_id", "filename", "pdf_hash", "pdf_path", "status",
        "created_at", "started_at", "completed_at", "expires_at",
        "total_pages", "fence_count", "non_fence_count", "results_dir",
        "error_msg", "config_json", "queue_position",
    }
    unknown = set(fields) - _KNOWN_COLUMNS
    if unknown:
        raise ValueError(f"update_job: unknown columns: {unknown!r}")

    assignments, values = _placeholders(fields)
    sql = f"UPDATE jobs SET {assignments} WHERE job_id = ?"
    values.append(job_id)

    db = _db()
    with _lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(sql, values)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise


def get_job(job_id: str) -> dict | None:
    """Fetch a single job by ID. Returns None if not found."""
    db = _db()
    with _lock:
        row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def get_user_jobs(user_id: str) -> list[dict]:
    """Return active + recently completed jobs for a user, not expired.

    Includes rows where:
      - expires_at > now()   (fresh enough)
      - OR status is queued / running / phases_ready (always shown)

    Ordered by created_at DESC.
    """
    now = int(time.time())
    active_in = ", ".join(f"'{s}'" for s in _ACTIVE_STATUSES)
    sql = f"""
        SELECT * FROM jobs
        WHERE user_id = ?
          AND (expires_at > ? OR status IN ({active_in}))
        ORDER BY created_at DESC
    """
    db = _db()
    with _lock:
        rows = db.execute(sql, (user_id, now)).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_active_user_jobs(user_id: str) -> list[dict]:
    """Return only queued + running jobs for a user."""
    active_in = ", ".join(f"'{s}'" for s in ("queued", "running"))
    sql = f"""
        SELECT * FROM jobs
        WHERE user_id = ?
          AND status IN ({active_in})
        ORDER BY created_at DESC
    """
    db = _db()
    with _lock:
        rows = db.execute(sql, (user_id,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_active_user_jobs(user_id: str) -> int:
    """Return the count of queued + running jobs for a user."""
    active_in = ", ".join(f"'{s}'" for s in ("queued", "running"))
    sql = f"""
        SELECT COUNT(*) FROM jobs
        WHERE user_id = ?
          AND status IN ({active_in})
    """
    db = _db()
    with _lock:
        (count,) = db.execute(sql, (user_id,)).fetchone()
    return count


def cleanup_expired_jobs() -> int:
    """Delete DB rows where expires_at < now() and remove their results dirs.

    Returns the number of rows deleted.
    """
    now = int(time.time())
    db = _db()

    with _lock:
        rows = db.execute(
            "SELECT job_id, results_dir FROM jobs WHERE expires_at < ?", (now,)
        ).fetchall()

        if not rows:
            return 0

        ids = [r["job_id"] for r in rows]
        dirs = [r["results_dir"] for r in rows if r["results_dir"]]

        placeholders = ", ".join("?" for _ in ids)
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(f"DELETE FROM jobs WHERE job_id IN ({placeholders})", ids)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise

    # Remove result directories outside the lock (slow I/O, no DB state at stake).
    for d in dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    return len(ids)


def write_progress(
    job_id: str,
    phase: str,
    pct: int,
    message: str,
    **extra: Any,
) -> None:
    """Atomically write progress.json for a job.

    Writes to <results_dir>/progress.json via a .tmp file + rename so readers
    never see a partially-written file.
    """
    job = get_job(job_id)
    if job is None or not job.get("results_dir"):
        # Best-effort: fall back to the canonical path.
        results_dir = _RESULTS_ROOT / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path(job["results_dir"])

    payload: dict[str, Any] = {
        "phase": phase,
        "pct": pct,
        "message": message,
        "updated_at": int(time.time()),
        **extra,
    }

    target = results_dir / "progress.json"
    tmp = results_dir / "progress.json.tmp"

    encoded = json.dumps(payload, default=str).encode("utf-8")
    tmp.write_bytes(encoded)
    os.replace(str(tmp), str(target))


def read_progress(job_id: str) -> dict | None:
    """Read progress.json for a job. Returns None if missing or corrupt."""
    job = get_job(job_id)
    if job is not None and job.get("results_dir"):
        progress_path = Path(job["results_dir"]) / "progress.json"
    else:
        progress_path = _RESULTS_ROOT / job_id / "progress.json"

    try:
        if not progress_path.exists():
            return None
        return json.loads(progress_path.read_bytes())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Queue management (used by api_server.py background workers)
# ---------------------------------------------------------------------------


def next_queued_job() -> dict | None:
    """Atomically claim the oldest queued job (lowest queue_position).

    Sets status='running' and started_at=now(). Returns the job dict,
    or None if the queue is empty.
    """
    db = _db()
    with _lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute(
                "SELECT * FROM jobs WHERE status = 'queued' "
                "ORDER BY queue_position ASC, created_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                db.execute("ROLLBACK")
                return None
            job = dict(row)
            now = int(time.time())
            db.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (now, job["job_id"]),
            )
            db.execute("COMMIT")
            job["status"] = "running"
            job["started_at"] = now
            return job
        except Exception:
            db.execute("ROLLBACK")
            raise


def count_running_jobs() -> int:
    """Return the total number of currently running jobs."""
    db = _db()
    with _lock:
        (count,) = db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'running'"
        ).fetchone()
    return count


def count_queued_jobs() -> int:
    """Return the total number of queued jobs."""
    db = _db()
    with _lock:
        (count,) = db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'queued'"
        ).fetchone()
    return count


def queue_position(job_id: str) -> int | None:
    """Return 1-based queue position for a queued job, or None if not queued."""
    db = _db()
    with _lock:
        rows = db.execute(
            "SELECT job_id FROM jobs WHERE status = 'queued' "
            "ORDER BY queue_position ASC, created_at ASC"
        ).fetchall()
    for i, r in enumerate(rows, 1):
        if r["job_id"] == job_id:
            return i
    return None


def mark_stale_running_as_failed(max_age_seconds: int = 7200) -> int:
    """Mark jobs stuck in 'running' status as 'failed'.

    Called on API server startup to recover from crashes. Jobs running
    longer than max_age_seconds are considered stale.
    """
    cutoff = int(time.time()) - max_age_seconds
    db = _db()
    with _lock:
        db.execute("BEGIN IMMEDIATE")
        try:
            cursor = db.execute(
                "UPDATE jobs SET status = 'failed', "
                "error_msg = 'Server restarted during analysis', "
                "completed_at = ? "
                "WHERE status = 'running' AND started_at < ?",
                (int(time.time()), cutoff),
            )
            count = cursor.rowcount
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    return count


def save_results(job_id: str, results: dict) -> None:
    """Save the analysis results JSON to the job's results directory."""
    job = get_job(job_id)
    if job is None:
        return
    results_dir = Path(job.get("results_dir") or str(_RESULTS_ROOT / job_id))
    results_dir.mkdir(parents=True, exist_ok=True)

    target = results_dir / "results.json"
    tmp = results_dir / "results.json.tmp"
    encoded = json.dumps(results, default=str).encode("utf-8")
    tmp.write_bytes(encoded)
    os.replace(str(tmp), str(target))


def load_results(job_id: str) -> dict | None:
    """Load results.json for a completed job. Returns None if missing."""
    job = get_job(job_id)
    if job is None:
        return None
    results_dir = Path(job.get("results_dir") or str(_RESULTS_ROOT / job_id))
    results_path = results_dir / "results.json"
    try:
        if not results_path.exists():
            return None
        return json.loads(results_path.read_bytes())
    except Exception:
        return None


def get_highlighted_pdf_path(job_id: str) -> Path | None:
    """Return the path to the highlighted PDF if it exists."""
    job = get_job(job_id)
    if job is None or not job.get("results_dir"):
        return None
    hl_path = Path(job["results_dir"]) / "highlighted.pdf"
    return hl_path if hl_path.exists() else None
