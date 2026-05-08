"""Postgres CRUD for the multi-user web app.

Mirrors the user-facing rows that power the dashboard (documents, jobs,
page_results, artifacts). The legacy SQLite job_registry.py still owns the
in-process worker queue; this module is the persistence layer the frontend
reads from.

Connection: a single psycopg ConnectionPool, lazily initialised.
DATABASE_URL is read from the environment (loaded by api_server.py via
python-dotenv from .env.local).

Every query that touches user data filters by user_id — there is no
"trust the caller" code path. For relations that don't carry user_id
directly (page_results, artifacts), ownership is enforced by joining
through documents.user_id.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = logging.getLogger("backend.db")


_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (expected in .env.local)")
    return url


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=_database_url(),
                    min_size=1,
                    max_size=8,
                    kwargs={"row_factory": dict_row},
                    open=True,
                )
                log.info("Postgres pool opened")
    return _pool


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def insert_document(
    user_id: str,
    original_filename: str,
    storage_path: str,
    total_pages: int | None = None,
    status: str = "uploaded",
) -> str:
    """Create a documents row. Returns the new document id (UUID string)."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            insert into documents
              (user_id, original_filename, storage_path, status, total_pages)
            values (%s, %s, %s, %s, %s)
            returning id
            """,
            (user_id, original_filename, storage_path, status, total_pages),
        ).fetchone()
    return str(row["id"])


def list_documents(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """List a user's documents with the latest job's status joined in."""
    with pool().connection() as conn:
        rows = conn.execute(
            """
            select
              d.id,
              d.original_filename,
              d.storage_path,
              d.status        as document_status,
              d.total_pages,
              d.created_at,
              j.id            as latest_job_id,
              j.status        as job_status,
              j.current_phase,
              j.progress_percent,
              j.error_message
            from documents d
            left join lateral (
              select id, status, current_phase, progress_percent, error_message
              from jobs
              where document_id = d.id
              order by created_at desc
              limit 1
            ) j on true
            where d.user_id = %s
            order by d.created_at desc
            limit %s
            """,
            (user_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_document(document_id: str, user_id: str) -> dict[str, Any] | None:
    """Fetch a single document, ownership-checked."""
    with pool().connection() as conn:
        row = conn.execute(
            "select * from documents where id = %s and user_id = %s",
            (document_id, user_id),
        ).fetchone()
    return _row_to_dict(row) if row else None


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def insert_job(
    document_id: str,
    user_id: str,
    status: str = "queued",
    job_id: str | None = None,
) -> str:
    """Create a jobs row. If job_id is given, override the default UUID
    (used to keep Postgres jobs.id in sync with the legacy SQLite job
    registry).
    """
    with pool().connection() as conn:
        if job_id:
            row = conn.execute(
                """
                insert into jobs (id, document_id, user_id, status)
                values (%s, %s, %s, %s)
                returning id
                """,
                (job_id, document_id, user_id, status),
            ).fetchone()
        else:
            row = conn.execute(
                """
                insert into jobs (document_id, user_id, status)
                values (%s, %s, %s)
                returning id
                """,
                (document_id, user_id, status),
            ).fetchone()
    return str(row["id"])


def insert_document_and_job(
    user_id: str,
    original_filename: str,
    storage_path: str,
    total_pages: int | None,
    job_id: str,
    document_status: str = "uploaded",
    job_status: str = "queued",
) -> tuple[str, str]:
    """Atomic upload mirror: create the document row and its initial job
    row in a single transaction. Returns (document_id, job_id).
    """
    with pool().connection() as conn:
        with conn.transaction():
            doc_row = conn.execute(
                """
                insert into documents
                  (user_id, original_filename, storage_path, status, total_pages)
                values (%s, %s, %s, %s, %s)
                returning id
                """,
                (user_id, original_filename, storage_path, document_status, total_pages),
            ).fetchone()
            doc_id = str(doc_row["id"])
            conn.execute(
                """
                insert into jobs (id, document_id, user_id, status)
                values (%s, %s, %s, %s)
                """,
                (job_id, doc_id, user_id, job_status),
            )
    return (doc_id, job_id)


def update_job_progress(
    job_id: str,
    *,
    status: str | None = None,
    current_phase: str | None = None,
    progress_percent: int | None = None,
    error_message: str | None = None,
    started_at_now: bool = False,
    finished_at_now: bool = False,
) -> None:
    """Patch a jobs row. Only non-None args are written."""
    sets: list[str] = []
    params: list[Any] = []

    if status is not None:
        sets.append("status = %s")
        params.append(status)
    if current_phase is not None:
        sets.append("current_phase = %s")
        params.append(current_phase)
    if progress_percent is not None:
        sets.append("progress_percent = %s")
        params.append(progress_percent)
    if error_message is not None:
        sets.append("error_message = %s")
        params.append(error_message)
    if started_at_now:
        sets.append("started_at = now()")
    if finished_at_now:
        sets.append("finished_at = now()")

    if not sets:
        return

    params.append(job_id)
    with pool().connection() as conn:
        conn.execute(
            f"update jobs set {', '.join(sets)} where id = %s",
            params,
        )


def get_job(job_id: str, user_id: str) -> dict[str, Any] | None:
    """Fetch a single job, ownership-checked."""
    with pool().connection() as conn:
        row = conn.execute(
            "select * from jobs where id = %s and user_id = %s",
            (job_id, user_id),
        ).fetchone()
    return _row_to_dict(row) if row else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert UUIDs / timestamps to JSON-friendly strings."""
    if row is None:
        return None  # type: ignore[return-value]
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = str(v) if k.endswith("_id") or k == "id" else v
    return out


def healthcheck() -> dict[str, Any]:
    """Cheap smoke test for the connection pool."""
    with pool().connection() as conn:
        row = conn.execute(
            "select current_database() as db, current_user as user, now() as now"
        ).fetchone()
    return _row_to_dict(row)
