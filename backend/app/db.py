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

from psycopg import Connection
from psycopg.pq import TransactionStatus
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


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _session_options() -> str:
    """Server-side per-connection timeouts, applied at connection startup via
    libpq's `options`. The pool's `timeout=5` only bounds how long we wait to
    *acquire* a connection; these bound what happens once we have one, so a
    query that blocks on a row lock or a slow scan can't wedge a request (and
    its pooled connection) indefinitely.

      * statement_timeout — cap any single query (default 30s).
      * lock_timeout — fail fast instead of queueing behind a held row lock,
        e.g. a delete racing a concurrent per-page upsert (default 10s).
      * idle_in_transaction_session_timeout — reap a transaction left open by
        a crashed/stuck caller so it stops holding locks (default 60s).

    All overridable via env for incident response without a redeploy. Set any
    to 0 to disable that individual timeout (Postgres treats 0 as "no limit").
    """
    statement_ms = _int_env("FENCE_DB_STATEMENT_TIMEOUT_MS", 30_000)
    lock_ms = _int_env("FENCE_DB_LOCK_TIMEOUT_MS", 10_000)
    idle_tx_ms = _int_env("FENCE_DB_IDLE_TX_TIMEOUT_MS", 60_000)
    return (
        f"-c statement_timeout={statement_ms} "
        f"-c lock_timeout={lock_ms} "
        f"-c idle_in_transaction_session_timeout={idle_tx_ms}"
    )


class _LeakSafePool(ConnectionPool):
    """ConnectionPool that closes the underlying socket of broken
    connections instead of silently dropping them.

    Background: psycopg-pool's _return_connection treats a connection
    in TransactionStatus.UNKNOWN as already-closed and just schedules
    a replacement. It does NOT call conn.close() on the broken one.
    libpq's TCP socket can stay ESTABLISHED for hours after the
    protocol layer goes UNKNOWN — and Supabase's session-mode pooler
    counts every ESTABLISHED socket against its 15-session per-project
    limit, so each leaked conn permanently consumes one slot until the
    process restarts.

    We catch the UNKNOWN-on-putconn case here and call close() ourselves
    before delegating to the parent. close() runs PQfinish which sends
    a Terminate (best-effort) and closes the file descriptor — that's
    what tells Supabase's pooler to free the session slot.

    The check-callback failure path is handled separately in
    _check_connection (which raises after closing).
    """

    def putconn(self, conn: Connection) -> None:  # type: ignore[override]
        try:
            if conn.pgconn.transaction_status == TransactionStatus.UNKNOWN:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass
        super().putconn(conn)


def _check_connection(conn: Connection) -> None:
    """psycopg-pool `check` callback. Run a trivial round-trip before
    handing a pooled connection back to the caller, so we don't reuse a
    corrupted/desynced one. Stale connections raise here, the pool drops
    them, and the next `getconn()` opens a fresh one.

    This is the fix for the long-running-job hang we saw in production:
    a Supabase pooler statement_timeout cancel left the connection's
    application-layer state out of sync (54 bytes in recv-q, never read),
    and the next user of the connection blocked forever waiting for a
    response that wasn't coming.

    On failure we explicitly call conn.close() before re-raising. This
    looks redundant — the pool catches the exception and discards the
    conn — but psycopg-pool's discard path for `from_getconn=True` +
    UNKNOWN-status connections (pool.py _return_connection lines 713-
    719) schedules a replacement task but never calls conn.close() on
    the failed connection. With Supabase's session-mode pooler the
    underlying TCP socket can stay ESTABLISHED for hours after the
    libpq-level state goes bad, so each check failure leaks one socket
    against the pooler's per-project session cap (15 on free tier). At
    16 leaks the whole project goes "max clients reached" and every
    new query times out. Closing the conn here forces the kernel to
    actually FIN the socket so it stops counting against the cap.
    """
    try:
        conn.execute("select 1").fetchone()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _LeakSafePool(
                    conninfo=_database_url(),
                    # Pool was exhausting at max_size=8 under realistic
                    # load: 1 pipeline worker doing rapid per-page
                    # upserts during Phase 3 + 3 Vercel edge IPs polling
                    # /api/documents and /pages every 3 s. Each conn
                    # round-trip also runs the `check` callback's
                    # SELECT 1, adding latency. Bumped to 20 — Supabase
                    # session-mode pooler allows up to 60 connections
                    # on free tier, plenty of headroom.
                    min_size=2,
                    max_size=20,
                    # `options` ships server-side statement/lock/idle-tx
                    # timeouts so a blocked query can't pin a connection
                    # forever — see _session_options().
                    kwargs={"row_factory": dict_row, "options": _session_options()},
                    # Run `check` on each getconn() to detect stale conns.
                    check=_check_connection,
                    # Recycle connections every 15 min so slow drift in
                    # the upstream pooler doesn't accumulate, AND so the
                    # check doesn't repeatedly re-validate the same
                    # already-flaky socket.
                    max_lifetime=900,
                    # If a connection sits idle 5 min, close it. Reduces
                    # the chance of NAT-timed-out sockets piling up in
                    # the pool.
                    max_idle=300,
                    # When all conns are busy, fail fast at 5 s instead
                    # of waiting 30 s — clients can retry sooner and we
                    # spot the issue in logs immediately.
                    timeout=5,
                    open=True,
                )
                log.info("Postgres pool opened (max_size=20, check-on-getconn)")
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
              j.error_message,
              j.started_at    as job_started_at,
              j.phase_started_at
            from documents d
            left join lateral (
              select id, status, current_phase, progress_percent, error_message,
                     started_at, phase_started_at
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
    """Fetch a single document with the latest job's status joined in,
    ownership-checked. Same row shape as list_documents() entries."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            select
              d.id,
              d.user_id,
              d.original_filename,
              d.storage_path,
              d.status        as document_status,
              d.total_pages,
              d.created_at,
              j.id            as latest_job_id,
              j.status        as job_status,
              j.current_phase,
              j.progress_percent,
              j.error_message,
              j.started_at    as job_started_at,
              j.phase_started_at
            from documents d
            left join lateral (
              select id, status, current_phase, progress_percent, error_message,
                     started_at, phase_started_at
              from jobs
              where document_id = d.id
              order by created_at desc
              limit 1
            ) j on true
            where d.id = %s and d.user_id = %s
            """,
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
    pdf_hash: str | None = None,
    document_status: str = "uploaded",
    job_status: str = "queued",
    document_id: str | None = None,
) -> tuple[str, str]:
    """Atomic upload mirror: create the document row and its initial job
    row in a single transaction. Returns (document_id, job_id).

    If `document_id` is provided, it is used as the documents.id; otherwise
    the DB default (gen_random_uuid()) generates one. The pre-allocated
    form lets the upload endpoint return the document_id to the client
    before the row is actually persisted (the persistence happens in a
    background task — see api_server.create_job).
    """
    with pool().connection() as conn:
        with conn.transaction():
            if document_id is not None:
                doc_row = conn.execute(
                    """
                    insert into documents
                      (id, user_id, original_filename, storage_path, status, total_pages, pdf_hash)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    returning id
                    """,
                    (document_id, user_id, original_filename, storage_path,
                     document_status, total_pages, pdf_hash),
                ).fetchone()
            else:
                doc_row = conn.execute(
                    """
                    insert into documents
                      (user_id, original_filename, storage_path, status, total_pages, pdf_hash)
                    values (%s, %s, %s, %s, %s, %s)
                    returning id
                    """,
                    (user_id, original_filename, storage_path, document_status, total_pages, pdf_hash),
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


def find_document_by_hash(user_id: str, pdf_hash: str) -> dict[str, Any] | None:
    """Return the most recent document for this user that matches the
    given pdf_hash, with the latest job's status joined in. Used for
    upload deduplication."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            select
              d.id,
              d.original_filename,
              d.status        as document_status,
              d.total_pages,
              d.created_at,
              d.pdf_hash,
              j.id            as latest_job_id,
              j.status        as job_status,
              j.current_phase,
              j.progress_percent,
              j.error_message,
              j.started_at    as job_started_at,
              j.phase_started_at
            from documents d
            left join lateral (
              select id, status, current_phase, progress_percent, error_message,
                     started_at, phase_started_at
              from jobs
              where document_id = d.id
              order by created_at desc
              limit 1
            ) j on true
            where d.user_id = %s and d.pdf_hash = %s
            order by d.created_at desc
            limit 1
            """,
            (user_id, pdf_hash),
        ).fetchone()
    return _row_to_dict(row) if row else None


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
        # Bump phase_started_at to now() iff the phase actually changes.
        # The CASE compares the OLD row's current_phase (pre-SET semantics
        # in PG) against the new value — same %s param used twice via the
        # two placeholders below.
        sets.append(
            "phase_started_at = case when current_phase is distinct from %s "
            "then now() else phase_started_at end"
        )
        params.append(current_phase)
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


def get_document_id_by_job(job_id: str) -> str | None:
    """Look up the document_id a job belongs to. Returns None for legacy
    `X-User-Id` jobs that never landed in Postgres."""
    with pool().connection() as conn:
        row = conn.execute(
            "select document_id from jobs where id = %s",
            (job_id,),
        ).fetchone()
    return str(row["document_id"]) if row else None


def list_document_jobs(
    document_id: str, user_id: str, only_completed: bool = True
) -> list[dict[str, Any]]:
    """List a document's jobs (ownership enforced via documents.user_id),
    newest first. Used to enumerate which modes a document has been analysed
    in — each job's trade is read from its saved results."""
    status_clause = "and j.status = 'completed'" if only_completed else ""
    with pool().connection() as conn:
        rows = conn.execute(
            f"""
            select j.id, j.status, j.created_at, j.finished_at
            from jobs j
            join documents d on d.id = j.document_id
            where j.document_id = %s and d.user_id = %s {status_clause}
            order by j.created_at desc
            """,
            (document_id, user_id),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# page_results — live per-page rows the worker upserts as the pipeline runs
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Make a pipeline result_json acceptable to Postgres JSONB.

    Two fixes:
      * Strip NUL (\\u0000) from any text — Postgres TEXT/JSONB rejects it
        outright (errors as 'unsupported Unicode escape sequence'). Some
        PDFs have null bytes embedded in their extracted page text.
      * Coerce non-JSON-serializable types (e.g. pipeline's VectorLine
        dataclass) by falling back to vars() / str() recursively.
    """
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        # Strip NUL — Postgres TEXT/JSONB cannot store \x00.
        return obj.replace("\x00", "") if "\x00" in obj else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # Custom dataclass / object: prefer __dict__ if available, else stringify.
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None


def upsert_page_result(
    document_id: str,
    page_number: int,
    is_fence_page: bool,
    result_json: dict | None,
) -> None:
    """Insert or update a per-page row for a document. Called from the
    pipeline `page_cb` so the frontend sees pages stream in.

    Phase 1c emits a stub; Phase 3 overwrites it with the enriched payload
    via ON CONFLICT — only fence pages get a second emission."""
    from psycopg.types.json import Json
    cleaned = _json_safe(result_json) if result_json is not None else None
    payload = Json(cleaned) if cleaned is not None else None
    with pool().connection() as conn:
        conn.execute(
            """
            insert into page_results
              (document_id, page_number, is_fence_page, result_json)
            values (%s, %s, %s, %s)
            on conflict (document_id, page_number) do update set
              is_fence_page = excluded.is_fence_page,
              result_json   = excluded.result_json
            """,
            (document_id, page_number, is_fence_page, payload),
        )


def list_page_results(document_id: str, user_id: str) -> list[dict[str, Any]]:
    """Return all page_results rows for a document the user owns. The
    join through documents enforces ownership — non-owners see []."""
    with pool().connection() as conn:
        rows = conn.execute(
            """
            select pr.page_number, pr.is_fence_page, pr.result_json,
                   pr.created_at
            from page_results pr
            join documents d on d.id = pr.document_id
            where pr.document_id = %s and d.user_id = %s
            order by pr.page_number asc
            """,
            (document_id, user_id),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


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
