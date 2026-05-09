"""Connection-leak regression tests for backend.app.db.

The bug we're protecting against: psycopg-pool drops broken connections
without closing their OS socket. Each leak counts permanently against
Supabase's 15-session per-project limit, eventually wedging the whole
service. The fixes — close-on-check-fail in `_check_connection` and a
`_LeakSafePool.putconn` override — must always release the socket
before letting the pool do its bookkeeping. Hence these tests.

These tests don't talk to a real Postgres; they drive fakes through
the same code paths so we can assert close() was called.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# db.py refuses to import without DATABASE_URL; we don't actually
# connect, but the module-level pool() call would fail otherwise.
os.environ.setdefault("DATABASE_URL", "postgresql://stub/stub")

# Ensure the repo root is on sys.path for the backend import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg.pq import TransactionStatus  # noqa: E402

from backend.app import db as dbmod  # noqa: E402


def _fake_conn(status: TransactionStatus) -> MagicMock:
    """Build a fake psycopg.Connection that reports the given pgconn
    transaction_status, with a tracked .close() and an .execute() that
    can be set to raise."""
    conn = MagicMock(name="FakeConn")
    conn.pgconn = SimpleNamespace(transaction_status=status)
    conn.close = MagicMock()
    return conn


def test_check_connection_passes_when_select_succeeds():
    """Healthy connection: SELECT 1 returns; close() is NOT called."""
    conn = _fake_conn(TransactionStatus.IDLE)
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    conn.execute.return_value = cursor

    dbmod._check_connection(conn)  # should not raise

    conn.execute.assert_called_once_with("select 1")
    conn.close.assert_not_called()


def test_check_connection_closes_socket_before_raising():
    """Stale connection: SELECT 1 raises. We must close the socket
    BEFORE re-raising so the pool's discard path doesn't leak the
    underlying TCP connection."""
    conn = _fake_conn(TransactionStatus.UNKNOWN)
    conn.execute.side_effect = RuntimeError("server gone")

    with pytest.raises(RuntimeError, match="server gone"):
        dbmod._check_connection(conn)

    conn.close.assert_called_once()


def test_check_connection_swallows_close_failure():
    """If the conn is so broken that close() itself raises, we still
    propagate the original check error rather than masking it with a
    secondary exception from cleanup."""
    conn = _fake_conn(TransactionStatus.UNKNOWN)
    conn.execute.side_effect = RuntimeError("primary failure")
    conn.close.side_effect = OSError("close failed too")

    with pytest.raises(RuntimeError, match="primary failure"):
        dbmod._check_connection(conn)

    conn.close.assert_called_once()


def test_leak_safe_pool_closes_unknown_on_putconn(monkeypatch):
    """A connection returned to the pool with UNKNOWN status (broke
    mid-query) must have close() called on it before the parent's
    putconn runs — otherwise the parent silently drops it without
    FIN'ing the socket."""

    parent_putconn_calls: list[object] = []

    def fake_super_putconn(self, conn):
        parent_putconn_calls.append(conn)

    # Stub out the parent class's putconn so we don't need a real pool.
    monkeypatch.setattr(
        dbmod.ConnectionPool, "putconn", fake_super_putconn, raising=True,
    )
    # Build an instance without going through __init__ (which would try
    # to connect to Postgres).
    pool = dbmod._LeakSafePool.__new__(dbmod._LeakSafePool)

    # UNKNOWN conn → must be closed before parent putconn is called.
    bad = _fake_conn(TransactionStatus.UNKNOWN)
    pool.putconn(bad)
    bad.close.assert_called_once()
    assert parent_putconn_calls == [bad]

    # Healthy conn → close() must NOT be called; the pool can recycle it.
    good = _fake_conn(TransactionStatus.IDLE)
    pool.putconn(good)
    good.close.assert_not_called()
    assert parent_putconn_calls == [bad, good]
