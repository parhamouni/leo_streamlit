"""Per-user spend tracking stub for app_ade_fast.py.

v1 goal: collect the data. Enforcement (block expensive users, auto-downgrade
to a cheaper model, etc.) is deferred to v2.

Records are appended as JSONL under ~/.cache/fence_ade/_spend/<user_id>.jsonl.
A future cron or dashboard consumes these files; nothing in the app path
reads them yet.

Writes are best-effort — any exception is swallowed so telemetry problems
don't break analysis.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _spend_dir() -> Path:
    p = Path(os.environ.get("FENCE_SPEND_DIR",
                            "~/.cache/fence_ade/_spend")).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        p = Path("/tmp/fence_spend")
        p.mkdir(parents=True, exist_ok=True)
    return p


def record_spend(user_id: str, provider: str,
                 tokens_in: int | None = None,
                 tokens_out: int | None = None,
                 cost_usd: float | None = None,
                 **extra: Any) -> None:
    """Append one spend record for a user. All numeric fields optional so
    callers can pass whatever the provider actually returned."""
    if not user_id:
        return
    try:
        safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
        if not safe:
            safe = "anon"
        rec = {
            "ts": round(time.time(), 3),
            "user_id": user_id,
            "provider": provider,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
        }
        if extra:
            rec.update(extra)
        path = _spend_dir() / f"{safe}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        # spend tracking must never break analysis
        pass


def check_spend_quota(user_id: str) -> tuple[bool, str]:
    """Stub — always allows. Wired up in v2 when per-user spend caps exist."""
    return (True, "")


def user_spend_today(user_id: str) -> dict:
    """Sum today's records for a user. Cheap enough to call on demand
    from a sidebar widget."""
    safe = "".join(c for c in (user_id or "") if c.isalnum() or c in "-_") or "anon"
    path = _spend_dir() / f"{safe}.jsonl"
    if not path.exists():
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    midnight = time.time() - 86400  # rolling 24 h
    calls = 0
    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if float(r.get("ts", 0)) < midnight:
                    continue
                calls += 1
                tokens_in += int(r.get("tokens_in") or 0)
                tokens_out += int(r.get("tokens_out") or 0)
                cost += float(r.get("cost_usd") or 0.0)
    except Exception:
        pass
    return {"calls": calls, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "cost_usd": round(cost, 4)}
