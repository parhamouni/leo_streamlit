"""Persistent disk cache for fence-analysis intermediate results.

Keyed by (pdf_sha256, phase, params_hash, [page_idx]). Stored at a durable
path (default ~/.cache/fence_ade) so it survives the fence.service 2-hour
watchdog restart — the whole point of this module.

Design:
  - `put` / `get` on a per-page granularity where possible, so partial
    progress survives a mid-run crash.
  - `probe` tells the caller which phases are fully cached for a given PDF,
    so the app can short-circuit straight to render on a re-upload.
  - Atomic writes via `*.tmp` + `os.replace`. Two sessions writing the same
    key is safe (last-writer-wins, content is idempotent).
  - JSON for small structured results, pickle (protocol 5) for chunk
    objects that aren't cleanly JSON-round-trippable.
  - `CACHE_SCHEMA_VERSION` is baked into every params_hash — bump it to
    invalidate every entry in one line.

Call sites live in app_ade_fast.py only; this module is not imported by
production app_ade.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional

CACHE_SCHEMA_VERSION = "v4"  # v4: pickle removed, all phases use JSON

# Phases that cache per-page. `phase1a` is whole-document (text + vector
# lines come out of one subprocess call), so it has no page granularity.
_PER_PAGE_PHASES = {
    "phase1b",
    "phase1c",
    "phase2",
    "phase3_scale",
    "phase3_legend",
    "phase3_measure",
    "ui_page",  # session-scoped JSON: definitions/instances for slim session_state
}
_WHOLE_DOC_PHASES = {"phase1a"}

# All phases now use JSON. Pickle was removed to eliminate deserialization
# risk from untrusted cache files.
_PICKLE_PHASES: set[str] = set()

# --- Ephemeral per-session cache ---
# Policy: a cache entry is only ever useful while processing one PDF. Once
# the analysis finishes (success or error) or the user uploads a new PDF,
# the cache is purged. Previously we had a shared directory that any
# session could read — that saved time on repeat uploads but every user
# file is unique in practice, so the shared dir was dead weight and a
# privacy risk. Now: every caller passes user_scope = session_<uuid>
# and its entries live under that one dir. Nothing shared, nothing reused.
SHARED_SCOPE = "_shared"  # kept for backward compat with old call sites
SHARED_PHASES = set()     # empty → no phase is shared anymore


def _effective_scope(phase: str, user_scope: str) -> str:
    """Return the caller's scope verbatim. SHARED_PHASES is empty in v3
    so even deterministic phases (1a, 1b, 1c, 2) are isolated."""
    return user_scope or SHARED_SCOPE


_CACHE_ROOT_CACHED: Optional[Path] = None
_CACHE_ROOT_LOCK = None  # lazy-init to avoid a module-load import of threading


def cache_root() -> Path:
    """Resolve the cache root directory. Memoized after first success so
    concurrent callers don't race on the writability probe (the previous
    implementation had a mkdir+probe+unlink dance that self-raced under
    Phase 3's 12-way worker pool).

    Always returns a writable path, or raises OSError if neither the
    configured path nor the fallback works.
    """
    global _CACHE_ROOT_CACHED, _CACHE_ROOT_LOCK

    # Fast path (common case): already resolved, just return.
    if _CACHE_ROOT_CACHED is not None:
        return _CACHE_ROOT_CACHED

    if _CACHE_ROOT_LOCK is None:
        import threading
        _CACHE_ROOT_LOCK = threading.Lock()

    with _CACHE_ROOT_LOCK:
        if _CACHE_ROOT_CACHED is not None:  # re-check after lock
            return _CACHE_ROOT_CACHED

        env = os.environ.get("FENCE_CACHE_DIR")
        candidates = []
        if env:
            candidates.append(Path(env).expanduser())
        candidates.append(Path("~/.cache/fence_ade").expanduser())
        candidates.append(Path(tempfile.gettempdir()) / "fence_ade_cache")
        last_err: Optional[Exception] = None
        for p in candidates:
            try:
                p.mkdir(parents=True, exist_ok=True)
                # writability probe, per-PID so concurrent writers don't
                # clobber or prematurely unlink each other's probe file.
                probe = p / f".write_probe_{os.getpid()}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                _CACHE_ROOT_CACHED = p
                return p
            except Exception as e:
                last_err = e
                continue
        raise OSError(f"No writable cache root among {candidates}: {last_err}")


def params_hash(**kwargs: Any) -> str:
    """Stable 12-char hash of the flags that affect cache validity.

    Baked in: CACHE_SCHEMA_VERSION. Bump the constant to invalidate
    everything.
    """
    payload = {"_schema": CACHE_SCHEMA_VERSION, **kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _phase_dir(pdf_sha256: str, phase: str, params: str,
               user_scope: str = SHARED_SCOPE) -> Path:
    """Resolve the on-disk directory for a (pdf, phase, params, user_scope) tuple.

    Layout: root/<effective_scope>/<pdf_sha[:2]>/<pdf_sha>/<phase>__<params>/
    Effective scope is always SHARED_SCOPE for deterministic phases,
    regardless of what the caller passed (see _effective_scope)."""
    root = cache_root()
    scope = _effective_scope(phase, user_scope)
    return root / scope / pdf_sha256[:2] / pdf_sha256 / f"{phase}__{params}"


def _entry_path(pdf_sha256: str, phase: str, params: str,
                page_idx: Optional[int],
                user_scope: str = SHARED_SCOPE) -> Path:
    d = _phase_dir(pdf_sha256, phase, params, user_scope=user_scope)
    ext = "pkl" if phase in _PICKLE_PHASES else "json"
    if page_idx is None:
        return d / f"data.{ext}"
    return d / f"page_{page_idx:04d}.{ext}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def get(phase: str, pdf_sha256: str, params: str,
        page_idx: Optional[int] = None,
        user_scope: str = SHARED_SCOPE) -> Any:
    """Return cached value, or None on miss / read error. Never raises."""
    try:
        path = _entry_path(pdf_sha256, phase, params, page_idx, user_scope=user_scope)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = f.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        print(f"[fence_cache] get({phase}, page={page_idx}) failed: {e}")
        return None


def put(phase: str, pdf_sha256: str, params: str, value: Any,
        page_idx: Optional[int] = None,
        user_scope: str = SHARED_SCOPE) -> None:
    """Write value to the cache. Logs and swallows errors (cache is best-effort)."""
    try:
        path = _entry_path(pdf_sha256, phase, params, page_idx, user_scope=user_scope)
        data = json.dumps(value, default=str).encode("utf-8")
        _atomic_write_bytes(path, data)
    except Exception as e:
        print(f"[fence_cache] put({phase}, page={page_idx}) failed: {e}")


def probe(pdf_sha256: str, phase: str, params: str,
          page_indices: Optional[Iterable[int]] = None,
          user_scope: str = SHARED_SCOPE) -> dict:
    """Inspect the cache for a given phase/pdf/params.

    Returns:
        {"complete": bool, "covered": set[int], "missing": set[int]}
        For whole-doc phases (phase1a), "covered" is {0} on hit, empty on miss;
        "complete" reflects the single-file presence.
    """
    d = _phase_dir(pdf_sha256, phase, params, user_scope=user_scope)
    if phase in _WHOLE_DOC_PHASES:
        hit = _entry_path(pdf_sha256, phase, params, None, user_scope=user_scope).exists()
        return {"complete": hit, "covered": {0} if hit else set(), "missing": set()}

    covered: set = set()
    if d.exists():
        ext = "pkl" if phase in _PICKLE_PHASES else "json"
        for p in d.glob(f"page_*.{ext}"):
            try:
                # filename: page_NNNN.ext
                idx_s = p.stem[5:]
                covered.add(int(idx_s))
            except Exception:
                continue
    if page_indices is None:
        # No expected-set given; just report what's there.
        return {"complete": False, "covered": covered, "missing": set()}
    expected = set(page_indices)
    missing = expected - covered
    return {
        "complete": not missing,
        "covered": covered & expected,
        "missing": missing,
    }


def purge_pdf(pdf_sha256: str, user_scope: Optional[str] = None) -> None:
    """Delete cached entries for one PDF hash.

    If user_scope is None, purges BOTH the shared entries and every
    per-user scope directory for this PDF. If user_scope is given,
    purges only that scope + the shared entries (typical "user A wants
    to re-run this PDF" path)."""
    try:
        root = cache_root()
        scopes_to_purge: list[str]
        if user_scope is None:
            scopes_to_purge = [p.name for p in root.iterdir() if p.is_dir()]
        else:
            scopes_to_purge = [SHARED_SCOPE, user_scope]
        for scope in set(scopes_to_purge):
            d = root / scope / pdf_sha256[:2] / pdf_sha256
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        print(f"[fence_cache] purge_pdf({pdf_sha256[:8]}) failed: {e}")


def sweep_old(ttl_days: float = 14.0) -> int:
    """Delete cache entries whose mtime is older than `ttl_days`. Returns
    the number of PDF directories removed. Safe to call at startup.

    New layout (v2): root/<scope>/<sha[:2]>/<sha>/...
    Also handles a transitional v1 layout: root/<sha[:2]>/<sha>/... —
    any two-char directories at the root are treated as v1 shards for
    sweeping purposes so stale pre-migration entries still age out."""
    try:
        root = cache_root()
    except OSError:
        return 0
    cutoff = time.time() - ttl_days * 86400.0
    removed = 0
    for top in root.iterdir() if root.exists() else []:
        if not top.is_dir():
            continue
        # Is this a scope dir (v2) or a 2-char shard (v1)?
        looks_like_shard = len(top.name) == 2 and all(c in "0123456789abcdef" for c in top.name)
        shard_iter = [top] if looks_like_shard else list(top.iterdir())
        for shard in shard_iter:
            if not shard.is_dir():
                continue
            for pdf_dir in shard.iterdir():
                if not pdf_dir.is_dir():
                    continue
                try:
                    newest = max(
                        (f.stat().st_mtime for f in pdf_dir.rglob("*") if f.is_file()),
                        default=0.0,
                    )
                    if newest and newest < cutoff:
                        shutil.rmtree(pdf_dir, ignore_errors=True)
                        removed += 1
                except Exception:
                    continue
    return removed
