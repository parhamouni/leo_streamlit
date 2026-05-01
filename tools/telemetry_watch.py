#!/usr/bin/env python3
"""Follow fence telemetry JSONL and flag likely problems (memory / Phase 3 / slots).

Runs until SIGINT/SIGTERM. Intended as a background subprocess, e.g.:

    nohup python3 tools/telemetry_watch.py >> /tmp/fence_telemetry_watch.log 2>&1 &

Env:
  FENCE_TELEMETRY_DIR  same as the app (default ~/.cache/fence_ade/_telemetry)

Exit code is always 0 when stopped by signal; use --strict-exit to exit non-zero
on the first problem (useful in CI with a bounded timeout).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterator


def _log_dir() -> Path:
    return Path(os.environ.get("FENCE_TELEMETRY_DIR",
                               "~/.cache/fence_ade/_telemetry")).expanduser()


def _today_file() -> Path:
    return _log_dir() / f"{time.strftime('%Y-%m-%d')}.jsonl"


def _emit(out, level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {level}: {msg}\n"
    out.write(line)
    out.flush()


def _check_record(
    rec: dict,
    *,
    rss_warn_mb: float,
    expect_phase3: str | None,
) -> list[str]:
    """Return list of problem strings (empty if OK)."""
    problems: list[str] = []
    kind = rec.get("kind")
    rss = rec.get("rss_mb")
    if isinstance(rss, (int, float)) and rss > rss_warn_mb:
        problems.append(f"high_rss rss_mb={rss} (threshold {rss_warn_mb})")

    if kind == "event":
        name = rec.get("name", "")
        if name == "slot_rejected":
            problems.append(f"slot_rejected {rec}")
        if name == "phase3_page_done":
            st = rec.get("status", "")
            if st and st != "ok":
                problems.append(f"phase3_page_done status={st} page={rec.get('page_idx')}")
            worker = rec.get("worker")
            if expect_phase3 == "subprocess" and worker == "thread":
                problems.append(
                    f"phase3_page_done worker=thread (expected subprocess) "
                    f"page={rec.get('page_idx')} rss_mb={rss}"
                )
        if name == "legend_prebatch_shard_timeout":
            problems.append(f"legend_prebatch_shard_timeout {rec}")

    return problems


def _follow_file(poll: float, out, stop: dict) -> Iterator[str]:
    """Yield complete lines appended to today's jsonl; reopen if date rolls or file appears."""
    fh = None
    cur: Path | None = None
    try:
        while not stop.get("flag"):
            path = _today_file()
            if not path.parent.is_dir():
                time.sleep(poll)
                continue
            if cur != path or fh is None:
                if fh:
                    fh.close()
                    fh = None
                cur = path
                if path.exists():
                    fh = open(path, encoding="utf-8")
                    fh.seek(0, 2)
                    _emit(out, "INFO", f"Following {path}")
                else:
                    _emit(out, "INFO", f"Waiting for {path} (directory ok)")
            if fh is None:
                time.sleep(poll)
                continue
            line = fh.readline()
            if not line:
                time.sleep(poll)
                continue
            if not line.endswith("\n"):
                continue
            yield line
    finally:
        if fh:
            fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch fence telemetry JSONL for problems.")
    ap.add_argument("--poll", type=float, default=1.0, help="Seconds when idle (default 1)")
    ap.add_argument("--heartbeat", type=int, default=300,
                    help="Seconds between OK heartbeats when no issues (default 300, 0=off)")
    ap.add_argument("--rss-warn-mb", type=float, default=6500.0,
                    help="Warn if any record has rss_mb above this (default 6500)")
    ap.add_argument(
        "--expect-phase3",
        choices=("subprocess", "any"),
        default="subprocess",
        help="subprocess: warn on phase3_page_done worker=thread (default). any: skip.",
    )
    ap.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit 1 on first problem (for bounded CI); default is run forever.",
    )
    args = ap.parse_args()
    expect = None if args.expect_phase3 == "any" else args.expect_phase3

    stop = {"flag": False}

    def _stop(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    out = sys.stdout
    problems_seen = 0
    last_hb = 0.0

    _emit(out, "INFO", f"telemetry_watch rss_warn_mb={args.rss_warn_mb} "
          f"expect_phase3={args.expect_phase3} dir={_log_dir()}")

    try:
        for line in _follow_file(args.poll, out, stop):
            if stop["flag"]:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            probs = _check_record(
                rec,
                rss_warn_mb=args.rss_warn_mb,
                expect_phase3=expect,
            )
            for p in probs:
                problems_seen += 1
                _emit(out, "PROBLEM", p)
                if args.strict_exit:
                    return 1
            now = time.time()
            if args.heartbeat > 0 and (now - last_hb) >= args.heartbeat:
                last_hb = now
                _emit(out, "OK", f"no issues in last {args.heartbeat}s "
                      f"(total problems since start: {problems_seen})")
    finally:
        _emit(out, "INFO", "telemetry_watch stopped")
    if args.strict_exit and problems_seen:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
