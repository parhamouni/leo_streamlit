#!/usr/bin/env python3
"""Summarise the telemetry JSONL produced by telemetry.py.

Usage:
    python3 tools/telemetry_report.py              # today's file
    python3 tools/telemetry_report.py 2026-04-22   # specific date
    python3 tools/telemetry_report.py --all        # every file in the dir
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _log_dir() -> Path:
    return Path(os.environ.get("FENCE_TELEMETRY_DIR",
                               "~/.cache/fence_ade/_telemetry")).expanduser()


def _pick_files(argv: list[str]) -> list[Path]:
    d = _log_dir()
    if not d.exists():
        print(f"No telemetry dir at {d}", file=sys.stderr)
        sys.exit(1)
    if "--all" in argv:
        return sorted(d.glob("*.jsonl"))
    if len(argv) > 1:
        p = d / f"{argv[1]}.jsonl"
        if not p.exists():
            print(f"No such file: {p}", file=sys.stderr)
            sys.exit(1)
        return [p]
    # default: today
    import time
    p = d / f"{time.strftime('%Y-%m-%d')}.jsonl"
    if not p.exists():
        print(f"No telemetry yet today at {p}", file=sys.stderr)
        sys.exit(1)
    return [p]


def _load(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _p(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round(pct / 100.0 * (len(vs) - 1)))))
    return vs[k]


def _fmt(x: float, unit: str = "") -> str:
    return f"{x:.2f}{unit}"


def report_timed(rows: list[dict]) -> None:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("kind") == "timed":
            by_label[r.get("label", "?")].append(r)
    if not by_label:
        print("\n(no @timed records)")
        return
    print("\n=== @timed functions ===")
    print(f"{'label':<40} {'n':>5} {'wall_p50':>10} {'wall_p95':>10} "
          f"{'cpu_sum':>10} {'rss_delta_p95':>14} {'peak_rss_p95':>14}")
    rows_out = []
    for label, recs in by_label.items():
        walls = [r["wall_s"] for r in recs if "wall_s" in r]
        cpus = [r["cpu_s"] for r in recs if "cpu_s" in r]
        rssd = [r["rss_delta_mb"] for r in recs if "rss_delta_mb" in r]
        peak = [r["peak_rss_mb"] for r in recs if "peak_rss_mb" in r]
        rows_out.append((sum(cpus), label, len(recs), _p(walls, 50), _p(walls, 95),
                         sum(cpus), _p(rssd, 95), _p(peak, 95)))
    rows_out.sort(reverse=True)  # highest cpu_sum first
    for _, label, n, w50, w95, csum, rd95, pr95 in rows_out:
        print(f"{label:<40} {n:>5} {_fmt(w50, 's'):>10} {_fmt(w95, 's'):>10} "
              f"{_fmt(csum, 's'):>10} {_fmt(rd95, 'MB'):>14} {_fmt(pr95, 'MB'):>14}")


def report_checkpoints(rows: list[dict]) -> None:
    by_label: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("kind") == "checkpoint":
            by_label[r.get("label", "?")].append(r.get("rss_mb", 0.0))
    if not by_label:
        print("\n(no checkpoint records)")
        return
    print("\n=== phase checkpoints (rss_mb) ===")
    print(f"{'label':<30} {'n':>5} {'rss_p50':>10} {'rss_p95':>10} {'rss_max':>10}")
    for label, rss in sorted(by_label.items()):
        print(f"{label:<30} {len(rss):>5} "
              f"{_fmt(_p(rss, 50), 'MB'):>10} {_fmt(_p(rss, 95), 'MB'):>10} "
              f"{_fmt(max(rss), 'MB'):>10}")


def report_events(rows: list[dict]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("kind") == "event":
            counts[r.get("name", "?")] += 1
    if not counts:
        print("\n(no event records)")
        return
    print("\n=== events ===")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{name:<40} {n:>5}")


def main() -> None:
    files = _pick_files(sys.argv)
    rows = _load(files)
    print(f"Loaded {len(rows)} records from {len(files)} file(s).")
    report_timed(rows)
    report_checkpoints(rows)
    report_events(rows)


if __name__ == "__main__":
    main()
