#!/usr/bin/env bash
set -euo pipefail

# Stale-analysis reaper.
#
# The app acquires an exclusive lock at /tmp/fence_analysis.lock (JSON with
# "session_id", "pid", "started_at" epoch seconds) when it begins a heavy
# analysis, and releases it on completion or error.
#
# Some analyses have been observed to wedge for hours (PDFs that push MuPDF
# into recovery loops, memory-throttled processes that can no longer make
# progress, sessions held open by disconnected browsers). When that happens
# the lock stays held, the service slowly consumes all its memory cap, and
# eventually the site stops responding entirely.
#
# This script is scheduled by fence-stale-restart.timer. On each tick it
# checks the lock: if the analysis has been running for longer than
# MAX_ANALYSIS_SECONDS, it restarts fence.service. systemd's Restart=always
# brings a clean fence.service back up within seconds, clearing the wedged
# memory and releasing the lock.
#
# If there is no active analysis (no lock file), the script is a no-op.

LOCK_FILE="${LOCK_FILE:-/tmp/fence_analysis.lock}"
MAX_ANALYSIS_SECONDS="${MAX_ANALYSIS_SECONDS:-7200}"   # 2 hours
SERVICE_NAME="${SERVICE_NAME:-fence.service}"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "stale-reaper: no active analysis lock — nothing to do"
  exit 0
fi

# Extract started_at from the lock JSON without needing jq.
# Lock format: {"session_id": "...", "pid": 12345, "started_at": 1729600000}
started_at="$(grep -oE '"started_at"[[:space:]]*:[[:space:]]*[0-9]+' "${LOCK_FILE}" | grep -oE '[0-9]+' | tail -1 || true)"

if [[ -z "${started_at}" || ! "${started_at}" =~ ^[0-9]+$ ]]; then
  echo "stale-reaper: lock file present but started_at unreadable — skipping" >&2
  exit 0
fi

now="$(date +%s)"
age=$((now - started_at))

if (( age <= MAX_ANALYSIS_SECONDS )); then
  echo "stale-reaper: analysis age ${age}s (cap ${MAX_ANALYSIS_SECONDS}s) — healthy, not acting"
  exit 0
fi

echo "stale-reaper: analysis has been running ${age}s (> ${MAX_ANALYSIS_SECONDS}s limit) — restarting ${SERVICE_NAME}" >&2
systemctl restart "${SERVICE_NAME}"
