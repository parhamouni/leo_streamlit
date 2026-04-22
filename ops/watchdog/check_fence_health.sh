#!/usr/bin/env bash
set -euo pipefail

# Health probe for fence Streamlit + nginx proxy.
#
# What this catches
# -----------------
# The "alive but unresponsive" failure mode: the Streamlit process is still
# running (systemd sees it as active) but it can't serve any HTTP requests —
# typically because it went memory-throttled after an analysis misbehaved
# (tight ADE retry loop, runaway logging, etc.). Systemd's Restart=always
# only fires on actual process death, so this state persists until a human
# notices and restarts.
#
# How it decides when to act
# --------------------------
# A single failed probe does NOT mean the app is dead. Streamlit is single-
# threaded and /_stcore/health can fail for 1–3 minutes during normal heavy
# work (large uploads, OCR, ADE batches that block briefly). We only act
# after FAIL_THRESHOLD consecutive failures have been observed — i.e. that
# many minutes of continuous unresponsiveness in a row. Any successful probe
# resets the counter.
#
# The default threshold is 15 (roughly 15 minutes). This is comfortably
# longer than any legitimate heavy phase after the Phase 1c parallelization
# (commit 8e49d39) and well short of the hour-long outage we saw in April.
#
# Actual process death is still handled by systemd's Restart=always, which
# is faster (≤ 3 s). This script is specifically for the hung-but-not-dead
# case that systemd can't see.

TARGET_URL="${1:-http://127.0.0.1/_stcore/health}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
RESTART_ON_FAIL="${RESTART_ON_FAIL:-0}"
SERVICE_NAME="${SERVICE_NAME:-fence.service}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-15}"
STATE_DIR="${STATE_DIR:-/run/fence-watchdog}"
STATE_FILE="${STATE_DIR}/consecutive_failures"

mkdir -p "${STATE_DIR}"

HTTP_CODE="$(curl --silent --show-error --max-time "${TIMEOUT_SECONDS}" \
  --output /dev/null --write-out "%{http_code}" "${TARGET_URL}" || true)"

if [[ "${HTTP_CODE}" == "200" || "${HTTP_CODE}" == "304" ]]; then
  if [[ -f "${STATE_FILE}" ]]; then
    prev="$(cat "${STATE_FILE}" 2>/dev/null || echo 0)"
    if [[ "${prev}" != "0" ]]; then
      echo "watchdog recovered after ${prev} consecutive failure(s)"
    fi
  fi
  echo 0 > "${STATE_FILE}"
  echo "watchdog ok: ${TARGET_URL} -> ${HTTP_CODE}"
  exit 0
fi

# Failure — increment streak.
prev="$(cat "${STATE_FILE}" 2>/dev/null || echo 0)"
if ! [[ "${prev}" =~ ^[0-9]+$ ]]; then prev=0; fi
curr=$((prev + 1))
echo "${curr}" > "${STATE_FILE}"

echo "watchdog fail ${curr}/${FAIL_THRESHOLD}: ${TARGET_URL} -> ${HTTP_CODE:-no-response}" >&2

if [[ "${RESTART_ON_FAIL}" != "1" ]]; then
  exit 1
fi

if (( curr >= FAIL_THRESHOLD )); then
  echo "watchdog action: restarting ${SERVICE_NAME} after ${curr} consecutive failures (${curr} min of unresponsiveness)" >&2
  echo 0 > "${STATE_FILE}"
  systemctl restart "${SERVICE_NAME}"
else
  echo "watchdog holding: ${curr} of ${FAIL_THRESHOLD} failures — not restarting yet" >&2
fi

exit 1
