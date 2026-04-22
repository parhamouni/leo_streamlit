#!/usr/bin/env bash
set -euo pipefail

# Health probe for fence Streamlit + nginx proxy.
#
# This is a MONITOR, not an executioner. A failed probe does NOT mean the app
# is dead — Streamlit is single-threaded, so /_stcore/health routinely returns
# 503/times out during normal heavy work (large uploads, OCR, ADE batching).
# Treating that as "dead" and restarting would kill valid in-progress user
# work, which is exactly what this script used to do (RESTART_ON_FAIL=1) and
# what caused the restart-loop we diagnosed in April 2026.
#
# Actual process death is handled by systemd's `Restart=always` on
# fence.service, so this script only needs to log health.
#
# RESTART_ON_FAIL is left as an opt-in knob (default off) so an operator can
# re-enable it if they build a smarter busy-detection system later. Today it
# should stay 0.

TARGET_URL="${1:-http://127.0.0.1/_stcore/health}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"
RESTART_ON_FAIL="${RESTART_ON_FAIL:-0}"
SERVICE_NAME="${SERVICE_NAME:-fence.service}"

HTTP_CODE="$(curl --silent --show-error --max-time "${TIMEOUT_SECONDS}" \
  --output /dev/null --write-out "%{http_code}" "${TARGET_URL}" || true)"

if [[ "${HTTP_CODE}" == "200" || "${HTTP_CODE}" == "304" ]]; then
  echo "watchdog ok: ${TARGET_URL} -> ${HTTP_CODE}"
  exit 0
fi

echo "watchdog fail: ${TARGET_URL} -> ${HTTP_CODE:-no-response} (probably busy, not dead — not acting)" >&2

if [[ "${RESTART_ON_FAIL}" == "1" ]]; then
  echo "watchdog action: restarting ${SERVICE_NAME} (RESTART_ON_FAIL=1)" >&2
  systemctl restart "${SERVICE_NAME}"
fi

exit 1
