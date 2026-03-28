#!/usr/bin/env bash
set -euo pipefail

# Simple watchdog probe for fence Streamlit + nginx proxy.
# Exits non-zero if health endpoint is not responsive.

TARGET_URL="${1:-http://127.0.0.1/_stcore/health}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-8}"
RESTART_ON_FAIL="${RESTART_ON_FAIL:-0}"
SERVICE_NAME="${SERVICE_NAME:-fence.service}"

HTTP_CODE="$(curl --silent --show-error --max-time "${TIMEOUT_SECONDS}" \
  --output /dev/null --write-out "%{http_code}" "${TARGET_URL}" || true)"

if [[ "${HTTP_CODE}" == "200" || "${HTTP_CODE}" == "304" ]]; then
  echo "watchdog ok: ${TARGET_URL} -> ${HTTP_CODE}"
  exit 0
fi

echo "watchdog fail: ${TARGET_URL} -> ${HTTP_CODE:-no-response}" >&2

if [[ "${RESTART_ON_FAIL}" == "1" ]]; then
  echo "watchdog action: restarting ${SERVICE_NAME}" >&2
  systemctl restart "${SERVICE_NAME}"
fi

exit 1
