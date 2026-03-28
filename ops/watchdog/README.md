# Fence Watchdog

This folder contains optional `systemd` watchdog units to auto-check Streamlit responsiveness and restart `fence.service` on health failures.

## Install

```bash
sudo cp /home/ubuntu/leo_streamlit/ops/watchdog/fence-watchdog.service /etc/systemd/system/
sudo cp /home/ubuntu/leo_streamlit/ops/watchdog/fence-watchdog.timer /etc/systemd/system/
sudo chmod +x /home/ubuntu/leo_streamlit/ops/watchdog/check_fence_health.sh
sudo systemctl daemon-reload
sudo systemctl enable --now fence-watchdog.timer
```

## Verify

```bash
systemctl status fence-watchdog.timer --no-pager
systemctl status fence-watchdog.service --no-pager
journalctl -u fence-watchdog.service -n 50 --no-pager
```

## Notes

- Health probe target is `http://127.0.0.1/_stcore/health` via nginx.
- The script treats HTTP `200` and `304` as healthy.
- Set `RESTART_ON_FAIL=0` in the service file to disable automatic restarts and use probe-only mode.
