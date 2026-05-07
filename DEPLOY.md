# Deploying leo_streamlit

The repo currently runs **two stacks side-by-side** on one host. Both are
managed by systemd and both share `pipeline.py` + `utils_ade` + `utils_vector`.

| Stack | systemd unit | Port | Process | Used by |
|---|---|---|---|---|
| **Prod (live)** | `fence-fast.service` | 8502 | `streamlit run app_ade_prod.py` | Real users today. Single-process monolith — analysis runs inline in Streamlit threads. |
| **Fast (next-gen)** | `fence-api.service` | 8503 | `uvicorn api_server:app` | Backend for the future-prod Streamlit frontend (`app_ade_fast.py`). Currently running but no Streamlit frontend pointed at it yet — that's the cutover target. |

`fence-fast.service` does **not** talk to `fence-api.service` — the
naming suggests they're related, but the prod monolith does its own
in-process analysis. The API backend sits there for the future
`app_ade_fast.py` frontend (see migration plan).

## Prerequisites

- Ubuntu 22.04+ (or any systemd Linux)
- Python 3.12 with `venv`
- `nginx`
- `git`, `build-essential`

## 1. Clone + Python env

```bash
git clone git@github.com:parhamouni/leo_streamlit.git /home/ubuntu/leo_streamlit
cd /home/ubuntu/leo_streamlit

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

## 2. Secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Fill in real OpenAI / LandingAI / Google Cloud / Google OIDC credentials.
```

The `.streamlit/secrets.toml` file is gitignored — fill it in fresh per host.

## 3. Local data dirs

```bash
mkdir -p ~/.leo/results        # job_registry result artifacts (fast stack)
mkdir -p ~/.cache/fence_ade    # disk cache for pipeline phases
mkdir -p /tmp/fence_pdfs       # uploaded PDFs (transient)
```

These are auto-created on first run too — listed for transparency.

## 4. systemd units

The repo ships unit files under `ops/systemd/`:

| Unit file | Purpose |
|---|---|
| `fence-fast.service` | Live prod Streamlit (`app_ade_prod.py`) on 8502 |
| `fence-api.service` | FastAPI backend (`api_server.py`) on 8503 |
| `fence-watchdog.service` + `.timer` | Health probe; restarts unhealthy services |
| `fence-stale-restart.service` + `.timer` | Periodic restart to release accumulated memory |

Install:

```bash
sudo cp ops/systemd/fence-fast.service          /etc/systemd/system/
sudo cp ops/systemd/fence-api.service           /etc/systemd/system/
sudo cp ops/systemd/fence-watchdog.service      /etc/systemd/system/
sudo cp ops/systemd/fence-watchdog.timer        /etc/systemd/system/
sudo cp ops/systemd/fence-stale-restart.service /etc/systemd/system/
sudo cp ops/systemd/fence-stale-restart.timer   /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now fence-fast.service fence-api.service \
                              fence-watchdog.timer fence-stale-restart.timer

sudo systemctl status fence-fast fence-api
```

Both `fence-fast` and `fence-api` cap memory at `MemoryMax=12G` /
`MemoryHigh=10G` so a runaway upload can't take the host down.

The API processes one job at a time (`FENCE_API_WORKER_COUNT=1`). Increase
via `Environment=FENCE_API_WORKER_COUNT=2` in the unit, then
`daemon-reload && systemctl restart fence-api`.

## 5. nginx

Create `/etc/nginx/sites-enabled/fence`:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 800M;

    # Production Streamlit (app_ade_prod.py) on 8502
    location / {
        proxy_pass http://127.0.0.1:8502/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }

    # FastAPI backend on 8503 — exposed under /api/ for the future fast frontend
    location /api/ {
        proxy_pass http://127.0.0.1:8503/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

Reload:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 6. Verify

```bash
# Prod Streamlit
curl -sI http://127.0.0.1:8502/ | head -1   # → HTTP/1.1 200 OK

# API health (direct)
curl -s http://127.0.0.1:8503/api/healthz | python3 -m json.tool

# Both services should be 'active'
systemctl is-active fence-fast.service fence-api.service
```

## Architecture today

```
                 ┌──────────────────────────────────────────┐
Browser ─ / ─→  │ fence-fast.service (Streamlit, 8502)      │
                │   app_ade_prod.py — monolithic, all       │
                │   phases run inline in Streamlit threads  │
                │   imports: utils_ade, utils_vector,       │
                │            fence_cache, telemetry,        │
                │            spend_tracker                  │
                └──────────────────────────────────────────┘

                 ┌──────────────────────────────────────────┐
                 │ fence-api.service (FastAPI, 8503)         │
                 │   api_server.py → pipeline.run_analysis() │
                 │   background worker (1 at a time)         │
                 │   SQLite job_registry + ~/.leo/results/   │
                 │   24h TTL on results                      │
                 │   ── currently NO Streamlit pointed at it │
                 └──────────────────────────────────────────┘
```

## Architecture target (post-migration)

```
Browser → Streamlit (app_ade_fast.py) ──HTTP──→ FastAPI (api_server.py)
              │                                       │
              │                                       └─ pipeline.run_analysis()
              └─ thin: auth, upload, render only         (background worker)
```

When the fast stack reaches feature parity with `app_ade_prod.py`, swap
`fence-fast.service`'s `ExecStart=` from `app_ade_prod.py` to
`app_ade_fast.py` and keep `app_ade_prod.py` available on a fallback port
for ~30 days before archiving. See `plans/how-to-refactor-and-groovy-mist.md`
for the migration sequence.

## Troubleshooting

- **Prod Streamlit won't start**: `journalctl -u fence-fast.service -e`
- **API won't start**: `journalctl -u fence-api.service -e`
- **Jobs stuck in `running` after restart**: self-healed by
  `requeue_orphaned_running()` on API startup (job_registry.py).
- **OAuth redirect loop** (fast stack only): confirm `[auth].redirect_uri`
  in `secrets.toml` exactly matches the one in Google Cloud Console.
- **All jobs fail with API key error**: check `secrets.toml` is readable
  by the systemd `User=ubuntu` and `OPENAI_API_KEY` / `LANDINGAI_API_KEY`
  are set.
- **Out-of-memory crash**: cap is `MemoryMax=12G`. The watchdog timer
  will restart automatically. If it's a regression, check
  `~/.cache/fence_ade/_telemetry/` for the RSS spike point.

## Tunables (env vars)

Set in the systemd unit's `Environment=` lines. All have safe defaults
declared in [config.py](config.py).

| Var | Default | Effect |
|---|---|---|
| `FENCE_API_WORKER_COUNT` | 1 | Concurrent jobs on the API |
| `FENCE_RESULTS_TTL_HOURS` | 24 | Result retention |
| `FENCE_MAX_PDF_MB` | 500 | Upload size limit |
| `FENCE_MAX_PAGES` | 300 | Per-PDF page limit |
| `FENCE_AUTH_MODE` | `streamlit_oidc` | `none` / `streamlit_oidc` / `streamlit_password` / `proxy_header` |
