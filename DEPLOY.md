# Deploying the dev (FastAPI + Streamlit) stack

This is the setup for the new architecture: persistent FIFO job queue, FastAPI
backend on port **8503**, Streamlit frontend on port **8501** under `/dev/`.

## Prerequisites on the host

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
# Edit .streamlit/secrets.toml with real OpenAI / LandingAI / Google Cloud /
# Google OIDC credentials. See the comments in the .example file.
```

The `.streamlit/secrets.toml` file is gitignored — fill it in fresh per host.

## 3. Local data dirs

```bash
mkdir -p ~/.leo/results
mkdir -p /tmp/fence_pdfs
```

These are auto-created on first run too — listed here for transparency.

## 4. systemd services

Copy the unit files from `ops/systemd/` into systemd:

```bash
sudo cp ops/systemd/fence.service        /etc/systemd/system/
sudo cp ops/systemd/fence-api.service    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fence.service fence-api.service
sudo systemctl status fence.service fence-api.service
```

What each does:

| Unit | Port | Process |
|---|---|---|
| `fence-api.service` | 8503 | `uvicorn api_server:app` — FastAPI worker pool |
| `fence.service`     | 8501 | `streamlit run app_ade_fast.py --server.baseUrlPath /dev` |

The API processes one job at a time (`FENCE_API_WORKER_COUNT=1`). To allow
parallel jobs, set the env var on the unit (`Environment=FENCE_API_WORKER_COUNT=2`)
and `daemon-reload + restart fence-api`.

## 5. nginx

Create `/etc/nginx/sites-enabled/fence`:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 800M;

    # Dev API → port 8503
    location /dev/api/ {
        proxy_pass http://127.0.0.1:8503/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }

    # Dev Streamlit → port 8501 (with /dev/ baseUrlPath)
    location /dev/ {
        proxy_pass http://127.0.0.1:8501/dev/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
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
# API health (direct)
curl -s http://127.0.0.1:8503/api/healthz | python3 -m json.tool

# API health (via nginx)
curl -s http://your-host/dev/api/healthz | python3 -m json.tool

# Streamlit
curl -sI http://your-host/dev/ | head -1   # → HTTP/1.1 200 OK
```

Then open `http://your-host/dev/` in a browser, sign in with Google, and
upload a test PDF.

## Architecture refresher

```
Browser ─ /dev/ ─→ Streamlit (8501)  ──HTTP──→  FastAPI (8503)
                       │                            │
                       │                            ├─ Background worker (1 at a time)
                       │                            └─ pipeline.py → utils_ade + utils_vector
                       └─ thin: auth, upload, render
                                                    SQLite job_registry + ~/.leo/results/{job_id}/
                                                    24h TTL on results
```

## Troubleshooting

- **Streamlit won't start**: check `journalctl -u fence.service -e`
- **API won't start**: check `journalctl -u fence-api.service -e`
- **Jobs stuck in `running` after restart**: this is now self-healed —
  `requeue_orphaned_running()` runs on API startup.
- **OAuth redirect loop**: confirm `[auth].redirect_uri` in `secrets.toml`
  exactly matches the one in Google Cloud Console.
- **All jobs fail with API key error**: check `secrets.toml` is readable
  by the systemd `User=ubuntu` and that `OPENAI_API_KEY` / `LANDINGAI_API_KEY`
  are set.

## Tunables (env vars)

Set in the systemd unit's `Environment=` lines. All have safe defaults.

| Var | Default | Effect |
|---|---|---|
| `FENCE_API_WORKER_COUNT` | 1 | Concurrent jobs |
| `FENCE_RESULTS_TTL_HOURS` | 24 | Result retention |
| `FENCE_MAX_PDF_MB` | 500 | Upload size limit |
| `FENCE_MAX_PAGES` | 300 | Per-PDF page limit |
| `FENCE_AUTH_MODE` | streamlit_oidc | `none` / `streamlit_oidc` / `streamlit_password` / `proxy_header` |
