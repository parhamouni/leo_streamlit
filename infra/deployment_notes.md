# Backend deployment — AWS systemd + nginx

The new multi-user backend (`fence-api-v2.service`, port 8513) runs alongside the legacy `fence-api.service` (port 8503) and the Streamlit `fence-fast.service` (port 8502) until cutover. **Do not modify the legacy units** — they're serving prod traffic from a different host today, but the box may host both during the transition.

Frontend deploys live separately on Vercel — see [deploy_frontend.md](deploy_frontend.md).

---

## Prerequisites

- Ubuntu box with systemd (the same OS pattern as legacy prod).
- Python venv at `/home/ubuntu/leo_streamlit/venv` with `requirements.txt` installed (FastAPI, uvicorn, pyjwt, psycopg, supabase, fitz, etc.).
- nginx + certbot.
- DNS A record pointing `api.<your-domain>` at the box's elastic IP.
- A Supabase project with the migrations under `backend/db/migrations/` already applied (001 schema, 002 dedup, 003 phase_started_at, 004 page_results.updated_at).

---

## One-time setup

### 1. Environment file

```bash
sudo mkdir -p /etc/fence-api-v2
sudo cp infra/systemd/fence-api-v2.env.example /etc/fence-api-v2/env
sudo chmod 600 /etc/fence-api-v2/env
sudoedit /etc/fence-api-v2/env   # fill in real values
```

If you use a Google Cloud service account for DocAI:

```bash
sudo cp /path/to/service-account.json /etc/fence-api-v2/google-sa.json
sudo chmod 600 /etc/fence-api-v2/google-sa.json
```

### 2. systemd unit

```bash
sudo cp infra/systemd/fence-api-v2.service /etc/systemd/system/fence-api-v2.service
sudo systemctl daemon-reload
sudo systemctl enable --now fence-api-v2
sudo systemctl status fence-api-v2
journalctl -u fence-api-v2 -f --since "5 min ago"
```

Health check:

```bash
curl -s http://127.0.0.1:8513/api/healthz
# → {"status":"ok",...}
```

### 3. nginx + TLS

```bash
sudo cp infra/nginx/fence-api-v2.conf /etc/nginx/sites-available/fence-api-v2.conf
# Replace `api.<your-domain>` with the real hostname before enabling.
sudoedit /etc/nginx/sites-available/fence-api-v2.conf
sudo ln -s /etc/nginx/sites-available/fence-api-v2.conf /etc/nginx/sites-enabled/

# TLS cert (port 80 must be reachable from the public internet first).
sudo certbot --nginx -d api.<your-domain>

sudo nginx -t && sudo systemctl reload nginx
```

External health check:

```bash
curl -sf https://api.<your-domain>/api/healthz
```

---

## Runtime ops

| Task | Command |
|---|---|
| View logs | `journalctl -u fence-api-v2 -f` |
| Restart service | `sudo systemctl restart fence-api-v2` |
| Reload env vars | `sudo systemctl restart fence-api-v2` (no graceful reload — uvicorn workers reread env on start) |
| Apply code update | `git pull && sudo systemctl restart fence-api-v2` |
| Tail nginx access | `sudo tail -f /var/log/nginx/fence-api-v2.access.log` |
| Inspect failed requests | `sudo grep ' 5[0-9][0-9] ' /var/log/nginx/fence-api-v2.access.log` |

---

## CORS configuration (Phase 11.3)

The backend reads `FENCE_CORS_ORIGINS` (comma-separated) from `/etc/fence-api-v2/env`. Update it to include the Vercel production domain plus any preview branches you actively test against, then `sudo systemctl restart fence-api-v2`.

Mirror the same list in **Supabase Studio → Authentication → URL Configuration**:
- **Site URL** = `https://app.<your-domain>` (the Vercel production domain)
- **Redirect URLs** = the production domain + any Vercel preview/branch URLs you sign in from (Supabase rejects post-OAuth redirects to unlisted hosts).

---

## Cutover checklist (after one prod day on the new stack)

1. Disable the legacy units:
   ```bash
   sudo systemctl disable --now fence-fast.service     # Streamlit on :8502
   sudo systemctl disable --now fence-api.service      # legacy FastAPI on :8503
   ```
2. Remove the `fence-fast` and `fence-api` nginx sites if any.
3. Rotate every secret listed in `STATUS.md` § "Secret-rotation checklist" (Supabase DB password, test user, Google OAuth client secret, every key in `.streamlit/secrets.toml`).
4. Tag the cutover commit on `main` for an easy rollback target:
   ```bash
   git tag -a v2-cutover -m "v2 stack cutover"
   git push origin v2-cutover
   ```

---

## Rollback

The legacy units stay installed (just disabled) so a regression rolls back in under a minute:

```bash
sudo systemctl stop fence-api-v2
sudo systemctl start fence-api.service fence-fast.service
# Revert the Vercel production deploy via the Vercel dashboard (Deployments → previous build → Promote to Production).
# Repoint api.<your-domain> at the old backend if the old box was on a different IP.
```

If a code-level regression is reproducible, also `git revert` the offending commit on `main` so the next clean deploy doesn't reintroduce it.

---

## What's deliberately NOT here yet

- **Redis + RQ worker** (Phase 7) — the in-process worker still handles jobs. Swap in `worker/jobs.py` + `fence-worker.service` when you go multi-instance.
- **S3 storage** — uploads still land on the box's `/tmp/fence_pdfs/<user_id>/`. Reboot wipes them. The Measurement PDF endpoint already 404s with a "re-upload" message when this happens. Move to S3 (Phase 5.1, deferred) for durability.
