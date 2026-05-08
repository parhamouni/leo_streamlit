# Leo Fence — Web-app Migration Status

> Snapshot for a fresh-context resumption. Read this first, then the plan at
> `.claude/plans/yes-i-get-you-refactored-hummingbird.md` for full detail.

**Branch:** `feat/web-app-migration` (12 commits, 30+ files)
**Plan file:** `/home/ubuntu/.claude/plans/yes-i-get-you-refactored-hummingbird.md`
**Decision log:** UMT confirmed **critical-path** (customers always correct measurements manually).

---

## What we're building

Replacing the Streamlit monolith ([app_ade_prod.py](app_ade_prod.py), 7,130 lines, port 8502) with a multi-user web app:

| Layer | Tech | Status |
|---|---|---|
| Frontend | **Next.js 14** App Router + TypeScript + Tailwind | Live in [frontend/](frontend/) |
| Auth | **Supabase** (email + Google OAuth, ES256 JWKS) | Live |
| Persistence | **Supabase Postgres** (`documents`, `jobs`, `page_results`, `artifacts`) | Live (migrations 001, 002 applied) |
| Backend | **FastAPI** ([api_server.py](api_server.py)) — already existed; evolved | Live |
| Pipeline | [pipeline.py](pipeline.py) — already existed, untouched | Reused as-is |
| Job queue | SQLite ([job_registry.py](job_registry.py)) for now (Redis/RQ deferred) | Live |
| Storage | **Local disk** (`/tmp/fence_pdfs/<user_id>/`) — S3 deferred to Phase 11 | Live |

The legacy Streamlit prod stays untouched per user instruction (they're serving prod from a different instance).

---

## Done so far (commits on `feat/web-app-migration`)

| Commit | What |
|---|---|
| `934a997` | Scaffold: `frontend/`, `infra/`, `backend/db/`. Migration 001 (documents/jobs/page_results/artifacts). Backend `auth.py` with Supabase JWKS. Login page (email+password). |
| `4ac5667` | Self-service signup + Google OAuth button on login page |
| `8f41f7c` | Phase 5: Postgres mirror in `POST /api/jobs`. New `GET /api/documents` and `/api/documents/{id}`. Worker mirrors progress to Postgres. |
| `8566386` | Dashboard document list + CORS middleware on backend |
| `30de7e3` | Next.js rewrites proxy `/api/*` → backend on `127.0.0.1:8513` (avoids second-port tunnel issue) |
| `d38fe50` | Drag-drop PDF upload component |
| `28c5374` | Real upload progress bar (XHR) + queueable uploads while busy |
| `fb110f3` | Live polling on dashboard (3s, paused when tab hidden) + inline progress bars |
| `a01851a` | Per-document detail page with summary stats + page table |
| `0613be4` | **Dedup** (migration 002 `pdf_hash`) + rich detail page (embedded PDF viewer, per-page accordion, scale info, legend, instances) |
| `c5c87f2` | **Cancel** running jobs and **Delete** terminal jobs (with Postgres cleanup via cascade) |
| `c574c0b` | **Sprint 1 of feature parity**: detection method badges, ADE chunk metrics, layer breakdown, dimension lines, scale debug expander, element specs table, non-fence reasoning panel, phase timings, three-way filter |

### Endpoints currently live on the backend

```
GET    /api/healthz                        — public
GET    /api/me                             — JWT only (verifies Supabase token)
POST   /api/jobs                           — JWT or X-User-Id; dedup-aware; mirrors Postgres
GET    /api/jobs                           — list user's jobs (legacy, used by old Streamlit)
GET    /api/jobs/{id}                      — single job
GET    /api/jobs/{id}/results              — full pipeline result JSON
GET    /api/jobs/{id}/highlighted-pdf      — FileResponse, FileDownload
GET    /api/jobs/{id}/progress             — SSE stream
DELETE /api/jobs/{id}                      — cancel (active) or hard-delete (terminal)
GET    /api/documents                      — JWT only; per-user list (powers dashboard)
GET    /api/documents/{id}                 — JWT only; single doc with latest job joined
```

### Frontend routes

```
/login              email+password OR Google OAuth; redirects to /dashboard if signed in
/dashboard          documents table, upload area, live polling, cancel/delete
/documents/[id]     detail page with embedded PDF viewer + rich per-page accordions
/                   redirects to /dashboard
```

---

## Hard constraints (do not break)

1. **No edits to [app_ade_prod.py](app_ade_prod.py)** or its `fence-fast.service` systemd unit. Production runs elsewhere; user said we can stop running prod locally.
2. **Shared library modules are additive-only**: [utils_ade/](utils_ade/), [utils_vector.py](utils_vector.py), [fence_cache.py](fence_cache.py), [telemetry.py](telemetry.py), [spend_tracker.py](spend_tracker.py). Don't change signatures.
3. **`requirements.txt` is additive-only**. Don't remove or downgrade existing pins.
4. **Free-to-evolve**: [api_server.py](api_server.py), [pipeline.py](pipeline.py), [job_registry.py](job_registry.py), [auth.py](auth.py), [config.py](config.py), [secrets_loader.py](secrets_loader.py), [exports.py](exports.py), [state.py](state.py).

---

## Sprint 1 verification (2026-05-08)

Verified against `frontend/app/documents/[id]/page.tsx`. **8 of 9 features fully implemented, 1 cosmetic-only difference:**

| | Feature | Status |
|---|---|---|
| A1 | Element specifications table | ✅ (lines 883–944) |
| A2 | Full Detail Text on row click | ✅ |
| A3 | Detection method badges | ✅ (557–570) |
| A4 | Non-fence reasoning | 🟡 Implemented inline in `NonFencePageCard` (727–767) — not as the separate "panel" the original doc described, but all data renders. Functional parity. |
| A6 | ADE chunk metrics | ✅ (589–596) |
| B1 | Layer breakdown expander | ✅ (802–843) |
| B2 | Dimension lines expander | ✅ (846–880) |
| B3 | Scale detection debug | ✅ (628–661) |
| — | Three-way filter (fence / non-fence / all) | ✅ (436–471) |
| — | Phase-timing grid | ✅ (377–390) |
| — | Highlighted-PDF download button | ✅ (line 353) |

No regressions found. Treat Sprint 1 as **done**.

## Parity audit vs `app_ade_prod.py` (2026-05-08)

Combed prod for user-workflow features the plan might have missed. The user clarified: parity goal = **what the customer touches**, not operator/queue/telemetry surfaces.

**Single net-new gap added to plan:** **C8 — Cross-page summary table by category** (prod 6935–7050). Grand totals across pages, auto vs manual line counts, ft per category, per-page breakdown. Ships with UMT in Sprint 4 — without it, the user has nothing to hand the customer at end of run.

Items the prod app has that **don't** belong in user-workflow parity (deliberately skipped):
- Queue / slot management UI (operator concern; `job_registry.py` enforces it server-side)
- Spend / telemetry / daily-cap displays (D1–D3 — internal)
- Sidebar API-key inputs (replaced by `.env` / Supabase secrets)

Items that initially looked missing but are already covered in the plan or in code:
- Highlighted PDF download — already on detail page
- Per-page image download / lazy-load button — same as A5 / C7 in Sprint 2/3
- Confidence / reasoning text (some) — Sprint 1 covers scale debug; B4/B5 cover the rest
- New-analysis / reset flow — uploading another PDF on the dashboard already does this; explicit "reset" button not needed

## What's left — prioritised

### Sprint 2 — live observability (~1 session)
Bridges what the user sees during analysis. Plan refs: A5 / A7 / A8.

| | What | Backend | Frontend |
|---|---|---|---|
| A7 | **Live per-page updates** while job runs (pages appear in detail view as the worker completes them) | Worker writes one `page_results` row per page when phase 1c classifies it; new `GET /api/documents/{id}/pages` endpoint | Detail page subscribes to `/pages` while running |
| A8 | **Phase-window progress with ETAs** (separate overall + within-phase bars; rate-based ETA per phase) | Add `phase_started_at` to `jobs` (migration 003); record phase transitions in `_progress` callback | Compute ETA from rate + remaining %; render dual bars in dashboard + detail page |
| A5 | **Per-page rasterized image** with overlays (lazy-loaded thumbnail / full image view) | New `GET /api/jobs/{id}/page-image/{n}?dpi=110` returning a rasterized PNG | Per-page card adds "🖼️ Load page image" button → fetches PNG → renders inline |

### Sprint 3 — settings + alternate exports (~1 session)
Plan refs: A9 / C5 / C6 / C7.

| | What | Backend | Frontend |
|---|---|---|---|
| A9 | **Settings panel** (Use ADE / Highlights / Low-DPI / Non-layer / fence keywords editor) | Already accepts `config` JSON on `POST /api/jobs` (currently empty); just wire the toggles into the FormData | Component above upload area: form, posts JSON `config` along with the file |
| C5 | **Measurements PDF download** | New `GET /api/jobs/{id}/measurement-pdf` wrapping `exports.py:generate_measurement_pdf` | Add download button to detail page header |
| C6 | **Measurements Excel download** | `GET /api/jobs/{id}/measurement-excel` wrapping `exports.py:generate_measurement_spreadsheet` | Add download button |
| C7 | **Per-page image downloads** ("DL HL Img", "DL Orig Img") | Re-uses A5's image endpoint with a `?download=1` param | Add small download links to per-page cards |

### Sprint 4 — UMT + summary report (CRITICAL PATH, biggest scope)
Plan refs: C1–C4, **C8**.

User confirmed customers *always* correct measurements manually — must ship before launch. C8 (cross-page summary table) ships with UMT — it's the report the user hands to the customer.

**Two paths to evaluate (decision needed at start of sprint):**

1. **(a) React rebuild** — port [umt.py](umt.py) (~1,248 lines) into a React canvas with `react-konva` or HTML5 canvas + custom hit-testing. New POST endpoints to persist line state per `(user_id, document_id, page_num)`. Highest fidelity. ~3-5 sessions.
2. **(b) Iframe-embed the existing Streamlit UMT** — keep `app_ade_fast.py`'s UMT page, embed via iframe in the new detail page. Fast, ugly, unblocks launch. ~1 session for the integration.

**Recommendation: Start with (b) for launch; collect React-rebuild requirements over time.** Iframe embedding requires:
- Passing `user_id` + `job_id` to the Streamlit page (cookie or query param)
- A small Streamlit shim page that auto-loads the right job and shows only the UMT (no rest of the prod app)
- CORS / `frame-ancestors` config on the Streamlit server
- Cookie sharing or token-based handoff between Next.js and Streamlit

### Sprint 5 — operational (low priority)
Plan refs: D1 / D2 / D3.
- D1 — usage / spend display (data already in [spend_tracker.py](spend_tracker.py))
- D2 — telemetry events display (data in [telemetry.py](telemetry.py) JSONL)
- D3 — daily-cap UI (`cfg.MAX_DAILY_SPEND_USD` is enforced silently today)

### Phase 10 — security & robustness
- 10.1 — `tests/test_authorization.py` covering cross-user 403/404 against all endpoints
- 10.2 — file-size + page-count limits already enforced; need cost/perf metadata logged to `jobs` (file_size, duration_ms, model)

### Phase 11 — deployment
- 11.1 — Vercel deploy of `frontend/` (env vars, branch deploys)
- 11.2 — `infra/systemd/fence-api-v2.service` (separate from prod's `fence-api.service`); nginx proxy for `api.<host>` → uvicorn
- 11.3 — Production CORS allowlist + Supabase Auth → URL Configuration with Vercel domain
- (deferred until later) S3 storage migration when going multi-instance — Sprint deferred from original Phase 5; storage_path keys already in S3-shape

### Phase 7 — Redis/RQ
Currently SQLite-based [job_registry.py](job_registry.py) handles queueing. Migration to Redis/RQ deferred until multi-instance deployment. Not blocking launch.

---

## Recommended next steps (in order)

1. **Sprint 2** — live per-page updates + phase ETAs + page images
2. **Sprint 3** — settings panel + Measurements PDF/Excel downloads
3. **Sprint 4** — UMT (start with iframe-embed for speed; revisit React rebuild post-launch)
4. **Phase 10.1** — authorization tests
5. **Phase 11** — Vercel + AWS deployment
6. (later) Sprint 5, S3 migration, Redis/RQ migration

---

## Secret-rotation checklist (do at end of migration)

These have appeared in chat or are saved in `.env.local` on this box:

| Secret | Source | How to rotate |
|---|---|---|
| Supabase database password (`Amarcord!973`) | chat | Supabase Studio → Project Settings → Database → Reset password; update `DATABASE_URL` in `.env.local` |
| Supabase test-user password (`parhamhamouni@gmail.com`) | chat | Studio → Authentication → Users → reset OR delete the test account |
| Google OAuth client secret (`GOCSPX-AyCb3g3ud-…`) | chat | Google Cloud Console → APIs → Credentials → +Add secret, disable old; update Supabase Auth providers AND `.streamlit/secrets.toml` (legacy prod uses same client) |
| `.streamlit/secrets.toml` contents (OpenAI / LandingAI / Azure CV / GCP service-account private key / cookie_secret) | exposed when Claude read the file | Each provider's console; coordinate with prod cutover |

Order: non-prod first (DB password, test user) → prod-affecting last (Google secret + secrets.toml keys).

---

## Where data lives right now

| Thing | Location |
|---|---|
| Uploaded PDFs | `/tmp/fence_pdfs/<user_id>/job_<hash>.pdf` (local disk) |
| Job state | `~/.leo/jobs.db` (SQLite, WAL mode) **plus** mirror in Supabase Postgres `jobs` table |
| Per-job results JSON | `~/.leo/results/<job_id>/results.json` |
| Highlighted PDF artifact | `~/.leo/results/<job_id>/highlighted.pdf` |
| Postgres documents/jobs/etc. | Supabase project `ssngaoyhxcghacdugdog` (us-west-2) |
| Frontend secrets | `frontend/.env.local` (gitignored) |
| Backend secrets | `.env.local` at repo root (gitignored) — `DATABASE_URL`, `SUPABASE_*`, etc. |

---

## How to resume

```bash
# Backend (terminal 1)
cd /home/ubuntu/leo_streamlit
fuser -k 8513/tcp 2>/dev/null   # in case old one is stuck
FENCE_API_AUTH_MODE=both ./venv/bin/uvicorn api_server:app \
  --host 127.0.0.1 --port 8513 --reload

# Frontend (terminal 2)
cd /home/ubuntu/leo_streamlit/frontend
npm run dev   # serves on http://localhost:3000

# Login at http://localhost:3000/login
# - email/password: anything you create (auto-confirm enabled in Supabase)
# - or Google OAuth via your existing leofence Google Cloud project
```

A fresh Claude session starting here can read this file + the plan, then ask the user which sprint to tackle next.
