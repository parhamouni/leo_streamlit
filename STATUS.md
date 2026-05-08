# Leo Fence — Web-app Migration Status

> Snapshot for a fresh-context resumption. Read this first, then the plan at
> `.claude/plans/yes-i-get-you-refactored-hummingbird.md` for full detail.

**Branch:** `feat/web-app-migration` (~25 commits, 30+ files)
**Plan file:** `/home/ubuntu/.claude/plans/yes-i-get-you-refactored-hummingbird.md`
**Stage-1 highlighting plan (now superseded — work below is done):** `/home/ubuntu/.claude/plans/fuck-you-you-are-splendid-beacon.md`
**Decision log:** UMT confirmed **critical-path** (customers always correct measurements manually).
**Last updated:** 2026-05-08 — Sprint 3 done + stage-1 highlighted-PDF parity with `app_ade_prod.py` shipped (definitions / instances / keyword_matches all from real bboxes incl. OCR-with-bboxes for scanned pages).

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
| `b0afa86` | docs: Sprint 1 verification + parity audit vs `app_ade_prod.py` (added C8 to plan) |
| `8651482` | **Sprint 2 / A7** — live per-page updates (Phase 1c stubs + Phase 3 enrichment via new `page_cb`) |
| `aac0cb2` | **Sprint 2 / A5** — `GET /api/jobs/{id}/page-image/{n}` + lazy "🖼️ Load page image" button + slim-results payload |
| `7fcda90` | Hardening: page-image renders in subprocess (MuPDF crash isolation) |
| `a120f17` | **Sprint 2 / A8** — phase-window dual progress bars + rate-based ETAs (needs migration 003) |
| `f9e4559` | (you) Detail-card cleanup: collapse "page text" + relabel "detected instances" |
| `e8467e3` | (you) Detail-card cleanup: drop sections prod never showed |
| `61d21c7` | **Bug fix**: serialize `VectorLine` → dict before saving results (was being repr'd to a string and breaking exports + highlighter) |
| `22dfead` | **Sprint 3 / C5+C6** — `GET /measurement-pdf` + `GET /measurement-excel` endpoints; download buttons on detail page |
| `6b66f7d` | **Bug fix**: wire the highlighted-PDF subprocess worker properly (it had been silently dead — `out_path` never set, JSON descriptor written to disk as PDF) + actually draw fence lines on the highlighted PDF |
| `10146db` | **Bug fix**: line endpoints now transform per-point on rotated pages (`reverse_rotation_point`). Rectangle-shaped formula was scrambling y-coords on rot=90/180/270 pages. |
| `f61e793` | Diagnostics: `_page_cb` log line + migration 004 (`page_results.updated_at`) for the live-pages debug |
| **(this commit)** | **Stage-1 highlighted-PDF parity with `app_ade_prod.py`.** Five wins, one file (`pipeline.py`) + one frontend cleanup: (1) `definitions` is now `legend_entries` (LLM per-row tight bboxes) instead of raw legend chunks; (2) new instance-finding step calls `find_instances_in_figures_fast` to get per-token indicator bboxes inside figures; (3) keyword scanner now runs against `get_native_pdf_lines` output (real per-line bboxes) so orange rectangles land at correct locations; (4) `phase1b` OCR step now preserves the per-line bboxes returned by Google DocAI (cache key bumped to `phase1b_v2`) — required for scanned pages where native PDF text is missing or has CID-without-CMap encoding; (5) `keyword_matches` always populated (deviates from prod's fallback semantics — user wants orange alongside green/purple). Frontend: removed the noisy "Detected Instances (bbox)" table from the detail page — only the legend definitions table remains. |

### Endpoints currently live on the backend

```
GET    /api/healthz                              — public
GET    /api/me                                   — JWT only (verifies Supabase token)
POST   /api/jobs                                 — JWT or X-User-Id; dedup-aware; mirrors Postgres; accepts `config` JSON form field
GET    /api/jobs                                 — list user's jobs (legacy, used by old Streamlit)
GET    /api/jobs/{id}                            — single job
GET    /api/jobs/{id}/results                    — full pipeline result JSON (slim by default; ?full=1 for raw)
GET    /api/jobs/{id}/highlighted-pdf            — FileResponse with cyan fence-line overlay (Sprint-2/3 fixes)
GET    /api/jobs/{id}/page-image/{n}?dpi=110     — PNG of one page from highlighted PDF (subprocess-isolated)
GET    /api/jobs/{id}/measurement-pdf            — measurement-overlay PDF (auto-only until UMT)
GET    /api/jobs/{id}/measurement-excel          — per-line workbook (auto-only until UMT)
GET    /api/jobs/{id}/progress                   — SSE stream
DELETE /api/jobs/{id}                            — cancel (active) or hard-delete (terminal)
GET    /api/documents                            — JWT only; per-user list (powers dashboard)
GET    /api/documents/{id}                       — JWT only; single doc + latest job (now includes started_at + phase_started_at)
GET    /api/documents/{id}/pages                 — JWT only; per-page rows (powers live "Pages so far")
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

### Sprint 2 — live observability ✅ DONE
| | What | Notes |
|---|---|---|
| A5 | Per-page rasterized image with overlays | Lazy "🖼️ Load page image" button; subprocess-isolated render |
| A7 | Live per-page updates while job runs | `page_cb` in pipeline; Phase 1c stubs + Phase 3 rich payloads upserted to `page_results`; `/api/documents/{id}/pages` polled at 3 s; **see [Open issues](#open-issues) — UI may not surface these live** |
| A8 | Phase-window dual progress bars + ETAs | `lib/eta.ts` with known phase ranges; "Overall" + "Within phase" bars on detail page; ETA on dashboard rows; **needs migration 003 in Supabase** |

### Sprint 3 — settings + alternate exports ✅ DONE
| | What | Notes |
|---|---|---|
| A9 | Analysis settings panel | `frontend/components/AnalysisSettings.tsx`; toggles persisted in localStorage; posted as `config` JSON on upload |
| C5 | Measurement PDF download | Wraps `exports.generate_measurement_pdf`; auto-populated UMT state from `layer_to_category` (with prod's partial-match fallback + "Auto-detected" bucket) |
| C6 | Measurement Excel download | Wraps `exports.generate_measurement_spreadsheet`; same caveats — fully populates only after UMT (Sprint 4) lands |
| C7 | Per-page image downloads | Covered by A5's "Load page image" button (right-click → save) |

## Open issues (need attention before Sprint 4)

### 1. Live "Pages so far" not visibly streaming during a run
User reported: "early on the pages don't load, then after the experiment is done it loads". The plumbing exists end-to-end (Phase 1c stubs *and* Phase 3 enrichment land in `page_results` — confirmed by inspecting the latest job's row content), but the UI behaviour suggests either:
- Phase 3 upserts are batched at job-end instead of per-page (worker-thread serialization?)
- Or the frontend polling effect resets too aggressively when `doc` changes every 3 s
- Or both

**Diagnostic shipped (`f61e793`):** `_page_cb` now logs one INFO line per upsert. The next run's `/tmp/uvicorn.log` will tell us exactly when each emission lands.

**Optional follow-up:** apply migration 004 (`page_results.updated_at`) so we can also see from the row itself when it was last touched. Migration is idempotent SQL in `backend/db/migrations/004_page_results_updated_at.sql`.

**Plan after the diagnostic run:** if Phase 3 emissions are landing live, the bug is in the frontend polling effect (likely the `doc`-in-deps causing constant resets); fix is to split the pages poll from the doc poll. If Phase 3 emissions are landing in a batch, the bug is in pipeline (e.g. ThreadPoolExecutor's `as_completed` accumulating before yielding); fix is in pipeline.py.

### 2. Migration 003 + 004 not yet applied in Supabase
Both are idempotent and ship with the code:

```sql
-- 003 (A8 ETA needs this)
alter table jobs
  add column if not exists phase_started_at timestamptz;

-- 004 (live-pages diagnostic; optional)
alter table page_results
  add column if not exists updated_at timestamptz not null default now();
```

Until 003 runs, A8's "within-phase" bar stays empty (overall bar still works because `started_at` already existed).

### 3. Highlighter geometry on rotated pages (FIXED 2026-05-08, needs visual verification)
Worker's `reverse_rotation()` was rectangle-shaped — applied a single (x0,y0,x1,y1) formula treating them as bbox corners. For lines, this scrambled the y-coords of the two endpoints on rotation=90/180/270 pages. Fix in `10146db` adds a per-point `reverse_rotation_point()` and applies it to each endpoint independently. Smoke-test with synthetic horizontal lines on all four rotations confirmed cyan stays centred. **Re-upload a rotated PDF to confirm visually.**

For the **measurement PDF** (the C5 download), `exports.generate_measurement_pdf` *also* uses the same rect-shaped formula on lines and has the same logical bug — but exports.py is shared with prod's flow, so we have a choice: leave it (matches prod's behaviour, even if both are wrong on rotated pages) or fork. Currently leaving it. Flag if the customer cares.

### 4. `app_ade_prod.py` colour-codes lines per category; we draw uniform cyan (stage-2 measurement only — not blocking)
Prod's drawing loop (lines 1774-1808) reads `categories[category]['color']` for each line. Ours uses cyan for all. **This applies to the stage-2 measurement-PDF download, NOT the stage-1 highlighted PDF.** Stage 1 is now prod-equivalent (definitions/instances/keyword_matches) — see "Stage-1 highlighted-PDF parity" commit above. To close stage-2: thread the same auto-assignment that the export builder produces (`_build_auto_export_state` in `api_server.py`) into the highlight worker as a `line_categories` field, then have the worker pick the colour from that map per line. Probably a 30-line addition; not blocking Sprint 4.

### 5. OCR-with-bboxes covers Google DocAI only
The new `ocr_lines_by_page` plumbing wires OCR-line bboxes into the keyword scan, legend extraction, and instance-finding — required for scanned pages whose native PDF text is missing/CID-without-CMap. Today this only fires when `google_cloud_config` is configured (DocAI). If the `<50` native-text fallback path is ever broadened (e.g. Tesseract local OCR), keep the same lines-with-bboxes shape.

---

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

1. **Apply migrations 003 + 004 in Supabase** (one paste in SQL Editor — both blocks above)
2. **Trigger one fresh upload, send the new uvicorn.log** so I can diagnose the live-pages issue from the `_page_cb` log lines
3. **Visually verify highlighter on rotated pages** (delete + re-upload `selected_pages_no_annotations.pdf`)
4. **Decide on per-category line colour** (Open issue #4) — if yes, ~30 lines of work to thread it through
5. **Sprint 4** — UMT (start with iframe-embed for speed; revisit React rebuild post-launch). C8 cross-page summary ships with UMT.
6. **Phase 10.1** — authorization tests
7. **Phase 11** — Vercel + AWS deployment
8. (later) Sprint 5 (operational/observability), S3 migration, Redis/RQ migration

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
