# Leo Fence — Web-app Migration Status

> Snapshot for a fresh-context resumption. Read this first, then the plan at
> `.claude/plans/yes-i-get-you-refactored-hummingbird.md` for full detail.

**Branch:** `feat/web-app-migration` (~25 commits, 30+ files)
**Plan file:** `/home/ubuntu/.claude/plans/yes-i-get-you-refactored-hummingbird.md`
**Stage-1 highlighting plan (now superseded — work below is done):** `/home/ubuntu/.claude/plans/fuck-you-you-are-splendid-beacon.md`
**Decision log:** UMT confirmed **critical-path** (customers always correct measurements manually).
**Last updated:** 2026-05-09 — **Sprint 4 (UMT) complete + non-fence UX polish + Phase 10.1 authorization tests.** Sprint 4: drawing mode (4a.5), zoom + pan + Ctrl/Cmd-wheel zoom-around-cursor (4a.6), C8 cross-page measurement summary (4a.7), Measurement PDF/Excel honor UMT edits (4a.8). UX: per-line popover, layer highlight + per-layer category dropdown, 🎯 smart auto-assign with layer-vote confidence thresholds. Non-fence pages: pipeline now persists classifier reason / confidence / signals; `NonFencePageCard` shows reasoning teaser + LLM signals + on-demand page image; **fixed page-image misalignment bug** that was serving wrong-page from highlighted PDF when page_num didn't match a fence page index. Phase 10.1: `tests/test_authorization.py` (19 tests, 50/50 suite green) — every job-scoped endpoint rejects cross-user requests with 403/404. Sprint-4 plan: `.claude/plans/sprint4-umt-react-canvas.md`. **Next:** Phase 10.2 (upload limits + metadata) or Phase 11 (deploy).

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
| `0ed52a8` | **Stage-1 highlighted-PDF parity with `app_ade_prod.py`.** Five wins, one file (`pipeline.py`) + one frontend cleanup: (1) `definitions` is now `legend_entries` (LLM per-row tight bboxes) instead of raw legend chunks; (2) new instance-finding step calls `find_instances_in_figures_fast` to get per-token indicator bboxes inside figures; (3) keyword scanner now runs against `get_native_pdf_lines` output (real per-line bboxes); (4) `phase1b` OCR step now preserves the per-line bboxes returned by Google DocAI (cache key bumped to `phase1b_v2`) — required for scanned pages where native PDF text is missing or has CID-without-CMap encoding; (5) `keyword_matches` always populated. Frontend: removed the noisy "Detected Instances (bbox)" table from the detail page. |
| `dec5e61` | **test(auth): Phase 10.1 — cross-user authorization tests.** 19 tests, all passing. 13 parametrized cross-user attempts (USER_B requests USER_A's resources) covering every job-scoped endpoint: `GET /api/jobs/{id}` + `/results`, `/highlighted-pdf`, `/page-image/{n}`, `/page-vector-lines/{n}[/smart-assign]`, `/measurement-pdf`, `/measurement-excel`, `/measurement-summary`, `/umt-state`; `PUT` and `DELETE /umt-state/{n}`; `DELETE /api/jobs/{id}`. All return 403/404 (never 200). Plus: unknown `job_id` → 404; no `X-User-Id` (anonymous default) blocked; `/api/jobs` filters by user; `/api/healthz` is public; JWT-only `/api/me` + `/api/documents` reject legacy `X-User-Id`. Uses `FENCE_API_AUTH_MODE=legacy_header` so the suite doesn't need real Supabase JWTs — same `_job_for_user` ownership check fires either way. Postgres-backed `/api/documents*` ownership tested separately when JWT mocking lands. |
| `b5bc4cf` | docs(status): mark Sprint 4 complete + log non-fence UX commits |
| `685d69c` | **fix(page-image): map page_num correctly when serving from highlighted PDF.** The highlighted PDF only contains fence-classified pages (renumbered sequentially). The page-image endpoint was naively passing original-document page_num to the renderer → requesting original page 7 (a non-fence floor plan) returned the *7th fence page in order*, with all its keyword/ADE overlays drawn on it. Now: build `{original_page_num → highlighted-PDF index}` from `results.fence_pages`; `source=auto` renders fence pages from the highlighted PDF at the mapped index and falls back to original PDF for non-fence pages; `source=highlighted` 404s with a clear "not in highlighted PDF" message when the page isn't fence; `source=original` serves the original PDF page-num as-is. Pass resolved 1-indexed `page_in_pdf` to the render subprocess. |
| `351a2f8` | **feat(detail): persist + display non-fence classifier reasoning.** Phase 1c was throwing away the LLM's reasoning before serializing — `non_fence_pages` was just `{page_idx, page_num}`. Now `classification_meta: dict[int, dict]` accumulates during Phase 1c and captures method + reason + confidence + signals (LLM path) / method + keyword_count (keyword path) / method + reason (no_llm + error fallbacks). Cache writes carry the rich payload; cache reads pull whatever's available (old entries stay thin). Both `result.non_fence_pages = [...]` build sites merge in the meta. Frontend `NonFencePage` type gains top-level confidence/signals/keyword_count; `NonFencePageCard` reads from top-level too, shows LLM signals row, prints a hint when neither reason nor method is present. Page-image button on non-fence cards now uses `source=auto` (highlighted has no overlays on non-fence pages anyway, so visually identical to original). |
| `8857b62` | **feat(detail): non-fence pages — load image + reasoning teaser.** `GET /api/jobs/{id}/page-image/{n}` accepts `source` query param: "auto" (default), "original", "highlighted". `PageImage` component takes optional `source` + `altText`. `NonFencePageCard` shows a one-line reasoning teaser inline on the closed summary row (truncates at ~110 chars), full reasoning + keywords-found + page-image button + page text on expand. |
| `b0f2061` | **Sprint 4 4a.8 — Measurement PDF/Excel honor UMT edits.** `_build_export_state(job_id, results, pdf_path)` merges `umt_state.json` per page: for pages with saved line_assignments or user_drawn_lines, recompute the export shape from UMT state (re-extract vector lines from PDF so saved indices resolve to length_pts; drop indicator-code placeholders + orphaned assignments; honor `scale_override`). Auto-only path moved to `_auto_export_for_page` helper. Both `/measurement-pdf` and `/measurement-excel` updated. PDF endpoint refuses (404) when source PDF is reaped — highlighted-PDF fallback would carry cyan overlays into the output and `tobytes(garbage=2, deflate=True)` on the doubly-decorated result hangs for minutes. Excel still works without source. |
| `1e495d0` | **Sprint 4 4a.7 — cross-page measurement summary (C8).** New `GET /api/jobs/{id}/measurement-summary` aggregates per-category totals across all fence pages. For each page: if umt_state has saved edits, recompute totals against saved state (re-extract vector lines from PDF); else use pipeline auto totals (mirrors `_build_auto_export_state` matching with partial-match + Auto-detected fallback). `ft = pts / verified_scale` (default 360, override per-page via `umt_state.scale_override`). New `MeasurementSummary` component on detail page: collapsible card above per-page cards, lazy-fetched. Grand-total table sorted by total ft (category, ft, auto count, manual count, grand-total row), expandable per-page breakdown with one column per category, 🔄 Refresh button. |
| `33bad86` | **Sprint 4 4a.5 + 4a.6 + UMT UX overhaul.** Drawing mode (click-drag on empty stage starts a new line; click on a line still opens its popover so Draw mode is sticky). Zoom slider 25–400% + ✋ Pan mode + Ctrl/Cmd+wheel zoom-around-cursor (anchors scrollLeft/Top to the cursor's PDF point post-zoom). Per-line popover for both vector and user-drawn lines (line idx / layer / length / current category, full category list with color swatches, ✕ Unassign or 🗑 Delete). Layers panel: click a row to highlight all its lines on the canvas in yellow; per-row category dropdown bulk-assigns the whole layer in one pick. 🎯 Smart auto-assign (`/page-vector-lines/{n}/smart-assign`): per-CAD-layer indicator-bbox proximity voting with confidence thresholds (≥3 votes, ≥50% share, ≥5% participation), strict layer-token fallback for layers with no votes. Page rotation applied via `fitz.page.rotation_matrix`. `_clean_legend_entries` filters indicator-code duplicates (kw==ind, desc=="Indicator Code") at the source; frontend `cleanCategoryMap` mirrors it and drops orphaned line_assignments — old saved umt_state self-heals on next save. Highlighted-PDF fallback when /tmp source PDF was reaped (returns source_missing=true; saved assignments still render). Dropped the over-defensive 15k LINE_CAP. |
| `e8956b7` | **Sprint 4 UMT React canvas (checkpoints 4a.1–4a.4) + side fixes.** New: `backend/app/umt_state.py` (per-job JSON persistence with size validation, atomic writes), `frontend/components/UMTCanvas.tsx` + `UMTCanvasInner.tsx` (Konva-based interactive canvas, dynamic-imported as a single client-only module to dodge react-konva named-export resolution issues; pinned `react-konva@^18` for React 18 compat). New API endpoints: `GET/PUT/DELETE /api/jobs/{id}/umt-state[/{page_num}]`, `GET /api/jobs/{id}/page-vector-lines/{n}` (returns ALL display-space vector lines + server-side `auto_categories` and `auto_assignments` mapping pipeline-detected fence lines to vector indices via rounded-coord exact match, with prod's partial-layer-match + "Auto-detected" fallback bucket). Canvas features: category panel (active selection, color swatches, add/delete), click-to-toggle assignment, debounced PUT-save status indicator, "Reassign Auto-detected → \<active>" bulk action, "Reset to auto" button, min-line-pts filter (default 20). Frontend: removed embedded `<iframe>` PDF viewer + dead `pdfBlobUrl` state — download button kept. Side fixes: (1) `_build_auto_export_state` was keying `line_assignments` by `enumerate(all_lines)` index but `exports.py` indexes into the post-filter `auto_lines`; layer-skipped lines silently shifted later lines out-of-bounds. Switched to `len(auto_lines)` (post-append). (2) `page-vector-lines` was swapping `pdf_width`/`pdf_height` for rotation 90/270 — but current PyMuPDF's `page.rect` already reflects display orientation, so the swap mis-sized the stage and lines didn't align with the page image. Removed the swap. (3) `FENCE_API_AUTH_MODE=supabase` set in `.env.local` — was defaulting to `legacy_header`, so JWT-authenticated frontend hits to `get_current_user`-using endpoints fell through to `"anonymous"` and tripped the ownership check. (4) Fixed page-index lookup that treated `0` as falsy. |

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
GET    /api/jobs/{id}/page-vector-lines/{n}      — all vector lines on a page + server-side auto_categories/auto_assignments (Sprint 4); falls back to highlighted PDF if /tmp source is gone (source_missing flag)
GET    /api/jobs/{id}/page-vector-lines/{n}/smart-assign — layer-vote indicator proximity assignment (Sprint 4 4a.5+)
GET    /api/jobs/{id}/umt-state                  — read user's saved UMT edits (Sprint 4)
PUT    /api/jobs/{id}/umt-state/{n}              — upsert one page's UMT state (Sprint 4)
DELETE /api/jobs/{id}/umt-state/{n}              — clear UMT edits for one page (Sprint 4)
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
/documents/[id]     detail page with rich per-page accordions, per-fence-page UMT canvas (Sprint 4), download buttons (no longer embeds the highlighted PDF inline)
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

### 4. ~~`app_ade_prod.py` colour-codes lines per category; we draw uniform cyan~~ — **resolved 2026-05-08**
The stage-2 measurement-PDF generator (`exports.generate_measurement_pdf`) was already wired for per-category colour via `_CATEGORY_PALETTE` in Sprint 3 — note was stale. Side bug found and fixed: `_build_auto_export_state` was keying `line_assignments` by `enumerate(all_lines)` index, but `exports.py` indexes into the post-filter `auto_lines`. Any layer-skipped line silently shifted later lines out of bounds. Switched to `len(auto_lines)` (post-append index).

### 5. OCR-with-bboxes covers Google DocAI only
The new `ocr_lines_by_page` plumbing wires OCR-line bboxes into the keyword scan, legend extraction, and instance-finding — required for scanned pages whose native PDF text is missing/CID-without-CMap. Today this only fires when `google_cloud_config` is configured (DocAI). If the `<50` native-text fallback path is ever broadened (e.g. Tesseract local OCR), keep the same lines-with-bboxes shape.

---

### Sprint 4 — UMT + summary report (IN PROGRESS, option (a) chosen)
Plan refs: C1–C4, **C8**. Detailed checkpoint plan: `.claude/plans/sprint4-umt-react-canvas.md`.

User picked **option (a)** native React canvas over iframe-embed. Done so far:

| Checkpoint | What |
|---|---|
| 4a.1 ✅ | Backend persistence: `backend/app/umt_state.py` + 3 endpoints (GET/PUT/DELETE umt-state). Atomic writes, validated payloads, max 50 cats / 2000 lines per page. |
| 4a.2 ✅ | `GET /api/jobs/{id}/page-vector-lines/{n}` returns lines + server-side `auto_categories` and `auto_assignments` (matches pipeline's `all_fence_lines` to vector indices via rounded-coord exact match; uses prod's partial-layer-match + "Auto-detected" fallback). |
| 4a.3 ✅ | Read-only Konva canvas mounted under each fence-page card. Page image as background, vector lines overlaid in display space. |
| 4a.4 ✅ | Selection wired: category panel (click to set active, add/delete with color swatches), click line to assign/unassign, debounced 500ms PUT, save status indicator, "Reassign Auto-detected → \<active>" bulk action, "Reset to auto", min-line-pts filter. Stage-1 cleanup: removed `<iframe>` PDF viewer (download button kept). |

**Sprint 4 — DONE (2026-05-08 → 09):**

| Checkpoint | Status |
|---|---|
| 4a.5 ✅ | Drawing mode + click-on-line popover both work in Draw mode (commit `33bad86`) |
| 4a.6 ✅ | Zoom slider + pan mode + Ctrl/Cmd+wheel zoom-around-cursor (commit `33bad86`). Manual scale-override numeric input deferred — pipeline's auto-detected scale used for now; add when user requests it. |
| 4a.7 ✅ | C8 cross-page measurement summary (commit `1e495d0`) |
| 4a.8 ✅ | Measurement PDF / Excel honor UMT edits (commit `b0f2061`) |

**Bonus shipped on top of the original plan (commits `33bad86` + `1e495d0` + `b0f2061` + `8857b62` + `351a2f8` + `685d69c`):**
- 🎯 Smart auto-assign endpoint with per-CAD-layer indicator-vote thresholds + page-rotation correction
- Per-line popover (click any vector or drawn line → category list + unassign/delete)
- Layers panel: row-click highlights its lines on canvas; per-row category dropdown bulk-assigns
- Indicator-code de-dup everywhere (canvas chips + Legend Definitions table + smart-assign source)
- Highlighted-PDF fallback when /tmp source PDF was reaped (source_missing flag; saved assignments still align)
- Removed the over-defensive 15k line cap that was emptying canvases on dense pages
- Non-fence pages: classifier reason / confidence / signals now persisted by Phase 1c; reasoning teaser on summary row + on-demand page-image button
- Bug fix (`685d69c`): page-image was returning the wrong page when serving from highlighted PDF (since highlighted only contains fence pages, renumbered) — now maps original page_num through fence_pages position correctly, with clean fallbacks for non-fence and missing-source cases.

### Sprint 5 — operational (low priority)
Plan refs: D1 / D2 / D3.
- D1 — usage / spend display (data already in [spend_tracker.py](spend_tracker.py))
- D2 — telemetry events display (data in [telemetry.py](telemetry.py) JSONL)
- D3 — daily-cap UI (`cfg.MAX_DAILY_SPEND_USD` is enforced silently today)

### Phase 10 — security & robustness
- 10.1 ✅ — `tests/test_authorization.py` covering cross-user 403/404 against all job-scoped endpoints (commit `dec5e61`, 19 tests, 50/50 suite green). Postgres-backed `/api/documents*` ownership tested separately when JWT mocking lands.
- 10.2 ⏳ — file-size + page-count limits already enforced; need cost/perf metadata logged to `jobs` (file_size, duration_ms, model)

### Phase 11 — deployment
- 11.1 — Vercel deploy of `frontend/` (env vars, branch deploys)
- 11.2 — `infra/systemd/fence-api-v2.service` (separate from prod's `fence-api.service`); nginx proxy for `api.<host>` → uvicorn
- 11.3 — Production CORS allowlist + Supabase Auth → URL Configuration with Vercel domain
- (deferred until later) S3 storage migration when going multi-instance — Sprint deferred from original Phase 5; storage_path keys already in S3-shape

### Phase 7 — Redis/RQ
Currently SQLite-based [job_registry.py](job_registry.py) handles queueing. Migration to Redis/RQ deferred until multi-instance deployment. Not blocking launch.

---

## Recommended next steps (in order)

1. **Apply migrations 003 + 004 in Supabase** (one paste in SQL Editor — both blocks above) — still pending
2. **Trigger one fresh upload, send the new uvicorn.log** so I can diagnose the live-pages issue from the `_page_cb` log lines
3. ~~**Sprint 4 remaining checkpoints (4a.7–4a.8)**~~ — DONE 2026-05-08/09. (Manual scale-override numeric input still deferred until asked.)
4. ~~**Phase 10.1** — authorization tests~~ — DONE 2026-05-09 (commit `dec5e61`).
5. **Phase 10.2** — upload limits + cost/perf metadata on `jobs` rows (file_size, duration_ms, model). Data is collected by `telemetry.py` / `spend_tracker.py`; just needs to be wired into the row.
6. **Phase 11** — Vercel + AWS deployment
7. (later) Sprint 5 (operational/observability), S3 migration, Redis/RQ migration

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
