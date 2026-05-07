# ADE Fence Detector

**Detect, classify, and measure fences in engineering / architectural PDF drawings.**

A B2B tool that ingests construction PDFs, identifies fence-related pages, extracts legend definitions, locates fence indicators on drawings, and calculates fence lengths from the underlying CAD geometry.

---

## Git workflow

- **`main`** is the single source of truth. Protected — changes land via PR.
- **Feature work** branches off `main` as `feat/<short-name>` (e.g. `feat/utils-ade-split`, `feat/cli`) and merges back via PR.
- **Hotfixes for `app_ade_prod.py`** (the live production app served by `fence-fast.service`) branch off the `prod-2026-05-07` tag as `hotfix/prod-<date>` and merge back to `main`. Do not edit `app_ade_prod.py` outside of this lane.
- **Tags worth knowing**:
  - `prod-2026-05-07` — exact source of `app_ade_prod.py` + `ops/analysis_worker.py` as deployed on 2026-05-07.
  - `archive/main-monolith-2026-05-07` — frozen reference to the old `app.py` monolith line of work (replaced by current `main`).
  - `archive/ocr-improvement` — frozen reference to a defunct OCR experiment branch.

There are no long-lived branches besides `main` and `prod-snapshot/<date>`.

---

## Repository layout

```
leo_streamlit/
├── app_ade_prod.py          # LIVE prod Streamlit app (port 8502)  ── do not edit ──
├── app_ade_fast.py          # Next-gen Streamlit frontend (talks to api_server.py)
├── api_server.py            # FastAPI backend for the fast stack (port 8503)
├── pipeline.py              # Streamlit-free analysis engine (Phase 1a → 3)
├── ops/
│   ├── analysis_worker.py    #   Phase 1 background worker (used by api_server)
│   ├── phase3_worker.py      #   Phase 3 subprocess (memory isolation)
│   ├── highlight_pdf_worker.py
│   ├── page_extractor.py
│   ├── systemd/              #   Production unit files
│   └── watchdog/             #   Health probes
├── utils_ade.py             # Core detection: ADE API, OCR, LLM, highlighting (3.1 KLOC)
├── utils_vector.py          # Vector line extraction, scale inference, measurement
├── umt.py                   # Unified Measurement Tool (interactive UI for Phase 3b)
├── job_registry.py          # SQLite-backed persistent job queue
├── fence_cache.py           # Disk cache for intermediate analysis phases
├── auth.py                  # Pluggable auth (none / OIDC / proxy / password)
├── config.py                # Central config (env-var-driven dataclass)
├── state.py                 # Typed Streamlit session state for app_ade_fast.py
├── exports.py               # PDF + Excel report generation
├── telemetry.py             # @timed / @checkpoint decorators → JSONL
├── spend_tracker.py         # Per-user API cost tracking → JSONL
├── tools/                   # Operator tools (telemetry_report, telemetry_watch)
├── notebooks/               # Development & analysis notebooks
├── subset_gold/             # Test PDF fixtures
├── archive/                 # Historical reference (predecessors, evaluation harnesses, debug)
├── DEPLOY.md                # Deployment + nginx + systemd setup
└── requirements.txt
```

`archive/` contains:
- `predecessors/` — `app_ade.py` (older monolith), `app_ade_fast_new.py`, `app_ade_fast_monolith.py` (pre-FastAPI-split backup)
- `evaluation/` — `fence_evaluator.py`, `fence_detection_comparison.py`, `fence_detector_agentic.py` (research harnesses)
- `debug/` — `debug_pipeline.py`, `debug_deep.py`, `debug_instances.py`
- `ade_backup/` — older ADE notebooks and experiments
- `old_versions/`, `debug_scripts/` — even earlier reference material

---

## Two stacks running side-by-side

| Stack | Status | Process | Port | Talks to API? |
|---|---|---|---|---|
| **Prod** | Live for users | `app_ade_prod.py` (Streamlit monolith) | 8502 | No — analysis runs inline |
| **Fast** | Backend live, frontend not yet cut over | `api_server.py` (FastAPI) + `app_ade_fast.py` (thin Streamlit) | 8503 | Yes — frontend submits jobs |

**Goal:** add features to `app_ade_fast.py` until it has parity with `app_ade_prod.py`, then swap `fence-fast.service`'s `ExecStart=` from `app_ade_prod.py` to `app_ade_fast.py`. See [plans/how-to-refactor-and-groovy-mist.md](.claude/plans/how-to-refactor-and-groovy-mist.md) for the migration sequence.

---

## What the system does

1. **Pre-filter**: scan each PDF page for fence keywords (`fence`, `gate`, `guardrail`, …) with word-boundary matching to avoid false positives like "aggregate".
2. **OCR + native text**: extract text from PDFs using both PyMuPDF (native) and Google Document AI (scanned). Coordinate transforms handle rotated pages.
3. **LLM classification**: ambiguous pages go to GPT for confirmation.
4. **ADE structured extraction**: fence pages are sent to LandingAI's ADE API; chunks are split into legend vs. figure regions.
5. **Legend extraction**: an LLM extracts indicator → description pairs from legend chunks.
6. **Instance detection**: indicators are located in figure regions, excluding legend areas.
7. **Measurement**: vector lines are extracted from the PDF; an LLM identifies fence-related CAD layers; scale is inferred from text or a scale bar; lengths are calculated per indicator.
8. **Visualization**: highlighted PDF + Excel report.

Output highlight colors:
| Element | Color |
|---|---|
| Definitions (legend entries) | Green |
| Instances (indicators in drawings) | Purple |
| Keyword matches (fallback) | Orange |
| Measured fence lines | Cyan |

---

## Local development

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Copy + fill in API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Run the prod monolith locally on a non-prod port
./venv/bin/streamlit run app_ade_prod.py --server.port 8510

# Or run the fast stack locally
./venv/bin/uvicorn api_server:app --port 8513 &
./venv/bin/streamlit run app_ade_fast.py --server.port 8511
```

For deployment (nginx + systemd), see [DEPLOY.md](DEPLOY.md).

---

## Required secrets (`.streamlit/secrets.toml`)

```toml
OPENAI_API_KEY = "sk-..."
LANDINGAI_API_KEY = "land_sk_..."

[google_cloud]
project_number = "123456789"
location = "us"
processor_id = "abc123..."

[gcp_service_account]
type = "service_account"
# ... full service account JSON

[auth]
# Only required for FENCE_AUTH_MODE=streamlit_oidc
client_id = "..."
client_secret = "..."
redirect_uri = "https://your-host/oauth2callback"
```

---

## License

Internal use only.
