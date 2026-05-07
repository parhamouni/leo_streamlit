# 🔍 ADE Fence Detector

**Intelligent fence detection and measurement in engineering/architectural PDF drawings.**

---

## Git workflow

- **`main`** is the single source of truth. It is protected — direct pushes are not allowed; changes land via PR.
- **Feature work** branches off `main` as `feat/<short-name>` (e.g. `feat/utils-ade-split`, `feat/cli`) and merges back via PR.
- **Hotfixes for `app_ade_prod.py`** (the live production app served by `fence-fast.service`) branch off the `prod-2026-05-07` tag as `hotfix/prod-<date>` and merge back to `main`. Do not touch `app_ade_prod.py` outside of this lane.
- **Tags worth knowing**:
  - `prod-2026-05-07` — exact source of `app_ade_prod.py` + `ops/analysis_worker.py` as deployed on 2026-05-07.
  - `archive/main-monolith-2026-05-07` — frozen reference to the old `app.py` monolith line of work (replaced by current `main`).
  - `archive/ocr-improvement` — frozen reference to a defunct OCR experiment branch.
- **Long-lived branch policy**: there are no long-lived branches besides `main` and `prod-snapshot/<date>`. Old `working-backup`, `fast-clean`, `claude/*`, etc. were retired in the 2026-05-07 cleanup.

> Note: this README's **Repository Structure** and **Key Files** sections below still reflect the old monolith layout. They are scheduled for rewrite in Phase 3 of the refactor (see `/home/ubuntu/.claude/plans/how-to-refactor-and-groovy-mist.md`).

---

## 📁 Repository Structure (After Cleanup)

```
leo_streamlit/
├── app_ade.py          # Main Streamlit application
├── utils_ade.py        # Core detection utilities
├── utils_vector.py     # Vector line extraction & measurement
├── requirements.txt    # Python dependencies
├── .streamlit/         # Streamlit configuration & secrets
├── notebooks/          # Development & analysis notebooks
├── subset_gold/        # Test PDF files
├── ade_backup/         # Historical backups & experiments
└── archive/            # Archived old versions & temp outputs
```

---

## 🚀 System Capabilities

### 1. **Multi-Layer Text Extraction**
- **PDF Native Text**: Extracts embedded text with precise bounding boxes
- **Google Document AI OCR**: For scanned/image-based content
- **Coordinate Transformation**: Handles rotated pages (0°, 90°, 180°, 270°)

### 2. **Intelligent Pre-Filtering**
- **Keyword Scanning**: Word-boundary matching to avoid false positives (e.g., "gate" not in "aggregate")
- **High-Signal Keywords**: Immediate detection for definitive terms (fence, gate, guardrail, etc.)
- **LLM Confirmation**: Optional GPT validation for ambiguous pages

### 3. **ADE (LandingAI) Structured Extraction**
- **Document Parsing**: Sends fence-related pages to ADE API
- **Chunk Segmentation**: Separates legend chunks from figure/drawing chunks
- **Legend Entry Extraction**: LLM-powered extraction of indicator→description pairs

### 4. **Instance Detection**
- **Indicator Matching**: Finds legend indicators (e.g., "1", "F-3") in drawing areas
- **Figure-Constrained Search**: Only searches within ADE-detected figure regions
- **Duplicate Filtering**: Excludes matches in legend areas

### 5. **Visual Highlighting**
| Element | Color | Description |
|---------|-------|-------------|
| Definitions | 🟢 Green | Legend entries with fence keywords |
| Instances | 🟣 Purple | Indicators found in drawings |
| Keywords | 🟠 Orange | Fallback keyword matches |
| Fence Lines | 🔵 Cyan | Measured vector lines |

### 6. **Smart Fence Measurement** (Experimental)
- **Layer Detection**: LLM identifies fence-related CAD layers
- **Vector Extraction**: Extracts line segments from PDF drawings
- **Scale Inference**: Auto-detects drawing scale from annotations
- **Connected-Line Grouping**: Groups continuous fence runs
- **Per-Indicator Measurement**: Associates lengths with specific fence types

---

## 🔧 Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF Upload                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Text Extraction                                        │
│  • PDF native text lines                                        │
│  • Google OCR (if configured)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Pre-Filter (Keyword + LLM)                             │
│  • Scan for fence keywords                                      │
│  • High-signal → immediate pass                                 │
│  • Low-signal → LLM confirmation                                │
│  • No keywords → skip page                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              fence_found=True    fence_found=False
                    │                   │
                    ▼                   ▼
┌──────────────────────────┐    ┌──────────────────┐
│  STEP 3: ADE Parsing     │    │  Non-Fence Page  │
│  • Send to LandingAI API │    │  (Skip)          │
│  • Get structured chunks │    └──────────────────┘
└──────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Chunk Processing                                       │
│  • Segment: legend vs figure chunks                             │
│  • Extract definitions from legends (LLM)                       │
│  • Find instances in figures                                    │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Measurement (Optional)                                 │
│  • Identify fence layers (LLM)                                  │
│  • Extract vector lines                                         │
│  • Calculate lengths per indicator                              │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Visualization & Output                                 │
│  • Highlight definitions (green)                                │
│  • Highlight instances (purple)                                 │
│  • Generate downloadable PDF                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Required Secrets (`.streamlit/secrets.toml`)

```toml
OPENAI_API_KEY = "sk-..."
LANDINGAI_API_KEY = "land_sk_..."

[google_cloud]
project_number = "123456789"
location = "us"
processor_id = "abc123..."

[gcp_service_account]
type = "service_account"
project_id = "your-project"
# ... full service account JSON
```

### Sidebar Options
- **🔍 Highlight text & indicators**: Toggle visual highlighting
- **🧠 Use ADE (LandingAI)**: Enable/disable ADE API calls
- **Fence Keywords**: Customizable keyword list

---

## 📊 Output Formats

1. **Live UI Display**: Real-time page-by-page results with expandable details
2. **Highlighted PDF Download**: Combined PDF with only fence-related pages
3. **Per-Page Analysis**: Definitions, instances, keyword matches, measurements

---

## 🛠️ Development

### Run the App
```bash
streamlit run app_ade.py
```

### Key Files
| File | Purpose |
|------|---------|
| `app_ade.py` | Streamlit UI, session management, display logic |
| `utils_ade.py` | ADE API, OCR, text extraction, LLM prompts, highlighting |
| `utils_vector.py` | PDF vector extraction, line measurement, scale inference |

---

## 📝 Version History

- **v2.0** (Current): Pre-filtering, ADE integration, smart measurement
- **v1.0**: Basic keyword detection with LLM classification

---

## 📄 License

Internal use only.
