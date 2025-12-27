# 🔍 ADE Fence Detector

**Intelligent fence detection and measurement in engineering/architectural PDF drawings.**

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
