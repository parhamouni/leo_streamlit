# utils.py
import time
import functools
import fitz  # PyMuPDF
import base64
import re
import random
import os
from typing import List, Dict, Tuple, Optional
from io import BytesIO
import json

import pdfplumber
from langchain_core.messages import HumanMessage
from openai import RateLimitError, APIError, APITimeoutError

# Google Cloud Document AI (optional, for OCR via comprehensive extractor)
from google.cloud import documentai_v1 as documentai
from pathlib import Path

# ===== Our prompt/JSON helpers =====
from llm_utils import json_invoke, make_analyze_page_prompt

# ===== Optional comprehensive extractor =====
try:
    from comprehensive_page_extractor import extract_comprehensive_page_text
    COMPREHENSIVE_EXTRACTION_AVAILABLE = True
    print("✅ Comprehensive text extraction available")
except ImportError:
    COMPREHENSIVE_EXTRACTION_AVAILABLE = False
    print("⚠️ Comprehensive text extraction not available - using fallback methods")


# =========================
# Timing Decorator (define BEFORE any @time_it usage)
# =========================
def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # start = time.time()
        result = func(*args, **kwargs)
        # dur = time.time() - start
        # print(f"TIMER LOG: {func.__module__}.{func.__name__} took {dur:.4f}s")
        return result
    return wrapper


# =========================
# Exceptions & Clients
# =========================
class UnrecoverableRateLimitError(Exception):
    """Raised when OpenAI RateLimitError persists after retries."""
    pass


# Legacy global (not used now; config comes from caller)
document_ai_client = None


def create_document_ai_client(google_cloud_config=None):
    """
    Create Document AI client from configuration (Streamlit secrets dict shape).
    google_cloud_config should include service_account_info.
    """
    try:
        if google_cloud_config and google_cloud_config.get("service_account_info"):
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                google_cloud_config["service_account_info"]
            )
            client = documentai.DocumentProcessorServiceClient(credentials=credentials)
            print("✅ Document AI client created from provided configuration")
            return client
        else:
            print("⚠️ No Google Cloud configuration provided.")
            return None
    except Exception as e:
        print(f"⚠️ Document AI client creation failed: {e}")
        return None


# =========================
# Utilities
# =========================
@time_it
def retry_with_backoff(llm_invoke_method, messages_list, retries=5, base_delay=2):
    """
    Retry wrapper for LLM .invoke with simple exponential backoff.
    Pass llm_invoke_method = llm.invoke (callable).
    """
    func_name = getattr(llm_invoke_method, "__qualname__", "llm_invoke")
    for attempt in range(retries):
        try:
            return llm_invoke_method(messages_list)
        except RateLimitError as rle:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (RateLimitError). Last error: {rle}")
                raise UnrecoverableRateLimitError(
                    f"OpenAI API rate limit. Please try later. (Details: {rle})"
                )
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: RateLimitError for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {rle}")
            time.sleep(delay)
        except (APIError, APITimeoutError) as apie:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (APIError/Timeout). Last error: {apie}")
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: API Error/Timeout for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {apie}")
            time.sleep(delay)
        except Exception as e:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (Unexpected). Last error: {e}")
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: Unexpected Error for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Exceeded max retries for {func_name}. Should have been caught earlier.")


@time_it
def extract_snippet(text: str, fence_keywords: List[str]) -> Optional[str]:
    """Return a small snippet around the first fence keyword hit."""
    for kw in fence_keywords:
        match = re.search(rf".{{0,50}}\b{re.escape(kw)}\b.{{0,50}}", text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


# =========================
# Text Extraction (page-level)
# =========================
@time_it
def extract_comprehensive_text_from_page(
    page_bytes: bytes,
    page_number: int = 1,
    google_cloud_config=None,
    ocr_dpi: int = 96,
    enable_tesseract_fallback: bool = False,
) -> Dict:
    """
    Prefer comprehensive extractor (text layer + annotations + GCP OCR).
    Fallback: text layer (+ optional local Tesseract OCR) for recall.
    """
    try:
        doc = fitz.open(stream=page_bytes, filetype="pdf")
        if not doc or len(doc) == 0:
            return {"error": "Could not open PDF page", "text": "", "all_text": [], "stats": {}}
        page = doc[0]

        # Try comprehensive extractor if available and config provided
        if COMPREHENSIVE_EXTRACTION_AVAILABLE and google_cloud_config:
            client = create_document_ai_client(google_cloud_config)
            processor_name = (
                f"projects/{google_cloud_config['project_number']}/"
                f"locations/{google_cloud_config['location']}/"
                f"processors/{google_cloud_config['processor_id']}"
            )
            if client:
                result = extract_comprehensive_page_text(
                    page=page,
                    client=client,
                    processor_name=processor_name,
                    use_ocr=True,
                    dpi=ocr_dpi,
                )
            else:
                result = None
        else:
            result = None

        if result:
            all_text_items = result.get("all_text", [])
            combined_text = " ".join([item["text"] for item in all_text_items]) if all_text_items else ""
            doc.close()
            return {
                "text": combined_text,
                "text_words": result.get("text_words", []),
                "text_annotations": result.get("text_annotations", []),
                "ocr_texts": result.get("ocr_texts", []),
                "all_text": all_text_items,
                "stats": result.get("stats", {}),
                "page_number": page_number,
                "extraction_method": "comprehensive",
            }

        # --------- Fallback path: PDF text layer + optional local OCR ----------
        text_words = page.get_text("words")
        base_text = " ".join([w[4] for w in text_words])

        ocr_added = []
        if enable_tesseract_fallback:
            try:
                import pytesseract
                from PIL import Image

                pix = page.get_pixmap(dpi=ocr_dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text:
                    ocr_added.append({"text": ocr_text, "source": "tesseract"})
                    base_text = (base_text + " " + ocr_text).strip()
            except Exception as _e:
                print(f"Tesseract fallback failed: {_e}")

        doc.close()
        return {
            "text": base_text,
            "text_words": [
                {"text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "source": "text_layer"}
                for w in text_words
            ],
            "text_annotations": [],
            "ocr_texts": ocr_added,
            "all_text": ([{"text": w[4], "source": "text_layer"} for w in text_words] + ocr_added),
            "stats": {
                "total_words": len(text_words),
                "total_elements": len(text_words) + len(ocr_added),
                "ocr_enabled": bool(ocr_added),
            },
            "page_number": page_number,
            "extraction_method": "fallback+tesseract" if ocr_added else "fallback",
        }

    except Exception as e:
        print(f"Error in comprehensive text extraction: {e}")
        return {"error": str(e), "text": "", "all_text": [], "stats": {}, "extraction_method": "error"}


@time_it
def get_comprehensive_text_for_analysis(page_bytes: bytes, google_cloud_config=None) -> str:
    """
    Convenience: return combined text for downstream logic.
    """
    try:
        extraction_result = extract_comprehensive_text_from_page(page_bytes, google_cloud_config=google_cloud_config)
        if extraction_result.get("error"):
            print(f"⚠️ Comprehensive extraction failed: {extraction_result['error']}")
            return ""
        return extraction_result.get("text", "")
    except Exception as e:
        print(f"❌ Error in comprehensive text extraction: {e}")
        return ""


# =========================
# Page Classification (recall-biased, JSON-enforced)
# =========================
@time_it
def analyze_page(
    page_data,
    llm_text,
    fence_keywords,
    google_cloud_config=None,
    ocr_dpi: int = 96,
    enable_tesseract_fallback: bool = False,
    neighbor_text_provider=None,
    recall_mode: str | None = None,   # NEW: optional override; else uses env var
):
    """
    Recall-aware page classification with JSON enforcement + optional neighbor context.
    Decision policy:
      - strict   : only answer=='yes'
      - balanced : answer=='yes' OR (confidence>=0.65 AND has_primary_signal)
      - high     : answer=='yes' OR (confidence>=0.55 AND has_primary_signal)
    """
    import json, os

    pg_num = page_data.get("page_number", "N/A")

    # ---- Extract text (same as before) ----
    page_bytes = page_data.get("page_bytes")
    if page_bytes:
        extraction_result = extract_comprehensive_text_from_page(
            page_bytes, pg_num, google_cloud_config, ocr_dpi=ocr_dpi,
            enable_tesseract_fallback=enable_tesseract_fallback
        )
        comprehensive_text = extraction_result.get("text", "")
        extraction_stats = extraction_result.get("stats", {})
        extraction_method = extraction_result.get("extraction_method", "unknown")
    else:
        comprehensive_text = page_data.get("text", "")
        extraction_stats = {}
        extraction_method = "legacy"

    # Optional neighbor context for sparse pages
    context_bits = []
    if neighbor_text_provider and extraction_stats.get("total_elements", 0) < 40:
        try:
            nbefore = neighbor_text_provider(pg_num - 1)
            nafter = neighbor_text_provider(pg_num + 1)
            if nbefore: context_bits.append(f"PrevPageExcerpt: {nbefore[:300]}")
            if nafter:  context_bits.append(f"NextPageExcerpt: {nafter[:300]}")
        except Exception:
            pass

    context_info = (
        f"Extraction: {extraction_method}, elements={extraction_stats.get('total_elements','n/a')}"
        + ("; " + " | ".join(context_bits) if context_bits else "")
    )
    prompt = make_analyze_page_prompt(context_info=context_info, page_text=comprehensive_text)

    # ---- LLM call ----
    try:
        raw_obj = retry_with_backoff(llm_text.invoke, [HumanMessage(content=prompt)])
        raw = raw_obj.content if hasattr(raw_obj, "content") else str(raw_obj)
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "no")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
        signals = parsed.get("signals", []) or []
        reason = parsed.get("reason", "")
    except Exception as e:
        print(f"analyze_page JSON parse failed pg {pg_num}: {e}")
        answer, confidence, signals, reason = "no", 0.0, [], f"error {e}"

    # ---- Decision policy (fixed) ----
    # Primary fence vocabulary for cross-checking signals
    primary_vocab = {k.lower() for k in (fence_keywords or [])}
    primary_vocab |= {
        "fence","fencing","gate","gates","barrier","guardrail",
        "chain link","chain-link","chainlink","screen wall","privacy screen"
    }
    signals_l = [s.lower() for s in signals if isinstance(s, str)]
    has_primary_signal = any(any(p in s for p in primary_vocab) for s in signals_l)

    mode = (recall_mode or os.getenv("RECALL_MODE", "balanced")).strip().lower()
    if mode == "strict":
        found = (answer == "yes")
    elif mode == "high":
        found = (answer == "yes") or (confidence >= 0.55 and has_primary_signal)
    else:  # balanced (default)
        found = (answer == "yes") or (confidence >= 0.65 and has_primary_signal)

    print(f"[ANALYZE_DEBUG] pg={pg_num} ans={answer} conf={confidence:.2f} "
          f"has_primary_signal={has_primary_signal} mode={mode} -> found={found}")

    snippet = extract_snippet(comprehensive_text, fence_keywords) if found else None

    return {
        "page_number": pg_num,
        "fence_found": found,
        "text_found": (answer == "yes"),
        "text_response": json.dumps({
            "answer": answer,
            "confidence": confidence,
            "signals": signals,
            "reason": reason
        }),
        "text_snippet": snippet,
        "extraction_stats": extraction_stats,
        "extraction_method": extraction_method,
        "comprehensive_text": comprehensive_text,
    }


# --- Token & geometry helpers for precise highlighting ---

def _pymupdf_words_from_page_bytes(page_bytes: bytes) -> List[Dict]:
    """Extract token-level words using PyMuPDF for tighter boxes."""
    try:
        with fitz.open(stream=BytesIO(page_bytes), filetype="pdf") as d:
            w = d[0].get_text("words")  # (x0, y0, x1, y1, text, block_no, line_no, word_no)
        return [
            {
                "text": t[4],
                "x0": float(t[0]),
                "y0": float(t[1]),
                "x1": float(t[2]),
                "y1": float(t[3]),
                "block": int(t[5]),
                "line": int(t[6]),
                "word": int(t[7]),
            }
            for t in w
            if str(t[4]).strip()
        ]
    except Exception as e:
        print(f"_pymupdf_words_from_page_bytes error: {e}")
        return []


def _rect_union(rects: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not rects:
        return None
    xs0 = min(r[0] for r in rects)
    ys0 = min(r[1] for r in rects)
    xs1 = max(r[2] for r in rects)
    ys1 = max(r[3] for r in rects)
    return (xs0, ys0, xs1, ys1)


def _pad_rect(x0, y0, x1, y1, pad: float = 1.5) -> Tuple[float, float, float, float]:
    """Slight visual padding for tiny tokens; keep it small (PDF units)."""
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _center(rect: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = rect
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _point_in_rect(px: float, py: float, rect: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = rect
    return (x0 <= px <= x1) and (y0 <= py <= y1)


def _norm_text(s: str) -> str:
    """Loose normalization for indicator matching."""
    return re.sub(r"[\s\.\-:_\(\)\[\]\{\}]+", "", s or "").upper()

# =========================
# Highlighting helper (unchanged core logic; kept for later steps)
# =========================

def _llm_extract_keynotes(lines_for_llm, llm, fence_keywords_from_app):
    """
    LLM-only parsing of possible KEYNOTES/LEGEND lines.
    Input: [{"id": "...","text":"...", "x0":..., "y0":..., "x1":..., "y1":...}, ...]
    Output: {"keynotes":[{"id":"<line_id>", "code":"0113","text":"MANUAL ROLLING GATES"}...]}
    Only keep items that are likely fence/gate/barrier/guardrail family.
    """
    import json, re
    # Send only the text + id to keep prompt compact
    compact = [{"id": it["id"], "text": it["text"]} for it in lines_for_llm if it.get("text")]
    payload = json.dumps(compact, separators=(",",":"))

    prompt = f"""
You are an engineering drawing analyst. You get a list of text lines from a plan sheet.
Many sheets have a "KEYNOTES" (or similar) list with entries like "0113 MANUAL ROLLING GATES" or "0401 CMU PARKING SCREEN WALL".
Goal: Extract only keynote-style items that are clearly about fences/gates/barriers/guardrails or very close cousins (e.g., CMU SCREEN WALL if used as a barrier).
Return STRICT JSON: {{"keynotes":[{{"id":"<original line id>","code":"<short code>","text":"<description>"}}...]}}
Rules:
- code: short identifier (often 3–5 chars), may include leading zeros or letters (e.g., 0113, 0401, F2A). Do NOT include dimensions or unit strings.
- text: the cleaned description (e.g., "MANUAL ROLLING GATES", "CMU TRUCK COURT SCREEN WALL").
- Only include items that seem fence/gate/barrier/guardrail family (use synonyms: {", ".join(fence_keywords_from_app + ['screen wall','cmu wall','security gate'])}).
- If nothing relevant, return {{"keynotes":[]}} strictly.

Lines:
{payload}
""".strip()

    try:
        resp = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt)])
        raw = resp.content.strip()
        # If the model wrapped in ```json, peel it
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m: raw = m.group(1).strip()
        data = json.loads(raw) if raw.startswith("{") else json.loads(raw[raw.find("{"): raw.rfind("}")+1])
        keynotes = data.get("keynotes", [])
        # Normalize code
        for kn in keynotes:
            if "code" in kn and isinstance(kn["code"], str):
                kn["code"] = kn["code"].strip()
        return keynotes
    except Exception as e:
        print(f"_llm_extract_keynotes failed: {e}")
        return []


def _llm_filter_indicators(candidate_words, llm, legend_context):
    """
    Given candidate short tokens (likely codes) and legend context, ask LLM which are TRUE indicators
    (not dimensions, scales, quantities, page numbers, etc.)
    candidate_words: [{"id":"word_12","text":"0113","x0":...,"y0":...,"x1":...,"y1":...}, ...]
    legend_context: [{"code":"0113","text":"MANUAL ROLLING GATES"}, ...]
    Returns subset of candidate_words with the same fields (whitelisted).
    """
    import json, re
    payload_words = json.dumps(
        [{"id":w["id"],"text":w["text"]} for w in candidate_words],
        separators=(",",":")
    )
    payload_legend = json.dumps(
        [{"code":l["code"],"text":l["text"]} for l in legend_context],
        separators=(",",":")
    )

    prompt = f"""
You get:
1) LEGEND CODES (context): {payload_legend}
2) CANDIDATE TOKENS on the plan (short text): {payload_words}

Task: Return ONLY the candidates that are TRUE indicator labels for one of the legend codes.
- TRUE indicator = a short standalone label (e.g. "0113", "F2A") tagging a drawing element.
- NOT indicators: dimensions (like 10', 59'-6", 1"=40'-0"), scale notes, quantities ("2 BOLTS"), page numbers, dates.

STRICT JSON:
{{"confirmed":[{{"id":"<candidate id>","code":"<matching code>","text":"<candidate text>"}}...]}}
""".strip()

    try:
        resp = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt)])
        raw = resp.content.strip()
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m: raw = m.group(1).strip()
        data = json.loads(raw) if raw.startswith("{") else json.loads(raw[raw.find("{"): raw.rfind("}")+1])
        confirmed = data.get("confirmed", [])
        # Retain geometry by joining back to candidate_words
        idx = {w["id"]: w for w in candidate_words}
        out = []
        for item in confirmed:
            wid = item.get("id")
            if wid in idx:
                w = idx[wid]
                out.append({**w, "matched_code": item.get("code")})
        return out
    except Exception as e:
        print(f"_llm_filter_indicators failed: {e}")
        return []

@time_it
def get_fence_related_text_boxes(
    page_bytes,
    llm,
    fence_keywords_from_app,
    extra_keywords=None,                   # << NEW: pass analyzer signals here
    selected_llm_model_name="gpt-3.5-turbo",
    google_cloud_config=None,              # optional DocAI fallback
):
    """
    v14-hybrid-ocr-signals:
      - Highlights ONLY user keywords + per-page 'signals'.
      - Text-layer path + opportunistic OCR on mixed pages.
      - Token-level boxes with ±context tokens; height clamped.
      - Geometric dedup (IoU) across text-layer & OCR.
    Returns: (boxes, page_width, page_height)
    Each box: {id, text, x0,y0,x1,y1, type_from_llm, tag_from_llm}
    """
    import re, time
    from io import BytesIO
    from statistics import median
    import fitz
    import pdfplumber

    t0 = time.time()
    print("TIMER LOG: (get_fence_related_text_boxes) v14-hybrid-ocr-signals")

    # ---------- Tunables ----------
    # text-layer
    PAD_TEXT = 1.8
    MAX_H_MULT_TEXT = 1.6
    GAP_SPLIT_FACTOR_TEXT = 0.9
    CONTEXT_TOKENS_TEXT = 2

    # OCR
    PAD_OCR = 1.4
    MAX_H_MULT_OCR = 1.6
    GAP_SPLIT_FACTOR_OCR = 0.9
    CONTEXT_TOKENS_OCR = 2

    # OCR “also-run” heuristics (mixed pages)
    OCR_IF_TEXT_MATCHES_LT = 2
    OCR_IF_PYMUPDF_TOKENS_LT = 25
    IMAGE_AREA_RATIO = 0.05

    # Geometric dedup
    IOU_DEDUP_THRESH = 0.85

    # ---------------- helpers ----------------
    def _pad_rect(x0, y0, x1, y1, pad=1.2):
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def _center(r):
        x0, y0, x1, y1 = r
        return ((x0 + x1)/2.0, (y0 + y1)/2.0)

    def _point_in_rect(px, py, r):
        x0, y0, x1, y1 = r
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _canon(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[\u2010-\u2015\-_]+", " ", s)  # hyphen/underscore -> space
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        au = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        bu = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = au + bu - inter
        return (inter / union) if union > 0 else 0.0

    def _dedup_geom(boxes, iou_thresh=IOU_DEDUP_THRESH):
        kept = []
        for b in boxes:
            br = (b["x0"], b["y0"], b["x1"], b["y1"])
            if any(_iou(br, (k["x0"], k["y0"], k["x1"], k["y1"])) >= iou_thresh for k in kept):
                continue
            kept.append(b)
        return kept

    def _pymupdf_tokens_from_bytes(pb: bytes):
        try:
            with fitz.open(stream=BytesIO(pb), filetype="pdf") as d:
                w = d[0].get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
            out = []
            for t in w:
                txt = str(t[4]).strip()
                if not txt: continue
                out.append({
                    "text": txt,
                    "x0": float(t[0]), "y0": float(t[1]),
                    "x1": float(t[2]), "y1": float(t[3]),
                    "block": int(t[5]), "line": int(t[6]), "word": int(t[7]),
                })
            return out
        except Exception as e:
            print(f"_pymupdf_tokens_from_bytes error: {e}")
            return []

    def _page_image_area_ratio(pb: bytes):
        try:
            with fitz.open(stream=BytesIO(pb), filetype="pdf") as d:
                page = d[0]
                info = page.get_text("rawdict")
                page_area = page.rect.width * page.rect.height
                img_area = 0.0
                for b in info.get("blocks", []):
                    if b.get("type") == 1:
                        x0, y0, x1, y1 = b["bbox"]
                        img_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
                return (img_area / page_area) if page_area > 0 else 0.0
        except Exception as e:
            print(f"_page_image_area_ratio error: {e}")
            return 0.0

    def _docai_tokens_and_lines(pb: bytes, cfg, dpi: int = 72):
        """Returns (lines, tokens, page_w, page_h)."""
        if not cfg: return [], [], 0.0, 0.0
        try:
            client = create_document_ai_client(cfg)
            if not client: return [], [], 0.0, 0.0

            with fitz.open(stream=BytesIO(pb), filetype="pdf") as d:
                page = d[0]
                pix = page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("png")
                page_w, page_h = float(page.rect.width), float(page.rect.height)
                sx, sy = page_w / float(pix.width), page_h / float(pix.height)

            from google.cloud import documentai_v1 as documentai
            raw_document = documentai.RawDocument(content=img_data, mime_type="image/png")
            name = f"projects/{cfg['project_number']}/locations/{cfg['location']}/processors/{cfg['processor_id']}"
            req = documentai.ProcessRequest(name=name, raw_document=raw_document)
            res = client.process_document(request=req)
            doc = res.document

            tokens = []
            if doc.pages:
                pg = doc.pages[0]
                full = doc.text or ""
                for tok in getattr(pg, "tokens", []):
                    text_content = ""
                    ta = tok.layout.text_anchor
                    if ta:
                        if getattr(ta, "text_segments", None):
                            seg = ta.text_segments[0]
                            text_content = full[seg.start_index:seg.end_index]
                        elif getattr(ta, "content", None):
                            text_content = ta.content
                    txt = (text_content or "").strip()
                    if not txt: continue
                    v = tok.layout.bounding_poly.vertices
                    if len(v) >= 4:
                        x0, y0 = float(v[0].x) * sx, float(v[0].y) * sy
                        x1, y1 = float(v[2].x) * sx, float(v[2].y) * sy
                        tokens.append({"text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

            if not tokens: return [], [], page_w, page_h

            # sort & group into lines with robust tolerances
            tokens.sort(key=lambda t: ((t["y0"] + t["y1"]) / 2.0, t["x0"]))
            heights = [t["y1"] - t["y0"] for t in tokens]
            widths  = [t["x1"] - t["x0"] for t in tokens]
            med_h = sorted(heights)[len(heights)//2] if heights else 8.0
            med_w = sorted(widths )[len(widths )//2] if widths  else 8.0

            y_tol     = max(6.0, med_h * 0.6)
            x_gap_tol = max(8.0, med_w * 3.0, page_w * 0.02)

            lines, cur = [], []
            prev_cy = prev_x1 = None

            def flush(group):
                if not group: return
                x0 = min(t["x0"] for t in group); y0 = min(t["y0"] for t in group)
                x1 = max(t["x1"] for t in group); y1 = max(t["y1"] for t in group)
                text = " ".join(t["text"] for t in sorted(group, key=lambda tt: tt["x0"]))
                lines.append({"text": text, "x0": x0, "top": y0, "x1": x1, "bottom": y1})

            for t in tokens:
                cy = (t["y0"] + t["y1"]) / 2.0
                if (prev_cy is None) or (abs(cy - prev_cy) <= y_tol and prev_x1 is not None and (t["x0"] - prev_x1) <= x_gap_tol):
                    cur.append(t)
                else:
                    flush(cur); cur = [t]
                prev_cy = cy; prev_x1 = t["x1"]
            flush(cur)

            return lines, tokens, page_w, page_h
        except Exception as e:
            print(f"[HL DEBUG] DocAI fallback failed: {e}")
            return [], [], 0.0, 0.0

    # --- keyword preparation (user + signals) ---
    def _augment_keywords(raw_list):
        """Add hyphen/space variants and singular/plural for multi-word phrases."""
        out = set()
        for s in raw_list:
            s = (s or "").strip()
            if not s: continue
            base = s
            out.add(base)
            # hyphen/space variants
            h2s = base.replace("-", " ")
            s2h = base.replace(" ", "-")
            out.add(h2s); out.add(s2h)
            # singular/plural if multi-word
            toks = base.split()
            if len(toks) >= 2:
                last = toks[-1]
                if last.endswith("s"):
                    sing = " ".join(toks[:-1] + [last[:-1]])
                    out.add(sing); out.add(sing.replace(" ", "-"))
                else:
                    plur = " ".join(toks[:-1] + [last + "s"])
                    out.add(plur); out.add(plur.replace(" ", "-"))
            # collapsed (“screenwall”) variant for two-word phrases
            if len(base.split()) == 2:
                out.add(base.replace(" ", ""))
        return sorted(set(out))

    raw_user = [k for k in (fence_keywords_from_app or []) if isinstance(k, str)]
    raw_signals = [k for k in (extra_keywords or []) if isinstance(k, str)]
    # normalize to canonical form used by matcher
    merged_raw = list({k.strip() for k in (raw_user + raw_signals) if k and k.strip()})
    # create variants, then canonicalize for the matcher
    expanded = _augment_keywords([_canon(k) for k in merged_raw])
    norm_kws = list({_canon(k) for k in expanded})
    phrase_lists = [k.split() for k in norm_kws if " " in k]
    single_set   = {k for k in norm_kws if " " not in k}

    def _find_spans(line_tokens_norm, phrase_lists, single_set):
        spans = []
        used = [False] * len(line_tokens_norm)
        # phrases (longest-first)
        for P in sorted(phrase_lists, key=lambda p: -len(p)):
            L = len(P); i = 0
            while i <= len(line_tokens_norm) - L:
                if line_tokens_norm[i:i+L] == P:
                    spans.append((i, i+L-1))
                    for k in range(i, i+L): used[k] = True
                    i += L
                else: i += 1
        # singles
        for i, tok in enumerate(line_tokens_norm):
            if not used[i] and tok in single_set:
                spans.append((i, i))
        spans.sort(key=lambda s: s[0])
        return spans

    def _expand_spans_with_context(spans, n_tokens, max_idx):
        return [(max(0, a - n_tokens), min(max_idx, b + n_tokens)) for a, b in spans]

    def _tight_boxes_from_spans(tokens_line, spans, pad_px, max_height_mult, gap_split_factor):
        if not spans: return []
        h_med = median([max(1.0, t["y1"] - t["y0"]) for t in tokens_line]) if tokens_line else 8.0
        gap_thresh = gap_split_factor * h_med
        out = []
        for a, b in spans:
            segs, start = [], a
            for j in range(a, b):
                gap = tokens_line[j+1]["x0"] - tokens_line[j]["x1"]
                if gap > gap_thresh:
                    segs.append((start, j)); start = j + 1
            segs.append((start, b))
            for sa, sb in segs:
                seg = tokens_line[sa:sb+1]
                x0 = min(t["x0"] for t in seg); x1 = max(t["x1"] for t in seg)
                ty0 = min(t["y0"] for t in seg); ty1 = max(t["y1"] for t in seg)
                height = min((ty1 - ty0), max_height_mult * h_med)
                cy = 0.5 * (ty0 + ty1)
                y0 = cy - 0.5 * height; y1 = cy + 0.5 * height
                x0, y0, x1, y1 = _pad_rect(x0, y0, x1, y1, pad=pad_px)
                out.append((x0, y0, x1, y1))
        return out

    def _should_run_ocr_also(text_layer_boxes, tokens_text, pb: bytes):
        if len(text_layer_boxes) < OCR_IF_TEXT_MATCHES_LT: return True
        if len(tokens_text)        < OCR_IF_PYMUPDF_TOKENS_LT: return True
        try:
            if _page_image_area_ratio(pb) >= IMAGE_AREA_RATIO: return True
        except Exception:
            pass
        return False

    # ---------------- page prep ----------------
    final_boxes = []
    page_width = page_height = 0

    tokens_text = _pymupdf_tokens_from_bytes(page_bytes)

    # pdfplumber objects (text-layer path)
    words_pl, lines_pl = [], []
    try:
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            if not pdf.pages: return [], 0, 0
            p = pdf.pages[0]
            page_width, page_height = float(p.width), float(p.height)
            try:
                words_pl = p.extract_words(
                    use_text_flow=True, split_at_punctuation=False,
                    x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False
                )
            except Exception as e:
                print(f"TIMER LOG: extract_words error: {e}")
            try:
                lines_pl = p.extract_text_lines(
                    layout=True, use_text_flow=True, strip=True, return_chars=False
                )
            except Exception as e:
                print(f"TIMER LOG: extract_text_lines error: {e}")
    except Exception as e:
        print(f"TIMER LOG: pdfplumber open error: {e}")
        return [], 0, 0

    # ---- (A) TEXT-LAYER keyword spans ----
    # map tokens to each line by center-in-rect
    line_tokens_map = {}
    for i, L in enumerate(lines_pl or []):
        lr = (float(L.get("x0", 0)), float(L.get("top", 0)), float(L.get("x1", 0)), float(L.get("bottom", 0)))
        ltoks = []
        for t in tokens_text:
            cx, cy = _center((t["x0"], t["y0"], t["x1"], t["y1"]))
            if _point_in_rect(cx, cy, lr):
                ltoks.append(t)
        ltoks.sort(key=lambda tt: tt["x0"])
        line_tokens_map[i] = ltoks

    for i, L in enumerate(lines_pl or []):
        txt = (L.get("text") or "").strip()
        if not txt: continue
        ltoks = line_tokens_map.get(i, [])
        if not ltoks: continue
        ln_norm = [_canon(t["text"]) for t in ltoks]
        spans = _find_spans(ln_norm, [pl for pl in phrase_lists], single_set)
        if not spans: continue
        spans = _expand_spans_with_context(spans, CONTEXT_TOKENS_TEXT, len(ltoks)-1)
        for j, (x0, y0, x1, y1) in enumerate(
            _tight_boxes_from_spans(ltoks, spans, PAD_TEXT, MAX_H_MULT_TEXT, GAP_SPLIT_FACTOR_TEXT)
        ):
            final_boxes.append({
                "id": f"kw_text_{i}_{j}",
                "text": txt,
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "type_from_llm": "keyword_line",
                "tag_from_llm": "KEYWORD_TEXT"
            })

    # Decide if we should ALSO run OCR (mixed page)
    run_ocr_anyway = _should_run_ocr_also(final_boxes, tokens_text, page_bytes)

    # ---- (B) OCR path ----
    if run_ocr_anyway or not final_boxes:
        if not final_boxes and not run_ocr_anyway:
            print("[HL DEBUG] No text-layer matches — using DocAI OCR fallback.")
        else:
            print("[HL DEBUG] Running OCR in addition to text-layer (mixed page).")

        ocr_lines, ocr_tokens, pw, ph = _docai_tokens_and_lines(page_bytes, google_cloud_config, dpi=72)
        if pw and ph: page_width, page_height = pw, ph

        # tokens per OCR line
        ocr_line_tokens = {}
        for i, L in enumerate(ocr_lines):
            lr = (float(L["x0"]), float(L["top"]), float(L["x1"]), float(L["bottom"]))
            ltoks = []
            for t in ocr_tokens:
                cx, cy = _center((t["x0"], t["y0"], t["x1"], t["y1"]))
                if _point_in_rect(cx, cy, lr):
                    ltoks.append(t)
            ltoks.sort(key=lambda tt: tt["x0"])
            ocr_line_tokens[i] = ltoks

        # keyword spans on OCR lines
        ocr_boxes = []
        for i, L in enumerate(ocr_lines):
            txt = (L.get("text") or "").strip()
            if not txt: continue
            ltoks = ocr_line_tokens.get(i, [])
            if not ltoks: continue
            ln_norm = [_canon(t["text"]) for t in ltoks]
            spans = _find_spans(ln_norm, [pl for pl in phrase_lists], single_set)
            if not spans: continue
            spans = _expand_spans_with_context(spans, CONTEXT_TOKENS_OCR, len(ltoks)-1)
            for j, (x0, y0, x1, y1) in enumerate(
                _tight_boxes_from_spans(ltoks, spans, PAD_OCR, MAX_H_MULT_OCR, GAP_SPLIT_FACTOR_OCR)
            ):
                ocr_boxes.append({
                    "id": f"kw_ocr_{i}_{j}",
                    "text": txt,
                    "x0": round(x0, 2), "y0": round(y0, 2),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "type_from_llm": "keyword_line",
                    "tag_from_llm": "KEYWORD_OCR"
                })

        if ocr_boxes:
            final_boxes = _dedup_geom(final_boxes + ocr_boxes, iou_thresh=IOU_DEDUP_THRESH)

    # stable IDs & debug
    for idx, it in enumerate(final_boxes):
        it["id"] = it.get("id") or f"kw_{idx+1}"

    print(f"[HL DEBUG] final_boxes={len(final_boxes)}  page=({page_width}x{page_height})")
    print(f"TIMER LOG: v14-hybrid-ocr-signals done. Boxes={len(final_boxes)} in {time.time() - t0:.3f}s")

    return final_boxes, page_width, page_height

# =========================
# Diagnostic helper
# =========================
def test_comprehensive_extraction(google_cloud_config=None):
    """
    Quick sanity test for Document AI configuration.
    """
    print("🧪 Testing comprehensive text extraction integration...")
    if not COMPREHENSIVE_EXTRACTION_AVAILABLE:
        print("❌ Comprehensive extraction not available")
        return False

    if google_cloud_config:
        test_client = create_document_ai_client(google_cloud_config)
        if test_client:
            print("✅ Document AI client created successfully from configuration")
            processor_name = (
                f"projects/{google_cloud_config['project_number']}/"
                f"locations/{google_cloud_config['location']}/"
                f"processors/{google_cloud_config['processor_id']}"
            )
            print(f"📄 Processor: {processor_name}")
            return True
        else:
            print("❌ Could not create Document AI client from configuration")
            return False
    elif document_ai_client:
        print("✅ Legacy Document AI client available (but configuration should be passed explicitly)")
        return True
    else:
        print("❌ No Document AI client available")
        return False


if __name__ == "__main__":
    test_comprehensive_extraction()
