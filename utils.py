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
    page_data: Dict,
    llm_text,
    fence_keywords: List[str],
    google_cloud_config=None,
    ocr_dpi: int = 96,
    enable_tesseract_fallback: bool = False,
    neighbor_text_provider=None,
) -> Dict:
    """
    Recall-biased classification. Uses json_invoke + safe prompt builder.
    Accepts "yes" or lower-confidence positive as True if confidence >= 0.40.
    """
    pg_num = page_data.get("page_number", "N/A")

    # Extract text (page bytes preferred)
    page_bytes = page_data.get("page_bytes")
    if page_bytes:
        extraction_result = extract_comprehensive_text_from_page(
            page_bytes,
            pg_num,
            google_cloud_config,
            ocr_dpi=ocr_dpi,
            enable_tesseract_fallback=enable_tesseract_fallback,
        )
        comprehensive_text = extraction_result.get("text", "")
        extraction_stats = extraction_result.get("stats", {})
        extraction_method = extraction_result.get("extraction_method", "unknown")
    else:
        comprehensive_text = page_data.get("text", "")
        extraction_stats = {}
        extraction_method = "legacy"

    # Neighbor context (±1 page) if sparse
    context_bits = []
    if neighbor_text_provider and extraction_stats.get("total_elements", 0) < 40:
        try:
            nbefore = neighbor_text_provider(pg_num - 1)
            nafter = neighbor_text_provider(pg_num + 1)
            if nbefore:
                context_bits.append(f"PrevPageExcerpt: {nbefore[:300]}")
            if nafter:
                context_bits.append(f"NextPageExcerpt: {nafter[:300]}")
        except Exception:
            pass

    context_info = f"Extraction: {extraction_method}, elements={extraction_stats.get('total_elements', 'n/a')}" + (
        "; " + " | ".join(context_bits) if context_bits else ""
    )

    prompt = make_analyze_page_prompt(context_info=context_info, page_text=comprehensive_text)

    try:
        # Prefer JSON-enforced invocation
        raw_obj = json_invoke(llm_text, [HumanMessage(content=prompt)])
        raw = raw_obj.content if hasattr(raw_obj, "content") else str(raw_obj)
        parsed_json = json.loads(raw)
        answer = parsed_json.get("answer", "no").lower()
        confidence = float(parsed_json.get("confidence", 0.0))
        signals = parsed_json.get("signals", [])
        reason = parsed_json.get("reason", "")
    except Exception as e:
        print(f"analyze_page JSON parse failed pg {pg_num}: {e}")
        answer, confidence, signals, reason = "no", 0.0, [], f"error {e}"

    found = (answer == "yes") or (float(confidence) >= 0.40)
    snippet = extract_snippet(comprehensive_text, fence_keywords) if found else None

    return {
        "page_number": pg_num,
        "fence_found": found,
        "text_found": answer == "yes",
        "text_response": json.dumps(
            {"answer": answer, "confidence": confidence, "signals": signals, "reason": reason}
        ),
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
    selected_llm_model_name="gpt-3.5-turbo",
    google_cloud_config=None,  # <-- NEW (optional) for DocAI fallback
):
    """
    v14 – keywords/legend ALWAYS highlighted; indicators LLM-filtered;
    + Document AI OCR fallback when the text layer is empty.

    Returns: (boxes, page_width, page_height)
    Each box: {id, text, x0,y0,x1,y1, type_from_llm, tag_from_llm}
    """
    import json, re, time
    from io import BytesIO

    t0 = time.time()
    print("TIMER LOG: (get_fence_related_text_boxes) v14 keywords-first + LLM-filtered indicators + DocAI OCR fallback")

    # ------------- helpers -------------
    def _pymupdf_tokens_from_bytes(pb: bytes):
        try:
            with fitz.open(stream=BytesIO(pb), filetype="pdf") as d:
                w = d[0].get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
            out = []
            for t in w:
                txt = str(t[4]).strip()
                if not txt:
                    continue
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

    def _rect_union(rects):
        if not rects: return None
        x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
        x1 = max(r[2] for r in rects); y1 = max(r[3] for r in rects)
        return (x0, y0, x1, y1)

    def _pad_rect(x0, y0, x1, y1, pad=1.5):
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def _center(rect):
        x0, y0, x1, y1 = rect
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def _point_in_rect(px, py, rect):
        x0, y0, x1, y1 = rect
        return (x0 <= px <= x1) and (y0 <= py <= y1)

    def _norm_text(s: str) -> str:
        return re.sub(r"[\s\.\-:_\(\)\[\]\{\}]+", "", (s or "")).upper()

    def _line_token_union(line_rect, tokens):
        tx = []
        for t in tokens:
            cx, cy = _center((t["x0"], t["y0"], t["x1"], t["y1"]))
            if _point_in_rect(cx, cy, line_rect):
                tx.append((t["x0"], t["y0"], t["x1"], t["y1"]))
        return _rect_union(tx) or line_rect

    # --- keynote detection (LLM) ---
    def _llm_extract_keynotes(lines_for_llm, llm_obj, fence_kws):
        compact = [{"id": it["id"], "text": it["text"]} for it in lines_for_llm if it.get("text")]
        payload = json.dumps(compact, separators=(",", ":"))
        prompt = f"""
You are an engineering drawing analyst. You receive text lines from a plan sheet.
Extract ONLY keynote-style items that are fences/gates/guardrails/barriers or close (e.g., SCREEN WALL used as a barrier).
STRICT JSON: {{"keynotes":[{{"id":"<original line id>","code":"<short code>","text":"<description>"}}]}}
Keep items relevant to: {", ".join(fence_kws + ['screen wall','cmu wall','security gate'])}.
If none: return {{"keynotes":[]}}.
Lines:
{payload}
""".strip()
        try:
            r = retry_with_backoff(llm_obj.invoke, [HumanMessage(content=prompt)])
            raw = r.content.strip()
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
            if m: raw = m.group(1).strip()
            if not raw.startswith("{"):
                b0, b1 = raw.find("{"), raw.rfind("}")
                if b0 != -1 and b1 > b0:
                    raw = raw[b0:b1+1]
            data = json.loads(raw)
            keynotes = data.get("keynotes", [])
            for kn in keynotes:
                if isinstance(kn.get("code"), str):
                    kn["code"] = kn["code"].strip()
            return keynotes
        except Exception as e:
            print(f"_llm_extract_keynotes failed: {e}")
            return []

    # --- indicator filter (LLM) ---
    def _llm_filter_indicators(candidate_words, llm_obj, legend_context):
        pw = json.dumps([{"id": w["id"], "text": w["text"]} for w in candidate_words], separators=(",", ":"))
        pl = json.dumps([{"code": c["code"], "text": c["text"]} for c in legend_context], separators=(",", ":"))
        prompt = f"""
You get LEGEND codes: {pl}
And CANDIDATE tokens on plan: {pw}
Return ONLY candidates that are TRUE indicator labels for one of the legend codes.
STRICT JSON: {{"confirmed":[{{"id":"<candidate id>","code":"<matching code>","text":"<candidate text>"}}]}}
""".strip()
        try:
            r = retry_with_backoff(llm_obj.invoke, [HumanMessage(content=prompt)])
            raw = r.content.strip()
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
            if m: raw = m.group(1).strip()
            if not raw.startswith("{"):
                b0, b1 = raw.find("{"), raw.rfind("}")
                if b0 != -1 and b1 > b0:
                    raw = raw[b0:b1+1]
            data = json.loads(raw)
            confirmed = data.get("confirmed", [])
            cmap = {w["id"]: w for w in candidate_words}
            out = []
            for it in confirmed:
                wid = it.get("id")
                if wid in cmap:
                    w = cmap[wid]
                    out.append({**w, "matched_code": it.get("code")})
            return out
        except Exception as e:
            print(f"_llm_filter_indicators failed: {e}")
            return []

    def _docai_tokens_and_lines(pb: bytes, cfg, dpi: int = 72):
        """
        Returns (ocr_lines, ocr_tokens).
        Better grouping: split lines on large X-gaps in addition to Y proximity.
        """
        if not cfg:
            return [], []
        try:
            client = create_document_ai_client(cfg)
            if not client:
                return [], []

            with fitz.open(stream=BytesIO(pb), filetype="pdf") as d:
                page = d[0]
                pix = page.get_pixmap(dpi=dpi)  # 72 worked well in your logs
                img_data = pix.tobytes("png")
                page_w, page_h = page.rect.width, page.rect.height
                scale_x = page_w / pix.width
                scale_y = page_h / pix.height

            from google.cloud import documentai_v1 as documentai
            raw_document = documentai.RawDocument(content=img_data, mime_type="image/png")
            processor_name = f"projects/{cfg['project_number']}/locations/{cfg['location']}/processors/{cfg['processor_id']}"
            request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
            result = client.process_document(request=request)
            doc = result.document

            tokens = []
            if doc.pages:
                pg = doc.pages[0]
                for tok in pg.tokens:
                    # text
                    text_content = ""
                    if tok.layout.text_anchor:
                        ta = tok.layout.text_anchor
                        if getattr(ta, "text_segments", None):
                            seg = ta.text_segments[0]
                            text_content = doc.text[seg.start_index:seg.end_index]
                        elif getattr(ta, "content", None):
                            text_content = ta.content
                    txt = (text_content or "").strip()
                    if not txt:
                        continue
                    # bbox
                    v = tok.layout.bounding_poly.vertices
                    if len(v) >= 4:
                        x0, y0 = v[0].x * scale_x, v[0].y * scale_y
                        x1, y1 = v[2].x * scale_x, v[2].y * scale_y
                        tokens.append({"text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

            if not tokens:
                return [], []

            # sort tokens top-to-bottom, then left-to-right
            tokens.sort(key=lambda t: ((t["y0"] + t["y1"]) / 2.0, t["x0"]))

            # robust thresholds
            heights = [t["y1"] - t["y0"] for t in tokens]
            widths  = [t["x1"] - t["x0"] for t in tokens]
            med_h = sorted(heights)[len(heights)//2] if heights else 8.0
            med_w = sorted(widths )[len(widths )//2] if widths  else 8.0

            y_tol      = max(6.0, med_h * 0.6)               # how close vertically to be same line
            x_gap_tol  = max(8.0, med_w * 3.0, page_w * 0.02)  # break line when horizontal gap is big

            lines = []
            cur = []
            prev_cy = None
            prev_x1 = None

            def flush(group):
                if not group: return
                # group is one real line slice (no huge X gaps)
                x0 = min(t["x0"] for t in group); y0 = min(t["y0"] for t in group)
                x1 = max(t["x1"] for t in group); y1 = max(t["y1"] for t in group)
                text = " ".join(t["text"] for t in sorted(group, key=lambda tt: tt["x0"]))
                lines.append({"text": text, "x0": x0, "top": y0, "x1": x1, "bottom": y1})

            for t in tokens:
                cy = (t["y0"] + t["y1"]) / 2.0
                if (prev_cy is None) or (abs(cy - prev_cy) <= y_tol and prev_x1 is not None and (t["x0"] - prev_x1) <= x_gap_tol):
                    cur.append(t)
                else:
                    flush(cur)
                    cur = [t]
                prev_cy = cy
                prev_x1 = t["x1"]
            flush(cur)

            return lines, tokens
        except Exception as e:
            print(f"[HL DEBUG] DocAI fallback failed: {e}")
            return [], []


    # ------------- main -------------
    final_boxes = []
    page_width = page_height = 0

    # tokens for tight geometry
    tokens = _pymupdf_tokens_from_bytes(page_bytes)

    # pdfplumber page/words/lines (for layout + candidate gen)
    words_pl, lines_pl = [], []
    try:
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            if not pdf.pages:
                return [], 0, 0
            p = pdf.pages[0]
            page_width, page_height = p.width, p.height
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

    # ---------- A) KEYWORD/LEGEND LINES: always highlight ----------
    primary = {k.lower() for k in (fence_keywords_from_app or [])}
    primary |= {"fence", "fencing", "gate", "gates", "barrier", "guardrail", "screen wall", "screenwall",
                "chain link", "chain-link", "chainlink"}
    secondary = {"mesh", "panel", "post"}  # needs a primary anchor on the same line

    for i, L in enumerate(lines_pl or []):
        txt = (L.get("text") or "").strip()
        if not txt: continue
        lt = txt.lower()
        has_primary = any(k in lt for k in primary)
        has_secondary = any(k in lt for k in secondary)
        if not has_primary and not has_secondary:
            continue
        if has_secondary and not has_primary:
            if not any(a in lt for a in ["fence","gate","guardrail","barrier","screen wall","chain link","chain-link","chainlink"]):
                continue
        if all(k in L for k in ["x0", "top", "x1", "bottom"]):
            line_rect = (float(L["x0"]), float(L["top"]), float(L["x1"]), float(L["bottom"]))
            union_rect = _line_token_union(line_rect, tokens)
            x0, y0, x1, y1 = _pad_rect(*union_rect, pad=1.5)
            final_boxes.append({
                "id": f"kwline_{i}",
                "text": txt,
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "type_from_llm": "keyword_line",
                "tag_from_llm": "KEYWORD"
            })

    # ---------- B) KEYNOTES (code + description): detect via LLM, then always highlight ----------
    line_payload = []
    line_lookup = {}
    for i, L in enumerate(lines_pl or []):
        txt = (L.get("text") or "").strip()
        if not txt: continue
        lid = f"ln_{i}"
        line_payload.append({"id": lid, "text": txt})
        line_lookup[lid] = L
    keynote_items = _llm_extract_keynotes(line_payload, llm, list(primary))
    keynote_codes = set()
    if keynote_items:
        for item in keynote_items:
            lid = item.get("id")
            code = (item.get("code") or "").strip()
            desc = (item.get("text") or "").strip()
            L = line_lookup.get(lid)
            if not (L and code): continue
            line_rect = (float(L["x0"]), float(L["top"]), float(L["x1"]), float(L["bottom"]))
            x0, y0, x1, y1 = _pad_rect(*_line_token_union(line_rect, tokens), pad=1.5) if tokens else _pad_rect(*line_rect, pad=1.5)
            final_boxes.append({
                "id": lid,
                "text": f"{code} {desc}" if desc else code,
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "type_from_llm": "legend_item",
                "tag_from_llm": code
            })
            keynote_codes.add(code)

    # ---------- C) INDICATORS: only LLM-confirmed ----------
    candidates = []
    if keynote_codes and words_pl:
        for idx, w in enumerate(words_pl):
            t = (w.get("text") or "").strip()
            if t in keynote_codes and len(t) <= 6:
                candidates.append({
                    "id": f"kw_{idx}",
                    "text": t,
                    "x0": round(float(w["x0"]), 2), "y0": round(float(w["top"]), 2),
                    "x1": round(float(w["x1"]), 2), "y1": round(float(w["bottom"]), 2),
                })
    if candidates:
        legend_ctx = [{"code": b["tag_from_llm"], "text": b["text"]}
                      for b in final_boxes if b.get("type_from_llm") == "legend_item"]
        confirmed = _llm_filter_indicators(candidates, llm, legend_ctx)
        for ci in confirmed:
            norm = _norm_text(ci["text"])
            matches = [t for t in tokens if _norm_text(t["text"]) == norm]
            chosen = None
            if matches:
                base_c = _center((ci["x0"], ci["y0"], ci["x1"], ci["y1"]))
                best_d, best_tok = 1e18, None
                for t in matches:
                    tc = _center((t["x0"], t["y0"], t["x1"], t["y1"]))
                    d = abs(tc[0] - base_c[0]) + abs(tc[1] - base_c[1])
                    if d < best_d:
                        best_d, best_tok = d, t
                if best_tok:
                    chosen = (best_tok["x0"], best_tok["y0"], best_tok["x1"], best_tok["y1"])
            rect = chosen or (ci["x0"], ci["y0"], ci["x1"], ci["y1"])
            x0, y0, x1, y1 = _pad_rect(*rect, pad=1.2)
            final_boxes.append({
                "id": ci["id"],
                "text": ci["text"],
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "type_from_llm": "indicator",
                "tag_from_llm": ci.get("matched_code")
            })

    # ---------- D) DocAI OCR fallback when nothing found ----------
    if not final_boxes:
        print("[HL DEBUG] No boxes from text-layer — using DocAI OCR fallback.")
        ocr_lines, ocr_tokens = _docai_tokens_and_lines(page_bytes, google_cloud_config, dpi=72)
        print(f"[HL DEBUG] docai_tokens={len(ocr_tokens)}  docai_lines={len(ocr_lines)}")

        if ocr_lines or ocr_tokens:
            # A) keyword lines
            for i, L in enumerate(ocr_lines):
                txt = (L.get("text") or "").strip()
                if not txt: continue
                lt = txt.lower()
                has_primary = any(k in lt for k in primary)
                has_secondary = any(k in lt for k in secondary)
                if not has_primary and not has_secondary:
                    continue
                if has_secondary and not has_primary:
                    if not any(a in lt for a in ["fence","gate","guardrail","barrier","screen wall","chain link","chain-link","chainlink"]):
                        continue
                line_rect = (float(L["x0"]), float(L["top"]), float(L["x1"]), float(L["bottom"]))
                x0, y0, x1, y1 = _pad_rect(*line_rect, pad=1.5)
                final_boxes.append({
                    "id": f"kwline_ocr_{i}",
                    "text": txt,
                    "x0": round(x0, 2), "y0": round(y0, 2),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "type_from_llm": "keyword_line",
                    "tag_from_llm": "KEYWORD_OCR"
                })

            # B) keynotes via LLM on OCR lines
            line_payload = [{"id": f"lnocr_{i}", "text": (L.get("text") or "").strip()} for i, L in enumerate(ocr_lines) if (L.get("text") or "").strip()]
            line_lookup = {f"lnocr_{i}": L for i, L in enumerate(ocr_lines)}
            keynote_items = _llm_extract_keynotes(line_payload, llm, list(primary))
            keynote_codes = set()
            for item in keynote_items:
                lid = item.get("id")
                code = (item.get("code") or "").strip()
                desc = (item.get("text") or "").strip()
                L = line_lookup.get(lid)
                if not (L and code): continue
                line_rect = (float(L["x0"]), float(L["top"]), float(L["x1"]), float(L["bottom"]))
                x0, y0, x1, y1 = _pad_rect(*line_rect, pad=1.5)
                final_boxes.append({
                    "id": lid,
                    "text": f"{code} {desc}" if desc else code,
                    "x0": round(x0, 2), "y0": round(y0, 2),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "type_from_llm": "legend_item",
                    "tag_from_llm": code
                })
                keynote_codes.add(code)

            # C) indicators from OCR tokens (LLM-filtered)
            if keynote_codes and ocr_tokens:
                candidates = []
                for idx, t in enumerate(ocr_tokens):
                    txt = (t.get("text") or "").strip()
                    if txt in keynote_codes and len(txt) <= 6:
                        candidates.append({
                            "id": f"kwocr_{idx}",
                            "text": txt,
                            "x0": round(t["x0"], 2), "y0": round(t["y0"], 2),
                            "x1": round(t["x1"], 2), "y1": round(t["y1"], 2),
                        })
                if candidates:
                    legend_ctx = [{"code": b["tag_from_llm"], "text": b["text"]}
                                  for b in final_boxes if b.get("type_from_llm") == "legend_item"]
                    confirmed = _llm_filter_indicators(candidates, llm, legend_ctx)
                    for ci in confirmed:
                        x0, y0, x1, y1 = _pad_rect(ci["x0"], ci["y0"], ci["x1"], ci["y1"], pad=1.2)
                        final_boxes.append({
                            "id": ci["id"],
                            "text": ci["text"],
                            "x0": round(x0, 2), "y0": round(y0, 2),
                            "x1": round(x1, 2), "y1": round(y1, 2),
                            "type_from_llm": "indicator",
                            "tag_from_llm": ci.get("matched_code")
                        })

    # ---------- E) dedup & return ----------
    dedup = {}
    for it in final_boxes:
        iid = it.get("id") or f"auto_{len(dedup)+1}"
        dedup[iid] = it
    final_boxes = list(dedup.values())

    # DEBUG
    print(f"[HL DEBUG] page_width={page_width}, page_height={page_height}")
    print(f"[HL DEBUG] tokens(PyMuPDF)={len(tokens)}  words_pl={len(words_pl)}  lines_pl={len(lines_pl)}")
    print(f"[HL DEBUG] keyword-lines={len([b for b in final_boxes if b.get('type_from_llm')=='keyword_line'])}")
    print(f"[HL DEBUG] legend-items={len([b for b in final_boxes if b.get('type_from_llm')=='legend_item'])}")
    print(f"[HL DEBUG] indicators={len([b for b in final_boxes if b.get('type_from_llm')=='indicator'])}")
    print(f"[HL DEBUG] total_boxes={len(final_boxes)}")
    print(f"TIMER LOG: v14 done. Boxes={len(final_boxes)} in {time.time() - t0:.3f}s")
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
