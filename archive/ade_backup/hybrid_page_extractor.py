from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF

from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

from utils_ade_official import (
    ade_parse_document_official,
    align_ade_chunks_to_page,
)

# -------------------------
# Document AI helpers
# -------------------------

def create_document_ai_client(service_account_info: Dict) -> documentai.DocumentProcessorServiceClient:
    """Create a Document AI client from a service account dictionary."""
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    return documentai.DocumentProcessorServiceClient(credentials=credentials)


def normalize_document_ai_config(config: Dict) -> Dict:
    """Ensure the Document AI configuration contains a ready-to-use client and processor name."""
    normalized = dict(config)

    client = normalized.get("client")
    if client is None and normalized.get("service_account_info"):
        normalized["client"] = create_document_ai_client(normalized["service_account_info"])

    if not normalized.get("processor_name"):
        project = normalized.get("project_number")
        location = normalized.get("location")
        processor_id = normalized.get("processor_id")
        if project and location and processor_id:
            normalized["processor_name"] = (
                f"projects/{project}/locations/{location}/processors/{processor_id}"
            )

    return normalized


def compute_safe_dpi(page: fitz.Page, target_dpi: int = 240, min_dpi: int = 72, max_cap: int = 300) -> int:
    max_dimension = max(page.rect.width, page.rect.height)
    max_dpi_allowed = int((10000 / max_dimension) * 72) if max_dimension else max_cap
    if max_dpi_allowed <= 0:
        max_dpi_allowed = max_cap
    working_dpi = min(max(target_dpi, min_dpi), max_dpi_allowed, max_cap)
    return max(working_dpi, min_dpi)


def preprocess_pixmap_for_ocr(pix: fitz.Pixmap, config: Dict) -> Tuple[bytes, Dict[str, float]]:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps

    orig_w, orig_h = pix.width, pix.height
    image_mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(image_mode, [orig_w, orig_h], pix.samples)
    if image_mode == "RGBA":
        image = image.convert("RGB")

    info = {
        "orig_width": float(orig_w),
        "orig_height": float(orig_h),
        "offset_x": 0.0,
        "offset_y": 0.0,
        "width": float(orig_w),
        "height": float(orig_h),
    }

    if config.get("auto_crop"):
        gray = image.convert("L")
        arr = np.array(gray)
        threshold = int(config.get("auto_crop_threshold", 245))
        mask = arr < threshold
        if mask.any():
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            top = int(np.argmax(rows))
            bottom = int(len(rows) - np.argmax(rows[::-1]))
            left = int(np.argmax(cols))
            right = int(len(cols) - np.argmax(cols[::-1]))
            margin_ratio = float(config.get("crop_margin", 0.02))
            margin_x = int(margin_ratio * orig_w)
            margin_y = int(margin_ratio * orig_h)
            crop_box = (
                max(left - margin_x, 0),
                max(top - margin_y, 0),
                min(right + margin_x, orig_w),
                min(bottom + margin_y, orig_h),
            )
            if crop_box[2] - crop_box[0] > 50 and crop_box[3] - crop_box[1] > 50:
                image = image.crop(crop_box)
                info["offset_x"] = float(crop_box[0])
                info["offset_y"] = float(crop_box[1])
                info["width"] = float(image.width)
                info["height"] = float(image.height)

    if config.get("grayscale"):
        image = image.convert("L")
        if config.get("invert_for_grayscale"):
            image = ImageOps.invert(image)
    else:
        image = image.convert("RGB")

    contrast_factor = float(config.get("contrast_factor", 1.0))
    if abs(contrast_factor - 1.0) > 1e-3:
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    sharpen_factor = float(config.get("sharpen_factor", 1.0))
    if abs(sharpen_factor - 1.0) > 1e-3:
        image = ImageEnhance.Sharpness(image).enhance(sharpen_factor)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), info


OCR_CACHE: Dict[str, List[Dict]] = {}


def run_document_ai_ocr(
    page: fitz.Page,
    client: Optional[documentai.DocumentProcessorServiceClient],
    processor_name: Optional[str],
    dpi: int = 240,
    config: Optional[Dict] = None,
) -> List[Dict]:
    if not client or not processor_name:
        return []

    cfg = dict(config or {})
    cache_config = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    target_dpi = int(cache_config.get("dpi", dpi))
    working_dpi = compute_safe_dpi(page, target_dpi)
    fallback_dpis = cache_config.get("fallback_dpis") or [
        working_dpi,
        240,
        210,
        180,
        150,
        120,
        96,
        72,
    ]

    cache_key_base = json.dumps(
        {
            "processor": processor_name,
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "config": cache_config,
        },
        sort_keys=True,
        default=str,
    )

    for attempt_dpi in fallback_dpis:
        attempt_dpi = compute_safe_dpi(page, int(attempt_dpi))
        cache_key = f"{cache_key_base}|dpi={attempt_dpi}"
        if cache_key in OCR_CACHE:
            return [token.copy() for token in OCR_CACHE[cache_key]]

        try:
            pix = page.get_pixmap(dpi=attempt_dpi)
            if pix.width > 10000 or pix.height > 10000:
                continue

            img_bytes, info = preprocess_pixmap_for_ocr(pix, cfg)
            raw_document = documentai.RawDocument(content=img_bytes, mime_type="image/png")

            process_options = None
            ocr_config = None
            if cfg.get("enable_native_pdf_parsing") or cfg.get("advanced_ocr_options"):
                ocr_config = documentai.OcrConfig()
                if cfg.get("enable_native_pdf_parsing"):
                    ocr_config.enable_native_pdf_parsing = True
                advanced_opts = cfg.get("advanced_ocr_options") or []
                if advanced_opts:
                    ocr_config.advanced_ocr_options.extend(list(advanced_opts))
            if ocr_config:
                process_options = documentai.ProcessOptions(ocr_config=ocr_config)

            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document,
                process_options=process_options,
            )
            doc = client.process_document(request=request).document

            tokens: List[Dict] = []
            if doc.pages:
                dpi_scale_x = page.rect.width / info["orig_width"]
                dpi_scale_y = page.rect.height / info["orig_height"]
                page_proto = doc.pages[0]

                for token in page_proto.tokens:
                    anchor = token.layout.text_anchor
                    text = ""
                    if anchor and anchor.text_segments:
                        segment = anchor.text_segments[0]
                        text = doc.text[segment.start_index : segment.end_index]
                    text = text.strip()
                    if not text:
                        continue

                    layout = token.layout
                    poly = getattr(layout, "bounding_poly", None)
                    if not poly:
                        continue

                    norm_vertices = list(getattr(poly, "normalized_vertices", [])) or []
                    raw_vertices = list(getattr(poly, "vertices", [])) or []

                    coords: List[Tuple[float, float]] = []

                    if norm_vertices:
                        width = info["width"] or info["orig_width"]
                        height = info["height"] or info["orig_height"]
                        for vert in norm_vertices[:4]:
                            if vert.x is None or vert.y is None:
                                continue
                            coords.append(
                                (
                                    info["offset_x"] + float(vert.x) * width,
                                    info["offset_y"] + float(vert.y) * height,
                                )
                            )
                    elif raw_vertices:
                        for vert in raw_vertices[:4]:
                            if vert.x is None or vert.y is None:
                                continue
                            coords.append(
                                (
                                    info["offset_x"] + float(vert.x),
                                    info["offset_y"] + float(vert.y),
                                )
                            )
                    if len(coords) < 2:
                        continue

                    xs = [pt[0] for pt in coords if pt[0] is not None]
                    ys = [pt[1] for pt in coords if pt[1] is not None]
                    if len(xs) < 2 or len(ys) < 2:
                        continue

                    x0 = min(xs) * dpi_scale_x
                    y0 = min(ys) * dpi_scale_y
                    x1 = max(xs) * dpi_scale_x
                    y1 = max(ys) * dpi_scale_y

                    tokens.append(
                        {
                            "text": text,
                            "normalized_text": normalize_text_for_matching(text),
                            "x0": float(x0),
                            "y0": float(y0),
                            "x1": float(x1),
                            "y1": float(y1),
                            "confidence": getattr(token.layout, "confidence", 1.0),
                            "attempt_dpi": attempt_dpi,
                        }
                    )

            if tokens:
                OCR_CACHE[cache_key] = [token.copy() for token in tokens]
                return tokens
        except Exception as exc:
            message = str(exc)
            if "enable_native_pdf_parsing" in message and cfg.pop("enable_native_pdf_parsing", None):
                cfg["_native_disabled"] = True
                return run_document_ai_ocr(page, client, processor_name, dpi=dpi, config=cfg)
            continue
    return []


def normalize_text_for_matching(text: Optional[str]) -> str:
    if not text:
        return ""
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "B": "8",
        "S": "5",
    }
    cleaned = "".join(replacements.get(ch, ch) for ch in str(text).strip())
    return cleaned.replace(" ", "")


# -------------------------
# PDF helpers
# -------------------------

def extract_pdf_text_words(page: fitz.Page) -> List[Dict]:
    words = []
    for x0, y0, x1, y1, text, block_no, line_no, word_no in page.get_text("words"):
        words.append(
            {
                "text": text,
                "normalized_text": normalize_text_for_matching(text),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "block_no": block_no,
                "line_no": line_no,
                "word_no": word_no,
            }
        )
    return words


def extract_pdf_blocks_and_annotations(page: fitz.Page) -> Dict[str, List[Dict]]:
    blocks = []
    for x0, y0, x1, y1, text, block_no, block_type in page.get_text("blocks"):
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        blocks.append(
            {
                "text": cleaned,
                "normalized_text": normalize_text_for_matching(cleaned),
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "block_no": block_no,
                "block_type": block_type,
            }
        )

    annotations = []
    if page.annots():
        for annot in page.annots():
            info = annot.info or {}
            contents = str(info.get("content", "")).strip()
            if not contents:
                continue
            annotations.append(
                {
                    "text": contents,
                    "normalized_text": normalize_text_for_matching(contents),
                    "x0": float(annot.rect.x0),
                    "y0": float(annot.rect.y0),
                    "x1": float(annot.rect.x1),
                    "y1": float(annot.rect.y1),
                    "type": annot.type[0],
                    "subtype": annot.type[1],
                    "title": info.get("title", ""),
                }
            )
    return {"blocks": blocks, "annotations": annotations}


def extract_page_text_layers(
    page: fitz.Page,
    doc_ai_config: Optional[Dict] = None,
) -> Dict[str, List[Dict]]:
    words = extract_pdf_text_words(page)
    blocks_and_annots = extract_pdf_blocks_and_annotations(page)

    ocr_tokens: List[Dict] = []
    if doc_ai_config:
        resolved = normalize_document_ai_config(doc_ai_config)
        ocr_tokens = run_document_ai_ocr(
            page,
            client=resolved.get("client"),
            processor_name=resolved.get("processor_name"),
            dpi=resolved.get("dpi", 240),
            config=resolved.get("ocr_config"),
        )

    return {
        "text_words": words,
        "text_blocks": blocks_and_annots["blocks"],
        "pdf_annotations": blocks_and_annots["annotations"],
        "ocr_tokens": ocr_tokens,
    }


# -------------------------
# ADE helpers
# -------------------------

def parse_ade_page(
    pdf_bytes: bytes,
    page_index: int,
    page_width: float,
    page_height: float,
    ade_api_key: str,
) -> List[Dict]:
    ade_result = ade_parse_document_official(pdf_bytes, ade_api_key)
    if not ade_result.get("success"):
        return []
    return align_ade_chunks_to_page(ade_result, page_index, page_width, page_height)


# -------------------------
# High level orchestration
# -------------------------


def extract_page_sources(
    pdf: Union[str, Path, bytes],
    page_number: int,
    ade_api_key: str,
    doc_ai_config: Optional[Dict] = None,
) -> Dict[str, Union[List[Dict], float]]:
    """Collect PDF text, annotations, Google Document AI OCR, and ADE chunks for a page."""
    if isinstance(pdf, (str, Path)):
        pdf_path = Path(pdf)
        pdf_bytes = pdf_path.read_bytes()
        doc = fitz.open(pdf_path)
    else:
        pdf_bytes = pdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_index = page_number - 1
    if page_index < 0 or page_index >= len(doc):
        raise IndexError(f"Page number {page_number} out of bounds for document with {len(doc)} pages")

    page = doc[page_index]
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    text_layers = extract_page_text_layers(page, doc_ai_config)

    ade_chunks = parse_ade_page(
        pdf_bytes=pdf_bytes,
        page_index=page_index,
        page_width=page_width,
        page_height=page_height,
        ade_api_key=ade_api_key,
    )

    doc.close()

    return {
        "page_index": page_index,
        "page_width": page_width,
        "page_height": page_height,
        "pdf_words": text_layers["text_words"],
        "pdf_blocks": text_layers["text_blocks"],
        "pdf_annotations": text_layers["pdf_annotations"],
        "ocr_tokens": text_layers["ocr_tokens"],
        "ade_chunks": ade_chunks,
    }
