"""Stage-3 of the agentic takeoff plan: vision verification of layer candidates.

The pipeline picks candidate CAD layers by NAME semantics. Names lie: a layer
that sounds fence-adjacent can hold tile hatch, signage or curbs, and one
wrong layer decision paints thousands of stray segments over the canvas (the
2026-08-05 "drawing lines randomly" report). This module renders the page
with each candidate layer's linework highlighted in a distinct colour and
asks a vision model ONE question per page: which colour groups are actually
fence-scope runs?

Verdicts are per-(page, layer) — never per line — and they are advisory
metadata: geometry and measurements are untouched. Consumers (canvas
auto-assign, measurement summary) skip layers with a "not_fence" verdict;
"unsure" layers stay visible. Gated by FENCE_VISION_VERIFY (default off —
it adds one vision call per fence page).
"""
from __future__ import annotations

import base64
import json
from typing import Dict, List, Optional

import fitz

from .llm import _parse_json_payload

# Distinct, high-contrast overlay colours (RGB 0-255).
_PALETTE = [
    (230, 0, 0), (0, 90, 255), (0, 170, 0), (255, 140, 0),
    (170, 0, 200), (0, 190, 190), (255, 0, 150), (120, 90, 0),
    (90, 120, 255), (0, 120, 60), (200, 160, 0), (130, 130, 130),
]

MAX_LAYERS_PER_CALL = len(_PALETTE)
MAX_SEGMENTS_PER_LAYER = 2000
MAX_RENDER_SIDE = 2000

_PROMPT = """You are an expert fencing-trade estimator reviewing an automatic takeoff.

This construction sheet has candidate CAD layers highlighted, one colour per
numbered group (legend below the question). For EACH numbered group decide
whether the highlighted linework is genuinely fence-scope geometry — fence
runs, gates, screen/site walls, site barriers, enclosure outlines — or other
content that merely lives on a fence-sounding layer (wall tile/hatch pattern,
building outlines, stairs, signage, symbols, curbs, drainage, dimension marks).

Judge by what the HIGHLIGHTED STROKES trace on the drawing, not by the layer
name. Dense short-fragment clusters filling an area are hatch/pattern, not
fence. Long connected runs along boundaries/enclosures are fence-like.

GROUPS:
{groups}

Return STRICT JSON:
{{"verdicts": {{"<group number>": {{"verdict": "fence"|"not_fence"|"unsure",
"reason": "<one short sentence>"}}}}}}
Use "unsure" when the image genuinely doesn't let you decide."""


def render_layer_overlay(page: fitz.Page, lines_by_layer: Dict[str, List],
                         layer_order: List[str]) -> bytes:
    """Render the page with each layer's segments drawn in its palette
    colour. Lines are display-space dicts with `start`/`end`."""
    from PIL import Image, ImageDraw

    rect = page.rect
    zoom = min(MAX_RENDER_SIDE / max(rect.width, rect.height), 4.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    for i, layer in enumerate(layer_order):
        color = _PALETTE[i % len(_PALETTE)]
        for ln in lines_by_layer.get(layer, [])[:MAX_SEGMENTS_PER_LAYER]:
            s, e = ln.get("start"), ln.get("end")
            if not s or not e:
                continue
            draw.line([float(s[0]) * zoom, float(s[1]) * zoom,
                       float(e[0]) * zoom, float(e[1]) * zoom],
                      fill=color, width=3)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def verify_page_layers(
    page: fitz.Page,
    measurements: Dict,
    client,
    model: str = "gpt-5.1",
) -> Optional[Dict]:
    """One vision call for one page. Returns
    {"verdicts": {layer: {"verdict", "reason"}}, "model": ...} or None when
    there is nothing to verify / the call failed (callers treat None as
    "no verdicts", never as rejection)."""
    all_lines = measurements.get("all_fence_lines") or []
    lines_by_layer: Dict[str, List] = {}
    for ln in all_lines:
        if isinstance(ln, dict):
            lines_by_layer.setdefault(ln.get("layer") or "", []).append(ln)
    layers = [l for l in (measurements.get("fence_layers") or [])
              if lines_by_layer.get(l)]
    if not layers or client is None:
        return None
    layers = layers[:MAX_LAYERS_PER_CALL]

    groups_txt = "\n".join(
        f'  {i + 1}. colour rgb{_PALETTE[i % len(_PALETTE)]} = layer '
        f'"{layer}" ({len(lines_by_layer[layer])} segments)'
        for i, layer in enumerate(layers)
    )
    try:
        png = render_layer_overlay(page, lines_by_layer, layers)
        b64 = base64.b64encode(png).decode()
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text",
                     "text": _PROMPT.format(groups=groups_txt)},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{b64}"},
                ],
            }],
        )
        parsed = _parse_json_payload(resp.output_text, opener="{") or {}
        raw = parsed.get("verdicts") or {}
    except Exception as e:
        print(f"[DEBUG] vision verify failed: {e}")
        return None

    verdicts: Dict[str, Dict] = {}
    for i, layer in enumerate(layers):
        v = raw.get(str(i + 1)) or raw.get(layer) or {}
        verdict = v.get("verdict")
        if verdict not in ("fence", "not_fence", "unsure"):
            continue
        verdicts[layer] = {"verdict": verdict,
                           "reason": (v.get("reason") or "")[:300]}
    if not verdicts:
        return None
    return {"verdicts": verdicts, "model": model}


def layer_rejected(measurements: Dict, layer: str) -> bool:
    """True when vision verification judged this layer's linework
    not-fence. Missing/absent verdicts never reject."""
    v = ((measurements or {}).get("vision_verdicts") or {}).get("verdicts") or {}
    return (v.get(layer) or {}).get("verdict") == "not_fence"
