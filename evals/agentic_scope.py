"""Agentic vision prototype: fence-scope identification on gold sheets.

Two-step "agent with a zoom tool", no page-specific logic:
  1. Full-page render → model returns the regions that list keynotes /
     legends / detail titles (relative bboxes).
  2. Each region cropped at high DPI → model (acting as a fencing
     estimator) returns the fence-scope items (id + description).
Then flag instances are located DETERMINISTICALLY: page words matching the
returned ids outside the table regions. LLM does semantics; geometry stays
code. Requires OPENAI key via secrets_loader.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils_vector import transform_coords_for_rotation  # noqa: E402

CLEAN_PDF = REPO / "subset_gold" / "selected_pages_no_annotations.pdf"
MODEL = "gpt-5.1"

SCOPE_DEF = (
    "You are an expert fencing-trade estimator doing takeoff scope marking. "
    "Fencing scope includes: fences of any material (chain link, wrought "
    "iron, wire mesh, wood), gates (swing, rolling, pedestrian, vehicular), "
    "screen walls (CMU or other), guardrails/handrails only if fence-like "
    "barriers, bollards, trash/recycle enclosures, dog runs and similar "
    "enclosed pens, and fence-mounted hardware. NOT in scope: buildings, "
    "paving, curbs, signage, landscaping, utilities, lighting."
)


def _client():
    from openai import OpenAI
    from secrets_loader import load_api_keys
    keys = load_api_keys()
    return OpenAI(api_key=keys["openai_key"])


def render_page_png(pno_1based: int, max_side: int = 2000,
                    clip_rel=None) -> tuple[bytes, float]:
    """Render (optionally a relative-coord crop of) a page; returns PNG
    bytes + the display-points-per-pixel scale used."""
    with fitz.open(str(CLEAN_PDF)) as doc:
        page = doc[pno_1based - 1]
        rect = page.rect
        clip = None
        if clip_rel:
            x0, y0, x1, y1 = clip_rel
            clip = fitz.Rect(rect.x0 + x0 * rect.width,
                             rect.y0 + y0 * rect.height,
                             rect.x0 + x1 * rect.width,
                             rect.y0 + y1 * rect.height)
        target = clip or rect
        zoom = min(max_side / max(target.width, target.height), 4.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False,
                              clip=clip)
        return pix.tobytes("png"), 1.0 / zoom


def _ask(client, png: bytes, prompt: str) -> str:
    b64 = base64.b64encode(png).decode()
    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{b64}"},
            ],
        }],
    )
    return resp.output_text


def _json_of(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def find_tables(client, pno: int) -> list:
    png, _ = render_page_png(pno)
    out = _ask(client, png, (
        "This is one sheet of a construction drawing set. Find every region "
        "that LISTS items: keynote tables, legends, or detail-title strips "
        "(the numbered titles under each detail drawing count as one region "
        "covering them). Return STRICT JSON: {\"regions\": [{\"kind\": "
        "\"keynotes|legend|detail_titles|schedule\", \"bbox\": [x0,y0,x1,y1] "
        "relative 0-1 coords of the full image}]} — nothing else."))
    return _json_of(out).get("regions", [])


def scope_items(client, pno: int, regions: list) -> list:
    items = []
    for reg in regions[:4]:
        bbox = reg.get("bbox") or [0, 0, 1, 1]
        try:
            png, _ = render_page_png(pno, max_side=1800, clip_rel=bbox)
        except Exception:
            continue
        out = _ask(client, png, (
            f"{SCOPE_DEF}\n\nThis image is the {reg.get('kind')} region of a "
            "sheet. Read every numbered/coded item. Return STRICT JSON: "
            "{\"items\": [{\"id\": \"<keynote number or code exactly as "
            "printed>\", \"description\": \"...\", \"in_scope\": true/false, "
            "\"reason\": \"...\"}]} listing ALL items you can read, with "
            "in_scope true only for fencing-trade scope as defined."))
        try:
            items.extend(_json_of(out).get("items", []))
        except Exception:
            pass
    return items


def locate_ids(pno: int, ids: set, exclude_rel_boxes: list) -> list:
    """Word tokens equal to a chosen id, outside the table regions."""
    with fitz.open(str(CLEAN_PDF)) as doc:
        page = doc[pno - 1]
        rot, mw, mh = page.rotation, page.mediabox.width, page.mediabox.height
        W, H = page.rect.width, page.rect.height
        hits = []
        for x0, y0, x1, y1, w, *_ in page.get_text("words"):
            t = w.strip().strip("()")
            if t not in ids:
                continue
            a = transform_coords_for_rotation(x0, y0, rot, mw, mh)
            b = transform_coords_for_rotation(x1, y1, rot, mw, mh)
            bb = (min(a[0], b[0]), min(a[1], b[1]),
                  max(a[0], b[0]), max(a[1], b[1]))
            cx, cy = (bb[0] + bb[2]) / 2 / W, (bb[1] + bb[3]) / 2 / H
            if any(rx0 <= cx <= rx1 and ry0 <= cy <= ry1
                   for rx0, ry0, rx1, ry1 in exclude_rel_boxes):
                continue
            hits.append({"id": t, "bbox": [round(v, 1) for v in bb]})
    return hits


def _display_words(pno: int) -> list:
    with fitz.open(str(CLEAN_PDF)) as doc:
        page = doc[pno - 1]
        rot, mw, mh = page.rotation, page.mediabox.width, page.mediabox.height
        out = []
        for x0, y0, x1, y1, w, *_ in page.get_text("words"):
            a = transform_coords_for_rotation(x0, y0, rot, mw, mh)
            b = transform_coords_for_rotation(x1, y1, rot, mw, mh)
            out.append((min(a[0], b[0]), min(a[1], b[1]),
                        max(a[0], b[0]), max(a[1], b[1]), w.strip()))
        return out, page.rect.width, page.rect.height


def verify_pass(client, pno: int, read_ids: set) -> list:
    """The agentic step the one-shot pass lacks: cross-check the page's own
    code tokens against what was read. Keynote tables print their ids in a
    vertical column — find columns of code-like tokens containing ids we
    have NOT read yet, crop each column's row-band, and read it. Fully
    generic: no page-specific coordinates."""
    words, W, H = _display_words(pno)
    codes = [w for w in words if re.fullmatch(r"\d{3,4}", w[4])]
    cols: dict[int, list] = {}
    for w in codes:
        cols.setdefault(int(w[0] // 25), []).append(w)
    extra = []
    for _, col in cols.items():
        if len(col) < 4:
            continue
        col_ids = {w[4] for w in col}
        if col_ids <= read_ids:
            continue                      # already fully read
        x0 = min(w[0] for w in col) - 15
        y0 = min(w[1] for w in col) - 15
        y1 = max(w[3] for w in col) + 15
        rel = (max(x0 / W, 0), max(y0 / H, 0),
               min((x0 + 900) / W, 1.0), min(y1 / H, 1.0))
        try:
            png, _ = render_page_png(pno, max_side=1800, clip_rel=rel)
        except Exception:
            continue
        out = _ask(client, png, (
            f"{SCOPE_DEF}\n\nThis crop shows a keynote/legend table column "
            "from a construction sheet. Read every coded item. Return "
            "STRICT JSON: {\"items\": [{\"id\": \"<code>\", \"description\": "
            "\"...\", \"in_scope\": true/false, \"reason\": \"...\"}]}"))
        try:
            extra.extend(_json_of(out).get("items", []))
        except Exception:
            pass
    return extra


def run(pages=(2, 3, 4, 5)) -> dict:
    client = _client()
    result = {}
    for pno in pages:
        regions = find_tables(client, pno)
        items = scope_items(client, pno, regions)
        read_ids = {str(it.get("id", "")).strip() for it in items}
        items += verify_pass(client, pno, read_ids)
        in_scope = [it for it in items if it.get("in_scope")]
        ids = {str(it.get("id", "")).strip() for it in in_scope}
        ids.discard("")
        flags = locate_ids(pno, ids, [r.get("bbox", [0, 0, 0, 0])
                                      for r in regions])
        result[pno] = {
            "regions": regions,
            "n_items_read": len(items),
            "in_scope": in_scope,
            "flags": flags,
        }
        print(f"page {pno}: regions={len(regions)} read={len(items)} "
              f"in_scope={sorted(ids)} flags_located={len(flags)}",
              flush=True)
    return result


if __name__ == "__main__":
    out = run()
    dest = sys.argv[1] if len(sys.argv) > 1 else "agentic_scope_result.json"
    Path(dest).write_text(json.dumps(out, indent=1))
    print("saved", dest)
