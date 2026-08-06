"""Ground-truth harness for fence detection against subset_gold.

Gold source: subset_gold/selected_pages.pdf — 5 real plan sheets annotated by
human estimators. Ink annotations (pages 4-5) are hand-traced fence/wall runs,
each stroke a polyline; numeric annotation contents are the estimator's
measured footage. Highlight annotations mark keynote flags / legend rows
(codes 0401/0402/0403 CMU screen walls, 0113 rolling gates, 3219 trash
enclosure, 0508 bollards).

Matching is POLYLINE-BUFFER based, not bbox IoU: a gold stroke is recalled
when detected lines cover >= coverage_frac of its resampled points within
buffer_pts; precision is the length-weighted fraction of detected linework
lying near any gold stroke. (The 2026-05 harness in archive/evaluation
scored 0.0 everywhere because IoU of thin, line-shaped bboxes is ~0 even
for perfect detections, and it dropped unlabeled ink strokes.)

Coordinates: everything is compared in DISPLAY space. Annot rects/vertices
are native (unrotated); we transform them with the same
utils_vector.transform_coords_for_rotation the pipeline uses, so harness and
pipeline can never disagree about rotation.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils_vector import extract_vector_lines, transform_coords_for_rotation  # noqa: E402

GOLD_PDF = REPO / "subset_gold" / "selected_pages.pdf"
CLEAN_PDF = REPO / "subset_gold" / "selected_pages_no_annotations.pdf"


@dataclass
class GoldStroke:
    page: int                      # 1-based
    points: list                   # [(x, y)] display space
    length_pts: float
    length_ft: float | None        # estimator-written footage, if any
    content: str


@dataclass
class GoldPage:
    page: int
    rotation: int
    width: float
    height: float
    strokes: list = field(default_factory=list)
    highlights: list = field(default_factory=list)   # (rect_display, content)


def _poly_len(pts) -> float:
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _resample(pts, step: float = 5.0) -> list:
    """Resample a polyline to ~equidistant points (always keeps endpoints)."""
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    carry = 0.0
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        seg = math.dist((x0, y0), (x1, y1))
        if seg == 0:
            continue
        t = step - carry
        while t <= seg:
            f = t / seg
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
            t += step
        carry = (carry + seg) % step
    out.append(pts[-1])
    return out


def load_gold(pdf_path: Path = GOLD_PDF) -> dict[int, GoldPage]:
    pages: dict[int, GoldPage] = {}
    with fitz.open(str(pdf_path)) as doc:
        for pno in range(doc.page_count):
            page = doc[pno]
            rot = page.rotation
            mw, mh = page.mediabox.width, page.mediabox.height

            def disp(x, y):
                return transform_coords_for_rotation(x, y, rot, mw, mh)

            gp = GoldPage(page=pno + 1, rotation=rot,
                          width=page.rect.width, height=page.rect.height)
            for a in page.annots() or []:
                kind = a.type[1]
                content = (a.info.get("content") or "").strip()
                if kind == "Ink":
                    for stroke in a.vertices or []:
                        pts = [disp(float(p[0]), float(p[1])) for p in stroke]
                        if len(pts) < 2:
                            continue
                        length_ft = None
                        try:
                            v = float(content)
                            if 0 < v < 2000:
                                length_ft = v
                        except ValueError:
                            pass
                        gp.strokes.append(GoldStroke(
                            page=gp.page, points=pts,
                            length_pts=_poly_len(pts),
                            length_ft=length_ft, content=content,
                        ))
                elif kind in ("Highlight", "Square", "FreeText"):
                    r = a.rect
                    c0 = disp(r.x0, r.y0)
                    c1 = disp(r.x1, r.y1)
                    rect = (min(c0[0], c1[0]), min(c0[1], c1[1]),
                            max(c0[0], c1[0]), max(c0[1], c1[1]))
                    gp.highlights.append((rect, content))
            pages[gp.page] = gp
    return pages


class _Grid:
    """Uniform spatial hash for nearest-point-within-radius queries."""

    def __init__(self, pts, cell: float):
        self.cell = cell
        self.d: dict[tuple, list] = {}
        for p in pts:
            self.d.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)

    def has_within(self, x, y, r) -> bool:
        c = self.cell
        r2 = r * r
        cx, cy = int(x // c), int(y // c)
        span = int(r // c) + 1
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                for (px, py) in self.d.get((gx, gy), ()):
                    dx, dy = px - x, py - y
                    if dx * dx + dy * dy <= r2:
                        return True
        return False


def score_page(
    detected: list,                 # [((x0,y0),(x1,y1)), ...] display space
    gold: GoldPage,
    buffer_pts: float = 15.0,
    coverage_frac: float = 0.5,
    min_len_pts: float = 10.0,
) -> dict:
    """Recall: gold strokes covered by detections. Precision: detected
    length lying near gold strokes (length-weighted). Both directions use
    resampled points against a spatial grid of the other side."""
    detected = [
        (a, b) for a, b in detected if math.dist(a, b) >= min_len_pts
    ]
    det_samples = []
    for a, b in detected:
        det_samples.extend(_resample([a, b]))
    det_grid = _Grid(det_samples, cell=buffer_pts)

    gold_samples = []
    for s in gold.strokes:
        gold_samples.extend(_resample(s.points))
    gold_grid = _Grid(gold_samples, cell=buffer_pts)

    recalled, per_stroke = 0, []
    for s in gold.strokes:
        pts = _resample(s.points)
        cov = (sum(det_grid.has_within(x, y, buffer_pts) for x, y in pts)
               / max(len(pts), 1))
        ok = cov >= coverage_frac
        recalled += ok
        per_stroke.append({"content": s.content, "length_ft": s.length_ft,
                           "coverage": round(cov, 2), "recalled": bool(ok)})

    matched_len = total_len = 0.0
    for a, b in detected:
        seg = math.dist(a, b)
        total_len += seg
        pts = _resample([a, b])
        near = sum(gold_grid.has_within(x, y, buffer_pts) for x, y in pts)
        matched_len += seg * near / max(len(pts), 1)

    n = len(gold.strokes)
    return {
        "page": gold.page,
        "gold_strokes": n,
        "recall": round(recalled / n, 3) if n else None,
        "precision_len": round(matched_len / total_len, 3) if total_len else None,
        "detected_lines": len(detected),
        "detected_len_pts": round(total_len),
        "matched_len_pts": round(matched_len),
        "per_stroke": per_stroke,
    }


def clean_page_lines(pno_1based: int, min_len: float = 10.0):
    """Display-space vector lines of the un-annotated gold PDF."""
    with fitz.open(str(CLEAN_PDF)) as doc:
        vlines = extract_vector_lines(doc[pno_1based - 1], apply_rotation=True)
    return [
        {"start": tuple(v.start), "end": tuple(v.end),
         "length_pts": v.length_pts, "layer": v.layer or ""}
        for v in vlines if v.length_pts >= min_len
    ]


def render_overlay(pno_1based: int, detected, out_png: str,
                   dpi: int = 72, gold_pages=None):
    """Visual sanity check: gold strokes green, detections red — overlap
    means the two coincide. ALWAYS look at this before trusting scores.
    get_pixmap of a rotated page is display-oriented, so display coords
    map straight to pixels via the zoom factor."""
    from PIL import Image, ImageDraw

    gold_pages = gold_pages or load_gold()
    gp = gold_pages[pno_1based]
    zoom = dpi / 72.0
    with fitz.open(str(CLEAN_PDF)) as doc:
        page = doc[pno_1based - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    for a, b in detected:
        draw.line([a[0] * zoom, a[1] * zoom, b[0] * zoom, b[1] * zoom],
                  fill=(255, 0, 0), width=2)
    for s in gp.strokes:
        pts = [(x * zoom, y * zoom) for x, y in s.points]
        draw.line(pts, fill=(0, 200, 0), width=3)
    img.save(out_png)
    return out_png
