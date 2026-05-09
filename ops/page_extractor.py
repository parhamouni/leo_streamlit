#!/usr/bin/env python3
"""Single-page text extractor run as a short-lived subprocess.

Why this exists
---------------
fitz.get_text("dict") on subtly-damaged pages can enter MuPDF's internal
recovery loop at the C level, burning 100% CPU without releasing the GIL and
without ever returning. No Python-level mechanism (signal.alarm, threading
timeout, asyncio cancellation) can interrupt it, because all of those require
the thread to check in at a Python bytecode boundary — and the C code never
yields.

The only reliable way to bound the time is to run the extraction in a separate
OS process and SIGKILL it if it doesn't respond in time. That is the contract
of this script:

    $ python ops/page_extractor.py /path/to/doc.pdf 42
    {"ok": true, "lines": [{"text": "...", "x0": ..., ...}, ...]}

If the page extracts cleanly, stdout is one JSON line with ok=True and the
extracted lines (same shape as utils_ade.get_native_pdf_lines).
If extraction raises, stdout is one JSON line with ok=False and an error.
If the process hangs, the parent kills it and interprets the timeout as a
damaged page.
"""
import json
import os
import sys

# Make the repo root importable so we can reuse utils_ade.get_native_pdf_lines
# without duplicating its logic here.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import fitz  # noqa: E402

# Silence MuPDF's C-level stderr spam so damaged PDFs don't flood our logs.
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

from utils_ade import get_native_pdf_lines  # noqa: E402


def _page_health_probe(doc, page_idx: int) -> tuple[bool, str]:
    """Cheap health check run BEFORE expensive text extraction.

    Catches most structurally-damaged pages without triggering MuPDF's
    recovery loop. Pages that pass the probe can still hang during
    extraction (if the content stream is subtly malformed), so the
    parent subprocess still needs a wall-clock timeout — but healthy
    pages finish in <1 ms here, so the probe is effectively free for
    the common case.

    Returns (ok, reason). ok=False means mark the page as damaged and
    skip extraction.
    """
    try:
        if page_idx < 0 or page_idx >= len(doc):
            return False, f"page index {page_idx} out of range (0..{len(doc) - 1})"
        page = doc[page_idx]
        # Page-dict + dims access. Breaks on missing xref, bad kids, etc.
        w, h = page.rect.width, page.rect.height
        if w <= 0 or h <= 0:
            return False, f"non-positive page dimensions: {w}x{h}"
        # Content-stream reference check. Broken pages often have an
        # empty or unresolvable contents array.
        contents = page.get_contents()
        if contents is None:
            return False, "page.get_contents() returned None"
        return True, ""
    except Exception as e:
        return False, f"probe failed: {e}"


def _extract_one(doc, page_idx: int) -> dict:
    """Extract one page from an already-open doc. Runs a cheap health
    probe first, skips expensive extraction if the probe fails."""
    ok, reason = _page_health_probe(doc, page_idx)
    if not ok:
        return {"page_idx": page_idx, "ok": False, "error": f"damaged: {reason}"}
    try:
        page = doc[page_idx]
        lines = get_native_pdf_lines(page)
        return {"page_idx": page_idx, "ok": True, "lines": lines}
    except Exception as e:
        return {"page_idx": page_idx, "ok": False, "error": f"extract failed: {e}"}


def _extract_one_phase1a(doc, page_idx: int) -> dict:
    """Phase 1a payload: plain page text + page dimensions.

    Same contract as _extract_one (probe-then-extract, returns ok+error
    on damaged pages) but tailored to what pipeline.py's Phase 1a needs.
    Kept separate from _extract_one so the legacy line-extraction
    callers (app_ade_prod) keep their exact response shape.
    """
    ok, reason = _page_health_probe(doc, page_idx)
    if not ok:
        return {"page_idx": page_idx, "ok": False, "error": f"damaged: {reason}"}
    try:
        page = doc[page_idx]
        text = page.get_text("text") or ""
        rect = page.rect
        return {
            "page_idx": page_idx,
            "ok": True,
            "text": text,
            "dims": {
                "width": rect.width,
                "height": rect.height,
                "rotation": page.rotation,
            },
        }
    except Exception as e:
        return {"page_idx": page_idx, "ok": False, "error": f"extract failed: {e}"}


def main() -> int:
    """Usage forms:
        page_extractor.py PDF_PATH PAGE_INDEX                  (single page, legacy)
        page_extractor.py PDF_PATH --pages 3,5,7,9,11          (batch, line extraction)
        page_extractor.py PDF_PATH --phase1a-batch 3,5,7,9     (batch, text+dims)

    Legacy output (single-page): {"ok": bool, "lines": [...]} or {"ok": false, "error": ...}
    --pages output:        streams {"page_result": {..., "lines": [...]}} per page, then {"done": true}
    --phase1a-batch output: streams {"page_result": {..., "text": "...", "dims": {...}}} per page, then {"done": true}

    The streaming forms exist so the parent can recover any pages that
    finished before a SIGKILL'd subprocess (e.g. on MuPDF C-level hang)
    without losing them.
    """
    args = sys.argv[1:]
    if len(args) < 2:
        sys.stdout.write(json.dumps({"ok": False,
                                     "error": "usage: page_extractor.py PDF_PATH PAGE_INDEX "
                                              "| PDF_PATH --pages N,N,N "
                                              "| PDF_PATH --phase1a-batch N,N,N"}))
        return 2
    pdf_path = args[0]

    # Parse mode + page indices.
    batch_mode = None  # None = legacy single, "lines" = --pages, "phase1a" = --phase1a-batch
    page_indices = []
    if args[1] == "--pages" and len(args) >= 3:
        batch_mode = "lines"
        list_str = args[2]
    elif args[1] == "--phase1a-batch" and len(args) >= 3:
        batch_mode = "phase1a"
        list_str = args[2]
    else:
        try:
            page_indices = [int(args[1])]
        except ValueError:
            sys.stdout.write(json.dumps({"ok": False,
                                         "error": f"invalid page index: {args[1]!r}"}))
            return 2

    if batch_mode is not None:
        try:
            page_indices = [int(x) for x in list_str.split(",") if x.strip() != ""]
        except ValueError:
            sys.stdout.write(json.dumps({"ok": False,
                                         "error": f"invalid page list: {list_str!r}"}))
            return 2

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": f"open failed: {e}"}))
        return 1

    try:
        if batch_mode is not None:
            extractor = _extract_one if batch_mode == "lines" else _extract_one_phase1a
            for pi in page_indices:
                r = extractor(doc, pi)
                sys.stdout.write(json.dumps({"page_result": r}) + "\n")
                sys.stdout.flush()
            sys.stdout.write(json.dumps({"done": True}) + "\n")
            sys.stdout.flush()
            return 0
        # Legacy single-page response shape (keep existing callers working).
        r = _extract_one(doc, page_indices[0])
        if r["ok"]:
            sys.stdout.write(json.dumps({"ok": True, "lines": r["lines"]}))
            return 0
        sys.stdout.write(json.dumps({"ok": False, "error": r["error"]}))
        return 1
    finally:
        try:
            doc.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
