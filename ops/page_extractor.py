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


def main() -> int:
    if len(sys.argv) != 3:
        sys.stdout.write(json.dumps({"ok": False, "error": "usage: page_extractor.py PDF_PATH PAGE_INDEX"}))
        return 2
    pdf_path = sys.argv[1]
    try:
        page_idx = int(sys.argv[2])
    except ValueError:
        sys.stdout.write(json.dumps({"ok": False, "error": f"invalid page index: {sys.argv[2]!r}"}))
        return 2

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": f"open failed: {e}"}))
        return 1

    try:
        if page_idx < 0 or page_idx >= len(doc):
            sys.stdout.write(json.dumps({"ok": False, "error": f"page index {page_idx} out of range (0..{len(doc) - 1})"}))
            return 1
        page = doc[page_idx]
        lines = get_native_pdf_lines(page)
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": f"extract failed: {e}"}))
        return 1
    finally:
        try:
            doc.close()
        except Exception:
            pass

    sys.stdout.write(json.dumps({"ok": True, "lines": lines}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
