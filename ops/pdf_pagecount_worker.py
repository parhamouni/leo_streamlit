#!/usr/bin/env python3
"""Page-count peek worker for the upload endpoint.

Opening an untrusted PDF just to read its page count can still drive PyMuPDF
into a C-level recovery loop on malformed content. Doing that inline in the
FastAPI handler would block the event loop (and therefore every other
request) until it gave up. Running it in a child process lets the API parent
enforce a hard wall-clock timeout (SIGKILL) — same isolation pattern as
export_vector_worker.py / measurement_pdf_worker.py.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _emit(obj) -> None:
    try:
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def main() -> int:
    t0 = time.perf_counter()
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise RuntimeError("empty stdin - expected JSON task")
        task = json.loads(raw)
        pdf_path = task["pdf_path"]

        import fitz

        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        doc = None
        try:
            doc = fitz.open(pdf_path)
            n_pages = int(doc.page_count)
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        _emit({
            "ok": True,
            "page_count": n_pages,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": None,
        })
        return 0

    except Exception as e:
        _emit({
            "ok": False,
            "page_count": 0,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
