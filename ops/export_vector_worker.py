#!/usr/bin/env python3
"""Vector-line extraction worker for export endpoints.

FastAPI uses this worker when UMT edits need saved line indices resolved
back to PDF vector geometry. PyMuPDF can hang in C on malformed content;
running this in a child process lets the API parent enforce a timeout.
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


def _line_to_dict(v) -> dict:
    return {
        "start": [float(v.start[0]), float(v.start[1])],
        "end": [float(v.end[0]), float(v.end[1])],
        "length_pts": float(v.length_pts),
        "layer": v.layer or "",
    }


def main() -> int:
    t0 = time.perf_counter()
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise RuntimeError("empty stdin - expected JSON task")
        task = json.loads(raw)
        pdf_path = task["pdf_path"]
        page_indices = [int(x) for x in (task.get("page_indices") or [])]

        import fitz
        from utils_vector import extract_vector_lines

        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        lines_by_page: dict[str, list[dict]] = {}
        doc = None
        try:
            doc = fitz.open(pdf_path)
            for page_idx in page_indices:
                if not (0 <= page_idx < len(doc)):
                    lines_by_page[str(page_idx)] = []
                    continue
                vlines = extract_vector_lines(doc[page_idx], apply_rotation=True)
                lines_by_page[str(page_idx)] = [_line_to_dict(v) for v in vlines]
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        _emit({
            "ok": True,
            "lines_by_page": lines_by_page,
            "page_count": len(page_indices),
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": None,
        })
        return 0

    except Exception as e:
        _emit({
            "ok": False,
            "lines_by_page": {},
            "page_count": 0,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
