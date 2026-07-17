#!/usr/bin/env python3
"""Measurement-PDF generator run as a short-lived subprocess.

Why this exists
---------------
`exports.generate_measurement_pdf` calls into PyMuPDF for `draw_line`,
`draw_rect`, etc. On certain malformed source PDFs MuPDF's C-level
content-stream balancer (`pdf_count_q_balance`) enters an unbounded
loop and never returns. Because that runs in C with the GIL held,
nothing else in the Python process can run — the entire async event
loop in the FastAPI server freezes until the process is killed. (See
STATUS.md "Damaged-page hang" entries from May 9 for the same class
of bug on the ingestion path; this worker is the export-side mirror.)

Running the export in a dedicated subprocess fixes that: the parent
imposes a wall-clock timeout via `subprocess.run(timeout=...)` and
SIGKILLs the child if it doesn't return. The parent stays responsive
the whole time.

IPC
---
Input  : one JSON object on stdin describing the task.
Output : one JSON object on stdout {"ok": bool, "out_path": str,
         "size_bytes": int, "wall_s": float, "error": str|null}.
         The actual PDF lives on disk at out_path; the parent reads
         + unlinks it.
Exit   : 0 on success, non-zero on fatal error.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to sys.path so `from exports import ...` resolves
# regardless of where Python was invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _emit(obj) -> None:
    try:
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def main() -> int:
    t0 = time.perf_counter()
    out_path = ""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise RuntimeError("empty stdin — expected JSON task")
        task = json.loads(raw)

        # Quiet MuPDF C-level error spam (matches highlight worker).
        import fitz
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        from exports import MIN_LINE_PTS, generate_measurement_pdf

        out_path = task["out_path"]

        # Stream the PDF to a sibling .tmp, then rename — atomic, so a
        # killed/crashed serialization can't leave a half file the parent
        # mistakes for a complete artifact. Saving to disk (instead of
        # returning bytes) keeps the finished PDF out of this process's
        # AND the parent's memory.
        tmp_path = out_path + ".tmp"
        generate_measurement_pdf(
            pdf_path=task["pdf_path"],
            fence_pages=task.get("fence_pages") or [],
            line_assignments=task.get("line_assignments") or {},
            user_drawn_lines=task.get("user_drawn_lines") or {},
            page_categories=task.get("page_categories") or {},
            uploaded_pdf_name=task.get("uploaded_pdf_name") or "document.pdf",
            min_line_pts=task.get("min_line_pts", MIN_LINE_PTS),
            max_labels_per_page=task.get("max_labels_per_page", 150),
            out_path=tmp_path,
        )
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(
                "generate_measurement_pdf produced no output — source PDF "
                "missing or no fence pages to render"
            )
        os.replace(tmp_path, out_path)

        _emit({
            "ok": True,
            "out_path": out_path,
            "size_bytes": os.path.getsize(out_path),
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": None,
        })
        return 0

    except Exception as e:
        _emit({
            "ok": False,
            "out_path": out_path,
            "size_bytes": 0,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
