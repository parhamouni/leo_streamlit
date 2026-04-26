#!/usr/bin/env python3
"""Combined-highlighted-PDF builder run as a short-lived subprocess.

Why this exists
---------------
generate_combined_highlighted_pdf used to run inline in the Streamlit
process. For a 144 MB source deck with 48 fence pages, fitz peaks the
parent's RSS at ~2.5 GB transiently while it copies pages, draws
rectangles, and serializes the new PDF. The bytes sit in
st.session_state.highlighted_pdf_bytes_for_download afterward. Even
once we've handed them to the user via st.download_button, glibc
typically doesn't return the full peak to the OS — so the main
process holds onto ~1 GB of "high-water mark" for the rest of the
session.

Running the work in a subprocess fixes that: the child does the
peak, then exits, and the OS reclaims every byte the child held.
The parent only ever sees the final PDF bytes (already much smaller
than the working memory the child needed) plus a small task-JSON.

IPC
---
Input  : one JSON object on stdin describing the task.
Output : one JSON object on stdout {"ok": bool, "out_path": str,
         "size_bytes": int, "error": str|null}. The actual PDF lives
         on disk at out_path; the parent reads + unlinks it.
Exit   : 0 on success, non-zero on fatal error.

Task schema
-----------
{
  "pdf_path":      "/tmp/fence_pdfs/<user>/<session>_<sha>.pdf",
  "out_path":      "/tmp/<sessionid>_combined.pdf",
  "uploaded_name": "Construction Documents (1).pdf",
  "fence_pages":   [
      { "page_index_in_original_doc": 0,
        "definitions": [{"x0":..., "y0":..., "x1":..., "y1":...}, ...],
        "instances":   [{"x0":..., "y0":..., "x1":..., "y1":...}, ...],
        "keyword_matches": [{"x0":..., "y0":..., "x1":..., "y1":...}, ...]
      },
      ...
  ]
}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback


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

        # Quiet MuPDF C-level error spam.
        import fitz
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

        pdf_path     = task["pdf_path"]
        out_path     = task["out_path"]
        fence_pages  = task.get("fence_pages", []) or []

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"source PDF not found: {pdf_path}")

        input_doc = fitz.open(pdf_path)
        output_doc = fitz.open()

        sorted_pages = sorted(
            fence_pages,
            key=lambda x: x.get("page_index_in_original_doc", float("inf")),
        )

        for res in sorted_pages:
            page_idx = res.get("page_index_in_original_doc")
            if page_idx is None:
                continue
            try:
                output_doc.insert_pdf(input_doc, from_page=page_idx, to_page=page_idx)
                page_out = output_doc.load_page(len(output_doc) - 1)

                rotation   = page_out.rotation
                mediabox_w = page_out.mediabox.width
                mediabox_h = page_out.mediabox.height

                def reverse_rotation(x0, y0, x1, y1,
                                     _rot=rotation, _w=mediabox_w, _h=mediabox_h):
                    """Display-space → MediaBox-space (same logic as the
                    inline version in app_ade_fast.py)."""
                    if _rot == 0:
                        return x0, y0, x1, y1
                    if _rot == 90:
                        return y0, _h - x1, y1, _h - x0
                    if _rot == 180:
                        return _w - x1, _h - y1, _w - x0, _h - y0
                    if _rot == 270:
                        return _w - y1, x0, _w - y0, x1
                    return x0, y0, x1, y1

                # Definitions — green rectangles.
                for d in res.get("definitions", []) or []:
                    if not all(k in d for k in ("x0", "y0", "x1", "y1")):
                        continue
                    mx0, my0, mx1, my1 = reverse_rotation(d["x0"], d["y0"], d["x1"], d["y1"])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(0, 0.9, 0), width=2.0, overlay=True)

                # Instances — purple rectangles.
                for inst in res.get("instances", []) or []:
                    if not all(k in inst for k in ("x0", "y0", "x1", "y1")):
                        continue
                    mx0, my0, mx1, my1 = reverse_rotation(inst["x0"], inst["y0"], inst["x1"], inst["y1"])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(0.9, 0, 0.9), width=2.0, overlay=True)

                # Keyword fallback — orange rectangles.
                for kw in res.get("keyword_matches", []) or []:
                    if not all(k in kw for k in ("x0", "y0", "x1", "y1")):
                        continue
                    mx0, my0, mx1, my1 = reverse_rotation(kw["x0"], kw["y0"], kw["x1"], kw["y1"])
                    r = fitz.Rect(mx0, my0, mx1, my1)
                    r.normalize()
                    if not r.is_empty and r.is_valid:
                        page_out.draw_rect(r, color=(1.0, 0.65, 0), width=2.0, overlay=True)
            except Exception as pg_e:
                print(f"[highlight_pdf_worker] page {page_idx + 1} draw error: {pg_e}",
                      file=sys.stderr)

        if len(output_doc) == 0:
            raise RuntimeError("output PDF has 0 pages — nothing to write")

        # Atomic write so a crashed serialization can't leave a half
        # file the parent then mistakes for a complete result.
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(output_doc.tobytes(garbage=2, deflate=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)

        try:
            input_doc.close()
        except Exception:
            pass
        try:
            output_doc.close()
        except Exception:
            pass

        size_bytes = os.path.getsize(out_path)
        _emit({
            "ok": True,
            "out_path": out_path,
            "size_bytes": size_bytes,
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
