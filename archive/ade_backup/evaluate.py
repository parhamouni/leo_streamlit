# evaluate.py
import os, sys, json, base64, time, argparse
from pathlib import Path
import fitz                        # PyMuPDF
import pandas as pd
from utils import analyze_page     # ← your existing function
from langchain_openai import ChatOpenAI
from rich import print as rprint   # for pretty console logs

# ------------- config -------------
EXCLUDE_ANNOTS = {"FreeText", "Square"}
FENCE_KEYWORDS = [
    "fence", "fencing", "gate", "barrier", "guardrail",
    "post", "mesh", "panel", "chain link"
]
MODEL_TEXT = "gpt-4o"
MODEL_VISION = "gpt-4-turbo"       # set None to skip vision
# ----------------------------------

def ground_truth_pages(pdf_path: Path) -> set[int]:
    """Return 1-based page numbers that contain *kept* annotations."""
    pages = set()
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc, start=1):
        for annot in (page.annots() or []):
            if annot.type[1] not in EXCLUDE_ANNOTS:
                pages.add(page_idx)
                break
    doc.close()
    return pages

def predict_pages(
    pdf_path: Path,
    llm_text,
    llm_vision=None,
    use_vision=False
) -> set[int]:
    pages_pred = set()
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc, start=1):
        page_dict = {
            "page_number": page_idx,
            "text": page.get_text("text")
        }
        if use_vision and llm_vision:
            pix = page.get_pixmap(dpi=72, alpha=False)
            page_dict["image_b64"] = base64.b64encode(pix.tobytes("png")).decode()
        res = analyze_page(
            page_dict,
            llm_text,
            llm_vision if use_vision else None,
            FENCE_KEYWORDS
        )
        if res["fence_found"]:
            pages_pred.add(page_idx)
    doc.close()
    return pages_pred

def evaluate_one(pdf_path: Path, llm_text, llm_vision, use_vision) -> dict:
    gt = ground_truth_pages(pdf_path)
    pred = predict_pages(pdf_path, llm_text, llm_vision, use_vision)
    total_pages = fitz.open(pdf_path).page_count
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    tn = total_pages - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc       = (tp + tn) / total_pages if total_pages else 0.0
    return dict(
        pdf=str(pdf_path),
        total_pages=total_pages,
        pages_with_annots=sorted(gt),
        pages_predicted=sorted(pred),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, accuracy=acc
    )

def main(root: Path, use_vision: bool, max_mb: int):
    """
    Evaluate every PDF in `root` (recursively).

    Skips any file larger than `max_mb` MB, builds per-PDF and overall
    metrics, and saves:
        • pdf_summary.csv
        • pages_eval.csv
    """
    start = time.time()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        sys.exit("OPENAI_API_KEY env var required.")

    llm_text   = ChatOpenAI(model=MODEL_TEXT,   temperature=0, openai_api_key=openai_key)
    llm_vision = ChatOpenAI(model=MODEL_VISION, temperature=0, openai_api_key=openai_key) if use_vision else None

    size_cap = max_mb * 1024 * 1024  # bytes
    pdf_paths = sorted(root.rglob("*.pdf"))

    all_rows = []
    for pdf in pdf_paths:
        if pdf.stat().st_size > size_cap:
            rprint(f"[dim]↷ Skipping {pdf} (> {max_mb} MB)")
            continue
        rprint(f"[yellow]→ Evaluating[/] {pdf}")
        row = evaluate_one(pdf, llm_text, llm_vision, use_vision)
        all_rows.append(row)

    pdf_df = pd.DataFrame(all_rows)

    # ---------- build page-level dataframe ----------
    page_rows = (
        [dict(pdf=r["pdf"], page=n, is_gt=1)   for r in all_rows for n in r["pages_with_annots"]] +
        [dict(pdf=r["pdf"], page=n, is_pred=1) for r in all_rows for n in r["pages_predicted"]]
    )
    page_df = (
        pd.DataFrame(page_rows)
          .groupby(["pdf", "page"], as_index=False)
          .sum(numeric_only=True)
    )

    # guarantee both flag columns exist
    for col in ("is_gt", "is_pred"):
        if col not in page_df.columns:
            page_df[col] = 0
    page_df = page_df.astype({"is_gt": "int8", "is_pred": "int8"})

    # ---------- overall confusion ----------
    tp = page_df.query("is_gt == 1 and is_pred == 1").shape[0]
    fp = page_df.query("is_gt == 0 and is_pred == 1").shape[0]
    fn = page_df.query("is_gt == 1 and is_pred == 0").shape[0]
    tn = page_df.query("is_gt == 0 and is_pred == 0").shape[0]

    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc  = (tp + tn) / page_df.shape[0] if page_df.shape[0] else 0

    # ---------- save ----------
    page_df.to_csv("pages_eval.csv", index=False)
    pdf_df.to_csv("pdf_summary.csv", index=False)

    rprint("\n[bold cyan]=== PER-PDF SUMMARY ===[/]")
    if not pdf_df.empty:
        rprint(pdf_df[["pdf", "precision", "recall", "f1", "accuracy"]])

    rprint("\n[bold green]=== OVERALL ===[/]")
    rprint(dict(precision=prec, recall=rec, f1=f1, accuracy=acc,
                tp=tp, fp=fp, fn=fn, tn=tn,
                total_pages=page_df.shape[0],
                elapsed=f"{time.time()-start:.1f}s"))


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate fence-detection accuracy")
    p.add_argument("--root",   required=True, help="Folder with PDFs (recursive)")
    p.add_argument("--vision", action="store_true", help="Enable vision model pass")
    p.add_argument("--max-mb", type=int, default=200,
                   help="Skip PDFs larger than this size (MB)")
    args = p.parse_args()

    main(Path(args.root).expanduser(), use_vision=args.vision, max_mb=args.max_mb)

