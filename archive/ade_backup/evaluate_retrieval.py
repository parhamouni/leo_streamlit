import os, sys, json, base64, time, argparse
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from utils import analyze_page
from langchain_openai import ChatOpenAI
from rich import print as rprint

# ----------------- config -----------------
POSITIVE_ANNOTS = {"Highlight", "Line", "Polygon", "Stamp", "Ink", "PolyLine"}
FENCE_KEYWORDS = [
    "fence", "fencing", "gate", "barrier", "guardrail",
    "post", "chain link", "mesh", "panel", "chain link"
]
MODEL_TEXT = "gpt-4o"
# ------------------------------------------

def load_google_cloud_config():
    """Load Google Cloud configuration from environment or secrets file."""
    try:
        # Try to load from Streamlit secrets file if available
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            if "google_cloud" in secrets and "gcp_service_account" in secrets:
                google_cloud_config = {
                    "project_number": secrets["google_cloud"]["project_number"],
                    "location": secrets["google_cloud"]["location"], 
                    "processor_id": secrets["google_cloud"]["processor_id"],
                    "service_account_info": dict(secrets["gcp_service_account"])
                }
                print("✅ Google Cloud config loaded from .streamlit/secrets.toml")
                return google_cloud_config
    except ImportError:
        print("⚠️ toml library not available, trying environment variables")
    except Exception as e:
        print(f"⚠️ Failed to load from secrets file: {e}")
    
    # Fallback: try environment variables
    try:
        project_number = os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us")
        processor_id = os.getenv("GOOGLE_CLOUD_PROCESSOR_ID")
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if project_number and processor_id and service_account_path and Path(service_account_path).exists():
            with open(service_account_path) as f:
                service_account_info = json.load(f)
            
            google_cloud_config = {
                "project_number": project_number,
                "location": location,
                "processor_id": processor_id,
                "service_account_info": service_account_info
            }
            print("✅ Google Cloud config loaded from environment variables")
            return google_cloud_config
    except Exception as e:
        print(f"⚠️ Failed to load from environment: {e}")
    
    print("⚠️ No Google Cloud configuration found. Comprehensive text extraction will be disabled.")
    return None

def ground_truth_pages(pdf_path: Path) -> set[int]:
    """Return 1‑based page numbers with at least one positive annotation."""
    pages: set[int] = set()
    doc = fitz.open(pdf_path)
    for idx, pg in enumerate(doc, start=1):
        for annot in (pg.annots() or []):
            if annot.type[1] in POSITIVE_ANNOTS:
                pages.add(idx)
                break
    doc.close()
    return pages


def analyze_all_pages(
    pdf_path: Path,
    llm_text,
    google_cloud_config=None,
) -> list[dict]:
    """Run the LLM pipeline over every page and collect rich diagnostics."""
    gt_pages = ground_truth_pages(pdf_path)
    doc = fitz.open(pdf_path)
    rows: list[dict] = []
    
    for page_idx in range(1, doc.page_count + 1):
        page = doc.load_page(page_idx - 1)
        
        # Create single-page PDF bytes for comprehensive text extraction
        single_page_pdf_bytes = None
        try:
            temp_doc = fitz.open()
            temp_doc.insert_pdf(doc, from_page=page_idx - 1, to_page=page_idx - 1)
            single_page_pdf_bytes = temp_doc.tobytes()
            temp_doc.close()
        except Exception as e:
            print(f"⚠️ Could not create single page PDF for page {page_idx}: {e}")
        
        page_dict = {
            "page_number": page_idx,
            "text": page.get_text("text"),
            "page_bytes": single_page_pdf_bytes
        }

        res = analyze_page(page_dict, llm_text, FENCE_KEYWORDS, google_cloud_config)

        rows.append(
            {
                "pdf": str(pdf_path),
                "page": page_idx,
                "is_gt": int(page_idx in gt_pages),
                "is_pred": int(res.get("fence_found", False)),
                "text_found": int(res.get("text_found", False)),
                "text_input": page_dict['text'],
                "text_response": res.get("text_response"),
                "text_snippet": res.get("text_snippet"),
                "extraction_method": res.get("extraction_method", "unknown"),
                "extraction_stats": res.get("extraction_stats", {}),
                "comprehensive_text": res.get("comprehensive_text", ""),
            }
        )
    doc.close()
    return rows


def build_pdf_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate page‑level GT/PRED flags into per‑PDF metrics."""
    summary_rows = []
    for pdf, grp in df.groupby("pdf"):
        tp = grp.query("is_gt == 1 and is_pred == 1").shape[0]
        fp = grp.query("is_gt == 0 and is_pred == 1").shape[0]
        fn = grp.query("is_gt == 1 and is_pred == 0").shape[0]
        tn = grp.query("is_gt == 0 and is_pred == 0").shape[0]
        total = grp.shape[0]
        
        # Text-only metrics
        tp_text = grp.query("is_gt == 1 and text_found == 1").shape[0]
        fp_text = grp.query("is_gt == 0 and text_found == 1").shape[0]
        fn_text = grp.query("is_gt == 1 and text_found == 0").shape[0]
        
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / total if total else 0.0
        
        prec_text = tp_text / (tp_text + fp_text) if tp_text + fp_text else 0.0
        rec_text = tp_text / (tp_text + fn_text) if tp_text + fn_text else 0.0
        f1_text = 2 * prec_text * rec_text / (prec_text + rec_text) if prec_text + rec_text else 0.0
        
        # Extraction method statistics
        comprehensive_pages = grp.query("extraction_method == 'comprehensive'").shape[0]
        fallback_pages = grp.query("extraction_method == 'fallback'").shape[0]
        
        summary_rows.append(
            {
                "pdf": pdf,
                "total_pages": total,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "precision_text": prec_text,
                "recall_text": rec_text,
                "f1_text": f1_text,
                "comprehensive_pages": comprehensive_pages,
                "fallback_pages": fallback_pages,
            }
        )
    return pd.DataFrame(summary_rows)


def print_overall(summary_df: pd.DataFrame, total_pages: int, elapsed_s: float):
    if summary_df.empty:
        rprint("[bold red]No PDFs processed.")
        return
    
    tp = summary_df["tp"].sum()
    fp = summary_df["fp"].sum()
    fn = summary_df["fn"].sum()
    tn = summary_df["tn"].sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / total_pages if total_pages else 0.0
    
    comprehensive_total = summary_df["comprehensive_pages"].sum()
    fallback_total = summary_df["fallback_pages"].sum()
    
    rprint("\n[bold green]=== OVERALL PERFORMANCE ===[/]")
    rprint({
        "precision": f"{prec:.3f}",
        "recall": f"{rec:.3f}",
        "f1": f"{f1:.3f}",
        "accuracy": f"{acc:.3f}",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pages": total_pages,
        "elapsed": f"{elapsed_s:.1f}s",
    })
    
    rprint("\n[bold blue]=== EXTRACTION METHODS ===[/]")
    rprint({
        "comprehensive_pages": comprehensive_total,
        "fallback_pages": fallback_total,
        "comprehensive_ratio": f"{comprehensive_total/total_pages:.3f}" if total_pages else "0.000"
    })

def main(root: Path, max_mb: int):
    start = time.time()
    
    # Load API key and config
    api_key = None
    try:
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            api_key = secrets.get("OPENAI_API_KEY")
            if api_key:
                print("✅ OpenAI API key loaded from .streamlit/secrets.toml")
    except Exception as e:
        print(f"⚠️ Failed to load OpenAI key from secrets: {e}")
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("✅ OpenAI API key loaded from environment variable")
    
    if not api_key:
        sys.exit("OPENAI_API_KEY not found in secrets file or environment variables.")

    google_cloud_config = load_google_cloud_config()

    llm_text = ChatOpenAI(model=MODEL_TEXT, temperature=0, openai_api_key=api_key)
    page_rows = []
    
    size_cap = max_mb * 1024 * 1024
    pdf_paths = sorted(root.rglob("*.pdf"))
    
    for pdf in pdf_paths:
        if pdf.stat().st_size > size_cap:
            rprint(f"[dim]↷ Skipping {pdf} (> {max_mb} MB)")
            continue
        rprint(f"[yellow]→ Evaluating[/] {pdf}")
        page_rows.extend(analyze_all_pages(pdf, llm_text, google_cloud_config))

    df_pages = pd.DataFrame(page_rows).sort_values(["pdf", "page"])
    df_pages.to_csv("pages_eval.csv", index=False)

    df_summary = build_pdf_summary(df_pages)
    df_summary.to_csv("pdf_summary.csv", index=False)

    rprint("\n[bold cyan]=== EVALUATION SUMMARY ===[/]")
    if not df_summary.empty:
        rprint(df_summary[["pdf", "precision", "recall", "f1", "accuracy", "f1_text", "comprehensive_pages"]])

    print_overall(df_summary, df_pages.shape[0], time.time() - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fence detection evaluation")
    parser.add_argument("--root", required=True, help="Folder with PDFs (recursive)")
    parser.add_argument("--max-mb", type=int, default=200, help="Skip PDFs larger than this size (MB)")
    args = parser.parse_args()

    main(Path(args.root).expanduser(), max_mb=args.max_mb)
