#!/usr/bin/env python3
"""
Quick evaluation script for fence detection and highlighting.
Uses Streamlit secrets for API key.
"""
import os, sys, json, time
from pathlib import Path
import fitz
import pandas as pd
from utils import analyze_page
from langchain_openai import ChatOpenAI

# Try to load OpenAI key from Streamlit secrets
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        openai_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
except:
    openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    print("❌ OPENAI_API_KEY not found. Please set it in .streamlit/secrets.toml or environment.")
    print("   Format: OPENAI_API_KEY = \"sk-...\"")
    sys.exit(1)

# Config
FENCE_KEYWORDS = [
    "fence", "fencing", "gate", "barrier", "guardrail",
    "post", "mesh", "panel", "chain link"
]
EXCLUDE_ANNOTS = {"FreeText", "Square"}

def get_ground_truth_pages(pdf_path: Path) -> set[int]:
    """Return 1-based page numbers with annotations (excluding FreeText/Square)."""
    pages = set()
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc, start=1):
        for annot in (page.annots() or []):
            if annot.type[1] not in EXCLUDE_ANNOTS:
                pages.add(page_idx)
                break
    doc.close()
    return pages

def predict_pages(pdf_path: Path, llm_text, recall_mode="balanced") -> tuple[set[int], dict]:
    """
    Predict which pages have fence content.
    Returns: (predicted_pages, detailed_results)
    """
    pages_pred = set()
    page_results = []
    
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc, start=1):
        # For evaluation without Document AI, just use the text layer directly
        # (Don't create PNG wrapper as it loses text layer)
        page_dict = {
            "page_number": page_idx,
            "text": page.get_text("text"),
            # Don't pass page_bytes - it would try to extract from PNG wrapper which has no text layer
        }
        
        # Analyze (no Document AI, just text-layer)
        res = analyze_page(
            page_dict,
            llm_text,
            FENCE_KEYWORDS,
            google_cloud_config=None,
            recall_mode=recall_mode
        )
        
        page_results.append({
            "page": page_idx,
            "fence_found": res["fence_found"],
            "text_found": res.get("text_found", False),
            "confidence": res.get("confidence", 0.0),
            "snippet": res.get("text_snippet", "")[:100] if res.get("text_snippet") else ""
        })
        
        if res["fence_found"]:
            pages_pred.add(page_idx)
    
    doc.close()
    return pages_pred, {"pages": page_results}

def evaluate_pdf(pdf_path: Path, llm_text, recall_mode="balanced"):
    """Evaluate a single PDF."""
    print(f"\n📄 Evaluating: {pdf_path.name}")
    print(f"   Recall mode: {recall_mode}")
    
    gt = get_ground_truth_pages(pdf_path)
    pred, details = predict_pages(pdf_path, llm_text, recall_mode)
    
    total_pages = fitz.open(pdf_path).page_count
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    tn = total_pages - tp - fp - fn
    
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / total_pages if total_pages else 0.0
    
    # Print results
    print(f"\n📊 Results:")
    print(f"   Total pages: {total_pages}")
    print(f"   Ground truth (pages with highlights): {sorted(gt)}")
    print(f"   Predicted (fence detected): {sorted(pred)}")
    print(f"\n   ✅ True Positives:  {tp}")
    print(f"   ❌ False Positives: {fp}")
    print(f"   ⚠️  False Negatives: {fn}")
    print(f"   ✓  True Negatives:  {tn}")
    print(f"\n   📈 Metrics:")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   Accuracy:  {acc:.3f}")
    
    # Show false positives and false negatives
    if fp > 0:
        fp_pages = sorted(pred - gt)
        print(f"\n   🔍 False Positives (predicted fence, but no highlights): {fp_pages}")
        for p in fp_pages[:3]:  # Show first 3
            page_info = next((r for r in details["pages"] if r["page"] == p), None)
            if page_info:
                print(f"      Page {p}: conf={page_info['confidence']:.2f}, snippet=\"{page_info['snippet']}\"")
    
    if fn > 0:
        fn_pages = sorted(gt - pred)
        print(f"\n   🔍 False Negatives (has highlights, but not detected): {fn_pages}")
    
    return {
        "pdf": pdf_path.name,
        "recall_mode": recall_mode,
        "total_pages": total_pages,
        "gt_pages": sorted(gt),
        "pred_pages": sorted(pred),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc
    }

def main():
    """Run evaluation on gold standard subset."""
    gold_dir = Path("/Users/parhamhamouni/Desktop/leo/data/gold_standard/subset_gold")
    
    if not gold_dir.exists():
        print(f"❌ Gold standard directory not found: {gold_dir}")
        sys.exit(1)
    
    pdf_files = sorted(gold_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {gold_dir}")
        sys.exit(1)
    
    print(f"🔍 Found {len(pdf_files)} PDF(s) to evaluate")
    print(f"🔑 Using OpenAI API key: {openai_key[:15]}...")
    
    # Initialize LLM
    llm_text = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key)
    
    # Test different recall modes
    recall_modes = ["strict", "balanced", "high"]
    all_results = []
    
    for pdf_path in pdf_files:
        # Skip the "no_annotations" version
        if "no_annotations" in pdf_path.name:
            print(f"\n⏭️  Skipping {pdf_path.name} (no annotations reference)")
            continue
        
        for mode in recall_modes:
            start = time.time()
            result = evaluate_pdf(pdf_path, llm_text, recall_mode=mode)
            result["time_sec"] = time.time() - start
            all_results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY - Performance by Recall Mode")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    summary = df.groupby("recall_mode").agg({
        "precision": "mean",
        "recall": "mean", 
        "f1": "mean",
        "accuracy": "mean",
        "time_sec": "sum"
    }).round(3)
    
    print("\n" + summary.to_string())
    
    # Save results
    output_file = gold_dir / "evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Best mode
    best_mode = df.loc[df["f1"].idxmax()]
    print(f"\n🏆 Best F1 Score: {best_mode['f1']:.3f} (mode={best_mode['recall_mode']})")
    print(f"   Precision: {best_mode['precision']:.3f}")
    print(f"   Recall: {best_mode['recall']:.3f}")

if __name__ == "__main__":
    main()

