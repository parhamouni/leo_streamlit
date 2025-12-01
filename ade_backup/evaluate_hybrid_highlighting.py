#!/usr/bin/env python3
"""
Evaluation script for hybrid ADE + Google OCR highlighting system.

Compares predicted highlights against gold standard annotations.
Calculates precision, recall, F1, and IoU metrics.
"""
import os
import sys
import json
import ast
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
import fitz  # PyMuPDF
from io import BytesIO

# Import hybrid highlighting functions
# Try to import ADE functions, but handle gracefully if missing
try:
    from utils_ade_official import (
        ade_parse_document_official,
        align_ade_chunks_to_page,
        get_google_ocr_results_with_boxes,
        extract_legend_keywords_and_indicators,
        find_indicators_in_figures,
        extract_indicators_from_legend_text,
        extract_indicators_from_legend_text_llm,
        identify_legend_tables,
        get_page_dimensions,
        create_single_page_pdf,
    )
    ADE_AVAILABLE = True
except ImportError:
    # ADE functions not available - use fallback implementations
    ADE_AVAILABLE = False
    print("⚠️ ADE functions not available - using LLM-only approach")
    
    # Create stub functions
    def ade_parse_document_official(*args, **kwargs):
        return {"success": False, "data": None, "error": "ADE not available"}
    
    def align_ade_chunks_to_page(*args, **kwargs):
        return []
    
    # Import other functions that should exist in utils_ade_official.py
    try:
        from utils_ade_official import (
            get_google_ocr_results_with_boxes,
            extract_legend_keywords_and_indicators,
            find_indicators_in_figures,
            get_page_dimensions,
            create_single_page_pdf,
        )
    except ImportError:
        # If these are also missing, import from utils.py or create stubs
        from utils import extract_comprehensive_text_from_page
        import fitz
        
        def get_google_ocr_results_with_boxes(*args, **kwargs):
            return []
        
        def extract_legend_keywords_and_indicators(*args, **kwargs):
            return []
        
        def find_indicators_in_figures(*args, **kwargs):
            return []
        
        def get_page_dimensions(page_bytes):
            doc = fitz.open(stream=BytesIO(page_bytes), filetype="pdf")
            page = doc[0]
            return page.rect.width, page.rect.height
        
        def create_single_page_pdf(pdf_bytes, page_idx):
            doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            temp_doc = fitz.open()
            temp_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            result = temp_doc.tobytes()
            temp_doc.close()
            doc.close()
            return result

# Import LLM extraction functions
try:
    from utils_ade_official import (
        extract_indicators_from_table_llm,
        extract_indicators_from_text_llm
    )
except ImportError:
    # Stub implementations
    def extract_indicators_from_table_llm(*args, **kwargs):
        return []
    def extract_indicators_from_text_llm(*args, **kwargs):
        return []

from langchain_openai import ChatOpenAI
from utils import UnrecoverableRateLimitError

# Configuration
FENCE_KEYWORDS = [
    'fence', 'fencing', 'gate', 'barrier', 'guardrail', 'post', 
    'mesh', 'panel', 'chain link', 'masonry', 'fence details', 
    'canopy shading', 'adot specifications', 'mag specifications', 
    'rail', 'railing', 'bollards', 'handrails', 'wall', 'cmu', 'keynote'
]

# IoU threshold for considering a match
# Lowered from 0.3 to 0.1 to account for:
# - OCR boxes being larger than gold annotations
# - Small annotation boxes (18x38 pixels) having low IoU with larger OCR boxes
IOU_THRESHOLD = 0.1


def load_google_cloud_config():
    """Load Google Cloud configuration from secrets."""
    try:
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            if "google_cloud" in secrets and "gcp_service_account" in secrets:
                return {
                    "project_number": secrets["google_cloud"]["project_number"],
                    "location": secrets["google_cloud"]["location"], 
                    "processor_id": secrets["google_cloud"]["processor_id"],
                    "service_account_info": dict(secrets["gcp_service_account"])
                }
    except Exception as e:
        print(f"⚠️ Failed to load Google Cloud config: {e}")
    return None


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bounding box string to (x0, y0, x1, y1)."""
    try:
        bbox_list = ast.literal_eval(bbox_str)
        if len(bbox_list) == 4:
            return tuple(float(x) for x in bbox_list)
    except Exception as e:
        print(f"⚠️ Error parsing bbox '{bbox_str}': {e}")
    return None


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x0_i >= x1_i or y0_i >= y1_i:
        return 0.0
    
    intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
    
    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def load_gold_standard(csv_path: Path) -> Dict[int, List[Dict]]:
    """
    Load gold standard annotations from CSV.
    
    Returns:
        Dict mapping page number (1-based) to list of annotation dicts with:
        - bbox: (x0, y0, x1, y1)
        - content: text content
        - type: annotation type
    """
    df = pd.read_csv(csv_path)
    gold_annotations = {}
    
    for _, row in df.iterrows():
        page_num = int(row['page'])
        bbox_str = row.get('bbox', '')
        content = str(row.get('content', '')).strip()
        annot_type = row.get('type_name', 'Highlight')
        
        # Skip empty annotations or non-highlight types if filtering
        if not bbox_str or annot_type not in ['Highlight', 'Ink']:
            continue
        
        bbox = parse_bbox(bbox_str)
        if bbox is None:
            continue
        
        if page_num not in gold_annotations:
            gold_annotations[page_num] = []
        
        gold_annotations[page_num].append({
            'bbox': bbox,
            'content': content,
            'type': annot_type
        })
    
    return gold_annotations


def run_hybrid_highlighting(
    pdf_path: Path,
    ade_api_key: str,
    google_cloud_config: Dict,
    llm_instance,
    fence_keywords: List[str]
) -> Dict[int, List[Dict]]:
    """
    Run hybrid ADE + Google OCR highlighting on PDF.
    
    Returns:
        Dict mapping page number (1-based) to list of highlight boxes with:
        - bbox: (x0, y0, x1, y1)
        - text: text content
        - type: highlight type
    """
    predictions = {}
    
    # Read PDF
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    total_pages = len(doc)
    
    print(f"📄 Processing {total_pages} pages with hybrid highlighting...")
    
    # Step 1: Process full document with ADE (if available)
    full_doc_ade_result = None
    if ADE_AVAILABLE:
        print("🔍 Running ADE on full document...")
        full_doc_ade_result = ade_parse_document_official(
            pdf_bytes,
            ade_api_key,
            zdr=False
        )
        
        if not full_doc_ade_result.get("success"):
            print(f"⚠️ ADE processing failed: {full_doc_ade_result.get('error')}, continuing with LLM-only approach")
            full_doc_ade_result = None
    else:
        print("⚠️ ADE not available - using LLM-only approach for extraction")
    
    if full_doc_ade_result and full_doc_ade_result.get("success"):
        print(f"✅ ADE processed {full_doc_ade_result['data'].get('total_pages', 0)} pages")
    
    # Step 2: Process each page
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        print(f"  Processing page {page_num}/{total_pages}...")
        
        # Get page dimensions
        single_page_bytes = create_single_page_pdf(pdf_bytes, page_idx)
        page_width, page_height = get_page_dimensions(single_page_bytes)
        
        # Align ADE chunks to this page
        page_chunks = []
        if full_doc_ade_result and full_doc_ade_result.get("success"):
            page_chunks = align_ade_chunks_to_page(
                full_doc_ade_result,
                page_idx,
                page_width,
                page_height
            )
        
        # If no ADE chunks, still try to process with OCR + LLM
        if not page_chunks:
            print(f"    ⚠️ No ADE chunks found for page {page_num}, using OCR-only approach")
            # Continue processing - we'll use OCR results directly
        
        # Run Google OCR on this page
        google_ocr_results = get_google_ocr_results_with_boxes(
            single_page_bytes,
            google_cloud_config,
            page_num
        )
        
        if not google_ocr_results:
            print(f"    ⚠️ No Google OCR results for page {page_num}")
            predictions[page_num] = []
            continue
        
        # Extract PDF text layer words as fallback source
        pdf_text_layer_words = []
        try:
            text_layer_doc = fitz.open(stream=BytesIO(single_page_bytes), filetype="pdf")
            if len(text_layer_doc) > 0:
                page = text_layer_doc[0]
                words = page.get_text("words")
                for word_tuple in words:
                    x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
                    pdf_text_layer_words.append({
                        "text": text,
                        "x0": float(x0),
                        "y0": float(y0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "block_no": block_no,
                        "line_no": line_no
                    })
            text_layer_doc.close()
        except Exception as e:
            print(f"    ⚠️ Error extracting PDF text layer: {e}")
        
        # Extract legend highlights (pass LLM for intelligent extraction)
        # If no ADE chunks, pass empty list - function will still work with OCR + text layer
        legend_highlights = extract_legend_keywords_and_indicators(
            page_chunks if page_chunks else [],
            google_ocr_results,
            fence_keywords,
            page_width,
            page_height,
            llm=llm_instance,
            pdf_text_layer_words=pdf_text_layer_words if pdf_text_layer_words else None
        )
        # Note: extract_legend_keywords_and_indicators now handles OCR-only mode internally
        
        # Extract indicators from the legend highlights (they're extracted inside the function now)
        # We need to get them separately for figure matching
        # Import the new extraction functions
        from utils_ade_official import extract_indicators_from_table_llm, extract_indicators_from_text_llm
        
        indicators_from_legend = []
        if llm_instance and page_chunks:
            # Process tables first - use markdown if available, fallback to text
            for chunk in page_chunks:
                chunk_type = chunk.get("type", "").lower()
                if chunk_type == "table":
                    table_markdown = chunk.get("markdown", chunk.get("text", ""))
                    if table_markdown:
                        # Debug: show table preview
                        if page_num <= 3:
                            preview = table_markdown[:200].replace('\n', ' ')
                            print(f"    DEBUG Page {page_num}: Processing table chunk ({len(table_markdown)} chars): {preview}...")
                        table_indicators = extract_indicators_from_table_llm(table_markdown, fence_keywords, llm_instance)
                        indicators_from_legend.extend(table_indicators)
                        if page_num <= 3 and table_indicators:
                            print(f"    DEBUG Page {page_num}: Table extracted {len(table_indicators)} indicators: {[ind['indicator'] for ind in table_indicators]}")
            
            # Process text chunks - use markdown if available
            for chunk in page_chunks:
                chunk_type = chunk.get("type", "").lower()
                if chunk_type == "text":
                    text_content = chunk.get("markdown", chunk.get("text", ""))
                    if text_content:
                        # Check if this text seems legend-related
                        text_lower = text_content.lower()
                        if any(kw in text_lower for kw in ["keynote", "legend", "note", "symbol"]) or len(text_content) < 500:
                            if page_num <= 3:
                                preview = text_content[:200].replace('\n', ' ')
                                print(f"    DEBUG Page {page_num}: Processing text chunk ({len(text_content)} chars): {preview}...")
                            text_indicators = extract_indicators_from_text_llm(text_content, fence_keywords, llm_instance)
                            indicators_from_legend.extend(text_indicators)
                            if page_num <= 3 and text_indicators:
                                print(f"    DEBUG Page {page_num}: Text extracted {len(text_indicators)} indicators: {[ind['indicator'] for ind in text_indicators]}")
            
            if page_num <= 3:
                print(f"    DEBUG Page {page_num}: Total LLM extracted {len(indicators_from_legend)} indicators: {[ind['indicator'] for ind in indicators_from_legend[:10]]}")
                if indicators_from_legend:
                    print(f"    DEBUG Page {page_num}: Indicator details: {[(ind['indicator'], ind.get('description', '')[:30]) for ind in indicators_from_legend[:5]]}")
        else:
            # Fallback - extract from legend text
            legend_tables = identify_legend_tables(page_chunks)
            legend_text_parts = list(legend_tables)
            for chunk in page_chunks:
                element_type = chunk.get("type", "").lower()
                if element_type == "text":
                    element_text = chunk.get("text", "")
                    if element_text:
                        text_lower = element_text.lower()
                        if any(kw in text_lower for kw in ["keynote", "legend", "note", "symbol"]):
                            legend_text_parts.append(element_text)
            legend_text = "\n".join(legend_text_parts)
            if legend_text:
                indicators_from_legend = extract_indicators_from_legend_text(legend_text, fence_keywords)
        
        # Find indicators in figures (pass LLM for intelligent matching)
        figure_highlights = find_indicators_in_figures(
            indicators_from_legend,
            page_chunks,
            google_ocr_results,
            page_width,
            page_height,
            llm=llm_instance
        )
        
        # Combine all highlights
        all_highlights = legend_highlights + figure_highlights
        
        # Convert to format for evaluation
        page_predictions = []
        for highlight in all_highlights:
            page_predictions.append({
                'bbox': (
                    highlight.get('x0', 0),
                    highlight.get('y0', 0),
                    highlight.get('x1', 0),
                    highlight.get('y1', 0)
                ),
                'text': highlight.get('text', ''),
                'type': highlight.get('tag_from_llm', 'UNKNOWN')
            })
        
        predictions[page_num] = page_predictions
        print(f"    ✅ Found {len(page_predictions)} highlights on page {page_num}")
    
    doc.close()
    return predictions


def evaluate_page(
    gold_annotations: List[Dict],
    predictions: List[Dict],
    page_num: int
) -> Dict:
    """
    Evaluate predictions against gold standard for a single page.
    
    Returns:
        Dict with metrics: tp, fp, fn, precision, recall, f1, avg_iou
    """
    if not gold_annotations and not predictions:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'avg_iou': 1.0}
    
    if not gold_annotations:
        return {'tp': 0, 'fp': len(predictions), 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'avg_iou': 0.0}
    
    if not predictions:
        return {'tp': 0, 'fp': 0, 'fn': len(gold_annotations), 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'avg_iou': 0.0}
    
    # Match predictions to gold standard
    matched_pred = set()
    matched_gold = set()
    ious = []
    
    # Debug: Show what we're comparing (for first few items)
    if page_num <= 3 and len(predictions) > 0 and len(gold_annotations) > 0:
        print(f"      [DEBUG] Comparing {len(predictions)} predictions vs {len(gold_annotations)} gold annotations")
        if len(predictions) <= 5 and len(gold_annotations) <= 5:
            for i, pred in enumerate(predictions[:3]):
                print(f"        Pred {i}: bbox={pred['bbox']}, text='{pred.get('text', '')[:20]}'")
            for i, gold in enumerate(gold_annotations[:3]):
                print(f"        Gold {i}: bbox={gold['bbox']}, content='{gold.get('content', '')[:20]}'")
    
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0.0
        best_gold_idx = None
        
        for gold_idx, gold in enumerate(gold_annotations):
            if gold_idx in matched_gold:
                continue
            
            iou = calculate_iou(pred['bbox'], gold['bbox'])
            if iou > best_iou:
                best_iou = iou
                if iou >= IOU_THRESHOLD:
                    best_gold_idx = gold_idx
        
        # Debug: Show best match for this prediction
        if page_num <= 3 and pred_idx < 3:
            print(f"      [DEBUG] Pred {pred_idx} best IoU: {best_iou:.3f} {'✓ MATCH' if best_gold_idx is not None else '✗ NO MATCH'}")
        
        if best_gold_idx is not None:
            matched_pred.add(pred_idx)
            matched_gold.add(best_gold_idx)
            ious.append(best_iou)
    
    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(gold_annotations) - len(matched_gold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'matched_count': len(matched_pred)
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate hybrid highlighting system')
    parser.add_argument('--pdf', type=Path, default='subset_gold/selected_pages_no_annotations.pdf',
                       help='Path to test PDF')
    parser.add_argument('--gold', type=Path, default='subset_gold/df_annotations_sub.csv',
                       help='Path to gold standard CSV')
    parser.add_argument('--output', type=Path, default='subset_gold/evaluation_results.csv',
                       help='Path to output CSV')
    args = parser.parse_args()
    
    # Load API keys and config
    ade_api_key = os.getenv("LANDINGAI_API_KEY")
    if not ade_api_key:
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            ade_api_key = secrets.get("LANDINGAI_API_KEY")
        except:
            pass
    
    if not ade_api_key:
        print("❌ LANDINGAI_API_KEY not found")
        sys.exit(1)
    
    google_cloud_config_raw = load_google_cloud_config()
    if not google_cloud_config_raw:
        print("❌ Google Cloud config not found")
        sys.exit(1)
    
    # Build proper Google Cloud config with client and processor_name
    from utils import create_document_ai_client
    client = create_document_ai_client(google_cloud_config_raw)
    if not client:
        print("❌ Could not create Document AI client")
        sys.exit(1)
    
    processor_name = (
        f"projects/{google_cloud_config_raw['project_number']}/"
        f"locations/{google_cloud_config_raw['location']}/"
        f"processors/{google_cloud_config_raw['processor_id']}"
    )
    
    google_cloud_config = {
        **google_cloud_config_raw,
        "client": client,
        "processor_name": processor_name
    }
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            openai_key = secrets.get("OPENAI_API_KEY")
        except:
            pass
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        sys.exit(1)
    
    llm_instance = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_key)
    
    # Load gold standard
    print(f"📊 Loading gold standard from {args.gold}...")
    gold_standard = load_gold_standard(args.gold)
    print(f"✅ Loaded {len(gold_standard)} pages with annotations")
    
    # Run predictions
    print(f"\n🚀 Running hybrid highlighting on {args.pdf}...")
    predictions = run_hybrid_highlighting(
        args.pdf,
        ade_api_key,
        google_cloud_config,
        llm_instance,
        FENCE_KEYWORDS
    )
    
    # Evaluate
    print(f"\n📈 Evaluating predictions...")
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    all_pages = set(gold_standard.keys()) | set(predictions.keys())
    
    for page_num in sorted(all_pages):
        gold_annots = gold_standard.get(page_num, [])
        preds = predictions.get(page_num, [])
        
        metrics = evaluate_page(gold_annots, preds, page_num)
        metrics['page'] = page_num
        metrics['gold_count'] = len(gold_annots)
        metrics['pred_count'] = len(preds)
        
        results.append(metrics)
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        
        print(f"  Page {page_num}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"F1={metrics['f1']:.3f}, IoU={metrics['avg_iou']:.3f} "
              f"(TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")
        
        # Debug: Show some examples
        if page_num <= 3 and (gold_annots or preds):
            print(f"    Gold samples: {[(g['content'][:20], g['bbox']) for g in gold_annots[:3]]}")
            print(f"    Pred samples: {[(p['text'][:20], p['bbox']) for p in preds[:3]]}")
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Precision: {overall_precision:.3f}")
    print(f"  Recall: {overall_recall:.3f}")
    print(f"  F1: {overall_f1:.3f}")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\n💾 Results saved to {args.output}")
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == "__main__":
    main()

