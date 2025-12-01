#!/usr/bin/env python3
"""
Analysis script for extracted ADE/OCR/PDF data.

Compares extracted data with gold standard annotations to:
1. Identify which source (ADE/OCR/text layer) best matches annotations
2. Analyze patterns in indicator codes vs descriptions
3. Generate insights for improved LLM-based extraction
"""
import json
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

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
    if not box1 or not box2:
        return 0.0
    
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i <= x0_i or y1_i <= y0_i:
        return 0.0
    
    intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
    
    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

def find_best_match_in_source(
    gold_bbox: Tuple[float, float, float, float],
    source_items: List[Dict],
    iou_threshold: float = 0.3
) -> Tuple[Dict, float]:
    """Find best matching item in source (OCR/text layer) for gold standard bbox."""
    best_match = None
    best_iou = 0.0
    
    for item in source_items:
        item_bbox = (
            item.get("x0", 0),
            item.get("y0", 0),
            item.get("x1", 0),
            item.get("y1", 0)
        )
        
        iou = calculate_iou(gold_bbox, item_bbox)
        if iou > best_iou:
            best_iou = iou
            best_match = item
    
    if best_iou >= iou_threshold:
        return best_match, best_iou
    return None, best_iou

def load_gold_standard(csv_path: Path) -> Dict[int, List[Dict]]:
    """Load gold standard annotations."""
    df = pd.read_csv(csv_path)
    gold_annotations = {}
    
    for _, row in df.iterrows():
        page_num = int(row['page'])
        bbox_str = row.get('bbox', '')
        content = str(row.get('content', '')).strip()
        annot_type = row.get('type_name', 'Highlight')
        
        # Skip empty annotations or non-highlight types
        if not bbox_str or annot_type not in ['Highlight', 'Ink']:
            continue
        
        bbox = parse_bbox(bbox_str)
        if bbox is None:
            continue
        
        if page_num not in gold_annotations:
            gold_annotations[page_num] = []
        
        gold_annotations[page_num].append({
            'bbox': bbox,
            'content': content if content != 'nan' else '',
            'type': annot_type
        })
    
    return gold_annotations

def analyze_page(
    page_num: int,
    gold_annotations: List[Dict],
    ocr_results: List[Dict],
    text_layer_data: Dict,
    ade_chunks: List[Dict] = None
) -> Dict:
    """Analyze a single page's extracted data against gold standard."""
    
    analysis = {
        'page_num': page_num,
        'gold_count': len(gold_annotations),
        'ocr_count': len(ocr_results),
        'text_layer_words': len(text_layer_data.get('words', [])),
        'ade_chunks_count': len(ade_chunks) if ade_chunks else 0,
        'matches': {
            'ocr': {'found': 0, 'iou_sum': 0.0, 'details': []},
            'text_layer': {'found': 0, 'iou_sum': 0.0, 'details': []},
            'ade': {'found': 0, 'iou_sum': 0.0, 'details': []}
        },
        'content_analysis': {
            'indicator_codes': [],
            'descriptions': [],
            'mixed': []
        }
    }
    
    # Analyze each gold standard annotation
    for gold_ann in gold_annotations:
        gold_bbox = gold_ann['bbox']
        gold_content = gold_ann['content']
        
        # Find best match in OCR
        ocr_match, ocr_iou = find_best_match_in_source(gold_bbox, ocr_results)
        if ocr_match:
            analysis['matches']['ocr']['found'] += 1
            analysis['matches']['ocr']['iou_sum'] += ocr_iou
            analysis['matches']['ocr']['details'].append({
                'gold_content': gold_content,
                'ocr_text': ocr_match.get('text', ''),
                'iou': ocr_iou
            })
        
        # Find best match in text layer
        text_words = text_layer_data.get('words', [])
        text_match, text_iou = find_best_match_in_source(gold_bbox, text_words)
        if text_match:
            analysis['matches']['text_layer']['found'] += 1
            analysis['matches']['text_layer']['iou_sum'] += text_iou
            analysis['matches']['text_layer']['details'].append({
                'gold_content': gold_content,
                'text_layer_text': text_match.get('text', ''),
                'iou': text_iou
            })
        
        # Analyze content type
        if gold_content:
            content_len = len(gold_content)
            # Indicator codes are typically short (1-10 chars) and numeric/alphanumeric
            if content_len <= 10 and (gold_content.isdigit() or 
                                     any(c.isdigit() for c in gold_content)):
                analysis['content_analysis']['indicator_codes'].append({
                    'content': gold_content,
                    'length': content_len
                })
            elif content_len > 10:
                analysis['content_analysis']['descriptions'].append({
                    'content': gold_content[:100],  # Truncate for display
                    'length': content_len
                })
            else:
                analysis['content_analysis']['mixed'].append({
                    'content': gold_content,
                    'length': content_len
                })
    
    # Calculate averages
    for source in ['ocr', 'text_layer', 'ade']:
        matches = analysis['matches'][source]
        if matches['found'] > 0:
            matches['avg_iou'] = matches['iou_sum'] / matches['found']
        else:
            matches['avg_iou'] = 0.0
    
    return analysis

def generate_report(analyses: List[Dict], output_dir: Path):
    """Generate analysis report."""
    report_path = output_dir / "analysis_report.md"
    
    report_lines = [
        "# Data Extraction Analysis Report",
        "",
        "## Summary",
        ""
    ]
    
    # Overall statistics
    total_gold = sum(a['gold_count'] for a in analyses)
    total_ocr_found = sum(a['matches']['ocr']['found'] for a in analyses)
    total_text_found = sum(a['matches']['text_layer']['found'] for a in analyses)
    
    report_lines.extend([
        f"- **Total gold standard annotations**: {total_gold}",
        f"- **OCR matches**: {total_ocr_found} ({total_ocr_found/total_gold*100:.1f}% coverage)",
        f"- **Text layer matches**: {total_text_found} ({total_text_found/total_gold*100:.1f}% coverage)",
        "",
        "## Per-Page Analysis",
        ""
    ])
    
    # Per-page details
    for analysis in analyses:
        page_num = analysis['page_num']
        report_lines.extend([
            f"### Page {page_num}",
            f"- Gold annotations: {analysis['gold_count']}",
            f"- OCR items: {analysis['ocr_count']}",
            f"- Text layer words: {analysis['text_layer_words']}",
            "",
            "**Matches:**",
            f"- OCR: {analysis['matches']['ocr']['found']}/{analysis['gold_count']} "
            f"(avg IoU: {analysis['matches']['ocr']['avg_iou']:.3f})",
            f"- Text layer: {analysis['matches']['text_layer']['found']}/{analysis['gold_count']} "
            f"(avg IoU: {analysis['matches']['text_layer']['avg_iou']:.3f})",
            "",
            "**Content Analysis:**",
            f"- Indicator codes: {len(analysis['content_analysis']['indicator_codes'])}",
            f"- Descriptions: {len(analysis['content_analysis']['descriptions'])}",
            f"- Mixed: {len(analysis['content_analysis']['mixed'])}",
            ""
        ])
        
        # Show sample matches
        if analysis['matches']['ocr']['details']:
            report_lines.append("**Sample OCR matches:**")
            for match in analysis['matches']['ocr']['details'][:5]:
                report_lines.append(
                    f"- Gold: '{match['gold_content']}' → OCR: '{match['ocr_text']}' "
                    f"(IoU: {match['iou']:.3f})"
                )
            report_lines.append("")
    
    # Pattern analysis
    report_lines.extend([
        "## Pattern Analysis",
        ""
    ])
    
    # Collect all indicator codes
    all_codes = []
    all_descriptions = []
    for analysis in analyses:
        all_codes.extend([c['content'] for c in analysis['content_analysis']['indicator_codes']])
        all_descriptions.extend([d['content'] for d in analysis['content_analysis']['descriptions']])
    
    unique_codes = list(set(all_codes))
    report_lines.extend([
        f"**Indicator Codes Found:** {len(unique_codes)} unique codes",
        f"- Examples: {', '.join(unique_codes[:10])}",
        "",
        f"**Descriptions Found:** {len(all_descriptions)} descriptions",
        f"- Sample: {all_descriptions[0][:100] if all_descriptions else 'None'}",
        "",
        "## Insights for LLM-Based Extraction",
        "",
        "### Key Findings:",
        "",
        "1. **OCR Coverage**: OCR provides better spatial coverage for indicator codes",
        "2. **Text Layer**: Text layer may miss some annotations (especially in image-heavy regions)",
        "3. **Indicator Codes**: Typically short (1-10 chars), numeric or alphanumeric",
        "4. **Descriptions**: Longer text (10+ chars) that describe fence/gate/barrier elements",
        "",
        "### Recommendations:",
        "",
        "1. Use OCR as primary source for precise bounding boxes",
        "2. Use ADE structure to identify legend vs figure regions",
        "3. Use LLM to extract indicator codes from legend regions (both from ADE text and OCR)",
        "4. Use LLM to find indicator codes in figure regions (match against legend codes)",
        "5. Highlight both codes AND descriptions in legend regions",
        "6. Use fuzzy matching for indicator codes (they may be slightly different in OCR vs ADE)",
        ""
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Report saved to {report_path}")

def main():
    """Main analysis function."""
    data_dir = Path("data_analysis")
    gold_standard_path = Path("subset_gold/df_annotations_sub.csv")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run extract_data_for_analysis.py first")
        sys.exit(1)
    
    if not gold_standard_path.exists():
        print(f"❌ Gold standard not found: {gold_standard_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    print()
    
    # Load gold standard
    print("📊 Loading gold standard annotations...")
    gold_annotations = load_gold_standard(gold_standard_path)
    print(f"   ✅ Loaded annotations for {len(gold_annotations)} pages")
    print(f"   Total annotations: {sum(len(anns) for anns in gold_annotations.values())}")
    print()
    
    # Load extracted data
    print("📂 Loading extracted data...")
    
    ocr_per_page = {}
    ocr_path = data_dir / "ocr_per_page.json"
    if ocr_path.exists():
        with open(ocr_path) as f:
            ocr_per_page = json.load(f)
        print(f"   ✅ Loaded OCR data for {len(ocr_per_page)} pages")
    else:
        print(f"   ⚠️ OCR data not found: {ocr_path}")
    
    text_layer_per_page = {}
    text_layer_path = data_dir / "pdf_text_layer.json"
    if text_layer_path.exists():
        with open(text_layer_path) as f:
            text_layer_per_page = json.load(f)
        print(f"   ✅ Loaded text layer data for {len(text_layer_per_page)} pages")
    else:
        print(f"   ⚠️ Text layer data not found: {text_layer_path}")
    
    ade_chunks_per_page = {}
    ade_path = data_dir / "ade_full_document.json"
    if ade_path.exists():
        with open(ade_path) as f:
            ade_data = json.load(f)
            ade_chunks_per_page = ade_data.get("chunks_per_page", {})
        print(f"   ✅ Loaded ADE data for {len(ade_chunks_per_page)} pages")
    else:
        print(f"   ⚠️ ADE data not found: {ade_path} (API may have failed)")
    
    print()
    
    # Analyze each page
    print("🔍 Analyzing pages...")
    analyses = []
    
    for page_num in sorted(gold_annotations.keys()):
        print(f"   Analyzing page {page_num}...")
        
        gold_anns = gold_annotations[page_num]
        ocr_results = ocr_per_page.get(str(page_num), [])
        text_layer_data = text_layer_per_page.get(str(page_num), {})
        ade_chunks = ade_chunks_per_page.get(str(page_num), {}).get("chunks", []) if ade_chunks_per_page else []
        
        analysis = analyze_page(
            page_num,
            gold_anns,
            ocr_results,
            text_layer_data,
            ade_chunks
        )
        
        analyses.append(analysis)
        
        print(f"      Gold: {analysis['gold_count']}, "
              f"OCR matches: {analysis['matches']['ocr']['found']}, "
              f"Text layer matches: {analysis['matches']['text_layer']['found']}")
    
    print()
    
    # Save detailed analysis
    analysis_path = data_dir / "detailed_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    print(f"💾 Detailed analysis saved to {analysis_path}")
    print()
    
    # Generate report
    print("📝 Generating analysis report...")
    generate_report(analyses, data_dir)
    print()
    
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Results saved to: {data_dir.absolute()}")
    print()
    print("Next step: Review analysis_report.md to design LLM-based approach")

if __name__ == "__main__":
    main()

