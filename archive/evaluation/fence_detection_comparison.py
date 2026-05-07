"""
Fence Detection Method Comparison

Implements and evaluates multiple fence detection approaches against ground truth.
"""

import fitz
import math
import csv
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


@dataclass
class DetectionResult:
    """Result from a fence detection method."""
    method_name: str
    regions: List[Dict]  # Each has: bbox, length_ft, orientation, etc.
    total_length_ft: float
    

@dataclass 
class GroundTruth:
    """Ground truth fence annotation."""
    page: int
    bbox: Tuple[float, float, float, float]
    length_ft: Optional[float]
    content: str


# =============================================================================
# GROUND TRUTH PARSING
# =============================================================================

def parse_ground_truth(csv_path: str) -> Dict[int, List[GroundTruth]]:
    """Parse ground truth annotations from CSV."""
    gt_by_page = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = int(row['page'])
            content = row['content'].strip() if row['content'] else ''
            
            try:
                bbox = tuple(ast.literal_eval(row['bbox']))
            except:
                continue
            
            # Check if content is a measurement
            length_ft = None
            try:
                num = float(content)
                if 0 < num < 1000:
                    length_ft = num
            except ValueError:
                pass
            
            # Include fence-related annotations
            fence_keywords = ['GATE', 'FENCE', 'WALL', '0113', '0401', '0402', '0403']
            is_fence = any(kw in content.upper() for kw in fence_keywords) or length_ft is not None
            
            if is_fence:
                gt_by_page[page].append(GroundTruth(
                    page=page,
                    bbox=bbox,
                    length_ft=length_ft,
                    content=content
                ))
    
    return dict(gt_by_page)


# =============================================================================
# METHOD 1: GLOBAL CLUSTERING (Baseline)
# =============================================================================

def method_global_clustering(
    page,
    scale_factor: float = 30.0,
    min_length: float = 15.0,
    distance_threshold: float = 25.0,
    angle_threshold: float = 20.0,
    min_cluster_lines: int = 5,
    min_aspect_ratio: float = 4.0
) -> DetectionResult:
    """
    Baseline: Cluster all lines by proximity/orientation, filter by shape.
    """
    from fence_detector_agentic import (
        extract_all_lines,
        cluster_by_proximity_and_orientation,
        filter_fence_like_clusters
    )
    
    lines = extract_all_lines(page, min_length=min_length)
    clusters = cluster_by_proximity_and_orientation(
        lines, distance_threshold=distance_threshold, angle_threshold=angle_threshold
    )
    fence_clusters = filter_fence_like_clusters(
        clusters, min_lines=min_cluster_lines, 
        min_length_pts=200.0, min_aspect_ratio=min_aspect_ratio
    )
    
    regions = []
    for c in fence_clusters:
        length_ft = (c.total_length_pts / 72.0) * scale_factor / 12.0
        regions.append({
            'bbox': c.bbox,
            'length_ft': length_ft,
            'orientation': c.orientation,
            'n_lines': len(c.lines)
        })
    
    return DetectionResult(
        method_name="Global Clustering",
        regions=regions,
        total_length_ft=sum(r['length_ft'] for r in regions)
    )


# =============================================================================
# METHOD 2: INDICATOR-DRIVEN (Text Keywords)
# =============================================================================

def method_indicator_driven(
    page,
    scale_factor: float = 30.0,
    search_radius: float = 150.0
) -> DetectionResult:
    """
    Find fence-related text indicators, then find lines near them.
    """
    from fence_detector_agentic import extract_all_lines
    
    # Keywords that indicate fence elements
    fence_keywords = [
        'FENCE', 'GATE', 'ROLLING GATE', 'CHAIN LINK', 'SCREEN WALL',
        '0113', '0401', '0402', '0403'
    ]
    
    # Extract text and find indicators
    text_dict = page.get_text('dict')
    indicators = []
    
    for block in text_dict.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip().upper()
                bbox = span['bbox']
                for kw in fence_keywords:
                    if kw in text:
                        indicators.append({
                            'text': text,
                            'bbox': bbox,
                            'center': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                        })
                        break
    
    # Find lines near each indicator
    all_lines = extract_all_lines(page, min_length=10.0)
    regions = []
    used_lines = set()
    
    for ind in indicators:
        px, py = ind['center']
        nearby_lines = []
        
        for i, line in enumerate(all_lines):
            if i in used_lines:
                continue
            sx, sy = line['start']
            ex, ey = line['end']
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            
            dist = min(
                math.sqrt((sx-px)**2 + (sy-py)**2),
                math.sqrt((ex-px)**2 + (ey-py)**2),
                math.sqrt((mx-px)**2 + (my-py)**2)
            )
            
            if dist <= search_radius:
                nearby_lines.append((i, line))
        
        if nearby_lines:
            # Mark lines as used
            for i, _ in nearby_lines:
                used_lines.add(i)
            
            total_pts = sum(l['length_pts'] for _, l in nearby_lines)
            length_ft = (total_pts / 72.0) * scale_factor / 12.0
            
            # Compute bbox of all lines
            all_x = []
            all_y = []
            for _, l in nearby_lines:
                all_x.extend([l['start'][0], l['end'][0]])
                all_y.extend([l['start'][1], l['end'][1]])
            
            regions.append({
                'bbox': (min(all_x), min(all_y), max(all_x), max(all_y)),
                'length_ft': length_ft,
                'indicator': ind['text'],
                'n_lines': len(nearby_lines)
            })
    
    return DetectionResult(
        method_name="Indicator-Driven (Text)",
        regions=regions,
        total_length_ft=sum(r['length_ft'] for r in regions)
    )


# =============================================================================
# METHOD 3: DIMENSION LINE DETECTION
# =============================================================================

def method_dimension_lines(
    page,
    scale_factor: float = 30.0,
    search_radius: float = 100.0
) -> DetectionResult:
    """
    Find numeric measurements in text, then find the line being measured.
    Looks for dimension-style annotations (numbers with units or standalone).
    """
    from fence_detector_agentic import extract_all_lines
    
    # Extract text and find numeric measurements
    text_dict = page.get_text('dict')
    measurements = []
    
    for block in text_dict.get('blocks', []):
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip()
                bbox = span['bbox']
                
                # Look for patterns like: "45", "45'", "45'-0\"", "45 LF"
                match = re.match(r'^(\d+\.?\d*)\s*[\'"\-LF]*', text)
                if match:
                    try:
                        num = float(match.group(1))
                        # Filter to reasonable fence lengths (2-500 ft)
                        if 2 <= num <= 500:
                            measurements.append({
                                'value': num,
                                'text': text,
                                'bbox': bbox,
                                'center': ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                            })
                    except:
                        pass
    
    # For each measurement, find the best matching line nearby
    all_lines = extract_all_lines(page, min_length=5.0)
    regions = []
    
    for meas in measurements:
        px, py = meas['center']
        gt_length_pts = meas['value'] * 12.0 / scale_factor * 72.0
        
        best_line = None
        best_score = float('inf')
        
        for line in all_lines:
            sx, sy = line['start']
            ex, ey = line['end']
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            
            dist = min(
                math.sqrt((sx-px)**2 + (sy-py)**2),
                math.sqrt((ex-px)**2 + (ey-py)**2),
                math.sqrt((mx-px)**2 + (my-py)**2)
            )
            
            if dist > search_radius:
                continue
            
            # Score: prefer length match + proximity
            length_diff_pct = abs(line['length_pts'] - gt_length_pts) / gt_length_pts if gt_length_pts > 0 else 1
            score = length_diff_pct + (dist / search_radius) * 0.3
            
            if score < best_score:
                best_score = score
                best_line = line
        
        if best_line and best_score < 1.0:  # Only include good matches
            length_ft = (best_line['length_pts'] / 72.0) * scale_factor / 12.0
            regions.append({
                'bbox': (best_line['start'][0], best_line['start'][1], 
                        best_line['end'][0], best_line['end'][1]),
                'length_ft': length_ft,
                'expected_ft': meas['value'],
                'measurement_text': meas['text']
            })
    
    return DetectionResult(
        method_name="Dimension Line Detection",
        regions=regions,
        total_length_ft=sum(r['length_ft'] for r in regions)
    )


# =============================================================================
# METHOD 4: AGGRESSIVE CLUSTERING (relaxed params)
# =============================================================================

def method_aggressive_clustering(
    page,
    scale_factor: float = 30.0
) -> DetectionResult:
    """
    More aggressive clustering with relaxed parameters.
    """
    return method_global_clustering(
        page, scale_factor,
        min_length=10.0,
        distance_threshold=35.0,
        angle_threshold=30.0,
        min_cluster_lines=3,
        min_aspect_ratio=3.0
    )


# =============================================================================
# METHOD 5: CONSERVATIVE CLUSTERING (strict params)
# =============================================================================

def method_conservative_clustering(
    page,
    scale_factor: float = 30.0
) -> DetectionResult:
    """
    Conservative clustering with strict parameters.
    """
    return method_global_clustering(
        page, scale_factor,
        min_length=20.0,
        distance_threshold=20.0,
        angle_threshold=15.0,
        min_cluster_lines=7,
        min_aspect_ratio=5.0
    )


# =============================================================================
# METHOD 6: ORACLE BEST-MATCH (Upper Bound)
# =============================================================================

def method_oracle_best_match(
    page,
    ground_truth: List[GroundTruth],
    scale_factor: float = 30.0,
    search_radius: float = 300.0
) -> DetectionResult:
    """
    Oracle method: uses GT locations to find best matching lines.
    This is an upper bound on achievable performance.
    """
    from fence_detector_agentic import extract_all_lines
    
    gt_with_length = [g for g in ground_truth if g.length_ft is not None]
    all_lines = extract_all_lines(page, min_length=5.0)
    
    regions = []
    
    for gt in gt_with_length:
        px = (gt.bbox[0] + gt.bbox[2]) / 2
        py = (gt.bbox[1] + gt.bbox[3]) / 2
        gt_length_pts = gt.length_ft * 12.0 / scale_factor * 72.0
        
        best_line = None
        best_score = float('inf')
        
        for line in all_lines:
            sx, sy = line['start']
            ex, ey = line['end']
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            
            dist = min(
                math.sqrt((sx-px)**2 + (sy-py)**2),
                math.sqrt((ex-px)**2 + (ey-py)**2),
                math.sqrt((mx-px)**2 + (my-py)**2)
            )
            
            if dist > search_radius:
                continue
            
            length_diff_pct = abs(line['length_pts'] - gt_length_pts) / gt_length_pts if gt_length_pts > 0 else 1
            score = length_diff_pct + (dist / search_radius) * 0.3
            
            if score < best_score:
                best_score = score
                best_line = line
        
        if best_line:
            length_ft = (best_line['length_pts'] / 72.0) * scale_factor / 12.0
            regions.append({
                'bbox': (best_line['start'][0], best_line['start'][1],
                        best_line['end'][0], best_line['end'][1]),
                'length_ft': length_ft,
                'gt_length_ft': gt.length_ft
            })
    
    return DetectionResult(
        method_name="Oracle Best-Match (Upper Bound)",
        regions=regions,
        total_length_ft=sum(r['length_ft'] for r in regions)
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_method(
    result: DetectionResult,
    ground_truth: List[GroundTruth],
    verbose: bool = False
) -> Dict:
    """
    Evaluate a detection result against ground truth.
    """
    gt_with_length = [g for g in ground_truth if g.length_ft is not None]
    gt_total = sum(g.length_ft for g in gt_with_length)
    detected_total = result.total_length_ft
    
    # Length accuracy
    if gt_total > 0:
        length_error_pct = abs(detected_total - gt_total) / gt_total * 100
        length_accuracy = max(0, 1.0 - length_error_pct / 100)
    else:
        length_error_pct = 0
        length_accuracy = 0
    
    # Region matching (IoU-based)
    matched_gt = 0
    for gt in gt_with_length:
        gt_center = ((gt.bbox[0]+gt.bbox[2])/2, (gt.bbox[1]+gt.bbox[3])/2)
        for region in result.regions:
            bbox = region['bbox']
            # Check if GT center is near any detected region
            if (bbox[0]-100 <= gt_center[0] <= bbox[2]+100 and
                bbox[1]-100 <= gt_center[1] <= bbox[3]+100):
                matched_gt += 1
                break
    
    recall = matched_gt / len(gt_with_length) if gt_with_length else 0
    
    metrics = {
        'method': result.method_name,
        'gt_total_ft': gt_total,
        'detected_total_ft': detected_total,
        'length_error_pct': length_error_pct,
        'length_accuracy': length_accuracy,
        'n_regions': len(result.regions),
        'n_gt': len(gt_with_length),
        'matched_gt': matched_gt,
        'recall': recall
    }
    
    if verbose:
        print(f"\n{result.method_name}")
        print("-" * 50)
        print(f"  GT Total:      {gt_total:.0f} ft")
        print(f"  Detected:      {detected_total:.0f} ft")
        print(f"  Length Error:  {length_error_pct:.1f}%")
        print(f"  Accuracy:      {length_accuracy:.1%}")
        print(f"  Regions:       {len(result.regions)} detected, {len(gt_with_length)} GT")
        print(f"  Recall:        {recall:.1%}")
    
    return metrics


def run_comparison(
    pdf_path: str,
    annotations_path: str,
    pages: List[int] = None,
    scale_factor: float = 30.0
) -> List[Dict]:
    """
    Run all methods on specified pages and return comparison results.
    """
    doc = fitz.open(pdf_path)
    gt_by_page = parse_ground_truth(annotations_path)
    
    if pages is None:
        pages = list(gt_by_page.keys())
    
    all_results = []
    
    for page_num in pages:
        page = doc[page_num - 1]  # 0-indexed
        gt = gt_by_page.get(page_num, [])
        
        if not gt:
            continue
        
        print(f"\n{'='*60}")
        print(f" PAGE {page_num}")
        print(f"{'='*60}")
        
        methods = [
            ("Global Clustering", lambda p: method_global_clustering(p, scale_factor)),
            ("Aggressive Clustering", lambda p: method_aggressive_clustering(p, scale_factor)),
            ("Conservative Clustering", lambda p: method_conservative_clustering(p, scale_factor)),
            ("Indicator-Driven", lambda p: method_indicator_driven(p, scale_factor)),
            ("Dimension Lines", lambda p: method_dimension_lines(p, scale_factor)),
            ("Oracle Best-Match", lambda p: method_oracle_best_match(p, gt, scale_factor)),
        ]
        
        for name, method_fn in methods:
            try:
                result = method_fn(page)
                result.method_name = name  # Ensure correct name
                metrics = evaluate_method(result, gt, verbose=True)
                metrics['page'] = page_num
                all_results.append(metrics)
            except Exception as e:
                print(f"\n{name}: ERROR - {e}")
    
    doc.close()
    return all_results


def print_summary(results: List[Dict]):
    """Print summary table of all results."""
    print("\n" + "=" * 80)
    print(" SUMMARY: METHOD COMPARISON")
    print("=" * 80)
    
    # Group by method
    by_method = defaultdict(list)
    for r in results:
        by_method[r['method']].append(r)
    
    print(f"\n{'Method':<30} {'Avg Accuracy':<15} {'Avg Error':<15} {'Avg Recall':<15}")
    print("-" * 80)
    
    method_scores = []
    for method, method_results in by_method.items():
        avg_acc = sum(r['length_accuracy'] for r in method_results) / len(method_results)
        avg_err = sum(r['length_error_pct'] for r in method_results) / len(method_results)
        avg_rec = sum(r['recall'] for r in method_results) / len(method_results)
        
        print(f"{method:<30} {avg_acc:>12.1%} {avg_err:>13.0f}% {avg_rec:>13.1%}")
        method_scores.append((method, avg_acc, avg_err, avg_rec))
    
    print("-" * 80)
    
    # Best method
    best = max(method_scores, key=lambda x: x[1])
    print(f"\nBEST METHOD: {best[0]} (Accuracy: {best[1]:.1%}, Error: {best[2]:.0f}%)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare fence detection methods')
    parser.add_argument('--pdf', default='gold_standard/subset_gold/selected_pages.pdf')
    parser.add_argument('--annotations', default='gold_standard/subset_gold/df_annotations_sub.csv')
    parser.add_argument('--pages', type=int, nargs='+', default=None, help='Pages to test (1-indexed)')
    parser.add_argument('--scale', type=float, default=30.0, help='Scale factor (1"=X\')')
    args = parser.parse_args()
    
    results = run_comparison(args.pdf, args.annotations, args.pages, args.scale)
    print_summary(results)
