"""
Fence Detection Evaluator

Compares detected fence regions against ground truth annotations to measure
detection accuracy and help tune parameters.
"""

import csv
import ast
import fitz
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math


@dataclass
class GroundTruthFence:
    """A ground truth fence annotation."""
    page: int
    bbox: Tuple[float, float, float, float]
    length_ft: Optional[float]  # Measured length if available
    element_type: str  # e.g., "0113", "GATE", etc.
    raw_content: str


@dataclass
class DetectedFence:
    """A detected fence region."""
    bbox: Tuple[float, float, float, float]
    length_ft: float
    orientation: str
    segment_count: int


def parse_annotations(csv_path: str) -> Dict[int, List[GroundTruthFence]]:
    """
    Parse ground truth annotations from CSV.
    Returns dict mapping page number -> list of fence annotations.
    """
    annotations_by_page = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = int(row['page'])
            content = row['content'].strip() if row['content'] else ''
            bbox_str = row['bbox']
            
            # Parse bbox
            try:
                bbox = tuple(ast.literal_eval(bbox_str))
            except:
                continue
            
            # Determine if this is a measurement (numeric) or element type
            length_ft = None
            element_type = ''
            
            # Check if content is a number (measurement)
            try:
                # Handle multi-digit numbers like "17", "45", "100", etc.
                num = float(content)
                if num > 0 and num < 1000:  # Reasonable fence length range
                    length_ft = num
            except ValueError:
                pass
            
            # Check for element type codes
            if content.startswith('0') or content.startswith('3'):
                element_type = content.split()[0] if ' ' in content else content
            
            # Only include if it has a measurement or is a fence-related element
            fence_keywords = ['GATE', 'FENCE', 'WALL', '0113', '0401', '0402', '0403']
            is_fence_related = any(kw in content.upper() for kw in fence_keywords)
            
            if length_ft is not None or is_fence_related:
                gt = GroundTruthFence(
                    page=page,
                    bbox=bbox,
                    length_ft=length_ft,
                    element_type=element_type,
                    raw_content=content
                )
                
                if page not in annotations_by_page:
                    annotations_by_page[page] = []
                annotations_by_page[page].append(gt)
    
    return annotations_by_page


def bbox_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union of two bboxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def bbox_overlap(box1: Tuple, box2: Tuple, threshold: float = 0.3) -> bool:
    """Check if two bboxes overlap significantly."""
    return bbox_iou(box1, box2) > threshold


def point_in_bbox(point: Tuple[float, float], bbox: Tuple, margin: float = 50) -> bool:
    """Check if a point is within or near a bbox."""
    x, y = point
    x0, y0, x1, y1 = bbox
    return (x0 - margin <= x <= x1 + margin) and (y0 - margin <= y <= y1 + margin)


def find_lines_near_point(
    page,
    point: Tuple[float, float],
    search_radius: float = 200.0,
    min_length: float = 10.0
) -> List[Dict]:
    """
    Find vector lines near a specific point on the page.
    Used to find fence segments near measurement labels.
    """
    from fence_detector_agentic import extract_all_lines
    
    all_lines = extract_all_lines(page, min_length=min_length)
    nearby = []
    
    px, py = point
    for line in all_lines:
        # Check if either endpoint or midpoint is near the target point
        sx, sy = line['start']
        ex, ey = line['end']
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        
        for x, y in [(sx, sy), (ex, ey), (mx, my)]:
            dist = math.sqrt((x - px)**2 + (y - py)**2)
            if dist <= search_radius:
                nearby.append(line)
                break
    
    return nearby


def find_best_matching_line(
    page,
    gt_length_ft: float,
    label_center: Tuple[float, float],
    scale_factor: float = 30.0,
    search_radius: float = 300.0,
    min_length: float = 5.0
) -> Tuple[Optional[Dict], float]:
    """
    Find the line whose length best matches the GT measurement.
    Returns (best_line, detected_length_ft).
    """
    from fence_detector_agentic import extract_all_lines
    
    all_lines = extract_all_lines(page, min_length=min_length)
    
    # Convert GT to pts
    gt_length_pts = gt_length_ft * 12.0 / scale_factor * 72.0
    
    px, py = label_center
    best_line = None
    best_score = float('inf')
    
    for line in all_lines:
        # Check distance to line
        sx, sy = line['start']
        ex, ey = line['end']
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        
        # Distance from label to line midpoint or endpoints
        min_dist = min(
            math.sqrt((sx - px)**2 + (sy - py)**2),
            math.sqrt((ex - px)**2 + (ey - py)**2),
            math.sqrt((mx - px)**2 + (my - py)**2)
        )
        
        if min_dist > search_radius:
            continue
        
        # Score based on length match + distance (prefer close + correct length)
        length_diff = abs(line['length_pts'] - gt_length_pts)
        length_diff_pct = length_diff / gt_length_pts if gt_length_pts > 0 else 1.0
        
        # Combined score: prioritize length match, penalize distance
        score = length_diff_pct + (min_dist / search_radius) * 0.3
        
        if score < best_score:
            best_score = score
            best_line = line
    
    if best_line:
        detected_ft = (best_line['length_pts'] / 72.0) * scale_factor / 12.0
        return best_line, detected_ft
    return None, 0.0


def evaluate_per_measurement(
    page,
    ground_truth: List[GroundTruthFence],
    scale_factor: float = 30.0,
    search_radius: float = 150.0,
    mode: str = 'best_match'  # 'best_match' or 'all_nearby'
) -> Dict:
    """
    Evaluate by finding lines near each GT measurement point.
    
    Modes:
    - 'best_match': Find single line whose length best matches GT
    - 'all_nearby': Sum all lines near measurement point
    """
    gt_with_length = [g for g in ground_truth if g.length_ft is not None]
    
    results = []
    total_gt = 0
    total_detected = 0
    
    for gt in gt_with_length:
        gt_center = ((gt.bbox[0] + gt.bbox[2]) / 2, (gt.bbox[1] + gt.bbox[3]) / 2)
        
        if mode == 'best_match':
            # Find single best-matching line
            best_line, detected_ft = find_best_matching_line(
                page, gt.length_ft, gt_center, scale_factor, search_radius
            )
            n_lines = 1 if best_line else 0
        else:
            # Sum all nearby lines (original approach)
            search_point = (gt_center[0], gt_center[1] - 100)
            nearby_lines = find_lines_near_point(page, search_point, search_radius)
            total_pts = sum(l['length_pts'] for l in nearby_lines)
            detected_ft = (total_pts / 72.0) * scale_factor / 12.0
            n_lines = len(nearby_lines)
        
        error_pct = abs(detected_ft - gt.length_ft) / gt.length_ft * 100 if gt.length_ft > 0 else 0
        
        results.append({
            'gt_length': gt.length_ft,
            'detected_length': detected_ft,
            'error_pct': error_pct,
            'n_lines': n_lines,
            'location': gt_center
        })
        
        total_gt += gt.length_ft
        total_detected += detected_ft
    
    # Overall metrics
    avg_error = sum(r['error_pct'] for r in results) / len(results) if results else 0
    length_accuracy = 1.0 - abs(total_detected - total_gt) / total_gt if total_gt > 0 else 0
    length_accuracy = max(0.0, length_accuracy)
    
    return {
        'per_measurement': results,
        'avg_error_pct': avg_error,
        'length_accuracy': length_accuracy,
        'gt_total_ft': total_gt,
        'detected_total_ft': total_detected,
        'n_measurements': len(gt_with_length)
    }


def evaluate_detection(
    detected: List[DetectedFence],
    ground_truth: List[GroundTruthFence],
    iou_threshold: float = 0.1
) -> Dict:
    """
    Evaluate detected fences against ground truth.
    
    Returns metrics dict with:
    - region_precision: detected regions that match GT / all detected
    - region_recall: GT regions matched / all GT regions
    - length_accuracy: how close total detected length is to GT total
    - per_region_errors: list of individual region length errors
    """
    if not ground_truth:
        return {
            'region_precision': 0.0,
            'region_recall': 0.0,
            'length_accuracy': 0.0,
            'gt_total_ft': 0.0,
            'detected_total_ft': sum(d.length_ft for d in detected),
            'matched_regions': 0,
            'gt_regions': 0,
            'detected_regions': len(detected)
        }
    
    # Filter GT to only those with measurements
    gt_with_length = [g for g in ground_truth if g.length_ft is not None]
    gt_total = sum(g.length_ft for g in gt_with_length)
    detected_total = sum(d.length_ft for d in detected)
    
    # Match detected regions to GT regions
    matched_gt = set()
    matched_det = set()
    matches = []
    
    for i, det in enumerate(detected):
        for j, gt in enumerate(gt_with_length):
            if j in matched_gt:
                continue
            
            # Check if detection bbox contains or is near the GT measurement point
            gt_center = ((gt.bbox[0] + gt.bbox[2]) / 2, (gt.bbox[1] + gt.bbox[3]) / 2)
            if point_in_bbox(gt_center, det.bbox, margin=100):
                matched_gt.add(j)
                matched_det.add(i)
                matches.append((det, gt))
                break
    
    # Calculate metrics
    n_matched = len(matches)
    precision = n_matched / len(detected) if detected else 0.0
    recall = n_matched / len(gt_with_length) if gt_with_length else 0.0
    
    # Length accuracy (how close is total detected to total GT)
    if gt_total > 0:
        length_accuracy = 1.0 - abs(detected_total - gt_total) / gt_total
        length_accuracy = max(0.0, length_accuracy)
    else:
        length_accuracy = 0.0
    
    # Per-region length errors
    region_errors = []
    for det, gt in matches:
        if gt.length_ft and gt.length_ft > 0:
            error_pct = abs(det.length_ft - gt.length_ft) / gt.length_ft * 100
            region_errors.append({
                'gt_length': gt.length_ft,
                'detected_length': det.length_ft,
                'error_pct': error_pct
            })
    
    return {
        'region_precision': precision,
        'region_recall': recall,
        'length_accuracy': length_accuracy,
        'gt_total_ft': gt_total,
        'detected_total_ft': detected_total,
        'length_error_pct': abs(detected_total - gt_total) / gt_total * 100 if gt_total > 0 else 0,
        'matched_regions': n_matched,
        'gt_regions': len(gt_with_length),
        'detected_regions': len(detected),
        'region_errors': region_errors
    }


def run_parameter_sweep(
    pdf_path: str,
    annotations_path: str,
    page_num: int,
    param_grid: Dict[str, List],
    scale_factor: float = 30.0
) -> List[Dict]:
    """
    Test different parameter combinations and return results.
    
    param_grid example:
    {
        'min_length': [10, 15, 20],
        'distance_threshold': [20, 25, 30],
        'angle_threshold': [15, 20, 25],
        'min_cluster_lines': [3, 5, 7],
        'min_aspect_ratio': [3, 4, 5]
    }
    """
    from fence_detector_agentic import (
        extract_all_lines,
        cluster_by_proximity_and_orientation,
        filter_fence_like_clusters
    )
    
    # Load ground truth
    gt_by_page = parse_annotations(annotations_path)
    gt = gt_by_page.get(page_num, [])
    
    # Load PDF page
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    results = []
    
    # Generate all parameter combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        
        # Run detection with these params
        lines = extract_all_lines(page, min_length=params.get('min_length', 15))
        clusters = cluster_by_proximity_and_orientation(
            lines,
            distance_threshold=params.get('distance_threshold', 25),
            angle_threshold=params.get('angle_threshold', 20)
        )
        fence_clusters = filter_fence_like_clusters(
            clusters,
            min_lines=params.get('min_cluster_lines', 5),
            min_length_pts=200.0,
            min_aspect_ratio=params.get('min_aspect_ratio', 4)
        )
        
        # Convert to DetectedFence objects
        detected = []
        for c in fence_clusters:
            length_ft = (c.total_length_pts / 72.0) * scale_factor / 12.0
            detected.append(DetectedFence(
                bbox=c.bbox,
                length_ft=length_ft,
                orientation=c.orientation,
                segment_count=len(c.lines)
            ))
        
        # Evaluate
        metrics = evaluate_detection(detected, gt)
        metrics['params'] = params
        results.append(metrics)
    
    doc.close()
    return results


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Results"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)
    print(f"  Ground Truth Regions:  {metrics['gt_regions']}")
    print(f"  Detected Regions:      {metrics['detected_regions']}")
    print(f"  Matched Regions:       {metrics['matched_regions']}")
    print()
    print(f"  Region Precision:      {metrics['region_precision']:.1%}")
    print(f"  Region Recall:         {metrics['region_recall']:.1%}")
    print()
    print(f"  GT Total Length:       {metrics['gt_total_ft']:.1f} ft")
    print(f"  Detected Total Length: {metrics['detected_total_ft']:.1f} ft")
    print(f"  Length Error:          {metrics.get('length_error_pct', 0):.1f}%")
    print(f"  Length Accuracy:       {metrics['length_accuracy']:.1%}")
    print('='*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate fence detection')
    parser.add_argument('--pdf', default='gold_standard/subset_gold/selected_pages.pdf')
    parser.add_argument('--annotations', default='gold_standard/subset_gold/df_annotations_sub.csv')
    parser.add_argument('--page', type=int, default=4, help='Page number (1-indexed)')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--per-measurement', action='store_true', help='Per-measurement evaluation')
    args = parser.parse_args()
    
    if args.per_measurement:
        # Per-measurement evaluation (more targeted)
        gt_by_page = parse_annotations(args.annotations)
        gt = gt_by_page.get(args.page, [])
        
        doc = fitz.open(args.pdf)
        page = doc[args.page - 1]
        
        print(f"\nPer-Measurement Evaluation for Page {args.page}")
        print("=" * 70)
        
        # Test BEST_MATCH mode with different search radii
        print("\nMode: BEST_MATCH (find single best-matching line per measurement)")
        for radius in [100, 200, 300, 500]:
            result = evaluate_per_measurement(page, gt, scale_factor=30.0, search_radius=radius, mode='best_match')
            print(f"\n  Radius={radius}: GT={result['gt_total_ft']:.0f}ft | Det={result['detected_total_ft']:.0f}ft | "
                  f"Acc={result['length_accuracy']:.1%} | AvgErr={result['avg_error_pct']:.0f}%")
        
        # Detailed results
        print("\n" + "=" * 70)
        print("Detailed Results (best_match, radius=300):")
        print("-" * 70)
        result = evaluate_per_measurement(page, gt, scale_factor=30.0, search_radius=300, mode='best_match')
        print(f"{'GT (ft)':<10} {'Detected':<10} {'Error %':<10} {'Lines':<8} {'Location'}")
        print("-" * 70)
        for r in result['per_measurement']:
            loc = r['location']
            print(f"{r['gt_length']:<10.0f} {r['detected_length']:<10.1f} {r['error_pct']:<10.0f} {r['n_lines']:<8} ({loc[0]:.0f}, {loc[1]:.0f})")
        print("-" * 70)
        print(f"{'TOTAL':<10} {result['detected_total_ft']:<10.1f} {result['avg_error_pct']:<10.0f}")
        
        doc.close()
    
    elif args.sweep:
        # Run parameter sweep
        param_grid = {
            'min_length': [10, 15, 20],
            'distance_threshold': [20, 25, 30, 35],
            'angle_threshold': [15, 20, 25],
            'min_cluster_lines': [3, 5],
            'min_aspect_ratio': [3, 4, 5]
        }
        
        print(f"Running parameter sweep on page {args.page}...")
        results = run_parameter_sweep(
            args.pdf, args.annotations, args.page, param_grid
        )
        
        # Sort by length accuracy
        results.sort(key=lambda x: x['length_accuracy'], reverse=True)
        
        print("\nTop 10 Parameter Configurations:")
        print("-" * 80)
        for i, r in enumerate(results[:10]):
            p = r['params']
            print(f"{i+1}. Accuracy={r['length_accuracy']:.1%} | "
                  f"Error={r.get('length_error_pct',0):.0f}% | "
                  f"Detected={r['detected_total_ft']:.0f}ft vs GT={r['gt_total_ft']:.0f}ft")
            print(f"   Params: min_len={p['min_length']}, dist={p['distance_threshold']}, "
                  f"angle={p['angle_threshold']}, min_lines={p['min_cluster_lines']}, aspect={p['min_aspect_ratio']}")
    else:
        # Single evaluation
        gt_by_page = parse_annotations(args.annotations)
        
        print(f"\nGround Truth Summary:")
        for page, annotations in sorted(gt_by_page.items()):
            lengths = [a.length_ft for a in annotations if a.length_ft]
            total = sum(lengths)
            print(f"  Page {page}: {len(annotations)} annotations, {len(lengths)} with measurements, total={total:.0f} ft")
