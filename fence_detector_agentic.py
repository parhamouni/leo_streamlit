"""
fence_detector_agentic.py - Agentic fence detection with iterative refinement

Uses an LLM agent that:
1. Analyzes the page structure
2. Identifies potential fence regions
3. Validates detections by examining line geometry
4. Iteratively refines and groups elements
"""

import fitz
import math
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LineCluster:
    """A cluster of connected/related lines."""
    lines: List[Dict] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    orientation: str = "unknown"  # horizontal, vertical, diagonal
    total_length_pts: float = 0
    
    def add_line(self, line: Dict):
        self.lines.append(line)
        self._update_stats()
    
    def _update_stats(self):
        if not self.lines:
            return
        
        xs = []
        ys = []
        for l in self.lines:
            xs.extend([l['start'][0], l['end'][0]])
            ys.extend([l['start'][1], l['end'][1]])
        
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.total_length_pts = sum(l['length_pts'] for l in self.lines)
        
        # Determine orientation
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        if width > height * 3:
            self.orientation = "horizontal"
        elif height > width * 3:
            self.orientation = "vertical"
        else:
            self.orientation = "diagonal"


def extract_all_lines(page: fitz.Page, min_length: float = 5.0) -> List[Dict]:
    """Extract all lines from page."""
    drawings = page.get_drawings()
    lines = []
    
    for d in drawings:
        layer = d.get('layer', '')
        for item in d.get('items', []):
            if item[0] == 'l':
                p1, p2 = item[1], item[2]
                length = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                if length >= min_length:
                    # Calculate angle
                    angle = math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / math.pi
                    lines.append({
                        'start': (p1.x, p1.y),
                        'end': (p2.x, p2.y),
                        'length_pts': length,
                        'angle': angle,
                        'layer': layer
                    })
    
    return lines


def cluster_by_proximity_and_orientation(
    lines: List[Dict],
    distance_threshold: float = 30.0,
    angle_threshold: float = 15.0
) -> List[LineCluster]:
    """
    Cluster lines that are:
    1. Spatially close (endpoints within distance_threshold)
    2. Similar orientation (angle difference within angle_threshold)
    """
    if not lines:
        return []
    
    used = set()
    clusters = []
    
    def points_close(p1, p2, thresh):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < thresh
    
    def angles_similar(a1, a2, thresh):
        diff = abs(a1 - a2) % 180
        return diff < thresh or (180 - diff) < thresh
    
    for i, line in enumerate(lines):
        if i in used:
            continue
        
        cluster = LineCluster()
        cluster.add_line(line)
        used.add(i)
        
        # Find connected lines with similar orientation
        queue = [line]
        while queue:
            current = queue.pop(0)
            
            for j, other in enumerate(lines):
                if j in used:
                    continue
                
                # Check proximity
                close = (points_close(current['start'], other['start'], distance_threshold) or
                        points_close(current['start'], other['end'], distance_threshold) or
                        points_close(current['end'], other['start'], distance_threshold) or
                        points_close(current['end'], other['end'], distance_threshold))
                
                # Check orientation similarity
                similar_angle = angles_similar(current['angle'], other['angle'], angle_threshold)
                
                if close and similar_angle:
                    cluster.add_line(other)
                    used.add(j)
                    queue.append(other)
        
        clusters.append(cluster)
    
    return clusters


def filter_fence_like_clusters(
    clusters: List[LineCluster],
    min_lines: int = 3,
    min_length_pts: float = 100.0,
    min_aspect_ratio: float = 3.0
) -> List[LineCluster]:
    """
    Filter clusters that have fence-like characteristics:
    - Multiple connected segments
    - Sufficient total length
    - Linear shape (high aspect ratio)
    """
    fence_clusters = []
    
    for cluster in clusters:
        if len(cluster.lines) < min_lines:
            continue
        if cluster.total_length_pts < min_length_pts:
            continue
        
        width = cluster.bbox[2] - cluster.bbox[0]
        height = cluster.bbox[3] - cluster.bbox[1]
        aspect = max(width, height) / (min(width, height) + 0.01)
        
        if aspect >= min_aspect_ratio:
            fence_clusters.append(cluster)
    
    return fence_clusters


def find_clusters_near_text(
    clusters: List[LineCluster],
    text_indicators: List[Dict],
    max_distance: float = 100.0
) -> List[Tuple[LineCluster, Dict]]:
    """
    Find clusters that are near fence text indicators.
    Returns pairs of (cluster, matching_indicator).
    """
    matches = []
    
    for cluster in clusters:
        cx = (cluster.bbox[0] + cluster.bbox[2]) / 2
        cy = (cluster.bbox[1] + cluster.bbox[3]) / 2
        
        for indicator in text_indicators:
            ix = (indicator.get('x0', 0) + indicator.get('x1', 0)) / 2
            iy = (indicator.get('y0', 0) + indicator.get('y1', 0)) / 2
            
            dist = math.sqrt((cx - ix)**2 + (cy - iy)**2)
            
            if dist < max_distance:
                matches.append((cluster, indicator))
                break
    
    return matches


def analyze_page_structure(page: fitz.Page) -> Dict:
    """
    Analyze page structure to understand layout.
    """
    w, h = page.rect.width, page.rect.height
    
    # Define regions
    regions = {
        'title_block': (w * 0.7, h * 0.8, w, h),  # Bottom right typically
        'legend': (w * 0.7, 0, w, h * 0.3),  # Top right often
        'main_drawing': (0, 0, w * 0.7, h * 0.8),  # Main area
        'notes': (0, h * 0.8, w * 0.7, h),  # Bottom notes
    }
    
    # Extract text to find indicators
    text_dict = page.get_text("dict")
    text_items = []
    
    fence_keywords = ['fence', 'fencing', 'gate', 'guardrail', 'railing', 
                      'barrier', 'chain link', 'f-1', 'f-2', 'f-3']
    
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").lower()
                    bbox = span.get("bbox", (0, 0, 0, 0))
                    
                    for kw in fence_keywords:
                        if kw in text:
                            text_items.append({
                                'text': span.get("text", ""),
                                'x0': bbox[0], 'y0': bbox[1],
                                'x1': bbox[2], 'y1': bbox[3],
                                'keyword': kw
                            })
                            break
    
    return {
        'width': w,
        'height': h,
        'regions': regions,
        'fence_indicators': text_items
    }


def find_lines_near_indicator(
    all_lines: List[Dict],
    indicator: Dict,
    search_radius: float = 150.0,
    connection_tolerance: float = 25.0
) -> List[Dict]:
    """
    INDICATOR-DRIVEN: Find all lines connected to an indicator.
    
    1. Find lines near the indicator bbox
    2. Trace connected lines from those starting points
    """
    # Get indicator center
    ix = (indicator['x0'] + indicator['x1']) / 2
    iy = (indicator['y0'] + indicator['y1']) / 2
    
    # Find seed lines near indicator
    seed_lines = []
    for line in all_lines:
        # Check distance from line endpoints to indicator center
        d1 = math.sqrt((line['start'][0] - ix)**2 + (line['start'][1] - iy)**2)
        d2 = math.sqrt((line['end'][0] - ix)**2 + (line['end'][1] - iy)**2)
        
        if min(d1, d2) <= search_radius:
            seed_lines.append(line)
    
    if not seed_lines:
        return []
    
    # Now trace all connected lines from seeds
    connected = set()
    used_ids = set()
    
    def points_close(p1, p2, tol):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < tol
    
    # BFS from seed lines
    queue = list(seed_lines)
    for line in seed_lines:
        connected.add(id(line))
        used_ids.add(id(line))
    
    while queue:
        current = queue.pop(0)
        
        for line in all_lines:
            if id(line) in used_ids:
                continue
            
            # Check if connected
            if (points_close(current['start'], line['start'], connection_tolerance) or
                points_close(current['start'], line['end'], connection_tolerance) or
                points_close(current['end'], line['start'], connection_tolerance) or
                points_close(current['end'], line['end'], connection_tolerance)):
                connected.add(id(line))
                used_ids.add(id(line))
                queue.append(line)
    
    # Return connected lines
    return [l for l in all_lines if id(l) in connected]


def agentic_fence_detection(
    page: fitz.Page,
    scale_factor: float = 1.0,
    verbose: bool = True,
    search_radius: float = 150.0,
    connection_tolerance: float = 25.0
) -> Dict:
    """
    INDICATOR-DRIVEN Agentic fence detection pipeline:
    
    Step 1: Find fence text indicators (F-1, CHAIN LINK, etc.)
    Step 2: For EACH indicator, find nearby lines
    Step 3: Trace connected lines from those seeds
    Step 4: Measure and report
    
    This uses indicators as the PRIMARY GUIDE, not post-hoc matching.
    """
    results = {
        'steps': [],
        'fence_regions': [],
        'indicator_regions': [],  # Indicator-driven detections
        'unmatched_regions': [],  # Fallback detections
        'total_length_feet': 0
    }
    
    # Step 1: Find indicators FIRST
    if verbose:
        print("Step 1: Finding fence text indicators...")
    structure = analyze_page_structure(page)
    indicators = structure['fence_indicators']
    results['steps'].append({
        'step': 1,
        'action': 'find_indicators',
        'found_indicators': len(indicators)
    })
    if verbose:
        print(f"  Found {len(indicators)} fence text indicators")
        for ind in indicators[:10]:
            print(f"    - '{ind['text']}' at ({ind['x0']:.0f}, {ind['y0']:.0f})")
    
    # Step 2: Extract all lines
    if verbose:
        print("\nStep 2: Extracting vector lines...")
    all_lines = extract_all_lines(page, min_length=10.0)
    results['steps'].append({
        'step': 2,
        'action': 'extract_lines',
        'total_lines': len(all_lines)
    })
    if verbose:
        print(f"  Extracted {len(all_lines)} lines")
    
    # Step 3: INDICATOR-DRIVEN - For each indicator, find connected lines
    if verbose:
        print("\nStep 3: Tracing lines from each indicator...")
    
    used_line_ids = set()  # Track which lines have been assigned
    
    for indicator in indicators:
        # Find lines connected to this indicator
        connected_lines = find_lines_near_indicator(
            all_lines, 
            indicator,
            search_radius=search_radius,
            connection_tolerance=connection_tolerance
        )
        
        # Filter out already-used lines
        new_lines = [l for l in connected_lines if id(l) not in used_line_ids]
        
        if len(new_lines) >= 3:  # Minimum lines to count as a region
            # Mark as used
            for l in new_lines:
                used_line_ids.add(id(l))
            
            # Calculate bbox
            xs = []
            ys = []
            for l in new_lines:
                xs.extend([l['start'][0], l['end'][0]])
                ys.extend([l['start'][1], l['end'][1]])
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            # Calculate length
            total_pts = sum(l['length_pts'] for l in new_lines)
            length_feet = (total_pts / 72.0) * scale_factor / 12.0
            
            # Determine orientation
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width > height * 2:
                orientation = "horizontal"
            elif height > width * 2:
                orientation = "vertical"
            else:
                orientation = "mixed"
            
            region = {
                'type': 'indicator_driven',
                'indicator': indicator['text'],
                'indicator_keyword': indicator['keyword'],
                'indicator_bbox': (indicator['x0'], indicator['y0'], indicator['x1'], indicator['y1']),
                'bbox': bbox,
                'orientation': orientation,
                'segment_count': len(new_lines),
                'length_pts': total_pts,
                'length_feet': round(length_feet, 2),
                'lines': new_lines
            }
            results['indicator_regions'].append(region)
            results['fence_regions'].append(region)
            
            if verbose:
                print(f"    '{indicator['text']}' → {len(new_lines)} lines, {length_feet:.1f} ft")
    
    results['steps'].append({
        'step': 3,
        'action': 'trace_from_indicators',
        'regions_found': len(results['indicator_regions'])
    })
    
    # Step 4: Fallback - find large unassigned line clusters (for pages with few/no indicators)
    if verbose:
        print("\nStep 4: Finding additional unassigned fence-like structures...")
    
    # Get remaining unassigned lines
    remaining_lines = [l for l in all_lines if id(l) not in used_line_ids]
    
    if remaining_lines:
        # Cluster remaining lines
        remaining_clusters = cluster_by_proximity_and_orientation(
            remaining_lines,
            distance_threshold=connection_tolerance,
            angle_threshold=20.0
        )
        
        # Filter for fence-like characteristics
        fence_like = filter_fence_like_clusters(
            remaining_clusters,
            min_lines=20,  # Higher threshold for unguided detection
            min_length_pts=500.0,
            min_aspect_ratio=5.0
        )
        
        for cluster in fence_like:
            total_pts = cluster.total_length_pts
            length_feet = (total_pts / 72.0) * scale_factor / 12.0
            
            region = {
                'type': 'fallback_detected',
                'indicator': None,
                'indicator_keyword': None,
                'indicator_bbox': None,
                'bbox': cluster.bbox,
                'orientation': cluster.orientation,
                'segment_count': len(cluster.lines),
                'length_pts': total_pts,
                'length_feet': round(length_feet, 2),
                'lines': cluster.lines
            }
            results['unmatched_regions'].append(region)
            results['fence_regions'].append(region)
            
            if verbose:
                print(f"    Fallback: {cluster.orientation} structure - {length_feet:.1f} ft ({len(cluster.lines)} segments)")
    
    results['steps'].append({
        'step': 4,
        'action': 'fallback_detection',
        'fallback_regions': len(results['unmatched_regions'])
    })
    
    # Step 5: Calculate totals
    total_length_pts = sum(r['length_pts'] for r in results['fence_regions'])
    results['total_length_feet'] = round((total_length_pts / 72.0) * scale_factor / 12.0, 2)
    results['scale_factor'] = scale_factor
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(results['indicator_regions'])} indicator-driven + {len(results['unmatched_regions'])} fallback")
        print(f"TOTAL: {results['total_length_feet']:.1f} ft")
        print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    import sys
    
    pdf_path = "/home/ec2-user/project/leo_streamlit/gold_standard/subset_gold/selected_pages_no_annotations.pdf"
    page_num = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    print(f"Analyzing page {page_num + 1} ({page.rect.width:.0f} x {page.rect.height:.0f})")
    print("=" * 60)
    
    result = agentic_fence_detection(page, scale_factor=30.0, verbose=True)
    
    print("\n" + "=" * 60)
    print(f"Found {len(result['fence_regions'])} fence regions")
    print(f"Total length: {result['total_length_feet']:.1f} feet")
    
    doc.close()
