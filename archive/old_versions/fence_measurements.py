"""
Fence Line Measurement Script
Extracts and measures fence-related elements from engineering drawings.
"""

import fitz
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class FenceLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_pts: float
    length_feet: float
    layer: str
    width: float
    dashes: str


def extract_fence_elements(pdf_path: str) -> Dict:
    """Extract fence-related elements from all pages."""
    doc = fitz.open(pdf_path)
    
    # Keywords for fence-related layers
    FENCE_KEYWORDS = ['FENC', 'WALL', 'BNDY', 'PROP', 'ESMT']
    
    all_results = {}
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        drawings = page.get_drawings()
        
        page_results = {
            'page_num': page_idx + 1,
            'page_size_inches': (page.rect.width / 72, page.rect.height / 72),
            'rotation': page.rotation,
            'elements': {}
        }
        
        for keyword in FENCE_KEYWORDS:
            # Filter drawings by layer keyword
            layer_drawings = [d for d in drawings if keyword in d.get('layer', '').upper()]
            
            lines = []
            for d in layer_drawings:
                layer = d.get('layer', '')
                width = d.get('width', 0)
                dashes = str(d.get('dashes', ''))
                
                for item in d.get('items', []):
                    if item[0] == 'l':  # Line
                        p1, p2 = item[1], item[2]
                        length_pts = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                        
                        lines.append(FenceLine(
                            start=(p1.x, p1.y),
                            end=(p2.x, p2.y),
                            length_pts=length_pts,
                            length_feet=length_pts / 72 / 12,
                            layer=layer,
                            width=width,
                            dashes=dashes
                        ))
            
            if lines:
                total_length_pts = sum(l.length_pts for l in lines)
                page_results['elements'][keyword] = {
                    'segment_count': len(lines),
                    'total_length_pts': total_length_pts,
                    'total_length_feet': total_length_pts / 72 / 12,
                    'layers': list(set(l.layer for l in lines)),
                    'lines': lines
                }
        
        all_results[page_idx + 1] = page_results
    
    doc.close()
    return all_results


def group_connected_lines(lines: List[FenceLine], tolerance: float = 2.0) -> List[List[FenceLine]]:
    """Group lines that share endpoints (connected segments)."""
    if not lines:
        return []
    
    def points_close(p1, p2, tol):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < tol
    
    groups = []
    used = set()
    
    for i, line in enumerate(lines):
        if i in used:
            continue
        
        group = [line]
        used.add(i)
        
        # Find connected lines
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(lines):
                if j in used:
                    continue
                
                # Check if any endpoint connects
                for g_line in group:
                    if (points_close(g_line.start, other.start, tolerance) or
                        points_close(g_line.start, other.end, tolerance) or
                        points_close(g_line.end, other.start, tolerance) or
                        points_close(g_line.end, other.end, tolerance)):
                        group.append(other)
                        used.add(j)
                        changed = True
                        break
        
        groups.append(group)
    
    return groups


def print_summary(results: Dict, scale_factor: float = 1.0):
    """Print a formatted summary of fence measurements.
    
    Args:
        scale_factor: Drawing scale (e.g., 30 means 1" = 30')
    """
    print("=" * 80)
    print("FENCE/BOUNDARY MEASUREMENT SUMMARY")
    if scale_factor != 1.0:
        print(f"Scale: 1\" = {scale_factor}'-0\" (applying {scale_factor}x multiplier)")
    print("=" * 80)
    
    grand_total = defaultdict(float)
    
    for page_num, page_data in results.items():
        print(f"\n{'─' * 80}")
        print(f"PAGE {page_num}")
        print(f"Size: {page_data['page_size_inches'][0]:.1f}\" x {page_data['page_size_inches'][1]:.1f}\"")
        print(f"Rotation: {page_data['rotation']}°")
        print(f"{'─' * 80}")
        
        if not page_data['elements']:
            print("  No fence/boundary elements found")
            continue
        
        for keyword, data in page_data['elements'].items():
            scaled_length = data['total_length_feet'] * scale_factor
            print(f"\n  {keyword}:")
            print(f"    Segments: {data['segment_count']}")
            print(f"    Total length: {scaled_length:.1f} ft (raw: {data['total_length_feet']:.1f} ft)")
            print(f"    Layers: {', '.join(data['layers'])}")
            
            # Group connected lines
            groups = group_connected_lines(data['lines'])
            if len(groups) > 1:
                print(f"    Connected runs: {len(groups)}")
                for i, group in enumerate(groups[:5]):  # Show first 5
                    group_len = sum(l.length_feet for l in group) * scale_factor
                    print(f"      Run {i+1}: {len(group)} segments, {group_len:.1f} ft")
                if len(groups) > 5:
                    print(f"      ... and {len(groups) - 5} more runs")
            
            grand_total[keyword] += scaled_length
    
    # Grand total
    print(f"\n{'=' * 80}")
    print("GRAND TOTAL (ALL PAGES)")
    print("=" * 80)
    for keyword, total_ft in grand_total.items():
        print(f"  {keyword}: {total_ft:.1f} ft")


if __name__ == "__main__":
    import sys
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/leo_streamlit/subset_gold/selected_pages_no_annotations.pdf"
    
    # Scale from drawing: 1" = 30'-0" means multiply by 30
    SCALE_FACTOR = 30  # feet per inch on drawing
    
    print(f"Analyzing: {pdf_path}\n")
    
    results = extract_fence_elements(pdf_path)
    print_summary(results, scale_factor=SCALE_FACTOR)
