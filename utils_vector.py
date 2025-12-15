"""
utils_vector.py - PDF Vector Drawing Extraction Utilities

Extracts and processes vector drawing elements (lines, paths) from PDF files.
Used for measuring fence-related elements identified by ADE detection.
"""

import fitz  # PyMuPDF
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


@dataclass
class VectorLine:
    """Represents a line extracted from a PDF drawing."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length_pts: float  # Length in PDF points (1/72 inch)
    color: Optional[Tuple]
    width: float
    dashes: str
    layer: str
    
    @property
    def length_inches(self) -> float:
        """Length in inches (unscaled)."""
        return self.length_pts / 72.0
    
    def length_feet(self, scale_factor: float = 1.0) -> float:
        """Length in feet with scale factor applied."""
        return self.length_inches * scale_factor / 12.0


def transform_coords_for_rotation(
    x: float, y: float, 
    rotation: int, 
    mediabox_width: float, 
    mediabox_height: float
) -> Tuple[float, float]:
    """
    Transform MediaBox coordinates to display coordinates based on page rotation.
    
    Args:
        x, y: Coordinates in MediaBox space
        rotation: Page rotation in degrees (0, 90, 180, 270)
        mediabox_width, mediabox_height: Dimensions of the MediaBox
    
    Returns:
        Transformed (x, y) coordinates in display space
    """
    if rotation == 0:
        return x, y
    elif rotation == 90:
        return mediabox_height - y, x
    elif rotation == 180:
        return mediabox_width - x, mediabox_height - y
    elif rotation == 270:
        return y, mediabox_width - x
    return x, y


def extract_layer_names(page: fitz.Page) -> List[str]:
    """
    Extract unique layer names from a PDF page's drawings.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        List of unique layer names
    """
    drawings = page.get_drawings()
    layers = set()
    for d in drawings:
        layer = d.get('layer', '')
        if layer:
            layers.add(layer)
    return sorted(list(layers))


def get_layer_summary(page: fitz.Page) -> Dict[str, int]:
    """
    Get a summary of layers and their drawing counts.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        Dictionary mapping layer names to drawing counts
    """
    drawings = page.get_drawings()
    return dict(Counter(d.get('layer', 'None') for d in drawings))


def extract_vector_lines(page: fitz.Page, apply_rotation: bool = True) -> List[VectorLine]:
    """
    Extract all line elements from a PDF page's vector drawings.
    
    Args:
        page: PyMuPDF page object
        apply_rotation: Whether to transform coordinates for page rotation
    
    Returns:
        List of VectorLine objects
    """
    drawings = page.get_drawings()
    rotation = page.rotation if apply_rotation else 0
    mbox_w = page.mediabox.width
    mbox_h = page.mediabox.height
    
    lines = []
    for d in drawings:
        layer = d.get('layer', '')
        color = d.get('color')
        width = d.get('width', 0)
        dashes = str(d.get('dashes', ''))
        
        for item in d.get('items', []):
            if item[0] == 'l':  # Line element
                p1, p2 = item[1], item[2]
                
                # Transform coordinates if needed
                if apply_rotation and rotation != 0:
                    x1, y1 = transform_coords_for_rotation(p1.x, p1.y, rotation, mbox_w, mbox_h)
                    x2, y2 = transform_coords_for_rotation(p2.x, p2.y, rotation, mbox_w, mbox_h)
                else:
                    x1, y1 = p1.x, p1.y
                    x2, y2 = p2.x, p2.y
                
                # Calculate length
                length_pts = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                lines.append(VectorLine(
                    start=(x1, y1),
                    end=(x2, y2),
                    length_pts=length_pts,
                    color=color,
                    width=width,
                    dashes=dashes,
                    layer=layer
                ))
    
    return lines


def extract_lines_by_layers(page: fitz.Page, layer_names: List[str]) -> List[VectorLine]:
    """
    Extract lines only from specified layers.
    
    Args:
        page: PyMuPDF page object
        layer_names: List of layer names to include
    
    Returns:
        List of VectorLine objects from matching layers
    """
    all_lines = extract_vector_lines(page)
    layer_set = set(layer_names)
    return [line for line in all_lines if line.layer in layer_set]


def group_lines_by_layer(lines: List[VectorLine]) -> Dict[str, List[VectorLine]]:
    """
    Group lines by their layer name.
    
    Args:
        lines: List of VectorLine objects
    
    Returns:
        Dictionary mapping layer names to lists of lines
    """
    groups = {}
    for line in lines:
        if line.layer not in groups:
            groups[line.layer] = []
        groups[line.layer].append(line)
    return groups


def points_close(p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float) -> bool:
    """Check if two points are within tolerance distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < tolerance


def group_connected_lines(lines: List[VectorLine], tolerance: float = 2.0) -> List[List[VectorLine]]:
    """
    Group lines that share endpoints (connected segments).
    
    Args:
        lines: List of VectorLine objects
        tolerance: Maximum distance to consider points connected
    
    Returns:
        List of line groups, where each group contains connected lines
    """
    if not lines:
        return []
    
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


def calculate_total_length(lines: List[VectorLine], scale_factor: float = 1.0) -> Dict:
    """
    Calculate total length of lines with various unit conversions.
    
    Args:
        lines: List of VectorLine objects
        scale_factor: Drawing scale (e.g., 30 means 1" on drawing = 30" actual)
    
    Returns:
        Dictionary with length measurements
    """
    total_pts = sum(line.length_pts for line in lines)
    total_inches_unscaled = total_pts / 72.0
    total_inches_scaled = total_inches_unscaled * scale_factor
    total_feet = total_inches_scaled / 12.0
    
    return {
        'segment_count': len(lines),
        'total_pts': total_pts,
        'total_inches_unscaled': total_inches_unscaled,
        'total_inches_scaled': total_inches_scaled,
        'total_feet': total_feet,
        'scale_factor': scale_factor
    }


def find_lines_near_point(
    lines: List[VectorLine], 
    point: Tuple[float, float], 
    radius: float = 50.0
) -> List[VectorLine]:
    """
    Find lines whose endpoints are within radius of a given point.
    
    Args:
        lines: List of VectorLine objects
        point: (x, y) coordinates to search around
        radius: Search radius in PDF points
    
    Returns:
        List of nearby VectorLine objects
    """
    nearby = []
    px, py = point
    
    for line in lines:
        # Check if either endpoint is near the point
        d1 = math.sqrt((line.start[0] - px)**2 + (line.start[1] - py)**2)
        d2 = math.sqrt((line.end[0] - px)**2 + (line.end[1] - py)**2)
        
        if d1 <= radius or d2 <= radius:
            nearby.append(line)
    
    return nearby


def find_lines_near_bbox(
    lines: List[VectorLine],
    bbox: Tuple[float, float, float, float],
    margin: float = 20.0
) -> List[VectorLine]:
    """
    Find lines that pass through or near a bounding box.
    
    Args:
        lines: List of VectorLine objects
        bbox: (x0, y0, x1, y1) bounding box
        margin: Distance margin around the bbox
    
    Returns:
        List of VectorLine objects near the bbox
    """
    x0, y0, x1, y1 = bbox
    x0 -= margin
    y0 -= margin
    x1 += margin
    y1 += margin
    
    nearby = []
    for line in lines:
        # Check if either endpoint is in the expanded bbox
        sx, sy = line.start
        ex, ey = line.end
        
        start_in = x0 <= sx <= x1 and y0 <= sy <= y1
        end_in = x0 <= ex <= x1 and y0 <= ey <= y1
        
        if start_in or end_in:
            nearby.append(line)
    
    return nearby


def infer_scale_from_text(text: str) -> Optional[float]:
    """
    Attempt to infer the drawing scale factor from text.
    
    Parses common scale notations like:
    - "SCALE: 1" = 30'-0""
    - "1:30"
    - "SCALE 1/30"
    
    Args:
        text: Text to search for scale notation
    
    Returns:
        Scale factor (e.g., 30.0) or None if not found
    """
    # Pattern 1: SCALE: 1" = XX'-Y" or 1" = XX'
    pattern1 = r'(?:SCALE[:\s]*)?1["\']?\s*=\s*(\d+)[\'′\-]'
    match1 = re.search(pattern1, text, re.IGNORECASE)
    if match1:
        return float(match1.group(1)) * 12  # Convert feet to inches
    
    # Pattern 2: 1:XX or 1/XX
    pattern2 = r'(?:SCALE[:\s]*)?1\s*[:/]\s*(\d+)'
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match2:
        return float(match2.group(1))
    
    # Pattern 3: SCALE: 1" = XX" (inches)
    pattern3 = r'(?:SCALE[:\s]*)?1["\']?\s*=\s*(\d+)["\'](?!\s*-)'
    match3 = re.search(pattern3, text, re.IGNORECASE)
    if match3:
        return float(match3.group(1))
    
    return None


def infer_scale_from_page(page: fitz.Page) -> Optional[float]:
    """
    Attempt to infer scale factor from a PDF page's text content.
    
    Args:
        page: PyMuPDF page object
    
    Returns:
        Scale factor or None if not found
    """
    # Extract all text from the page
    text = page.get_text()
    return infer_scale_from_text(text)


if __name__ == "__main__":
    # Simple test
    import sys
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/leo_streamlit/subset_gold/selected_pages_no_annotations.pdf"
    
    doc = fitz.open(pdf_path)
    print(f"Analyzing: {pdf_path}")
    print(f"Pages: {len(doc)}")
    
    for page_idx in range(min(2, len(doc))):  # First 2 pages
        page = doc[page_idx]
        print(f"\n--- Page {page_idx + 1} ---")
        print(f"Size: {page.rect.width:.1f} x {page.rect.height:.1f}, Rotation: {page.rotation}°")
        
        # Layer summary
        layer_summary = get_layer_summary(page)
        print(f"Layers ({len(layer_summary)}): {list(layer_summary.keys())[:5]}...")
        
        # Extract lines
        lines = extract_vector_lines(page)
        print(f"Total lines: {len(lines)}")
        
        # Try to infer scale
        scale = infer_scale_from_page(page)
        print(f"Inferred scale: {scale}")
        
        # Calculate total length
        if lines:
            measurements = calculate_total_length(lines, scale or 1.0)
            print(f"Total length: {measurements['total_feet']:.1f} ft (scale={scale or 1.0})")
    
    doc.close()
