"""
utils_vector.py - PDF Vector Drawing Extraction Utilities

Extracts and processes vector drawing elements (lines, paths) from PDF files.
Used for measuring fence-related elements identified by ADE detection.
"""

import fitz  # PyMuPDF
import math
import re
import json
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Any
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


def distance_point_to_line_segment(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """
    Calculate the shortest distance from a point to a line segment.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line is a point
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t for the closest point on the line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # Closest point on the segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def find_closest_line_to_point(
    lines: List[VectorLine],
    point: Tuple[float, float]
) -> Optional[Tuple[VectorLine, float]]:
    """
    Find the line closest to a given point.
    
    Args:
        lines: List of VectorLine objects
        point: (x, y) coordinates
    
    Returns:
        Tuple of (closest_line, distance) or None if no lines
    """
    if not lines:
        return None
    
    closest = None
    min_dist = float('inf')
    
    for line in lines:
        dist = distance_point_to_line_segment(point, line.start, line.end)
        if dist < min_dist:
            min_dist = dist
            closest = line
    
    return (closest, min_dist) if closest else None


def trace_connected_lines_from_start(
    start_line: VectorLine,
    all_lines: List[VectorLine],
    tolerance: float = 5.0,
    max_lines: int = 500
) -> List[VectorLine]:
    """
    Trace all lines connected to a starting line using flood-fill.
    
    Args:
        start_line: The line to start tracing from
        all_lines: All available lines to search
        tolerance: Maximum distance to consider points connected
        max_lines: Maximum lines to include (safety limit)
    
    Returns:
        List of connected VectorLine objects
    """
    connected = [start_line]
    used_indices = {id(start_line)}
    
    # Create index for faster lookup
    line_ids = {id(line): line for line in all_lines}
    
    # BFS to find connected lines
    queue = [start_line]
    
    while queue and len(connected) < max_lines:
        current = queue.pop(0)
        
        for line in all_lines:
            if id(line) in used_indices:
                continue
            
            # Check if this line connects to current
            if (points_close(current.start, line.start, tolerance) or
                points_close(current.start, line.end, tolerance) or
                points_close(current.end, line.start, tolerance) or
                points_close(current.end, line.end, tolerance)):
                connected.append(line)
                used_indices.add(id(line))
                queue.append(line)
    
    return connected


def find_fence_run_from_indicator(
    all_lines: List[VectorLine],
    indicator_bbox: Tuple[float, float, float, float],
    max_initial_distance: float = 50.0,
    connection_tolerance: float = 5.0
) -> List[VectorLine]:
    """
    Smart fence run detection: find the closest line to an indicator,
    then trace all connected lines to get the complete fence run.
    
    Args:
        all_lines: All vector lines on the page
        indicator_bbox: Bounding box of the indicator (x0, y0, x1, y1)
        max_initial_distance: Maximum distance to the starting line
        connection_tolerance: Tolerance for connecting lines
    
    Returns:
        List of VectorLine objects that form the fence run
    """
    # Get indicator center point
    cx = (indicator_bbox[0] + indicator_bbox[2]) / 2
    cy = (indicator_bbox[1] + indicator_bbox[3]) / 2
    indicator_center = (cx, cy)
    
    # Find the closest line to the indicator
    result = find_closest_line_to_point(all_lines, indicator_center)
    
    if not result:
        return []
    
    closest_line, distance = result
    
    # If the closest line is too far, skip
    if distance > max_initial_distance:
        return []
    
    # Trace all connected lines from this starting point
    connected_lines = trace_connected_lines_from_start(
        closest_line, 
        all_lines, 
        tolerance=connection_tolerance
    )
    
    return connected_lines


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


def infer_scale_from_page(page: fitz.Page, ocr_text: str = None) -> Optional[float]:
    """
    Attempt to infer scale factor from a PDF page's text content.
    
    Args:
        page: PyMuPDF page object
        ocr_text: OCR text as fallback when page has no embedded text
    
    Returns:
        Scale factor or None if not found
    """
    # Extract all text from the page
    text = page.get_text()
    result = infer_scale_from_text(text)
    if result:
        return result
    # Fallback: try OCR text if embedded text didn't yield a scale
    if ocr_text:
        return infer_scale_from_text(ocr_text)
    return None


def get_page_size_info(page: fitz.Page) -> Dict:
    """
    Get PDF page dimensions and detect if it matches standard architectural sizes.
    
    Standard architectural sizes (in points, 1 inch = 72 pts):
    - ARCH A: 9x12" (648x864)
    - ARCH B: 12x18" (864x1296)
    - ARCH C: 18x24" (1296x1728)
    - ARCH D: 24x36" (1728x2592)
    - ARCH E: 36x48" (2592x3456)
    """
    rect = page.rect
    width_pts = rect.width
    height_pts = rect.height
    width_inches = width_pts / 72.0
    height_inches = height_pts / 72.0
    
    # Standard architectural sizes (width x height in inches)
    arch_sizes = {
        'ARCH A': (9, 12),
        'ARCH B': (12, 18),
        'ARCH C': (18, 24),
        'ARCH D': (24, 36),
        'ARCH E': (36, 48),
        'ANSI A (Letter)': (8.5, 11),
        'ANSI B (Tabloid)': (11, 17),
        'ANSI C': (17, 22),
        'ANSI D': (22, 34),
    }
    
    # Check both orientations (landscape/portrait)
    detected_size = None
    scale_factor = 1.0
    
    for name, (w, h) in arch_sizes.items():
        # Check landscape
        if abs(width_inches - w) < 0.5 and abs(height_inches - h) < 0.5:
            detected_size = name
            scale_factor = 1.0
            break
        # Check portrait
        if abs(width_inches - h) < 0.5 and abs(height_inches - w) < 0.5:
            detected_size = f"{name} (portrait)"
            scale_factor = 1.0
            break
        # Check if rescaled from this size
        ratio_w = width_inches / w
        ratio_h = height_inches / h
        if abs(ratio_w - ratio_h) < 0.05 and 0.3 < ratio_w < 3.0:
            # Proportionally scaled
            detected_size = f"{name} (scaled {ratio_w:.1%})"
            scale_factor = ratio_w
            break
    
    return {
        'width_pts': width_pts,
        'height_pts': height_pts,
        'width_inches': width_inches,
        'height_inches': height_inches,
        'detected_size': detected_size,
        'scale_factor': scale_factor
    }


def detect_scale_with_vision(page: fitz.Page, llm: Any) -> Dict:
    """
    Use GPT-4V to visually detect scale notation from PDF page image.
    
    Args:
        page: PyMuPDF page object
        llm: LangChain ChatOpenAI instance (must support vision)
    
    Returns:
        Dict with scale info
    """
    from langchain_core.messages import HumanMessage
    
    # Get page size info
    page_info = get_page_size_info(page)
    
    result = {
        'success': False,
        'verified_scale': None,
        'scale_text': None,
        'confidence': 'low',
        'message': '',
        'page_size': page_info,
        'method': 'vision'
    }
    
    try:
        # Render page to image (lower DPI for efficiency, but readable)
        pix = page.get_pixmap(dpi=100)
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt_text = f"""Look at this architectural/engineering drawing and find the DRAWING SCALE notation.

PAGE SIZE: {page_info['width_inches']:.1f}" x {page_info['height_inches']:.1f}"

The scale is usually shown:
- Near a title block (bottom right corner)
- Near "SCALE:" label
- As a graphic scale bar with measurements
- Common formats: "1" = 30'-0"", "1:48", "SCALE: 1/4" = 1'-0""

Look carefully at ALL areas of the drawing, especially:
- Title block area
- Near plan/view labels
- Bottom of the page near graphic scale bars

Respond in JSON format only:
{{
    "found": true/false,
    "scale_text": "the exact scale notation you see (e.g., '1\" = 30'-0\"')",
    "scale_inches": number (what 1 inch on paper equals in inches, e.g., 360 for 30 feet, 48 for 4 feet),
    "location": "where you found it (e.g., 'bottom left near FINAL SITE PLAN')",
    "confidence": "high"/"medium"/"low",
    "reasoning": "brief explanation"
}}

If no scale found, set found=false and scale_inches=null."""

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        )
        
        response = llm.invoke([message])
        content = response.content.strip()
        result['raw_response'] = content
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        if data.get('found') and data.get('scale_inches'):
            result['success'] = True
            result['verified_scale'] = float(data['scale_inches'])
            result['scale_text'] = data.get('scale_text', '')
            result['confidence'] = data.get('confidence', 'medium')
            result['message'] = f"{data.get('reasoning', '')} (Found at: {data.get('location', 'unknown')})"
        else:
            result['message'] = data.get('reasoning', 'No scale found visually')
            
    except json.JSONDecodeError as e:
        result['message'] = f"Vision response parse error: {e}"
    except Exception as e:
        result['message'] = f"Vision error: {e}"
    
    return result


def detect_scale_with_llm(page: fitz.Page, llm: Any, use_vision: bool = True) -> Dict:
    """
    Detect scale using vision (preferred) or text-based LLM.
    
    Args:
        page: PyMuPDF page object
        llm: LangChain ChatOpenAI instance
        use_vision: If True, use GPT-4V to analyze the page image
    
    Returns:
        Dict with scale info
    """
    # Try vision-based detection first (more reliable)
    if use_vision:
        result = detect_scale_with_vision(page, llm)
        if result.get('success'):
            return result
        # Fall back to text if vision fails
        result['message'] = f"Vision failed ({result.get('message')}), trying text..."
    
    # Get page size info
    page_info = get_page_size_info(page)
    
    result = {
        'success': False,
        'verified_scale': None,
        'scale_text': None,
        'confidence': 'low',
        'message': '',
        'page_size': page_info,
        'method': 'text'
    }
    
    # Extract page text
    page_text = page.get_text()
    result['extracted_text_sample'] = page_text[:1500] if page_text else "No text extracted"
    
    if not page_text.strip():
        result['message'] = "No text found on page"
        return result
    
    if len(page_text) > 8000:
        page_text = page_text[:8000]
    
    prompt = f"""Analyze this architectural/engineering drawing text and find the DRAWING SCALE notation.

PAGE SIZE: {page_info['width_inches']:.1f}" x {page_info['height_inches']:.1f}" ({page_info.get('detected_size', 'unknown size')})

The scale tells you how measurements on paper relate to real-world measurements.
Common formats:
- "1" = 30'-0"" means 1 inch on paper = 30 feet in reality
- "1:48" means 1 unit on paper = 48 units in reality  
- "SCALE: 1/4" = 1'-0"" means quarter inch = 1 foot (= 1" = 4')

IMPORTANT: Only identify actual scale notations, NOT random text that happens to contain numbers.
Look for text near words like "SCALE", "GRAPHIC SCALE", or standard scale bar labels.

Page text:
{page_text}

Respond in JSON format only:
{{
    "found": true/false,
    "scale_text": "the exact text containing the scale (e.g., '1\" = 30'-0\"')",
    "scale_inches": number (what 1 inch on paper equals in inches, e.g., 360 for 30 feet),
    "confidence": "high"/"medium"/"low",
    "reasoning": "brief explanation"
}}

If no scale found, set found=false and scale_inches=null."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        result['raw_response'] = content
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        if data.get('found') and data.get('scale_inches'):
            result['success'] = True
            result['verified_scale'] = float(data['scale_inches'])
            result['scale_text'] = data.get('scale_text', '')
            result['confidence'] = data.get('confidence', 'medium')
            result['message'] = data.get('reasoning', 'Scale detected by LLM')
        else:
            result['message'] = data.get('reasoning', 'No scale found by LLM')
            
    except json.JSONDecodeError as e:
        result['message'] = f"LLM response parse error: {e}"
    except Exception as e:
        result['message'] = f"LLM error: {e}"
    
    return result


def verify_scale_with_bar(page: fitz.Page, llm: Any = None) -> Dict:
    """
    Detect scale factor using LLM (preferred) or regex fallback.
    
    Args:
        page: PyMuPDF page object
        llm: Optional LangChain ChatOpenAI instance for intelligent detection
    
    Returns:
        Dict with:
        - 'success': bool
        - 'verified_scale': scale in inches (what 1" on paper equals)
        - 'confidence': 'high', 'medium', 'low'
        - 'message': explanation
        - 'scale_text': the detected scale text
    """
    # Use LLM if provided
    if llm is not None:
        return detect_scale_with_llm(page, llm)
    
    # Fallback to regex-based detection
    result = {
        'success': False,
        'text_scale': None,
        'verified_scale': None,
        'scale_bar_length_pts': None,
        'confidence': 'low',
        'message': '',
        'scale_text_bbox': None,
        'scale_bar_line': None
    }
    
    # Regex patterns for scale notation
    text_dict = page.get_text("dict")
    scale_text_bbox = None
    scale_text_value = None
    
    scale_patterns = [
        (r'1["\']?\s*=\s*(\d+)[\'′\-]', 'feet'),      # 1" = 30'
        (r'1\s*[:/]\s*(\d+)', 'ratio'),               # 1:30 or 1/30
        (r'1["\']?\s*=\s*(\d+)["\'](?!\s*-)', 'inches')  # 1" = 30"
    ]
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            line_bbox = None
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                if line_bbox is None:
                    line_bbox = list(span.get("bbox", [0,0,0,0]))
                else:
                    sb = span.get("bbox", [0,0,0,0])
                    line_bbox[0] = min(line_bbox[0], sb[0])
                    line_bbox[1] = min(line_bbox[1], sb[1])
                    line_bbox[2] = max(line_bbox[2], sb[2])
                    line_bbox[3] = max(line_bbox[3], sb[3])
            
            for pattern, unit_type in scale_patterns:
                match = re.search(pattern, line_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    if unit_type == 'feet':
                        scale_text_value = value * 12
                    else:
                        scale_text_value = value
                    scale_text_bbox = line_bbox
                    break
            if scale_text_bbox:
                break
        if scale_text_bbox:
            break
    
    result['text_scale'] = scale_text_value
    result['scale_text_bbox'] = scale_text_bbox
    
    if not scale_text_bbox:
        result['message'] = "No scale text found on page"
        return result
    
    # Step 2: Find lines near the scale text
    # Detect if text bbox is rotated (tall and narrow = vertical text)
    text_width = abs(scale_text_bbox[2] - scale_text_bbox[0])
    text_height = abs(scale_text_bbox[3] - scale_text_bbox[1])
    is_rotated = text_height > text_width * 2  # vertical text indicates rotated page
    
    # Expand search area around the scale text (larger area for rotated pages)
    padding = 150 if is_rotated else 100
    search_bbox = (
        min(scale_text_bbox[0], scale_text_bbox[2]) - padding,
        min(scale_text_bbox[1], scale_text_bbox[3]) - padding,
        max(scale_text_bbox[0], scale_text_bbox[2]) + padding,
        max(scale_text_bbox[1], scale_text_bbox[3]) + padding
    )
    
    # Extract all vector lines from page
    all_lines = extract_vector_lines(page)
    
    # Find scale bar lines in search area
    # For rotated pages, look for vertical lines; otherwise horizontal
    candidate_lines = []
    for line in all_lines:
        sx, sy = line.start
        ex, ey = line.end
        
        dy = abs(ey - sy)
        dx = abs(ex - sx)
        
        # Check if line is roughly aligned (horizontal or vertical depending on rotation)
        if is_rotated:
            # Look for vertical lines (scale bar appears vertical in rotated PDF)
            if dy > 10 and dx < dy * 0.1:
                is_aligned = True
            else:
                is_aligned = False
        else:
            # Look for horizontal lines
            if dx > 10 and dy < dx * 0.1:
                is_aligned = True
            else:
                is_aligned = False
        
        if is_aligned:
            # Check if line intersects search bbox
            min_x, max_x = min(sx, ex), max(sx, ex)
            min_y, max_y = min(sy, ey), max(sy, ey)
            
            if (min_x < search_bbox[2] and max_x > search_bbox[0] and
                min_y < search_bbox[3] and max_y > search_bbox[1]):
                candidate_lines.append(line)
    
    # Rename for compatibility
    horizontal_lines = candidate_lines
    
    # Simplified approach: just use the text-based scale, no bar line detection needed
    # The scale bar is often a complex graphic (rectangles, not lines)
    result['verified_scale'] = scale_text_value
    result['success'] = True
    result['confidence'] = 'medium'
    result['message'] = f"Scale from text: 1\" = {scale_text_value/12:.0f}'"
    
    # We don't need to find the bar line - just return the text-based scale
    return result
    
    # Step 3: Find the most likely scale bar (typically a clean horizontal line)
    # Sort by length and pick the longest reasonable one
    horizontal_lines.sort(key=lambda l: l.length_pts, reverse=True)
    
    # The scale bar is typically 1 inch (72 pts) or a fraction thereof
    # Look for lines close to common scale bar lengths
    common_lengths = [72, 36, 144, 18, 90]  # 1", 0.5", 2", 0.25", 1.25"
    
    best_line = None
    best_score = float('inf')
    
    for line in horizontal_lines[:10]:  # Check top 10 longest
        for common_len in common_lengths:
            score = abs(line.length_pts - common_len)
            if score < best_score:
                best_score = score
                best_line = line
    
    if not best_line:
        best_line = horizontal_lines[0]  # fallback to longest
    
    bar_length_pts = best_line.length_pts
    result['scale_bar_length_pts'] = bar_length_pts
    result['scale_bar_line'] = (best_line.start[0], best_line.start[1], 
                                 best_line.end[0], best_line.end[1])
    
    # Step 4: Calculate verified scale
    # If text says "1" = X'" and bar is supposed to be 1", 
    # then: verified_scale = text_scale * (72 / bar_length_pts)
    # This corrects for any PDF resizing
    
    if scale_text_value and bar_length_pts > 0:
        # Assume the scale bar represents 1 inch on drawing
        # If bar measures 72 pts, no correction needed
        # If bar measures 36 pts (PDF scaled 50%), multiply scale by 2
        correction_factor = 72.0 / bar_length_pts
        verified_scale = scale_text_value * correction_factor
        
        result['verified_scale'] = verified_scale
        result['success'] = True
        
        # Determine confidence
        if 0.9 <= correction_factor <= 1.1:
            result['confidence'] = 'high'
            result['message'] = f"Scale verified: bar={bar_length_pts:.1f}pts (≈1\"), scale={verified_scale:.0f}\""
        elif 0.5 <= correction_factor <= 2.0:
            result['confidence'] = 'medium'
            result['message'] = f"PDF appears resized ({correction_factor:.2f}x). Adjusted scale: {verified_scale:.0f}\""
        else:
            result['confidence'] = 'low'
            result['message'] = f"Unusual correction ({correction_factor:.2f}x). Using adjusted scale: {verified_scale:.0f}\""
    
    return result


def find_lines_in_bbox(
    lines: List[VectorLine],
    bbox: Tuple[float, float, float, float],
    require_both_endpoints: bool = False
) -> List[VectorLine]:
    """
    Find all lines that are inside or intersect a bounding box.
    
    Args:
        lines: List of VectorLine objects
        bbox: (x0, y0, x1, y1) bounding box in PDF coordinates
        require_both_endpoints: If True, both endpoints must be inside bbox.
                                If False, at least one endpoint must be inside.
    
    Returns:
        List of VectorLine objects inside the bbox
    """
    x0, y0, x1, y1 = bbox
    # Normalize bbox (ensure x0 < x1, y0 < y1)
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    
    matched = []
    for line in lines:
        sx, sy = line.start
        ex, ey = line.end
        
        start_in = x0 <= sx <= x1 and y0 <= sy <= y1
        end_in = x0 <= ex <= x1 and y0 <= ey <= y1
        
        if require_both_endpoints:
            if start_in and end_in:
                matched.append(line)
        else:
            if start_in or end_in:
                matched.append(line)
    
    return matched


def measure_lines_in_selection(
    page: fitz.Page,
    selection_bbox: Tuple[float, float, float, float],
    scale_factor: float = 1.0,
    min_line_length: float = 5.0
) -> Dict:
    """
    Measure all vector lines within a user-selected bounding box.
    
    Args:
        page: PyMuPDF page object
        selection_bbox: (x0, y0, x1, y1) selection area in PDF coordinates
        scale_factor: Drawing scale factor (e.g., 360 means 1" = 30')
        min_line_length: Minimum line length in points to include
    
    Returns:
        Dictionary with measurement results
    """
    # Extract all lines from the page
    all_lines = extract_vector_lines(page, apply_rotation=True)
    
    # Filter to lines within selection
    selected_lines = find_lines_in_bbox(all_lines, selection_bbox, require_both_endpoints=False)
    
    # Filter out tiny lines (hatching, etc.)
    selected_lines = [l for l in selected_lines if l.length_pts >= min_line_length]
    
    if not selected_lines:
        return {
            'success': True,
            'line_count': 0,
            'total_length_pts': 0,
            'total_length_feet': 0,
            'scale_factor': scale_factor,
            'lines': [],
            'layers': []
        }
    
    # Calculate measurements
    total_pts = sum(l.length_pts for l in selected_lines)
    total_inches_scaled = (total_pts / 72.0) * scale_factor
    total_feet = total_inches_scaled / 12.0
    
    # Get layer breakdown
    layer_lengths = {}
    for line in selected_lines:
        layer = line.layer or 'Unknown'
        if layer not in layer_lengths:
            layer_lengths[layer] = {'count': 0, 'length_pts': 0}
        layer_lengths[layer]['count'] += 1
        layer_lengths[layer]['length_pts'] += line.length_pts
    
    # Convert layer lengths to feet
    for layer in layer_lengths:
        pts = layer_lengths[layer]['length_pts']
        layer_lengths[layer]['length_feet'] = (pts / 72.0) * scale_factor / 12.0
    
    return {
        'success': True,
        'line_count': len(selected_lines),
        'total_length_pts': total_pts,
        'total_length_feet': total_feet,
        'scale_factor': scale_factor,
        'lines': selected_lines,
        'layers': layer_lengths
    }


def measure_at_click_point(
    page: fitz.Page,
    click_point: Tuple[float, float],
    scale_factor: float = 1.0,
    search_radius: float = 30.0,
    trace_connected: bool = True
) -> Dict:
    """
    Measure lines near a click point, optionally tracing connected lines.
    
    Args:
        page: PyMuPDF page object
        click_point: (x, y) in PDF coordinates
        scale_factor: Drawing scale factor
        search_radius: Radius to search for lines near click
        trace_connected: If True, trace all connected lines from nearest line
    
    Returns:
        Dictionary with measurement results
    """
    all_lines = extract_vector_lines(page, apply_rotation=True)
    
    # Find closest line to click point
    result = find_closest_line_to_point(all_lines, click_point)
    
    if not result or result[1] > search_radius:
        return {
            'success': False,
            'error': 'No line found near click point',
            'line_count': 0,
            'total_length_feet': 0
        }
    
    closest_line, distance = result
    
    if trace_connected:
        # Trace all connected lines
        connected = trace_connected_lines_from_start(closest_line, all_lines, tolerance=5.0)
    else:
        connected = [closest_line]
    
    # Calculate measurements
    total_pts = sum(l.length_pts for l in connected)
    total_feet = (total_pts / 72.0) * scale_factor / 12.0
    
    return {
        'success': True,
        'line_count': len(connected),
        'total_length_pts': total_pts,
        'total_length_feet': total_feet,
        'scale_factor': scale_factor,
        'click_distance': distance,
        'lines': connected
    }


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
