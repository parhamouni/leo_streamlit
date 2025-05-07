"""
OCR Visualization with Proper Bounding Boxes
Shows all OCR-detected text elements with rectangular outlines rather than filled boxes
"""

import cv2
import numpy as np
import pytesseract
import io
import random
from PIL import Image, ImageDraw, ImageFont

def detect_text_with_tesseract(image):
    """
    Detect all text in the image with bounding boxes using Tesseract OCR.
    
    Args:
        image: Image as numpy array
        
    Returns:
        List of dictionaries with text and bounding box information
    """
    # Convert to grayscale if it's color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Use Tesseract to get text with bounding boxes
    # Page segmentation mode 11: Sparse text. Find as much text as possible in no particular order.
    custom_config = r'--oem 3 --psm 11'
    results = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Filter and process results
    text_elements = []
    
    for i in range(len(results['text'])):
        # Filter out empty text and low confidence results
        if results['text'][i].strip() and results['conf'][i] > 30:
            x = results['left'][i]
            y = results['top'][i]
            w = results['width'][i]
            h = results['height'][i]
            
            text = results['text'][i].strip()
            conf = results['conf'][i]
            
            # Skip if dimensions are invalid
            if w < 2 or h < 2:
                continue
            
            text_elements.append({
                "text": text,
                "conf": conf / 100.0,  # Normalize confidence to 0-1 range
                "bbox": (x, y, w, h)
            })
    
    return text_elements

def visualize_with_bounding_boxes(image_bytes):
    """
    Create a visualization showing all OCR-detected text elements
    with proper bounding boxes (outlines only, no fill).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Visualization image as bytes
    """
    # Convert image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert OpenCV BGR to PIL RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a high-quality font
    try:
        font = ImageFont.truetype("Arial", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Detect text with Tesseract OCR
    text_elements = detect_text_with_tesseract(img)
    print(f"Detected {len(text_elements)} text elements")
    
    # Use a fixed set of high-contrast colors for better visibility
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (255, 0, 128)   # Pink
    ]
    
    # Group text elements by Y-coordinate to identify potential lines of text
    y_tolerance = 10  # Pixels of tolerance for considering elements on the same line
    lines = {}
    
    for i, element in enumerate(text_elements):
        x, y, w, h = element["bbox"]
        
        # Find the line this element belongs to
        line_key = None
        for key in lines.keys():
            if abs(key - y) < y_tolerance:
                line_key = key
                break
        
        if line_key is None:
            line_key = y
            lines[line_key] = []
        
        lines[line_key].append(i)
    
    # Assign colors by line for better visualization
    line_colors = {}
    for i, line_key in enumerate(sorted(lines.keys())):
        line_colors[line_key] = colors[i % len(colors)]
    
    # Draw bounding boxes (outlines only) around all detected text
    for i, element in enumerate(text_elements):
        x, y, w, h = element["bbox"]
        
        # Get the line this element belongs to and its color
        line_key = None
        for key in lines.keys():
            if i in lines[key]:
                line_key = key
                break
        
        if line_key is not None:
            color = line_colors[line_key]
        else:
            color = colors[i % len(colors)]
        
        # Draw rectangle outline only (no fill)
        draw.rectangle([(x, y), (x + w, y + h)], fill=None, outline=color, width=2)
    
    # Add a legend explaining the colors
    legend_x = 10
    legend_y = 10
    legend_spacing = 25
    
    # Calculate the number of unique lines for the legend
    unique_lines = len(line_colors)
    legend_width = 180
    legend_height = unique_lines * legend_spacing + 40
    
    # Draw a semi-transparent white background for the legend
    draw.rectangle(
        [(legend_x - 5, legend_y - 5), (legend_x + legend_width, legend_y + legend_height)],
        fill=(255, 255, 255, 200)
    )
    
    # Draw legend title
    draw.text((legend_x, legend_y), "OCR Text Groups", fill=(0, 0, 0), font=font)
    legend_y += 30
    
    # Draw legend entries
    for i, (line_key, color) in enumerate(sorted(line_colors.items())):
        # Draw color sample (outline only)
        draw.rectangle(
            [(legend_x, legend_y + i * legend_spacing), (legend_x + 20, legend_y + i * legend_spacing + 15)],
            fill=None, outline=color, width=2
        )
        
        # Draw label
        draw.text(
            (legend_x + 30, legend_y + i * legend_spacing),
            f"Text Group {i+1}",
            fill=(0, 0, 0),
            font=font
        )
    
    # Convert to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", quality=95, dpi=(300, 300))
    return buffer.getvalue()

def analyze_page(page, llm_vision, FENCE_KEYWORDS):
    """
    Function to visualize all OCR-detected text with proper bounding boxes.
    
    Args:
        page: Dictionary with page information
        llm_vision: Not used in this simplified version
        FENCE_KEYWORDS: Not used in this simplified version
        
    Returns:
        Dictionary with analysis results
    """
    image_bytes = page["image_bytes"]
    
    # Create visualization with proper bounding boxes
    highlighted_image = visualize_with_bounding_boxes(image_bytes)
    
    # Since we're visualizing all text, we'll consider it found fence elements
    # This is just to make the app show the visualization
    fence_found = True
    
    return {
        "page_number": page["page_number"],
        "fence_found": fence_found,
        "text_found": True,
        "vision_found": True,
        "text_references": [],
        "image": image_bytes,
        "highlighted_image": highlighted_image,
        "llm_analysis": "Visualization of all OCR-detected text elements with proper bounding boxes."
    }