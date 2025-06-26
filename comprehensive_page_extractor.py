import fitz  # PyMuPDF
import pandas as pd
from google.cloud import documentai_v1 as documentai
from typing import List, Dict, Tuple, Optional
import numpy as np

def extract_comprehensive_page_text(
    page: fitz.Page, 
    client: Optional[documentai.DocumentProcessorServiceClient] = None,
    processor_name: Optional[str] = None,
    use_ocr: bool = True,
    dpi: int = 150,
    overlap_threshold: float = 0.3,
    text_similarity_threshold: float = 0.8,
    exclude_annotation_types: Optional[List[str]] = None
) -> Dict:
    """
    Comprehensive text extraction from a single PDF page.
    
    Extracts text from:
    1. PDF text layer (native text)
    2. Text annotations (excluding visual-only annotations)
    3. OCR (with deduplication against existing text)
    
    Args:
        page: PyMuPDF page object
        client: Google Cloud Document AI client (optional)
        processor_name: Document AI processor name (required if using OCR)
        use_ocr: Whether to run OCR analysis
        dpi: DPI for OCR image conversion
        overlap_threshold: Spatial overlap threshold for deduplication
        text_similarity_threshold: Text similarity threshold for deduplication
        exclude_annotation_types: Annotation types to exclude
    
    Returns:
        Dictionary containing:
        - text_words: List of text layer words
        - text_annotations: List of text annotations
        - ocr_texts: List of unique OCR results
        - all_text: Combined list of all text elements
        - stats: Statistics about extraction results
    """
    
    # ============================================================================
    # HELPER FUNCTIONS (embedded for self-containment)
    # ============================================================================
    
    def _get_all_text_words(page: fitz.Page) -> List[Dict]:
        """Extract all text words from PDF text layer."""
        words_data = []
        words = page.get_text("words")
        
        for word_tuple in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word_tuple
            words_data.append({
                'text': text,
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'bbox': (x0, y0, x1, y1),
                'block_no': block_no,
                'line_no': line_no,
                'word_no': word_no,
                'source': 'text_layer'
            })
        return words_data
    
    def _get_text_annotations(page: fitz.Page, exclude_types: List[str]) -> List[Dict]:
        """Extract text from annotations, excluding visual-only types."""
        annotations_data = []
        
        for annot in page.annots():
            annot_type = annot.type[1]
            
            if annot_type in exclude_types:
                continue
                
            content = annot.info.get("content", "").strip()
            if content:
                rect = annot.rect
                annotations_data.append({
                    'text': content,
                    'x0': rect.x0, 'y0': rect.y0, 'x1': rect.x1, 'y1': rect.y1,
                    'bbox': (rect.x0, rect.y0, rect.x1, rect.y1),
                    'annot_type': annot_type,
                    'source': 'annotation'
                })
        return annotations_data
    
    def _get_ocr_text(page: fitz.Page, client, processor_name: str, dpi: int) -> List[Dict]:
        """Extract text using OCR."""
        if not client or not processor_name:
            return []
        
        try:
            # Use 72 DPI (minimal) - this worked in our debug test!
            working_dpi = 72
            
            print(f"🔍 Using working DPI: {working_dpi} (based on debug results)")
            
            # Create image with 72 DPI (proven to work)
            pix = page.get_pixmap(dpi=working_dpi)
            img_data = pix.tobytes("png")
            
            print(f"📷 Image size: {len(img_data)} bytes, dimensions: {pix.width}x{pix.height}")
            
            # Send to Document AI
            raw_document = documentai.RawDocument(
                content=img_data,
                mime_type="image/png",
            )
            
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document,
            )
            
            result = client.process_document(request=request)
            doc_result = result.document
            
            print(f"📄 Document AI response: {len(doc_result.text)} chars, {len(doc_result.pages)} pages")
            
            ocr_data = []
            if doc_result.pages:
                pg = doc_result.pages[0]
                # Scale coordinates back to PDF space
                scale_x = page.rect.width / pix.width
                scale_y = page.rect.height / pix.height
                
                # Extract tokens using text_segments (preferred method)
                for token in pg.tokens:
                    # Get text content from text_segments
                    text_content = ""
                    if hasattr(token.layout, 'text_anchor') and token.layout.text_anchor:
                        anchor = token.layout.text_anchor
                        if hasattr(anchor, 'text_segments') and anchor.text_segments:
                            # Use text_segments to extract text from full document
                            for segment in anchor.text_segments:
                                if hasattr(segment, 'start_index') and hasattr(segment, 'end_index'):
                                    start_idx = segment.start_index
                                    end_idx = segment.end_index
                                    if start_idx < len(doc_result.text) and end_idx <= len(doc_result.text):
                                        text_content = doc_result.text[start_idx:end_idx]
                                        break
                        elif hasattr(anchor, 'content') and anchor.content:
                            # Fallback to content if available
                            text_content = anchor.content
                    
                    # Clean and validate text
                    text_content = text_content.strip() if text_content else ""
                    
                    if text_content and len(text_content) > 0:  # Only include non-empty text
                        v = token.layout.bounding_poly.vertices
                        if len(v) >= 4:
                            x0, y0 = v[0].x * scale_x, v[0].y * scale_y
                            x1, y1 = v[2].x * scale_x, v[2].y * scale_y
                            
                            ocr_data.append({
                                'text': text_content,
                                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                'bbox': (x0, y0, x1, y1),
                                'confidence': getattr(token.layout, 'confidence', 1.0),
                                'source': 'ocr'
                            })
            
            print(f"📝 Document AI found {len(ocr_data)} valid text elements via OCR")
            return ocr_data
            
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            
            # Fallback: Try with even lower DPI
            try:
                print("🔄 Trying fallback: lower DPI OCR...")
                fallback_dpi = 72  # Minimal DPI
                pix = page.get_pixmap(dpi=fallback_dpi)
                img_data = pix.tobytes("png")
                
                print(f"📷 Fallback image: {len(img_data)} bytes, {pix.width}x{pix.height}")
                
                raw_document = documentai.RawDocument(
                    content=img_data,
                    mime_type="image/png",
                )
                
                request = documentai.ProcessRequest(
                    name=processor_name,
                    raw_document=raw_document,
                )
                
                result = client.process_document(request=request)
                doc_result = result.document
                
                ocr_data = []
                if doc_result.pages:
                    pg = doc_result.pages[0]
                    scale_x = page.rect.width / pix.width
                    scale_y = page.rect.height / pix.height
                    
                    for token in pg.tokens:
                        if token.layout.text_anchor.content.strip():
                            v = token.layout.bounding_poly.vertices
                            if len(v) >= 4:
                                x0, y0 = v[0].x * scale_x, v[0].y * scale_y
                                x1, y1 = v[2].x * scale_x, v[2].y * scale_y
                                
                                ocr_data.append({
                                    'text': token.layout.text_anchor.content.strip(),
                                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                    'bbox': (x0, y0, x1, y1),
                                    'confidence': getattr(token.layout, 'confidence', 1.0),
                                    'source': 'ocr'
                                })
                
                print(f"📝 Fallback OCR found {len(ocr_data)} text elements")
                return ocr_data
                
            except Exception as e2:
                print(f"❌ Fallback OCR also failed: {e2}")
                return []
    
    def _calculate_overlap(bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Return overlap ratio relative to smaller box
        smaller_area = min(area1, area2)
        if smaller_area == 0:
            return 0.0
        
        return intersection_area / smaller_area
    
    def _deduplicate_text(
        existing_texts: List[Dict], 
        new_texts: List[Dict], 
        overlap_thresh: float, 
        similarity_thresh: float
    ) -> List[Dict]:
        """Remove duplicates from new_texts that overlap with existing_texts."""
        unique_texts = []
        
        for new_item in new_texts:
            is_duplicate = False
            new_text = new_item['text'].lower().strip()
            new_bbox = new_item['bbox']
            
            for existing_item in existing_texts:
                existing_text = existing_item['text'].lower().strip()
                existing_bbox = existing_item['bbox']
                
                # Check exact text match
                if new_text == existing_text:
                    is_duplicate = True
                    break
                
                # Check spatial overlap for similar texts
                overlap_ratio = _calculate_overlap(new_bbox, existing_bbox)
                if overlap_ratio > overlap_thresh:
                    # Calculate text similarity
                    common_chars = set(new_text) & set(existing_text)
                    similarity = len(common_chars) / max(len(set(new_text)), len(set(existing_text)), 1)
                    
                    if similarity > similarity_thresh:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_texts.append(new_item)
        
        return unique_texts
    
    # ============================================================================
    # MAIN EXTRACTION LOGIC
    # ============================================================================
    
    # Set default exclusions for annotations
    if exclude_annotation_types is None:
        exclude_annotation_types = ['Highlight', 'Ink', 'Line', 'Square', 'Circle', 'Polygon']
    
    # Extract text from different sources
    text_words = _get_all_text_words(page)
    text_annotations = _get_text_annotations(page, exclude_annotation_types)
    
    # Combine existing text sources
    existing_texts = text_words + text_annotations
    
    # Extract OCR text with deduplication
    ocr_texts = []
    total_ocr_found = 0
    
    if use_ocr and client and processor_name:
        ocr_results = _get_ocr_text(page, client, processor_name, dpi)
        total_ocr_found = len(ocr_results)
        
        # Deduplicate OCR results against existing text
        ocr_texts = _deduplicate_text(
            existing_texts, 
            ocr_results, 
            overlap_threshold, 
            text_similarity_threshold
        )
    
    # Combine all text sources
    all_text = existing_texts + ocr_texts
    
    # Calculate statistics
    stats = {
        'total_words': len(text_words),
        'total_annotations': len(text_annotations),
        'total_ocr_found': total_ocr_found,
        'total_ocr_unique': len(ocr_texts),
        'total_ocr_duplicates_removed': total_ocr_found - len(ocr_texts),
        'total_elements': len(all_text),
        'ocr_enabled': use_ocr and client is not None,
        'page_dimensions': (page.rect.width, page.rect.height),
        'page_rotation': page.rotation
    }
    
    return {
        'text_words': text_words,
        'text_annotations': text_annotations,
        'ocr_texts': ocr_texts,
        'all_text': all_text,
        'stats': stats
    }


def create_page_text_dataframe(page_result: Dict, page_number: Optional[int] = None) -> pd.DataFrame:
    """
    Convert page text extraction results to a pandas DataFrame.
    
    Args:
        page_result: Result from extract_comprehensive_page_text()
        page_number: Optional page number to add to DataFrame
    
    Returns:
        pandas DataFrame with all text elements
    """
    all_text_data = page_result['all_text'].copy()
    
    # Add page number if provided
    if page_number is not None:
        for item in all_text_data:
            item['page_number'] = page_number
    
    if not all_text_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_text_data)
    
    # Add calculated columns
    if 'x0' in df.columns:
        df['width'] = df['x1'] - df['x0']
        df['height'] = df['y1'] - df['y0']
        df['center_x'] = (df['x0'] + df['x1']) / 2
        df['center_y'] = (df['y0'] + df['y1']) / 2
        df['area'] = df['width'] * df['height']
    
    return df


def print_page_extraction_summary(page_result: Dict, page_number: Optional[int] = None):
    """
    Print a summary of page text extraction results.
    
    Args:
        page_result: Result from extract_comprehensive_page_text()
        page_number: Optional page number for display
    """
    stats = page_result['stats']
    page_info = f"Page {page_number}" if page_number else "Page"
    
    print(f"📄 {page_info} Text Extraction Summary")
    print("=" * 50)
    print(f"Text Layer Words: {stats['total_words']}")
    print(f"Text Annotations: {stats['total_annotations']}")
    
    if stats['ocr_enabled']:
        print(f"OCR Texts Found: {stats['total_ocr_found']}")
        print(f"OCR Unique Texts: {stats['total_ocr_unique']}")
        print(f"OCR Duplicates Removed: {stats['total_ocr_duplicates_removed']}")
    else:
        print("OCR: Disabled")
    
    print(f"Total Text Elements: {stats['total_elements']}")
    
    if stats['total_elements'] > 0 and stats['ocr_enabled']:
        ocr_contribution = (stats['total_ocr_unique'] / stats['total_elements']) * 100
        print(f"OCR Contribution: {ocr_contribution:.1f}%")
    
    print(f"Page Dimensions: {stats['page_dimensions'][0]:.0f} x {stats['page_dimensions'][1]:.0f}")
    print(f"Page Rotation: {stats['page_rotation']}°")


# Example usage function
def example_usage():
    """
    Example of how to use the comprehensive page text extraction.
    """
    # Setup (you'll need to provide your own values)
    PDF_FILE = "your_pdf_file.pdf"
    SA_KEY = "your_service_account_key.json"
    PROCESSOR_NAME = "projects/YOUR_PROJECT/locations/us/processors/YOUR_PROCESSOR_ID"
    
    # Initialize Document AI client (optional, for OCR)
    try:
        client = documentai.DocumentProcessorServiceClient.from_service_account_file(SA_KEY)
        print("✅ Document AI client initialized")
    except Exception as e:
        print(f"⚠️ Document AI not available: {e}")
        client = None
    
    # Open PDF and process a single page
    doc = fitz.open(PDF_FILE)
    page = doc[0]  # First page
    
    # Extract comprehensive text from the page
    result = extract_comprehensive_page_text(
        page=page,
        client=client,
        processor_name=PROCESSOR_NAME,
        use_ocr=True,  # Set to False to skip OCR
        dpi=150,
        overlap_threshold=0.3,
        text_similarity_threshold=0.8
    )
    
    # Print summary
    print_page_extraction_summary(result, page_number=1)
    
    # Create DataFrame
    df = create_page_text_dataframe(result, page_number=1)
    print(f"\n📊 DataFrame shape: {df.shape}")
    print(f"Text sources: {df['source'].value_counts().to_dict()}")
    
    # Access specific text types
    text_layer_words = result['text_words']
    annotations = result['text_annotations'] 
    unique_ocr_texts = result['ocr_texts']
    all_texts = result['all_text']
    
    print(f"\n📝 First 5 text elements:")
    for i, text_item in enumerate(all_texts[:5]):
        print(f"{i+1}. [{text_item['source']}] {text_item['text'][:50]}...")
    
    doc.close()
    return result, df


if __name__ == "__main__":
    example_usage() 