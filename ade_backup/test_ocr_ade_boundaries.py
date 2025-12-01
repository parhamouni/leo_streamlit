#!/usr/bin/env python3
"""
Test script to verify OCR-ADE boundary alignment and evaluate detection capabilities.
Checks if OCR elements are correctly identified as inside/outside ADE layout regions.
"""

import fitz
import json
from pathlib import Path
from utils_ade_official import (
    ade_parse_document_official,
    align_ade_chunks_to_page,
    filter_ocr_by_ade_regions,
    get_google_ocr_results_with_boxes
)
from utils import create_document_ai_client
import sys

def test_boundary_overlap(ocr_box, ade_box):
    """Test if two boxes overlap (same logic as filter_ocr_by_ade_regions)."""
    ocr_x0, ocr_y0, ocr_x1, ocr_y1 = ocr_box
    ade_x0, ade_y0, ade_x1, ade_y1 = ade_box
    
    # Check for box overlap
    overlaps = (ocr_x0 < ade_x1 and ocr_x1 > ade_x0 and
                ocr_y0 < ade_y1 and ocr_y1 > ade_y0)
    
    # Calculate overlap details
    overlap_x0 = max(ocr_x0, ade_x0)
    overlap_y0 = max(ocr_y0, ade_y0)
    overlap_x1 = min(ocr_x1, ade_x1)
    overlap_y1 = min(ocr_y1, ade_y1)
    
    if overlaps:
        overlap_area = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        ocr_area = (ocr_x1 - ocr_x0) * (ocr_y1 - ocr_y0)
        ade_area = (ade_x1 - ade_x0) * (ade_y1 - ade_y0)
        overlap_ratio_ocr = overlap_area / ocr_area if ocr_area > 0 else 0
        overlap_ratio_ade = overlap_area / ade_area if ade_area > 0 else 0
        return True, overlap_ratio_ocr, overlap_ratio_ade, overlap_area
    else:
        return False, 0.0, 0.0, 0.0

def analyze_ocr_ade_alignment(pdf_path, page_idx=0):
    """Analyze OCR-ADE boundary alignment for a single page."""
    print(f"\n{'='*80}")
    print(f"Analyzing OCR-ADE Alignment for Page {page_idx}")
    print(f"{'='*80}\n")
    
    # Load API key
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            for line in f:
                if 'LANDINGAI_API_KEY' in line:
                    api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break
    except Exception as e:
        print(f"Error loading API key: {e}")
        return
    
    # Load PDF
    doc = fitz.open(str(pdf_path))
    if page_idx >= len(doc):
        print(f"Error: Page {page_idx} not found (PDF has {len(doc)} pages)")
        return
    
    page = doc[page_idx]
    page_width = page.rect.width
    page_height = page.rect.height
    print(f"Page dimensions: {page_width:.1f} x {page_height:.1f} points")
    
    # Create single-page PDF
    temp_doc = fitz.open()
    temp_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
    single_page_bytes = temp_doc.tobytes()
    temp_doc.close()
    
    # Get ADE chunks
    print("\n1. Parsing with ADE...")
    ade_result = ade_parse_document_official(single_page_bytes, api_key, zdr=False)
    
    if not ade_result.get("success"):
        print(f"   ❌ ADE parsing failed: {ade_result.get('error')}")
        return
    
    page_chunks = align_ade_chunks_to_page(ade_result, page_idx, page_width, page_height)
    print(f"   ✅ Found {len(page_chunks)} ADE chunks")
    
    # Get Google OCR
    print("\n2. Getting Google OCR results...")
    # Load Google Cloud config (similar to evaluate_hybrid_highlighting.py)
    try:
        import toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            google_cloud_raw = {}
            if "google_cloud" in secrets:
                google_cloud_raw["project_number"] = secrets["google_cloud"].get("project_number")
                google_cloud_raw["location"] = secrets["google_cloud"].get("location")
                google_cloud_raw["processor_id"] = secrets["google_cloud"].get("processor_id")
            if "gcp_service_account" in secrets:
                google_cloud_raw["service_account_info"] = dict(secrets["gcp_service_account"])
            
            if google_cloud_raw.get("service_account_info"):
                client = create_document_ai_client(google_cloud_raw)
                processor_name = f"projects/{google_cloud_raw.get('project_number')}/locations/{google_cloud_raw.get('location')}/processors/{google_cloud_raw.get('processor_id')}"
                google_cloud_config = {
                    "client": client,
                    "processor_name": processor_name
                }
            else:
                print("   ❌ Google Cloud config incomplete")
                return
        else:
            print("   ❌ secrets.toml not found")
            return
    except Exception as e:
        print(f"   ❌ Error loading Google Cloud config: {e}")
        return
    
    google_ocr_results = get_google_ocr_results_with_boxes(
        single_page_bytes,
        google_cloud_config
    )
    print(f"   ✅ Found {len(google_ocr_results)} OCR items")
    
    # Analyze each ADE chunk
    print("\n3. Analyzing OCR-ADE boundary alignment...\n")
    
    total_ocr_in_regions = 0
    total_ocr_outside = 0
    boundary_issues = []
    
    for chunk_idx, chunk in enumerate(page_chunks):
        chunk_type = chunk.get("type", "unknown")
        chunk_id = chunk.get("id", f"chunk_{chunk_idx}")
        chunk_box = (chunk.get("x0", 0), chunk.get("y0", 0), 
                     chunk.get("x1", 0), chunk.get("y1", 0))
        
        print(f"   Chunk {chunk_idx+1}: {chunk_type} (ID: {chunk_id[:20]}...)")
        print(f"      ADE Box: ({chunk_box[0]:.1f}, {chunk_box[1]:.1f}) - ({chunk_box[2]:.1f}, {chunk_box[3]:.1f})")
        
        # Filter OCR for this region
        filtered_ocr = filter_ocr_by_ade_regions(google_ocr_results, [chunk])
        
        print(f"      OCR items in region: {len(filtered_ocr)}")
        
        # Detailed analysis
        ocr_in_region = []
        ocr_outside = []
        low_overlap_items = []
        
        for ocr_item in google_ocr_results:
            ocr_box = (ocr_item.get("x0", 0), ocr_item.get("y0", 0),
                      ocr_item.get("x1", 0), ocr_item.get("y1", 0))
            ocr_text = ocr_item.get("text", "").strip()
            
            overlaps, overlap_ratio_ocr, overlap_ratio_ade, overlap_area = test_boundary_overlap(ocr_box, chunk_box)
            
            if overlaps:
                ocr_in_region.append((ocr_item, overlap_ratio_ocr, overlap_ratio_ade))
                total_ocr_in_regions += 1
                
                # Check for low overlap (potential boundary issue)
                if overlap_ratio_ocr < 0.5:  # Less than 50% of OCR box overlaps
                    low_overlap_items.append((ocr_text, overlap_ratio_ocr, overlap_ratio_ade))
            else:
                ocr_outside.append((ocr_item, ocr_text))
                total_ocr_outside += 1
        
        # Show sample OCR items
        if ocr_in_region:
            print(f"      Sample OCR in region (top 5):")
            for ocr_item, ratio_ocr, ratio_ade in ocr_in_region[:5]:
                text = ocr_item.get("text", "")[:30]
                print(f"         '{text}' - OCR overlap: {ratio_ocr:.1%}, ADE overlap: {ratio_ade:.1%}")
        
        if low_overlap_items:
            print(f"      ⚠️  Low overlap items ({len(low_overlap_items)}):")
            for text, ratio_ocr, ratio_ade in low_overlap_items[:3]:
                print(f"         '{text}' - OCR: {ratio_ocr:.1%}, ADE: {ratio_ade:.1%}")
            boundary_issues.extend(low_overlap_items)
        
        print()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total ADE chunks: {len(page_chunks)}")
    print(f"Total OCR items: {len(google_ocr_results)}")
    print(f"OCR items in ADE regions: {total_ocr_in_regions}")
    print(f"OCR items outside ADE regions: {total_ocr_outside}")
    print(f"Coverage ratio: {total_ocr_in_regions / len(google_ocr_results):.1%}" if google_ocr_results else "N/A")
    print(f"Boundary issues (low overlap): {len(boundary_issues)}")
    
    # Check if OCR can detect elements that should be detected
    print(f"\n{'='*80}")
    print("DETECTION ANALYSIS")
    print(f"{'='*80}\n")
    
    # Find legend/keynote chunks
    legend_chunks = []
    figure_chunks = []
    
    for chunk in page_chunks:
        chunk_type = chunk.get("type", "").lower()
        chunk_text = (chunk.get("markdown") or chunk.get("text", "")).lower()
        
        if chunk_type == "table" or any(kw in chunk_text for kw in ["keynote", "legend", "note", "symbol"]):
            legend_chunks.append(chunk)
        elif chunk_type in ["figure", "architectural_drawing"]:
            figure_chunks.append(chunk)
    
    print(f"Legend/Keynote chunks: {len(legend_chunks)}")
    print(f"Figure chunks: {len(figure_chunks)}")
    
    # Check OCR coverage for legends
    if legend_chunks:
        legend_ocr = filter_ocr_by_ade_regions(google_ocr_results, legend_chunks)
        print(f"OCR items in legend regions: {len(legend_ocr)}")
        
        # Sample OCR text from legends
        if legend_ocr:
            sample_texts = [item.get("text", "")[:50] for item in legend_ocr[:10]]
            print(f"Sample OCR text from legends:")
            for i, text in enumerate(sample_texts, 1):
                print(f"   {i}. '{text}'")
    
    # Check OCR coverage for figures
    if figure_chunks:
        figure_ocr = filter_ocr_by_ade_regions(google_ocr_results, figure_chunks)
        print(f"\nOCR items in figure regions: {len(figure_ocr)}")
        
        # Sample OCR text from figures
        if figure_ocr:
            sample_texts = [item.get("text", "")[:50] for item in figure_ocr[:10]]
            print(f"Sample OCR text from figures:")
            for i, text in enumerate(sample_texts, 1):
                print(f"   {i}. '{text}'")
    
    doc.close()
    return {
        "ade_chunks": len(page_chunks),
        "ocr_items": len(google_ocr_results),
        "ocr_in_regions": total_ocr_in_regions,
        "ocr_outside": total_ocr_outside,
        "legend_chunks": len(legend_chunks),
        "figure_chunks": len(figure_chunks),
        "boundary_issues": len(boundary_issues)
    }

if __name__ == "__main__":
    # Test with the evaluation PDF
    pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Analyze first page
    result = analyze_ocr_ade_alignment(pdf_path, page_idx=0)
    
    if result:
        print(f"\n✅ Analysis complete!")
        print(f"   Check boundary_issues count to see if overlap logic needs improvement")

