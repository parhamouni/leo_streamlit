#!/usr/bin/env python3
"""
Full PDF Keyword and Indicator Extraction Script

Processes all pages in a PDF:
1. Extracts keywords and indicators from table/text chunks
2. Finds indicators in figure regions
3. Generates visualization images per page
4. Saves images and results to disk
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from matplotlib.patches import Rectangle, Patch

from hybrid_page_extractor import extract_page_text_layers, normalize_document_ai_config
from utils_ade_official import (
    find_indicators_in_figures,
    filter_ocr_by_ade_regions,
    extract_keywords_and_indicators_llm,
    group_items_by_line,
    combine_bbox,
    make_highlight,
    match_text_in_lines,
    find_text_in_ocr_and_pdf,
    extract_keywords_and_indicators_from_chunks,
    ade_parse_document_official,
    align_ade_chunks_to_page,
)
from utils import retry_with_backoff
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Default fence keywords
FENCE_KEYWORDS = [
    "fence", "fencing", "gate", "barrier", "guardrail", "post", "mesh",
    "panel", "chain link", "masonry", "bollard", "wall", "railing",
    "enclosure", "perimeter", "screen", "security",
]

RENDER_DPI = 144


def create_visualization(
    pdf_page: fitz.Page,
    page_width: float,
    page_height: float,
    extraction_results: List[Dict],
    figure_highlights: List[Dict],
    output_path: Path,
    page_num: int,
):
    """Create and save visualization image for a page."""
    # Render page image
    render_start = time.time()
    page_pix = pdf_page.get_pixmap(dpi=RENDER_DPI)
    render_time = time.time() - render_start
    page_img = np.frombuffer(page_pix.samples, dtype=np.uint8).reshape(
        page_pix.height, page_pix.width, page_pix.n
    )
    if page_pix.n == 4:
        page_img = page_img[:, :, :3]

    # Scale factors
    scale_x = page_pix.width / page_width
    scale_y = page_pix.height / page_height
    img_height = page_pix.height

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))
    # PyMuPDF pixmap has y=0 at top (standard image coordinates)
    # So we can use imshow directly without flipping
    ax.imshow(page_img, origin='upper')
    ax.set_axis_off()
    ax.set_title(f"Page {page_num} - All Keywords and Indicators", fontsize=14)

    # Draw boxes - show ALL extraction results together
    def add_boxes(records: List[Dict], color: str, label_key: Optional[str] = None, linewidth: float = 1.4, alpha: float = 0.5):
        """Add bounding boxes to axis. Coordinates are in PDF points, converted to pixels using scale factors."""
        for rec in records:
            # Convert PDF points to rendered image pixels using scale factors
            x0 = float(rec.get("x0", 0.0)) * scale_x
            y0 = float(rec.get("y0", 0.0)) * scale_y
            x1 = float(rec.get("x1", 0.0)) * scale_x
            y1 = float(rec.get("y1", 0.0)) * scale_y
            width = max(x1 - x0, 1.0)
            height = max(y1 - y0, 1.0)

            rect = Rectangle(
                (x0, y0), width, height,
                linewidth=linewidth, edgecolor=color, facecolor=(0, 0, 0, 0), alpha=alpha
            )
            ax.add_patch(rect)

            if label_key:
                text = rec.get(label_key)
                if text:
                    ax.text(
                        x0, max(y0 - 6, 8), str(text)[:42],
                        color="black", fontsize=8,
                        bbox=dict(facecolor=color, alpha=alpha, edgecolor="none", pad=1.5),
                    )

    # Draw ALL extraction results together (green) - both keywords and indicators
    if extraction_results:
        add_boxes(extraction_results, "green", "keyword", linewidth=1.6, alpha=0.6)

    # Draw figure highlights (magenta) - indicators found in figures
    if figure_highlights:
        add_boxes(figure_highlights, "magenta", "text", linewidth=1.6, alpha=0.65)

    # Add legend
    legend_elements = []
    if extraction_results:
        legend_elements.append(Patch(facecolor='green', edgecolor='green', alpha=0.6, label='Keywords & Indicators'))
    if figure_highlights:
        legend_elements.append(Patch(facecolor='magenta', edgecolor='magenta', alpha=0.65, label='Indicators in figures'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    save_start = time.time()
    plt.tight_layout()
    plt.savefig(output_path, dpi=RENDER_DPI, bbox_inches='tight')
    plt.close()
    save_time = time.time() - save_start

    # Only print detailed timing if it's significant (avoid cluttering output)
    if render_time > 0.5 or save_time > 0.5:
        print(f"     (Render: {render_time:.2f}s, Save: {save_time:.2f}s)")


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    ade_api_key: str,
    openai_api_key: str,
    doc_ai_config: Optional[Dict] = None,
    fence_keywords: Optional[List[str]] = None,
):
    """Process all pages in PDF and save results."""
    if fence_keywords is None:
        fence_keywords = FENCE_KEYWORDS

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key,
        timeout=180,
        max_retries=2,
    )

    # Open PDF
    pdf_doc = fitz.open(str(pdf_path))
    total_pages = len(pdf_doc)

    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    images_dir = output_dir / "images"
    results_dir = output_dir / "results"
    images_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    all_page_results = []
    total_start_time = time.time()

    # Call ADE once for the entire PDF (this is the big optimization!)
    print(f"\n{'='*60}")
    print("Step 0: Parsing entire PDF with ADE (one-time call)...")
    print(f"{'='*60}")
    step_start = time.time()
    pdf_bytes = pdf_path.read_bytes()
    full_ade_result = ade_parse_document_official(pdf_bytes, ade_api_key)
    step_time = time.time() - step_start
    if full_ade_result.get("success"):
        total_ade_pages = full_ade_result.get("data", {}).get("total_pages", total_pages)
        print(f"  ✅ ADE parsed entire PDF in {step_time:.2f}s ({total_ade_pages} pages)")
    else:
        print(f"  ❌ ADE parsing failed: {full_ade_result.get('error')}")
        print(f"  ⚠️  Continuing without ADE chunks...")
        full_ade_result = None

    # Process each page
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Processing page {page_num}/{total_pages}")
        print(f"{'='*60}")

        pdf_page = pdf_doc[page_idx]
        page_width = float(pdf_page.rect.width)
        page_height = float(pdf_page.rect.height)

        # Extract page sources (OCR + PDF text per page, ADE chunks from full result)
        print("  ⏱️  Step 1: Extracting page sources (OCR + PDF text + ADE chunks)...")
        step_start = time.time()
        
        # Extract OCR and PDF text layers (per-page)
        text_layers = extract_page_text_layers(pdf_page, doc_ai_config)
        pdf_blocks = text_layers["text_blocks"]
        pdf_words = text_layers["text_words"]
        ocr_tokens = text_layers["ocr_tokens"]
        
        # Extract ADE chunks for this page from the full document result
        if full_ade_result and full_ade_result.get("success"):
            ade_chunks = align_ade_chunks_to_page(
                full_ade_result,
                page_idx,  # positional argument, not keyword
                page_width,
                page_height,
            )
        else:
            ade_chunks = []
        
        step_time = time.time() - step_start
        print(f"     ✅ Completed in {step_time:.2f}s")
        print(f"     📊 Found: {len(ade_chunks)} ADE chunks, {len(ocr_tokens)} OCR tokens, {len(pdf_words)} PDF words")

        # Extract keywords and indicators
        print("  ⏱️  Step 2: Extracting keywords and indicators from chunks...")
        step_start = time.time()
        extraction_results = extract_keywords_and_indicators_from_chunks(
            ade_chunks=ade_chunks,
            ocr_tokens=ocr_tokens,
            pdf_blocks=pdf_blocks,
            pdf_words=pdf_words,
            fence_keywords=fence_keywords,
            llm=llm,
            page_width=page_width,
            page_height=page_height,
        )
        step_time = time.time() - step_start
        print(f"     ✅ Completed in {step_time:.2f}s - Found {len(extraction_results)} results")

        # Find indicators in figure regions
        print("  ⏱️  Step 3: Finding indicators in figure regions...")
        step_start = time.time()
        figure_chunks = [
            chunk for chunk in ade_chunks
            if chunk.get("type", "").lower() in ["figure", "architectural_drawing"]
        ]

        figure_highlights = []
        if figure_chunks and extraction_results:
            indicators_for_figures = [
                {"indicator": r.get("indicator"), "description": r.get("description", "")}
                for r in extraction_results if r.get("indicator")
            ]

            if indicators_for_figures:
                figure_highlights = find_indicators_in_figures(
                    indicators=indicators_for_figures,
                    page_chunks=figure_chunks,
                    google_ocr_results=ocr_tokens,
                    page_width=page_width,
                    page_height=page_height,
                    llm=llm,
                    fence_keywords=fence_keywords,
                )
        step_time = time.time() - step_start
        print(f"     ✅ Completed in {step_time:.2f}s - Found {len(figure_highlights)} figure highlights")

        # Save results JSON
        print("  ⏱️  Step 4: Saving results JSON...")
        step_start = time.time()
        page_result = {
            "page_number": page_num,
            "extraction_results": extraction_results,
            "figure_highlights": figure_highlights,
            "summary": {
                "total_extractions": len(extraction_results),
                "with_indicators": sum(1 for r in extraction_results if r.get("indicator")),
                "figure_highlights": len(figure_highlights),
            }
        }
        all_page_results.append(page_result)

        results_file = results_dir / f"page_{page_num:03d}_results.json"
        with open(results_file, "w") as f:
            json.dump(page_result, f, indent=2)
        step_time = time.time() - step_start
        print(f"     ✅ Completed in {step_time:.2f}s")

        # Create and save visualization
        print("  ⏱️  Step 5: Creating visualization...")
        step_start = time.time()
        image_file = images_dir / f"page_{page_num:03d}_visualization.png"
        create_visualization(
            pdf_page=pdf_page,
            page_width=page_width,
            page_height=page_height,
            extraction_results=extraction_results,
            figure_highlights=figure_highlights,
            output_path=image_file,
            page_num=page_num,
        )
        step_time = time.time() - step_start
        print(f"     ✅ Completed in {step_time:.2f}s")

        page_time = time.time() - page_start_time
        print(f"\n  ⏱️  Total page processing time: {page_time:.2f}s")
        print(f"  📊 Average time per page so far: {(time.time() - total_start_time) / page_num:.2f}s")

    # Save combined results
    combined_results_file = output_dir / "all_pages_results.json"
    with open(combined_results_file, "w") as f:
        json.dump(all_page_results, f, indent=2)

    # Save summary
    summary = {
        "pdf_path": str(pdf_path),
        "total_pages": total_pages,
        "total_extractions": sum(len(r["extraction_results"]) for r in all_page_results),
        "total_figure_highlights": sum(len(r["figure_highlights"]) for r in all_page_results),
        "pages_processed": len(all_page_results),
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    pdf_doc.close()

    total_time = time.time() - total_start_time
    avg_time_per_page = total_time / total_pages if total_pages > 0 else 0

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"⏱️  Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"⏱️  Average time per page: {avg_time_per_page:.2f}s")
    print(f"📊 Results saved to: {output_dir}")
    print(f"  - Images: {images_dir}")
    print(f"  - Results: {results_dir}")
    print(f"  - Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract keywords and indicators from PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--secrets-file", type=Path, default=Path(".streamlit/secrets.toml"), help="Secrets file path")
    args = parser.parse_args()

    # Load secrets
    if not args.secrets_file.exists():
        raise FileNotFoundError(f"Secrets file not found: {args.secrets_file}")

    secrets = toml.load(args.secrets_file)
    ade_api_key = secrets.get("LANDINGAI_API_KEY")
    openai_api_key = secrets.get("OPENAI_API_KEY") or secrets.get("openai_api_key")

    if not ade_api_key:
        raise ValueError("Missing LANDINGAI_API_KEY in secrets.toml")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in secrets.toml")

    # Setup Document AI config
    doc_ai_config = None
    if "google_cloud" in secrets and "gcp_service_account" in secrets:
        raw_doc_ai_config = {
            "project_number": secrets["google_cloud"].get("project_number"),
            "location": secrets["google_cloud"].get("location"),
            "processor_id": secrets["google_cloud"].get("processor_id"),
            "service_account_info": dict(secrets["gcp_service_account"]),
        }
        doc_ai_config = normalize_document_ai_config(raw_doc_ai_config)

    # Process PDF
    process_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        ade_api_key=ade_api_key,
        openai_api_key=openai_api_key,
        doc_ai_config=doc_ai_config,
    )


if __name__ == "__main__":
    main()

