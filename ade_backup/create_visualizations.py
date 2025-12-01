#!/usr/bin/env python3
"""
Create visualizations for extracted data analysis.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_coverage_plot(analyses: list, output_dir: Path):
    """Create coverage heatmap showing ADE/OCR/text layer coverage per page."""
    pages = [a['page_num'] for a in analyses]
    ocr_coverage = [a['matches']['ocr']['found'] / a['gold_count'] * 100 if a['gold_count'] > 0 else 0 for a in analyses]
    text_coverage = [a['matches']['text_layer']['found'] / a['gold_count'] * 100 if a['gold_count'] > 0 else 0 for a in analyses]
    
    x = np.arange(len(pages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ocr_coverage, width, label='OCR', color='#1f77b4')
    bars2 = ax.bar(x + width/2, text_coverage, width, label='Text Layer', color='#ff7f0e')
    
    ax.set_xlabel('Page Number')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Annotation Coverage by Source')
    ax.set_xticks(x)
    ax.set_xticklabels(pages)
    ax.legend()
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_by_source.png', dpi=150)
    print(f"📊 Saved coverage plot to {output_dir / 'coverage_by_source.png'}")
    plt.close()

def create_content_distribution(analyses: list, output_dir: Path):
    """Create plot showing distribution of indicator codes vs descriptions."""
    all_codes = []
    all_descriptions = []
    
    for analysis in analyses:
        all_codes.extend([c['content'] for c in analysis['content_analysis']['indicator_codes']])
        all_descriptions.extend([d['content'] for d in analysis['content_analysis']['descriptions']])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Indicator codes length distribution
    code_lengths = [len(c) for c in all_codes if c]
    if code_lengths:
        ax1.hist(code_lengths, bins=range(1, max(code_lengths)+2), color='#2ca02c', edgecolor='black')
        ax1.set_xlabel('Code Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Indicator Code Length Distribution')
        ax1.set_xticks(range(1, max(code_lengths)+1))
    
    # Description lengths
    desc_lengths = [len(d) for d in all_descriptions if d]
    if desc_lengths:
        ax2.hist(desc_lengths, bins=20, color='#d62728', edgecolor='black')
        ax2.set_xlabel('Description Length (characters)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Description Length Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'content_distribution.png', dpi=150)
    print(f"📊 Saved content distribution plot to {output_dir / 'content_distribution.png'}")
    plt.close()

def create_iou_comparison(analyses: list, output_dir: Path):
    """Create plot comparing IoU scores between OCR and text layer."""
    ocr_ious = []
    text_ious = []
    
    for analysis in analyses:
        ocr_details = analysis['matches']['ocr']['details']
        text_details = analysis['matches']['text_layer']['details']
        
        ocr_ious.extend([d['iou'] for d in ocr_details if d.get('iou', 0) > 0])
        text_ious.extend([d['iou'] for d in text_details if d.get('iou', 0) > 0])
    
    if not ocr_ious and not text_ious:
        print("⚠️ No IoU data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    if ocr_ious:
        data_to_plot.append(ocr_ious)
        labels.append('OCR')
    if text_ious:
        data_to_plot.append(text_ious)
        labels.append('Text Layer')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('#ff7f0e')
        
        ax.set_ylabel('IoU Score')
        ax.set_title('Bounding Box Alignment Quality (IoU)')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_comparison.png', dpi=150)
    print(f"📊 Saved IoU comparison plot to {output_dir / 'iou_comparison.png'}")
    plt.close()

def main():
    """Main visualization function."""
    data_dir = Path("data_analysis")
    viz_dir = data_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    analysis_path = data_dir / "detailed_analysis.json"
    
    if not analysis_path.exists():
        print(f"❌ Analysis file not found: {analysis_path}")
        print("   Run analyze_extracted_data.py first")
        sys.exit(1)
    
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    with open(analysis_path) as f:
        analyses = json.load(f)
    
    print(f"📊 Creating visualizations from {len(analyses)} page analyses...")
    print()
    
    create_coverage_plot(analyses, viz_dir)
    create_content_distribution(analyses, viz_dir)
    create_iou_comparison(analyses, viz_dir)
    
    print()
    print("=" * 60)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"📁 Visualizations saved to: {viz_dir.absolute()}")

if __name__ == "__main__":
    main()























