"""
OCR Input/Output Size Profiler
Analyzes what Document AI receives and returns for each page
"""
import fitz
import os
import sys
from io import BytesIO

# Setup path
sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

# Mock streamlit for local execution
class MockSecrets:
    def __init__(self):
        try:
            import toml
            secrets_path = '/Users/parhamhamouni/Desktop/leo/leo_streamlit/.streamlit/secrets.toml'
            if os.path.exists(secrets_path):
                self.data = toml.load(secrets_path)
            else:
                self.data = {}
        except:
            self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __contains__(self, key):
        return key in self.data

class MockStreamlit:
    secrets = MockSecrets()

sys.modules['streamlit'] = MockStreamlit()
import streamlit as st

# Now import utils
from utils import extract_comprehensive_text_from_page

# Configuration
PDF_PATH = "/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf"
OUTPUT_LOG = "/Users/parhamhamouni/Desktop/leo/leo_streamlit/ocr_size_profile.txt"
PAGES_TO_ANALYZE = 30  # Focus on first 30 pages including problem area

# Load Google Cloud config
google_cloud_config = None
if "google_cloud" in st.secrets and "gcp_service_account" in st.secrets:
    google_cloud_config = {
        "project_number": st.secrets["google_cloud"]["project_number"],
        "location": st.secrets["google_cloud"]["location"],
        "processor_id": st.secrets["google_cloud"]["processor_id"],
        "service_account_info": dict(st.secrets["gcp_service_account"])
    }
    print("✓ Google Cloud config loaded")
else:
    print("ERROR: No Google Cloud config found!")
    sys.exit(1)

def analyze_ocr_sizes(page_bytes, dpi_label):
    """Analyze OCR input and output sizes"""
    size_kb = len(page_bytes) / 1024
    
    # Extract just to see response size
    result = extract_comprehensive_text_from_page(
        page_bytes,
        page_number=1,
        google_cloud_config=google_cloud_config,
        ocr_dpi=96
    )
    
    text = result.get("text", "")
    stats = result.get("stats", {})
    
    # Estimate response size (text + stats)
    response_size_kb = (len(text) + len(str(stats))) / 1024
    
    return {
        "input_size_kb": size_kb,
        "response_text_kb": len(text) / 1024,
        "response_total_kb": response_size_kb,
        "ocr_elements": stats.get("total_ocr_found", 0),
        "text_length": len(text)
    }

# Open output file
log_file = open(OUTPUT_LOG, 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("="*100)
log("OCR INPUT/OUTPUT SIZE PROFILER")
log("="*100)
log(f"\nPDF: {PDF_PATH}")
log(f"Analyzing first {PAGES_TO_ANALYZE} pages")
log(f"Output: {OUTPUT_LOG}\n")
log("="*100)

# Open PDF
doc = fitz.open(PDF_PATH)
total_pages = len(doc)
log(f"\nTotal pages in PDF: {total_pages}\n")

# Track totals
total_input_size = 0
total_response_size = 0

for page_idx in range(min(PAGES_TO_ANALYZE, total_pages)):
    page_num = page_idx + 1
    log(f"\n{'='*100}")
    log(f"PAGE {page_num}")
    log(f"{'='*100}")
    
    page = doc[page_idx]
    page_width = page.rect.width
    page_height = page.rect.height
    is_large_page = page_width > 2000 or page_height > 2000
    
    log(f"\nPage dimensions: {page_width:.0f} x {page_height:.0f}")
    log(f"Is large page: {is_large_page}")
    log(f"Page area: {(page_width * page_height):.0f} sq pts")
    
    # Test different DPIs
    dpis_to_test = []
    if is_large_page:
        dpis_to_test = [20, 25, 30]
        log("\nTesting DPIs: 20, 25, 30 (large page)")
    else:
        dpis_to_test = [45]
        log("\nTesting DPI: 45 (small page)")
    
    for dpi in dpis_to_test:
        log(f"\n  --- DPI={dpi} ---")
        
        try:
            # Generate image at this DPI
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            pix_width, pix_height = pix.width, pix.height
            log(f"  Image size: {pix_width}x{pix_height} pixels")
            
            # Convert to PNG
            img_bytes = pix.tobytes("png")
            img_size_mb = len(img_bytes) / (1024*1024)
            log(f"  PNG size: {img_size_mb:.2f} MB")
            del pix
            
            # Wrap in PDF
            temp_doc = fitz.open()
            temp_page = temp_doc.new_page(width=pix_width, height=pix_height)
            temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
            page_bytes = temp_doc.tobytes(deflate=True, garbage=4)
            temp_doc.close()
            del img_bytes, temp_page
            
            input_size_kb = len(page_bytes) / 1024
            log(f"  PDF wrapper size: {input_size_kb:.1f} KB")
            
            # Analyze OCR
            log(f"  Sending to Document AI...")
            sizes = analyze_ocr_sizes(page_bytes, f"DPI={dpi}")
            
            log(f"  ✓ Input to DocAI: {sizes['input_size_kb']:.1f} KB")
            log(f"  ✓ Response text: {sizes['response_text_kb']:.1f} KB ({sizes['text_length']} chars)")
            log(f"  ✓ OCR elements: {sizes['ocr_elements']}")
            log(f"  ✓ Estimated total response: {sizes['response_total_kb']:.1f} KB")
            
            # Track for totals
            if dpi == (30 if is_large_page else 45):  # Default DPI
                total_input_size += sizes['input_size_kb']
                total_response_size += sizes['response_total_kb']
            
            del page_bytes
            
        except Exception as e:
            log(f"  ERROR: {e}")
    
    log(f"\n{'='*100}")

doc.close()

# Summary
log(f"\n\n{'='*100}")
log("SUMMARY")
log(f"{'='*100}")
log(f"\nTotal pages analyzed: {min(PAGES_TO_ANALYZE, total_pages)}")
log(f"Total INPUT size (at default DPI): {total_input_size:.1f} KB ({total_input_size/1024:.1f} MB)")
log(f"Total RESPONSE size (estimated): {total_response_size:.1f} KB ({total_response_size/1024:.1f} MB)")
log(f"\nAverage INPUT per page: {total_input_size/min(PAGES_TO_ANALYZE, total_pages):.1f} KB")
log(f"Average RESPONSE per page: {total_response_size/min(PAGES_TO_ANALYZE, total_pages):.1f} KB")
log(f"\n{'='*100}")

log_file.close()
print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_LOG}")

