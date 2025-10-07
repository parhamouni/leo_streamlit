#!/usr/bin/env python3
"""
QUICK Memory Profiling Test - Only first 5 pages for fast results
"""
import sys
import os
import gc
import io
import json
import time
from pathlib import Path

# Add to path
sys.path.insert(0, '/Users/parhamhamouni/Desktop/leo/leo_streamlit')

# Memory tracking
try:
    import psutil
    def _rss_mb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
except:
    def _rss_mb():
        return 0.0

# Import required modules
import fitz

# Load API keys
import toml
secrets_path = '/Users/parhamhamouni/Desktop/leo/leo_streamlit/.streamlit/secrets.toml'
if os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    os.environ['OPENAI_API_KEY'] = secrets.get('OPENAI_API_KEY', '')
    print(f"✅ Loaded API key")
else:
    print(f"⚠️  No secrets file, will skip LLM calls")
    secrets = {}

from utils import analyze_page, get_fence_related_text_boxes
from langchain_openai import ChatOpenAI

# Google Cloud config
try:
    google_cloud_config = {
        'project_id': secrets['google_cloud']['project_id'],
        'location': secrets['google_cloud']['location'],
        'processor_id': secrets['google_cloud']['processor_id'],
        'service_account_info': dict(secrets['google_cloud']['service_account_info'])
    }
    print("✅ Google Cloud config loaded")
except Exception as e:
    print(f"⚠️  No Google Cloud config: {e}")
    google_cloud_config = None

# MemoryProfiler class
class MemoryProfiler:
    def __init__(self):
        self.page_data = []
        self.current_page = None
    
    def start_page(self, page_num):
        self.current_page = {'page': page_num, 'steps': [], 'start_mem': _rss_mb()}
        print(f"\n📄 Page {page_num} START: {self.current_page['start_mem']:.1f}MB")
    
    def record_step(self, step_name, details=""):
        if not self.current_page:
            return
        mem_now = _rss_mb()
        last_mem = self.current_page['steps'][-1]['mem'] if self.current_page['steps'] else self.current_page['start_mem']
        delta = mem_now - last_mem
        self.current_page['steps'].append({
            'name': step_name,
            'mem': mem_now,
            'delta': delta,
            'details': details
        })
        sign = "+" if delta >= 0 else ""
        print(f"      {step_name}: {mem_now:.1f}MB ({sign}{delta:.1f}MB) {details}")
    
    def end_page(self):
        if not self.current_page:
            return 0
        end_mem = _rss_mb()
        net = end_mem - self.current_page['start_mem']
        self.page_data.append(self.current_page)
        print(f"   📊 Page {self.current_page['page']} NET: {net:+.1f}MB (start={self.current_page['start_mem']:.1f}MB, end={end_mem:.1f}MB)")
        return net

profiler = MemoryProfiler()

FENCE_KEYWORDS = [
    "fence", "fencing", "chain link", "wood fence", "vinyl fence", 
    "wrought iron", "ornamental", "gate", "gates", "railing",
    "F-", "F1", "F2", "F3", "F4", "F5", "F-1", "F-2", "F-3",
]

PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'
MAX_PAGES = 5  # Only first 5 pages for quick test

def main():
    print("="*80)
    print("QUICK MEMORY PROFILING TEST (First 5 Pages Only)")
    print("="*80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Max pages: {MAX_PAGES}")
    print(f"Start memory: {_rss_mb():.1f}MB\n")
    
    # Check PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        sys.exit(1)
    
    # Initialize LLM
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ LLM initialized")
    except Exception as e:
        print(f"❌ LLM init failed: {e}")
        sys.exit(1)
    
    # Open PDF
    print(f"Opening PDF...")
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    pages_to_process = min(MAX_PAGES, total_pages)
    print(f"✅ PDF opened: {total_pages} pages (will process {pages_to_process})")
    print(f"Memory after opening: {_rss_mb():.1f}MB\n")
    
    fence_pages = []
    non_fence_pages = []
    
    print("="*80)
    print(f"PROCESSING PAGES 1-{pages_to_process}")
    print("="*80)
    
    for i in range(pages_to_process):
        curr_pg_num = i + 1
        
        profiler.start_page(curr_pg_num)
        
        # 1. GC cleanup
        gc.collect()
        profiler.record_step("1. GC cleanup")
        
        # 3. Load page
        page_obj = doc.load_page(i)
        profiler.record_step("3. load_page()")
        
        # 4. Extract text
        text_content = page_obj.get_text("text")
        profiler.record_step("4. get_text()", f"len={len(text_content)}")
        
        # Adaptive DPI for large pages
        page_width = page_obj.rect.width
        page_height = page_obj.rect.height
        if page_width > 2000 or page_height > 2000:
            dpi = 60
            profiler.record_step("→ Large page", f"{page_width:.0f}×{page_height:.0f}, DPI=60")
        else:
            dpi = 72
        
        # 5. Create pixmap
        pix = page_obj.get_pixmap(dpi=dpi, alpha=False)
        pix_width, pix_height = pix.width, pix.height
        profiler.record_step("5. get_pixmap()", f"size={pix_width}x{pix_height}")
        
        # 6. Convert to PNG and FREE pixmap immediately
        img_bytes = pix.tobytes("png")
        del pix  # Free NOW
        gc.collect()
        profiler.record_step("6. tobytes() + free pixmap", f"{len(img_bytes)/(1024*1024):.2f}MB")
        
        # 7. PDF wrapper
        temp_img_doc = fitz.open()
        temp_page = temp_img_doc.new_page(width=pix_width, height=pix_height)
        temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
        single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
        temp_img_doc.close()
        profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")
        
        # 8. Cleanup img_bytes immediately
        del img_bytes  # Free NOW
        gc.collect()
        profiler.record_step("8. Cleanup img_bytes")
        
        # 9. Analyze page
        page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
        print(f"      → Calling LLM API for page {curr_pg_num}...")
        try:
            analysis_res_core = analyze_page(
                page_data_an, llm, FENCE_KEYWORDS, google_cloud_config,
                recall_mode="strict"
            )
            profiler.record_step("9. analyze_page()", f"fence={analysis_res_core.get('fence_found')}")
        except Exception as e:
            print(f"      ⚠️  analyze_page() error: {str(e)[:100]}")
            analysis_res_core = {"fence_found": False, "text_found": False, "text_response": "{}"}
            profiler.record_step("9. analyze_page() FAILED", str(e)[:30])
        
        # 10. Extract signals
        try:
            jr = json.loads(analysis_res_core.get("text_response", "{}"))
            signals = jr.get("signals", [])
        except:
            signals = []
        profiler.record_step("10. Extract signals", f"count={len(signals)}")
        
        # 11. OCR highlighting (if text found)
        analysis_result = {**analysis_res_core, 'page_number': curr_pg_num, 'fence_text_boxes_details': []}
        if analysis_result.get('text_found') and google_cloud_config:
            print(f"      → Calling Google Document AI for OCR...")
            try:
                boxes, _, _ = get_fence_related_text_boxes(
                    single_page_pdf_bytes,
                    llm,
                    FENCE_KEYWORDS,
                    signals,
                    "gpt-4o-mini",
                    google_cloud_config
                )
                if boxes:
                    analysis_result['fence_text_boxes_details'] = boxes
                profiler.record_step("11. OCR highlighting", f"boxes={len(boxes) if boxes else 0}")
            except Exception as e:
                print(f"      ⚠️  OCR error: {str(e)[:100]}")
                profiler.record_step("11. OCR highlighting FAILED", str(e)[:30])
        else:
            profiler.record_step("11. OCR highlighting", "skipped (no text or no config)")
        
        # 12. Store result
        if analysis_result.get('fence_found'):
            fence_pages.append(analysis_result)
        else:
            non_fence_pages.append(analysis_result)
        profiler.record_step("12. Store result", f"size={len(str(analysis_result))/(1024):.1f}KB")
        
        # 13. Cleanup variables
        del page_obj, text_content, single_page_pdf_bytes
        gc.collect()
        profiler.record_step("13. Cleanup variables")
        
        profiler.end_page()
    
    doc.close()
    print(f"\n✅ PDF closed")
    print(f"Final memory: {_rss_mb():.1f}MB")
    
    # Print summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY - Memory Delta Per Function")
    print("="*80)
    
    if profiler.page_data:
        summary = {}
        for page_data in profiler.page_data:
            for step in page_data['steps']:
                name = step['name']
                if name not in summary:
                    summary[name] = {'count': 0, 'total': 0, 'max': 0, 'min': 999}
                summary[name]['count'] += 1
                summary[name]['total'] += step['delta']
                summary[name]['max'] = max(summary[name]['max'], step['delta'])
                summary[name]['min'] = min(summary[name]['min'], step['delta'])
        
        print(f"\n{'Step':<40} {'Calls':<8} {'Avg Δ':<12} {'Max Δ':<12} {'Min Δ':<12}")
        print("-"*88)
        
        # Sort by total delta (biggest impact first)
        sorted_steps = sorted(summary.items(), key=lambda x: abs(x[1]['total']), reverse=True)
        
        for name, s in sorted_steps:
            avg = s['total'] / s['count'] if s['count'] > 0 else 0
            print(f"{name:<40} {s['count']:<8} {avg:>+11.1f}MB {s['max']:>+11.1f}MB {s['min']:>+11.1f}MB")
        
        total_net = sum((p['steps'][-1]['mem'] if p['steps'] else p['start_mem']) - p['start_mem'] for p in profiler.page_data)
        avg_net = total_net / len(profiler.page_data) if profiler.page_data else 0
        print(f"\n{'='*88}")
        print(f"Pages profiled: {len(profiler.page_data)}")
        print(f"Fence pages: {len(fence_pages)}")
        print(f"Non-fence pages: {len(non_fence_pages)}")
        print(f"Total NET memory: {total_net:+.1f}MB")
        print(f"Avg NET per page: {avg_net:+.1f}MB")
        print("="*88)
        
        # Analysis
        print("\n🔍 MEMORY LEAK ANALYSIS:")
        print("-"*88)
        
        # Find biggest positive leak
        positive_leaks = [(name, s) for name, s in sorted_steps if s['total'] > 0]
        if positive_leaks:
            biggest = positive_leaks[0]
            avg_leak = biggest[1]['total'] / biggest[1]['count']
            print(f"🥇 #1 CULPRIT: {biggest[0]}")
            print(f"   Average: {avg_leak:+.1f}MB per call")
            print(f"   Max spike: {biggest[1]['max']:+.1f}MB")
            print(f"   Total impact: {biggest[1]['total']:+.1f}MB")
            
            if len(positive_leaks) > 1:
                second = positive_leaks[1]
                avg_leak = second[1]['total'] / second[1]['count']
                print(f"\n🥈 #2 CULPRIT: {second[0]}")
                print(f"   Average: {avg_leak:+.1f}MB per call")
                print(f"   Total impact: {second[1]['total']:+.1f}MB")
            
            if len(positive_leaks) > 2:
                third = positive_leaks[2]
                avg_leak = third[1]['total'] / third[1]['count']
                print(f"\n🥉 #3 CULPRIT: {third[0]}")
                print(f"   Average: {avg_leak:+.1f}MB per call")
                print(f"   Total impact: {third[1]['total']:+.1f}MB")
        
        # Check cleanup
        cleanup_steps = [s for s in sorted_steps if 'cleanup' in s[0].lower() or 'gc' in s[0].lower()]
        if cleanup_steps:
            print(f"\n🧹 CLEANUP EFFECTIVENESS:")
            for name, s in cleanup_steps:
                avg = s['total'] / s['count'] if s['count'] > 0 else 0
                status = "✅ Working" if avg < 0 else "❌ NOT working"
                print(f"   {name}: {avg:+.1f}MB avg ({status})")
        
        # Projection for 84 pages
        if avg_net > 0:
            projected_84 = avg_net * 84
            print(f"\n📊 PROJECTION FOR 84 PAGES:")
            print(f"   At {avg_net:+.1f}MB per page × 84 pages = {projected_84:+.1f}MB total")
            if projected_84 > 1000:
                print(f"   ⚠️  WILL CRASH! (exceeds 1000MB limit)")
            elif projected_84 > 800:
                print(f"   ⚠️  RISKY (close to 800MB limit)")
            else:
                print(f"   ✅ Should be OK (under 800MB)")
        
        print("="*88)
    
    print(f"\n✅ Quick test complete!")

if __name__ == "__main__":
    main()

