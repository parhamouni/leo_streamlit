#!/usr/bin/env python3
"""
Automated Memory Profiling Test
Processes the 84-page PDF and generates profiling report WITHOUT Streamlit UI
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
    import sys
    def _rss_mb():
        return 0.0

# Import required modules
import fitz
from utils import analyze_page, get_fence_related_text_boxes
from langchain_openai import ChatOpenAI

# Load API keys from Streamlit secrets
import toml
secrets_path = '/Users/parhamhamouni/Desktop/leo/leo_streamlit/.streamlit/secrets.toml'
if os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    os.environ['OPENAI_API_KEY'] = secrets.get('OPENAI_API_KEY', '')
    print(f"✅ Loaded API key from {secrets_path}")
else:
    print(f"⚠️  No secrets file found at {secrets_path}")
    secrets = {}

# MemoryProfiler class (same as in app.py)
class MemoryProfiler:
    """Track memory per function to find leaks"""
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

# Global profiler
profiler = MemoryProfiler()

# Configuration
PDF_PATH = '/Users/parhamhamouni/Desktop/leo/data/Marked up plans 2/Takeoff/Combined_TO.pdf'
FENCE_KEYWORDS = [
    "fence", "fencing", "chain link", "wood fence", "vinyl fence", 
    "wrought iron", "ornamental", "gate", "gates", "railing",
    "F-", "F1", "F2", "F3", "F4", "F5", "F-1", "F-2", "F-3",
]

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

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("✅ LLM initialized")
except Exception as e:
    print(f"❌ LLM init failed: {e}")
    sys.exit(1)

def main():
    print("="*80)
    print("AUTOMATED MEMORY PROFILING TEST")
    print("="*80)
    print(f"\nPDF: {PDF_PATH}")
    print(f"Start memory: {_rss_mb():.1f}MB\n")
    
    # Check PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        sys.exit(1)
    
    # Open PDF
    print(f"Opening PDF...")
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"✅ PDF opened: {total_pages} pages")
    print(f"Memory after opening: {_rss_mb():.1f}MB\n")
    
    # Process pages
    fence_pages = []
    non_fence_pages = []
    
    print("="*80)
    print("PROCESSING PAGES")
    print("="*80)
    
    for i in range(total_pages):
        curr_pg_num = i + 1
        
        # Start profiling this page
        profiler.start_page(curr_pg_num)
        
        # 1. GC cleanup
        gc.collect()
        profiler.record_step("1. GC cleanup")
        
        # 2. Cache clear (every 5 pages)
        if i % 5 == 0 and i > 0:
            profiler.record_step("2. Cache clear (every 5)")
        
        # 3. Load page
        page_obj = doc.load_page(i)
        profiler.record_step("3. load_page()")
        
        # 4. Extract text
        text_content = page_obj.get_text("text")
        profiler.record_step("4. get_text()", f"len={len(text_content)}")
        
        # 5. Create pixmap
        pix = page_obj.get_pixmap(dpi=72, alpha=False)
        profiler.record_step("5. get_pixmap()", f"size={pix.width}x{pix.height}")
        
        # 6. Convert to PNG
        img_bytes = pix.tobytes("png")
        profiler.record_step("6. tobytes('png')", f"{len(img_bytes)/(1024*1024):.2f}MB")
        
        # 7. PDF wrapper
        temp_img_doc = fitz.open()
        temp_page = temp_img_doc.new_page(width=pix.width, height=pix.height)
        temp_page.insert_image(temp_page.rect, stream=img_bytes, keep_proportion=False)
        single_page_pdf_bytes = temp_img_doc.tobytes(deflate=True, garbage=4)
        temp_img_doc.close()
        profiler.record_step("7. PDF wrapper", f"{len(single_page_pdf_bytes)/(1024*1024):.2f}MB")
        
        # 8. Cleanup pixmap
        del pix, img_bytes
        gc.collect()
        profiler.record_step("8. Cleanup pixmap")
        
        # 9. Analyze page
        page_data_an = {"page_number": curr_pg_num, "text": text_content, "page_bytes": single_page_pdf_bytes}
        try:
            analysis_res_core = analyze_page(
                page_data_an, llm, FENCE_KEYWORDS, google_cloud_config,
                recall_mode="strict"
            )
            profiler.record_step("9. analyze_page()", f"fence={analysis_res_core.get('fence_found')}")
        except Exception as e:
            print(f"      ⚠️  analyze_page() error: {e}")
            analysis_res_core = {"fence_found": False, "text_found": False}
            profiler.record_step("9. analyze_page() FAILED", str(e)[:30])
        
        # 10. Extract signals
        try:
            jr = json.loads(analysis_res_core["text_response"])
            signals = jr.get("signals", [])
        except:
            signals = []
        profiler.record_step("10. Extract signals", f"count={len(signals)}")
        
        # 11. OCR highlighting (if text found)
        analysis_result = {**analysis_res_core, 'page_number': curr_pg_num, 'fence_text_boxes_details': []}
        if analysis_result.get('text_found') and google_cloud_config:
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
                print(f"      ⚠️  OCR error: {e}")
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
        
        # End page profiling
        net_change = profiler.end_page()
        
        # Check memory limit
        current_mem = _rss_mb()
        if current_mem > 1200:
            print(f"\n⚠️  MEMORY LIMIT REACHED: {current_mem:.1f}MB")
            print(f"Stopping at page {curr_pg_num} to prevent crash")
            break
    
    # Close PDF
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
        
        print(f"\n{'Step':<35} {'Calls':<8} {'Avg Δ':<10} {'Max Δ':<10} {'Min Δ':<10}")
        print("-"*80)
        
        # Sort by total delta (biggest impact first)
        sorted_steps = sorted(summary.items(), key=lambda x: abs(x[1]['total']), reverse=True)
        
        for name, s in sorted_steps:
            avg = s['total'] / s['count'] if s['count'] > 0 else 0
            print(f"{name:<35} {s['count']:<8} {avg:>+9.1f}MB {s['max']:>+9.1f}MB {s['min']:>+9.1f}MB")
        
        total_net = sum((p['steps'][-1]['mem'] if p['steps'] else p['start_mem']) - p['start_mem'] for p in profiler.page_data)
        avg_net = total_net / len(profiler.page_data) if profiler.page_data else 0
        print(f"\n{'='*80}")
        print(f"Total pages profiled: {len(profiler.page_data)}")
        print(f"Fence pages: {len(fence_pages)}")
        print(f"Non-fence pages: {len(non_fence_pages)}")
        print(f"Total NET memory change: {total_net:+.1f}MB")
        print(f"Average NET per page: {avg_net:+.1f}MB")
        print("="*80)
        
        # Identify culprits
        print("\n🔍 MEMORY LEAK ANALYSIS:")
        print("-"*80)
        
        biggest_leak = sorted_steps[0]
        if biggest_leak[1]['total'] > 0:
            avg_leak = biggest_leak[1]['total'] / biggest_leak[1]['count']
            print(f"🥇 #1 CULPRIT: {biggest_leak[0]}")
            print(f"   Average: {avg_leak:+.1f}MB per call")
            print(f"   Max spike: {biggest_leak[1]['max']:+.1f}MB")
            print(f"   Total impact: {biggest_leak[1]['total']:+.1f}MB")
        
        if len(sorted_steps) > 1:
            second_leak = sorted_steps[1]
            if second_leak[1]['total'] > 0:
                avg_leak = second_leak[1]['total'] / second_leak[1]['count']
                print(f"\n🥈 #2 CULPRIT: {second_leak[0]}")
                print(f"   Average: {avg_leak:+.1f}MB per call")
                print(f"   Total impact: {second_leak[1]['total']:+.1f}MB")
        
        # Check if cleanup is working
        cleanup_steps = [s for s in sorted_steps if 'cleanup' in s[0].lower() or 'gc' in s[0].lower()]
        if cleanup_steps:
            print(f"\n🧹 CLEANUP EFFECTIVENESS:")
            for name, s in cleanup_steps:
                avg = s['total'] / s['count'] if s['count'] > 0 else 0
                status = "✅ Working" if avg < 0 else "❌ NOT working"
                print(f"   {name}: {avg:+.1f}MB avg ({status})")
        
        print("="*80)
    
    print(f"\n✅ Test complete!")
    print(f"Final results: {len(fence_pages)} fence pages, {len(non_fence_pages)} non-fence pages")

if __name__ == "__main__":
    main()

