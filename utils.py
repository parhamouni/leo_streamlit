import time
import functools
import fitz  # PyMuPDF
import base64
import re
import random 
import os
from langchain_core.messages import HumanMessage
from openai import RateLimitError, APIError, APITimeoutError
import pdfplumber
from io import BytesIO
import json
# pip install --upgrade google-cloud-documentai
from google.cloud import documentai_v1 as documentai
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import comprehensive text extraction
try:
    from comprehensive_page_extractor import extract_comprehensive_page_text
    COMPREHENSIVE_EXTRACTION_AVAILABLE = True
    print("✅ Comprehensive text extraction available")
except ImportError:
    COMPREHENSIVE_EXTRACTION_AVAILABLE = False
    print("⚠️ Comprehensive text extraction not available - using fallback methods")

# Legacy global Document AI client - deprecated, use google_cloud_config parameter instead
document_ai_client = None

def create_document_ai_client(google_cloud_config=None):
    """Create Document AI client from configuration - only supports Streamlit secrets."""
    try:
        if google_cloud_config and google_cloud_config.get("service_account_info"):
            # Use service account info from configuration (Streamlit secrets)
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                google_cloud_config["service_account_info"]
            )
            client = documentai.DocumentProcessorServiceClient(credentials=credentials)
            print("✅ Document AI client created from Streamlit secrets configuration")
            return client
        else:
            print("⚠️ No Google Cloud configuration provided. Please ensure secrets are configured in Streamlit.")
            return None
            
    except Exception as e:
        print(f"⚠️ Document AI client creation failed: {e}")
        return None

# No legacy initialization - all configuration should come from app.py


# Custom Exception for unrecoverable rate limit
class UnrecoverableRateLimitError(Exception):
    pass



# --- Timing Decorator ---
def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        # print(f"TIMER LOG: ---> Entering {func_name}") 
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # print(f"TIMER LOG: <--- Exiting  {func_name} (Duration: {duration:.4f} seconds)")
        return result
    return wrapper

# --- Core Functions ---

@time_it
def extract_comprehensive_text_from_page(page_bytes: bytes, page_number: int = 1, google_cloud_config=None) -> Dict:
    """
    Extract comprehensive text from a PDF page using multiple methods:
    1. PDF text layer (native text)
    2. Text annotations (excluding visual-only annotations)  
    3. OCR (with deduplication against existing text)
    
    Args:
        page_bytes: PDF page as bytes
        page_number: Page number for reference
        
    Returns:
        Dictionary with extracted text data and statistics
    """
    try:
        # Open PDF page from bytes
        doc = fitz.open(stream=page_bytes, filetype="pdf")
        if not doc or len(doc) == 0:
            return {"error": "Could not open PDF page", "text": "", "all_text": [], "stats": {}}
        
        page = doc[0]  # Get first (and only) page
        
        if COMPREHENSIVE_EXTRACTION_AVAILABLE and google_cloud_config:
            # Create Document AI client and processor name from config
            client = create_document_ai_client(google_cloud_config)
            processor_name = f"projects/{google_cloud_config['project_number']}/locations/{google_cloud_config['location']}/processors/{google_cloud_config['processor_id']}"
            
            if client:
                # Use comprehensive extraction with OCR
                result = extract_comprehensive_page_text(
                    page=page,
                    client=client,
                    processor_name=processor_name,
                    use_ocr=True
                )
            else:
                # Fallback if client creation failed
                result = None
        elif COMPREHENSIVE_EXTRACTION_AVAILABLE and document_ai_client:
            # Fallback to legacy global client - but this should not be used anymore
            print("⚠️ Using legacy global Document AI client - configuration should be passed explicitly")
            result = None
        else:
            result = None
            
        if result:
            # Combine all text for analysis
            all_text_items = result['all_text']
            combined_text = ' '.join([item['text'] for item in all_text_items])
            
            doc.close()
            
            return {
                "text": combined_text,
                "text_words": result['text_words'],
                "text_annotations": result['text_annotations'],
                "ocr_texts": result['ocr_texts'],
                "all_text": all_text_items,
                "stats": result['stats'],
                "page_number": page_number,
                "extraction_method": "comprehensive"
            }
        else:
            # Fallback to basic text extraction
            text_words = page.get_text("words")
            combined_text = ' '.join([word[4] for word in text_words])  # word[4] is the text
            
            doc.close()
            
            return {
                "text": combined_text,
                "text_words": [{"text": word[4], "x0": word[0], "y0": word[1], 
                              "x1": word[2], "y1": word[3], "source": "text_layer"} 
                              for word in text_words],
                "text_annotations": [],
                "ocr_texts": [],
                "all_text": [{"text": word[4], "source": "text_layer"} for word in text_words],
                "stats": {"total_words": len(text_words), "total_elements": len(text_words), 
                         "ocr_enabled": False},
                "page_number": page_number,
                "extraction_method": "fallback"
            }
            
    except Exception as e:
        print(f"Error in comprehensive text extraction: {e}")
        return {"error": str(e), "text": "", "all_text": [], "stats": {}, 
                "extraction_method": "error"}

@time_it
def extract_snippet(text, fence_keywords):
    for kw in fence_keywords:
        match = re.search(rf".{{0,50}}\b{re.escape(kw)}\b.{{0,50}}", text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def is_positive_response(response: str) -> bool:
    response = response.strip().lower()
    return response.startswith("yes") or response.startswith('"yes')

@time_it
def retry_with_backoff(llm_invoke_method, messages_list, retries=5, base_delay=2):
    func_name = llm_invoke_method.__qualname__ if hasattr(llm_invoke_method, '__qualname__') else "llm_invoke"
    for attempt in range(retries):
        try:
            return llm_invoke_method(messages_list)
        except RateLimitError as rle:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (RateLimitError). Last error: {rle}")
                raise UnrecoverableRateLimitError(f"OpenAI API rate limit. Please try later. (Details: {rle})")
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: RateLimitError for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {rle}")
            time.sleep(delay)
        except (APIError, APITimeoutError) as apie:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (APIError/Timeout). Last error: {apie}")
                raise 
            delay = base_delay * (2 ** attempt) + random.uniform(0,1)
            print(f"TIMER LOG: API Error/Timeout for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {apie}")
            time.sleep(delay)
        except Exception as e:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (Unexpected). Last error: {e}")
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0,1)
            print(f"TIMER LOG: Unexpected Error for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Exceeded max retries for {func_name}. Should have been caught earlier.")


@time_it
def analyze_page(page_data, llm_text, fence_keywords, google_cloud_config=None):
    """
    Enhanced page analysis using comprehensive text extraction.
    
    • Extracts text from multiple sources (text layer + annotations + OCR)
    • Model must reply with JSON {"answer":"yes|no","reason":"…"}
    • Returns fence_found = True if answer==yes (case-insensitive)
    """

    pg_num = page_data.get("page_number", "N/A")
    
    # ---- Extract comprehensive text from page bytes ----
    page_bytes = page_data.get("page_bytes")
    if page_bytes:
        print(f"🔍 Running comprehensive text extraction for page {pg_num}...")
        extraction_result = extract_comprehensive_text_from_page(page_bytes, pg_num, google_cloud_config)
        
        comprehensive_text = extraction_result.get("text", "")
        extraction_stats = extraction_result.get("stats", {})
        extraction_method = extraction_result.get("extraction_method", "unknown")
        
        print(f"📊 Page {pg_num} extraction: {extraction_stats.get('total_elements', 0)} total texts "
              f"({extraction_stats.get('total_ocr_unique', 0)} from OCR) via {extraction_method}")
    else:
        # Fallback to original text if no page_bytes
        comprehensive_text = page_data.get("text", "")
        extraction_stats = {}
        extraction_method = "legacy"

    # ---- Build focused excerpt for analysis ----
    # Use more text since we have comprehensive extraction
    max_excerpt_length = 1200 if extraction_method == "comprehensive" else 800
    excerpt = comprehensive_text
    
    # if len(comprehensive_text) > max_excerpt_length:
    #     excerpt += "... [text truncated]"

    # ---- Compose enhanced prompt ----
    system = (
        "You are an assistant that returns STRICT JSON like "
        '{"answer":"yes","reason":"…"}  or {"answer":"no","reason":"…"}.\n'
        "• yes  = the sheet is mainly about fences / gates / guardrails / barriers\n"
        "• no   = everything else (title, lighting, schedules, etc.)\n\n"
        f"Text extracted via {extraction_method} method with {extraction_stats.get('total_elements', 'unknown')} elements."
    )

    few_shot = """
{"answer":"yes","reason":"Legend line for F-2 chain-link fence."}
Text: "F-2  8' CL FENCE  SEE DETAIL 3/L2-01"
--
{"answer":"yes","reason":"Specifies a security gate."}
Text: "NEW SECURITY GATE (12' SWING) AT SERVICE ENTRY"
--
{"answer":"yes","reason":"Contains fence post specifications and details."}
Text: "FENCE POST FOOTING DETAIL  CONCRETE FOOTING 24\" DEEP  GALVANIZED POST"
--
{"answer":"no","reason":"Only a title / index sheet."}
Text: "TITLE SHEET  •  DRAWING INDEX  •  SCALE 1:200"
--
{"answer":"no","reason":"Electrical plan with no fence elements."}
Text: "LIGHTING PLAN  ELECTRICAL OUTLETS  SWITCH LOCATIONS"
"""

    prompt = (
        f"{system}\n\n{few_shot}\n"
        "-----\n"
        f'Text: "{excerpt}"'
    )

    # ---- LLM call ----
    try:
        raw = retry_with_backoff(llm_text.invoke,
                                 [HumanMessage(content=prompt)]).content
    except UnrecoverableRateLimitError:
        raise
    except Exception as e:
        raw = f'{{"answer":"no","reason":"error {e}"}}'

    # ---- Parse yes/no ----
    answer = "no"
    
    # Debug logging to track the issue
    print(f"🔍 Page {pg_num} DEBUG: Raw LLM response: '{raw}'")
    
    try:
        clean_raw = raw
        if "```" in raw:
            # Extract content between code blocks
            parts = raw.split("```")
            if len(parts) >= 3:
                clean_raw = parts[1]
            else:
                clean_raw = parts[1] if len(parts) > 1 else raw
            print(f"🔍 Page {pg_num} DEBUG: Extracted from code block: '{clean_raw}'")
        
        clean_raw = clean_raw.strip()
        
        # Remove language identifier if present (e.g., "json" at the beginning)
        if clean_raw.lower().startswith('json'):
            clean_raw = clean_raw[4:].strip()
            print(f"🔍 Page {pg_num} DEBUG: Removed 'json' prefix: '{clean_raw}'")
        
        print(f"🔍 Page {pg_num} DEBUG: Attempting to parse JSON: '{clean_raw}'")
        
        parsed_json = json.loads(clean_raw)
        answer = parsed_json.get("answer", "no").lower()
        
        print(f"🔍 Page {pg_num} DEBUG: Parsed JSON successfully. Answer field: '{answer}'")
        
    except Exception as e:
        print(f"⚠️ Page {pg_num} DEBUG: JSON parsing failed with error: {e}")
        # Fallback to simple text parsing if JSON fails
        answer = raw.strip().split()[0].lower() if raw.strip() else "no"
        print(f"🔍 Page {pg_num} DEBUG: Fallback answer: '{answer}'")

    text_yes = answer.startswith("y")
    found = text_yes
    
    print(f"📋 Page {pg_num} DEBUG: Final decision - answer='{answer}', text_yes={text_yes}, found={found}")
    snippet = extract_snippet(comprehensive_text, fence_keywords) if found else None

    return {
        "page_number": pg_num,
        "fence_found": found,
        "text_found": text_yes,
        "text_response": raw,
        "text_snippet": snippet,
        "extraction_stats": extraction_stats,
        "extraction_method": extraction_method,
        "comprehensive_text": comprehensive_text
    }

@time_it  
def get_comprehensive_text_for_analysis(page_bytes: bytes, google_cloud_config=None) -> str:
    """
    Extract comprehensive text from a page for use in text box analysis.
    Returns combined text from all sources.
    """
    try:
        extraction_result = extract_comprehensive_text_from_page(page_bytes, google_cloud_config=google_cloud_config)
        if extraction_result.get("error"):
            print(f"⚠️ Comprehensive extraction failed: {extraction_result['error']}")
            return ""
        
        # Get comprehensive text
        comprehensive_text = extraction_result.get("text", "")
        stats = extraction_result.get("stats", {})
        method = extraction_result.get("extraction_method", "unknown")
        
        print(f"📝 Comprehensive text extraction: {stats.get('total_elements', 0)} elements "
              f"via {method} method")
        
        return comprehensive_text
        
    except Exception as e:
        print(f"❌ Error in comprehensive text extraction: {e}")
        return ""

@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app, selected_llm_model_name="gpt-3.5-turbo"):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting REFINED TWO-PASS - v10 (Noise Reduction for Sparse Pages).")
    overall_gfrtb_start_time = time.time()
    
    page_width, page_height = 0, 0
    final_highlight_boxes_list = []
    words = [] 

    try:
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            if not pdf.pages: return [], 0, 0
            page_obj = pdf.pages[0] 
            page_width, page_height = page_obj.width, page_obj.height
            if not page_width or not page_height: return [], 0, 0

            try:
                words = page_obj.extract_words(use_text_flow=True, split_at_punctuation=False, 
                                           x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
            except Exception as e:
                 print(f"TIMER LOG: (get_fence_related_text_boxes) Error during page_obj.extract_words: {type(e).__name__}: {e}")

            extracted_lines = []
            try:
                extracted_lines = page_obj.extract_text_lines(layout=True, use_text_flow=True, strip=True, return_chars=False)
            except Exception as e_lines:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Error during page_obj.extract_text_lines: {type(e_lines).__name__}: {e_lines}")

            print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Pass 1: Legend/Description Identification.")
            candidate_legend_lines_for_llm1 = []
            if extracted_lines:
                for line_idx, line_obj_data in enumerate(extracted_lines):
                    line_text = line_obj_data.get('text', '').strip()
                    if not line_text: continue
                    
                    # --- Stricter Pre-filtering for Pass 1 Candidates (V10) ---
                    if len(line_text) < 5: # Ignore very short fragments (e.g., "ft.", "mm", single characters)
                        continue
                    num_words = len(line_text.split())
                    # If less than 2 words, it must look like a code (e.g. "F1:", "1.", "NTS") to be considered
                    # Allow single word if it's a very strong fence keyword AND looks like a title/label (e.g. all caps)
                    is_short_code_like = bool(re.match(r"^[A-Z0-9]{1,6}([:\.\-\s]*)$", line_text.strip())) or \
                                         bool(re.match(r"^\s*\(?[A-Za-z0-9]{1,4}\)?[:\.\-]\s*", line_text))
                    
                    if num_words < 1: # Should not happen if len(line_text) >=5, but good check
                        continue
                    if num_words == 1 and len(line_text) < 10 and not is_short_code_like: # Single short word, not code-like
                        # Check if this single word is a strong, specific fence keyword (not just "post")
                        is_strong_single_fence_keyword = line_text.lower() in ['fence', 'fencing', 'gate', 'gates', 'barrier', 'guardrail']
                        if not is_strong_single_fence_keyword:
                            continue 
                    elif num_words < 2 and not is_short_code_like: # (e.g. two very short words, like "TO FENCE")
                         if len(line_text) < 10: # If it's very short and not code-like
                            continue
                    # --- End of Stricter Pre-filtering for V10 ---

                    line_text_lower = line_text.lower()
                    has_fence_keyword = any(kw in line_text_lower for kw in fence_keywords_from_app)
                    is_drawing_term_or_identifier = any(term in line_text_lower for term in ["detail", "type", "schedule", "item", "legend", "notes", "spec", "assy", "section", "elevation", "matl", "constr", "view", "plan", "typical", "standard", "description", "callout"]) or \
                                                    bool(re.match(r"^\s*\(?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\)?[\s:.)\-]", line_text))
                    
                    if (has_fence_keyword or is_drawing_term_or_identifier):
                        if (1 < num_words < 80) and len(line_text) < 500 : # Use num_words from above
                            if all(k in line_obj_data for k in ['x0', 'top', 'x1', 'bottom']):
                                candidate_legend_lines_for_llm1.append({
                                    "id": f"line_{line_idx}", "text": line_text,
                                    "x0": round(line_obj_data['x0'], 2), "y0": round(line_obj_data['top'], 2),
                                    "x1": round(line_obj_data['x1'], 2), "y1": round(line_obj_data['bottom'], 2)
                                })
            print(f"TIMER LOG: (get_fence_related_text_boxes) Pre-filtering lines for Pass 1. Found {len(candidate_legend_lines_for_llm1)} candidates.")

            identified_legends_from_pass1_llm_output = [] 
            confirmed_legend_core_ids = set()  
            processed_legends_for_pass2_prompt_context = [] 

            if candidate_legend_lines_for_llm1:
                lines_json_str_pass1 = json.dumps(candidate_legend_lines_for_llm1, separators=(',', ':'))
                pass1_examples = """
Example 1 Input Line: {"id": "line_23", "text": "1. 6' HIGH INTERIOR COURT FENCE, COLOR: BLACK VINYL"}
Example 1 Output: {"id": "line_23", "full_text": "1. 6' HIGH INTERIOR COURT FENCE, COLOR: BLACK VINYL", "core_identifier_text": "1", "type": "legend_item"}
Example 2 Input Line: {"id": "line_45", "text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS PER DETAIL 3/CD-2."}
Example 2 Output: {"id": "line_45", "full_text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS PER DETAIL 3/CD-2.", "core_identifier_text": "FENCE_POST_FOOTING_NOTE", "type": "note"}
Example 3 Input Line: {"id": "line_67", "text": "6. POST TENSION SLAB - REFER TO DETAIL 3, CD-1"}
Example 3 Output: {} (OMIT - NOT fence-related)
Example 4 Input Line: {"id": "line_101", "text": "VISITORS' LOW FENCE"}
Example 4 Output: {"id": "line_101", "full_text": "VISITORS' LOW FENCE", "core_identifier_text": "VISITORS_LOW_FENCE", "type": "description"}
Example 5 Input Line: {"id": "line_102", "text": "DATE PRINTED: 11/4/2024"} 
Example 5 Output: {} (OMIT - Not fence-related, likely boilerplate/noise)
Example 6 Input Line: {"id": "line_103", "text": "ft"} 
Example 6 Output: {} (OMIT - Too short, likely OCR noise or irrelevant fragment)
"""
                prompt_pass1 = f"""You are an engineering drawing analyst. Your goal is to identify text elements SPECIFICALLY about FENCES, GATES, or their components.
You are given a JSON list of TEXT LINES. Fence-related keywords: {', '.join(fence_keywords_from_app)}.
Your task:
1.  Analyze each input line. Identify ONLY lines that EXPLICITLY describe FENCES, GATES, BARRIERS, GUARDRAILS, or their direct attributes (e.g., fence height, gate type, post material).
    - IGNORE lines that are clearly not fence-related (e.g., "LIGHT POLE", "POST TENSION SLAB", "PROJECT BOUNDARY", "DATE PRINTED").
    - IGNORE very short (e.g. < 4 characters), fragmented, or noisy text (e.g., standalone "ft", "mm", random characters) unless it's a very clear and known fence identifier type/code (e.g. "F1").
2.  For each **fence-specific and meaningful** line identified:
    a.  Determine **`core_identifier_text`**: 
        - **P1:** Exact tag (e.g., "F1", "1", "NOTE 3"), remove trailing punctuation. 
        - **P2:** ALL_CAPS_SNAKE_CASE summary for descriptive text (e.g., "FENCE_HEIGHT_SPEC").
        - **P3:** "GENERAL_FENCE_NOTE" or "N/A_DESC".
    b.  Provide **`type`**: "legend_item", "specification", "note", or "description".

Examples (output shows items to be included in 'identified_fences' list):
{pass1_examples}

Output ONLY a single valid JSON object: {{"identified_fences": [{{...output for fence-related line 1...}}, ...]}}.
'id' and 'full_text' must match input. OMIT non-fence-related or noisy lines from the output list (i.e., do not create an entry for them).
If NO valid fence-specific items are found, return {{"identified_fences": []}}. Strictly JSON.

Input Text Lines:
{lines_json_str_pass1}"""
                
                response_content_pass1 = ""
                try:
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Initiating Pass 1 Text LLM call ({len(candidate_legend_lines_for_llm1)} lines).")
                    pass1_llm_call_start_time = time.time()
                    response_obj_pass1 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass1)])
                    response_content_pass1 = response_obj_pass1.content
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 1 LLM call completed (took {time.time() - pass1_llm_call_start_time:.4f}s).")
                    clean_resp_pass1 = response_content_pass1.strip()
                    match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_pass1, re.IGNORECASE)
                    if match_json_block: clean_resp_pass1 = match_json_block.group(1).strip()
                    else:
                        first_brace = clean_resp_pass1.find('{'); last_brace = clean_resp_pass1.rfind('}')
                        if first_brace != -1 and last_brace > first_brace: clean_resp_pass1 = clean_resp_pass1[first_brace : last_brace+1]
                    parsed_data_pass1 = json.loads(clean_resp_pass1)
                    identified_legends_from_pass1_llm_output = parsed_data_pass1.get("identified_fences", [])
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 1 Parsing. Found {len(identified_legends_from_pass1_llm_output)} fence items from LLM.")
                    if identified_legends_from_pass1_llm_output:
                        temp_legend_map = {item['id']: item for item in candidate_legend_lines_for_llm1} 
                        for legend_data_llm in identified_legends_from_pass1_llm_output:
                            if not isinstance(legend_data_llm, dict) or not legend_data_llm: continue
                            original_line_id = legend_data_llm.get("id")
                            if not original_line_id: continue
                            original_line_obj_from_map = temp_legend_map.get(original_line_id)
                            if not original_line_obj_from_map: continue
                            item_text_to_use = legend_data_llm.get("full_text", original_line_obj_from_map.get("text","")) 
                            core_id_from_llm = legend_data_llm.get("core_identifier_text")
                            item_type = legend_data_llm.get("type", "unknown") 
                            core_id_processed = ""
                            if core_id_from_llm and isinstance(core_id_from_llm, str) and core_id_from_llm.strip() and core_id_from_llm != "N/A_DESC":
                                core_id_processed = core_id_from_llm.strip().upper()
                                core_id_processed = re.sub(r"[.:\s]+$", "", core_id_processed) 
                                confirmed_legend_core_ids.add(core_id_processed)
                            if all(original_line_obj_from_map.get(k) is not None for k in ['text', 'x0', 'y0', 'x1', 'y1']):
                                final_highlight_boxes_list.append({'id': original_line_id, 'text': item_text_to_use, 'x0': original_line_obj_from_map['x0'], 'y0': original_line_obj_from_map['y0'],'x1': original_line_obj_from_map['x1'], 'y1': original_line_obj_from_map['y1'],'type_from_llm': item_type, 'tag_from_llm': core_id_processed if core_id_processed else core_id_from_llm })
                                processed_legends_for_pass2_prompt_context.append({"id": original_line_id, "core_identifier_text": core_id_processed if core_id_processed else core_id_from_llm, "full_text": item_text_to_use, "type": item_type})
                except UnrecoverableRateLimitError: raise 
                except Exception as e_p1: print(f"TIMER LOG: Error in Pass 1: {e_p1}"); print(f"TIMER LOG: Pass 1 Resp: {response_content_pass1[:500] if response_content_pass1 else 'None'}")
            print(f"TIMER LOG: After Pass 1, core_ids for Pass 2: {confirmed_legend_core_ids}")

            # --- PASS 2 Logic (Identical to V9) ---
            if confirmed_legend_core_ids and words: 
                print(f"TIMER LOG: Starting Pass 2: Indicator Identification ({len(words)} Words).")
                candidate_indicator_words_for_llm2 = []
                for idx, word_obj_data in enumerate(words):
                    text_val = word_obj_data['text'].strip()
                    if not text_val: continue
                    core_word_text_match = re.match(r"^\s*\(?\s*([A-Za-z0-9]+(?:[\.\-_][A-Za-z0-9]+)*)\s*\)?\.?\s*$", text_val)
                    raw_core_word_text = core_word_text_match.group(1) if core_word_text_match else text_val
                    core_word_text_for_match = raw_core_word_text.upper()
                    core_word_text_for_match = re.sub(r"[.:\s]+$", "", core_word_text_for_match)
                    if core_word_text_for_match in confirmed_legend_core_ids:
                        if 0 < len(text_val) <= 15: 
                            is_already_part_of_identified_item = False
                            for identified_item_box in final_highlight_boxes_list: 
                                if identified_item_box.get('tag_from_llm', '').upper() == core_word_text_for_match or \
                                   text_val.upper() in identified_item_box.get('text','').upper() :
                                    lx0, ly0, lx1, ly1 = identified_item_box.get('x0'), identified_item_box.get('y0'), identified_item_box.get('x1'), identified_item_box.get('y1')
                                    if all(v is not None for v in [lx0, ly0, lx1, ly1]):
                                        word_center_x = (word_obj_data['x0'] + word_obj_data['x1']) / 2
                                        word_center_y = (word_obj_data['top'] + word_obj_data['bottom']) / 2
                                        if (lx0 - 2) <= word_center_x <= (lx1 + 2) and \
                                           (ly0 - 2) <= word_center_y <= (ly1 + 2):
                                            is_already_part_of_identified_item = True; break
                            if not is_already_part_of_identified_item:
                                candidate_indicator_words_for_llm2.append({
                                    "id": f"word_{idx}", "text": text_val, 
                                    "core_text_matched": core_word_text_for_match, 
                                    "x0": round(word_obj_data['x0'], 2), "y0": round(word_obj_data['top'], 2),
                                    "x1": round(word_obj_data['x1'], 2), "y1": round(word_obj_data['bottom'], 2)
                                })
                print(f"TIMER LOG: Pass 2: Found {len(candidate_indicator_words_for_llm2)} candidate indicators.")

                if candidate_indicator_words_for_llm2:
                    indicators_json_str_pass2 = json.dumps(candidate_indicator_words_for_llm2, separators=(',',':'))
                    pass1_context_json_for_pass2 = []
                    for leg_detail in processed_legends_for_pass2_prompt_context:
                        leg_text_snippet = leg_detail.get("full_text", "")[:70]; 
                        if len(leg_detail.get("full_text", "")) > 70: leg_text_snippet += "..."
                        pass1_context_json_for_pass2.append({"identifier": leg_detail.get("core_identifier_text"),"description_snippet": leg_text_snippet, "type" : leg_detail.get("type")})
                    confirmed_legends_context_str_for_prompt = json.dumps(pass1_context_json_for_pass2, separators=(',',':'))
                    pass2_examples = """
Example 1 (Good Indicator): {"id": "word_105", "text": "1", "core_text_matched": "1"} -> Output: {"id": "word_105", "matched_legend_identifier": "1", "text_content": "1"}
Example 2 (Bad Indicator - Dimension): {"id": "word_200", "text": "10'", "core_text_matched": "10"} -> Output: []
Example 3 (Bad Indicator - Part of text): {"id": "word_10", "text": "Fence", "core_text_matched": "FENCE_POST_FOOTING_NOTE"} -> Output: []
Example 4 (Good Alphanumeric): {"id": "word_77", "text": "F2A", "core_text_matched": "F2A"} -> Output: {"id": "word_77", "matched_legend_identifier": "F2A", "text_content": "F2A"}
Example 5 (Bad Numerical - Quantity): {"id": "word_301", "text": "2", "core_text_matched": "2"} (Context: "2. GATE HARDWARE", Word context: "PROVIDE 2 ANCHOR BOLTS") -> Output: []
Example 6 (Bad Numerical - Dimension Value): {"id": "word_401", "text": "150", "core_text_matched": "150"} (Context: "150. FENCE MODEL X", Word context: "150mm typ.") -> Output: []"""
                    prompt_pass2 = f"""You are an engineering drawing analyst.
Context: Fence-related items & identifiers: {confirmed_legends_context_str_for_prompt}
You are given CANDIDATE INDICATOR text elements. Their 'core_text_matched' matches an identifier from context.
Task: Determine if each is a TRUE standalone graphical callout/indicator.
A TRUE indicator: Is short, spatially distinct, labels a drawing part, NOT part of sentence/dimension (e.g., "10'-0\"", "1:20", values with units like ', mm, LF), title, or main body of context items.
NUMBERS: An indicator if clearly tagging a legend item (e.g., "1" in circle with leader). NOT if measurement ("10'"), quantity ("2 POSTS"), scale, page number. Words with units (', ", mm, LF) are almost NEVER indicators.
Examples (output shows items for 'confirmed_indicators'): {pass2_examples}
Input Candidates: {indicators_json_str_pass2}
Output ONLY JSON: {{"confirmed_indicators": [{{...}}]}}. OMIT non-indicators. If none, {{"confirmed_indicators": []}}. Strictly JSON."""
                    response_content_pass2 = ""
                    try:
                        print(f"TIMER LOG: Initiating Pass 2 Text LLM call ({len(candidate_indicator_words_for_llm2)} candidates).")
                        pass2_llm_call_start_time = time.time()
                        response_obj_pass2 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass2)])
                        response_content_pass2 = response_obj_pass2.content
                        print(f"TIMER LOG: Pass 2 LLM call completed (took {time.time() - pass2_llm_call_start_time:.4f}s).")
                        clean_resp_pass2 = response_content_pass2.strip()
                        match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_pass2, re.IGNORECASE)
                        if match_json_block: clean_resp_pass2 = match_json_block.group(1).strip()
                        else:
                            first_brace = clean_resp_pass2.find('{'); last_brace = clean_resp_pass2.rfind('}')
                            if first_brace != -1 and last_brace > first_brace: clean_resp_pass2 = clean_resp_pass2[first_brace : last_brace+1]
                        parsed_data_pass2 = json.loads(clean_resp_pass2)
                        confirmed_indicators_from_llm = parsed_data_pass2.get("confirmed_indicators", [])
                        print(f"TIMER LOG: Pass 2 Parsing. Found {len(confirmed_indicators_from_llm)} confirmed indicators.")
                        original_pass2_candidates_map = {item['id']: item for item in candidate_indicator_words_for_llm2}
                        for ind_data in confirmed_indicators_from_llm:
                            if not isinstance(ind_data, dict): continue
                            el_id = ind_data.get("id")
                            if not el_id: continue
                            original_box_data = original_pass2_candidates_map.get(el_id)
                            if original_box_data and all(original_box_data.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({'id': el_id, 'text': original_box_data['text'], 'x0': original_box_data['x0'], 'y0': original_box_data['y0'],'x1': original_box_data['x1'], 'y1': original_box_data['y1'],'type_from_llm': "indicator", 'tag_from_llm': ind_data.get("matched_legend_identifier") })
                    except UnrecoverableRateLimitError: raise 
                    except Exception as e_p2: print(f"TIMER LOG: Error in Pass 2: {e_p2}"); print(f"TIMER LOG: Pass 2 Resp: {response_content_pass2[:500] if response_content_pass2 else 'None'}")
        
        dedup_map_final = {}
        for item in final_highlight_boxes_list:
            item_id = item.get('id') 
            if item_id: dedup_map_final[item_id] = item 
        final_highlight_boxes_list = list(dedup_map_final.values())
        print(f"TIMER LOG: (get_fence_related_text_boxes) Finished. Returning {len(final_highlight_boxes_list)} boxes. Total time: {time.time() - overall_gfrtb_start_time:.4f}s.")
        return final_highlight_boxes_list, page_width, page_height

    except Exception as e_outer:
        print(f"TIMER LOG: (get_fence_related_text_boxes) CRITICAL Outer Error: {e_outer}. Time: {time.time() - overall_gfrtb_start_time:.4f}s.")
        if 'final_highlight_boxes_list' in locals() and isinstance(final_highlight_boxes_list, list):
            dedup_map_final_except = {item.get('id'): item for item in final_highlight_boxes_list if item.get('id')}
            return list(dedup_map_final_except.values()), page_width, page_height
        return [], page_width, page_height


# Test function for comprehensive text extraction
def test_comprehensive_extraction(google_cloud_config=None):
    """
    Test function to verify comprehensive text extraction is working.
    """
    print("🧪 Testing comprehensive text extraction integration...")
    
    if not COMPREHENSIVE_EXTRACTION_AVAILABLE:
        print("❌ Comprehensive extraction not available")
        return False
    
    # Test client creation
    if google_cloud_config:
        test_client = create_document_ai_client(google_cloud_config)
        if test_client:
            print("✅ Document AI client created successfully from configuration")
            processor_name = f"projects/{google_cloud_config['project_number']}/locations/{google_cloud_config['location']}/processors/{google_cloud_config['processor_id']}"
            print(f"📄 Processor: {processor_name}")
            return True
        else:
            print("❌ Could not create Document AI client from configuration")
            return False
    elif document_ai_client:
        print("✅ Legacy Document AI client available (but configuration should be passed explicitly)")
        return True
    else:
        print("❌ No Document AI client available")
        return False

# Run test on import
if __name__ == "__main__":
    test_comprehensive_extraction()