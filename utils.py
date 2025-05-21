import time
import functools
import fitz  # PyMuPDF
import base64
import re
import random 
from langchain_core.messages import HumanMessage
from openai import RateLimitError 
import pdfplumber
from io import BytesIO
import json
# For token counting, if you want to add it:
# import tiktoken 

# --- Timing Decorator ---
def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        print(f"TIMER LOG: ---> Entering {func_name}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"TIMER LOG: <--- Exiting  {func_name} (Duration: {duration:.4f} seconds)")
        return result
    return wrapper

# --- Core Functions ---

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
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: RateLimitError for '{func_name}'. Retrying in {delay:.2f}s (Attempt {attempt+1}/{retries}). Error: {rle}")
            time.sleep(delay)
        except Exception as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0,1)
            print(f"TIMER LOG: API Error for '{func_name}' (Attempt {attempt+1}/{retries}). Retrying in {delay:.2f}s. Error: {type(e).__name__}: {e}")
            time.sleep(delay)
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries exceeded for '{func_name}'. Last error: {type(e).__name__}: {e}")
                raise
    raise RuntimeError(f"Max retries exceeded for {func_name} after multiple attempts.")

@time_it
def analyze_page(page, llm_text, llm_vision, fence_keywords):
    page_num = page.get('page_number', 'N/A')
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Starting analysis.")
    text = page["text"]
    text_response_content = None
    text_snippet = extract_snippet(text, fence_keywords)
    text_found = False
    vision_found = False
    vision_response_content = None

    text_prompt = f"""Analyze the following text from an engineering drawing page.
Does this text contain any information, descriptions, or specifications related to fences, fencing, gates, barriers, or guardrails?
Consider any mention of these terms or types of these structures.
Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning. Text: {text}"""
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Preparing for text LLM call.")
    try:
        response_obj = retry_with_backoff(llm_text.invoke, [HumanMessage(content=text_prompt)])
        text_response_content = response_obj.content
        text_found = is_positive_response(text_response_content)
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Text LLM call successful.")
    except Exception as e:
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Error during text analysis LLM call: {type(e).__name__}: {e}")
        text_response_content = f"Error in text analysis: {e}"

    if not text_found and llm_vision:
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Text did not confirm fence. Preparing for vision LLM call.")
        image_url = f"data:image/png;base64,{page['image_b64']}"
        vision_prompt_messages = [ HumanMessage(content=[ {"type": "text", "text": "Analyze this engineering drawing image. Does it visually depict any fences, fencing structures, gates, or barriers? Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning."}, {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}} ]) ]
        try:
            response_obj = retry_with_backoff(llm_vision.invoke, vision_prompt_messages)
            vision_response_content = response_obj.content
            vision_found = is_positive_response(vision_response_content)
            print(f"TIMER LOG: (analyze_page) Page {page_num} - Vision LLM call successful.")
        except Exception as e:
            print(f"TIMER LOG: (analyze_page) Page {page_num} - Error during vision analysis LLM call: {type(e).__name__}: {e}")
            vision_response_content = f"Error in vision analysis: {e}"

    fence_found_overall = text_found or vision_found
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Analysis complete. Fence found: {fence_found_overall}.")
    return {
        "page_number": page_num, "fence_found": fence_found_overall, "text_found": text_found,
        "vision_found": vision_found, "text_response": text_response_content,
        "vision_response": vision_response_content, "text_snippet": text_snippet, "image": page["image_bytes"]
    }

# Optional: Function to estimate token count if you install tiktoken
# def count_tokens(text, model_name="gpt-3.5-turbo"):
#     try:
#         encoding = tiktoken.encoding_for_model(model_name)
#         return len(encoding.encode(text))
#     except Exception:
#         return len(text) // 4 # Fallback rough estimate

# utils.py
@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app, selected_llm_model_name="gpt-3.5-turbo"):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting REFINED TWO-PASS (Lines for Legends, Words for Indicators) - v4 (Robust ID).")
    overall_gfrtb_start_time = time.time()
    
    page_width, page_height = 0, 0
    final_highlight_boxes_list = [] # Use a list name to avoid confusion with the var name inside loops
    words = [] 

    try:
        pdfplumber_parse_start_time = time.time()
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            print(f"TIMER LOG: (get_fence_related_text_boxes) pdfplumber.open took {time.time() - pdfplumber_parse_start_time:.4f}s.")
            if not pdf.pages: return [], 0, 0
            page = pdf.pages[0]
            page_width, page_height = page.width, page.height
            if not page_width or not page_height: return [], 0, 0

            word_extraction_start_time = time.time()
            try:
                words = page.extract_words(use_text_flow=True, split_at_punctuation=False, 
                                           x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
                print(f"TIMER LOG: (get_fence_related_text_boxes) pdfplumber.extract_words found {len(words)} words (took {time.time() - word_extraction_start_time:.4f}s).")
            except Exception as e:
                 print(f"TIMER LOG: (get_fence_related_text_boxes) Error during page.extract_words: {type(e).__name__}: {e}")

            extracted_lines_start_time = time.time()
            extracted_lines = []
            try:
                extracted_lines = page.extract_text_lines(layout=True, use_text_flow=True, strip=True, return_chars=False)
                print(f"TIMER LOG: (get_fence_related_text_boxes) pdfplumber.extract_text_lines found {len(extracted_lines)} lines (took {time.time() - extracted_lines_start_time:.4f}s).")
            except Exception as e_lines:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Error during page.extract_text_lines: {type(e_lines).__name__}: {e_lines}")

            # --- PASS 1: Legend Identification using Text Lines ---
            print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Pass 1: Legend Identification.")
            pass1_logic_start_time = time.time()
            candidate_legend_lines_for_llm1 = []
            if extracted_lines:
                for line_idx, line_obj in enumerate(extracted_lines):
                    line_text = line_obj.get('text', '').strip()
                    if not line_text: continue
                    line_text_lower = line_text.lower()
                    has_fence_keyword = any(kw in line_text_lower for kw in fence_keywords_from_app)
                    has_drawing_term = any(term in line_text_lower for term in ["detail", "type", "schedule", "item", "legend", "notes", "spec", "assy", "section", "elevation", "matl", "constr", "view", "plan", "typical", "standard"])
                    starts_like_identifier = bool(re.match(r"^\s*\(?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\)?[\s:.)\-]", line_text))
                    if (has_fence_keyword or has_drawing_term or starts_like_identifier) and \
                       (1 < len(line_text.split()) < 60) and len(line_text) < 400:
                        if all(k in line_obj for k in ['x0', 'top', 'x1', 'bottom']):
                            candidate_legend_lines_for_llm1.append({
                                "id": f"line_{line_idx}", "text": line_text,
                                "x0": round(line_obj['x0'], 2), "y0": round(line_obj['top'], 2),
                                "x1": round(line_obj['x1'], 2), "y1": round(line_obj['bottom'], 2)
                            })
            print(f"TIMER LOG: (get_fence_related_text_boxes) Pre-filtering lines for Pass 1. Found {len(candidate_legend_lines_for_llm1)} candidates.")

            identified_legends_from_pass1_llm_output = [] 
            confirmed_legend_core_ids = set()  
            processed_legends_for_pass2_prompt_context = [] 

            if candidate_legend_lines_for_llm1:
                lines_json_str_pass1 = json.dumps(candidate_legend_lines_for_llm1, separators=(',', ':'))
                prompt_pass1 = f"""You are an engineering drawing analyst.
You are provided with a JSON list of TEXT LINES extracted from a drawing page (width {page_width:.2f}, height {page_height:.2f}). Each line has "id", "text", and coordinates.
Fence-related keywords: {', '.join(fence_keywords_from_app)}.
Your task is to:
1.  Identify lines representing **Fence Legend Items**.
2.  For each, determine its **`core_identifier_text`** (e.g., "1" from "1. TYPE A FENCE", "F1" from "FENCE TYPE F1"). If no tag, use "N/A_DESC".
Output ONLY a single valid JSON object: {{"identified_legends": [{{"id": "line_id", "full_legend_text": "text", "core_identifier_text": "id_val"}}]}}.
If no legends, return {{"identified_legends": []}}. Adhere strictly to JSON.
Input Text Lines:
{lines_json_str_pass1}""" # Slightly shortened prompt for brevity if needed, but keep essentials.
                response_content_pass1 = ""
                try:
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Initiating Pass 1 Text LLM call for legends ({len(candidate_legend_lines_for_llm1)} lines).")
                    response_obj_pass1 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass1)])
                    response_content_pass1 = response_obj_pass1.content
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 1 LLM call completed (took {time.time() - pass1_logic_start_time:.4f}s).")
                    
                    clean_resp_pass1 = response_content_pass1.strip()
                    match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_pass1, re.IGNORECASE)
                    if match_json_block: clean_resp_pass1 = match_json_block.group(1).strip()
                    else:
                        first_brace = clean_resp_pass1.find('{'); last_brace = clean_resp_pass1.rfind('}')
                        if first_brace != -1 and last_brace > first_brace: clean_resp_pass1 = clean_resp_pass1[first_brace : last_brace+1]
                    
                    parsed_data_pass1 = json.loads(clean_resp_pass1)
                    identified_legends_from_pass1_llm_output = parsed_data_pass1.get("identified_legends", [])
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 1 Parsing. Found {len(identified_legends_from_pass1_llm_output)} legend items from LLM.")

                    if identified_legends_from_pass1_llm_output:
                        temp_legend_map = {item['id']: item for item in candidate_legend_lines_for_llm1} 
                        for legend_data_llm in identified_legends_from_pass1_llm_output:
                            if not isinstance(legend_data_llm, dict): continue
                            
                            original_line_id = legend_data_llm.get("id")
                            if not original_line_id: # LLM MUST return the ID
                                print("TIMER LOG: (get_fence_related_text_boxes) Warning: LLM Pass 1 legend item missing 'id'. Skipping.")
                                continue
                                
                            original_line_obj_from_map = temp_legend_map.get(original_line_id)
                            if not original_line_obj_from_map:
                                print(f"TIMER LOG: (get_fence_related_text_boxes) Warning: LLM Pass 1 returned ID '{original_line_id}' not in candidate lines.")
                                continue

                            legend_text_to_use = legend_data_llm.get("full_legend_text", original_line_obj_from_map.get("text","")) 
                            core_id = legend_data_llm.get("core_identifier_text")

                            if core_id and isinstance(core_id, str) and core_id.strip() and core_id != "N/A_DESC":
                                id_match = re.match(r"([A-Za-z0-9]+)", core_id.strip())
                                if id_match: confirmed_legend_core_ids.add(id_match.group(1))
                                else: confirmed_legend_core_ids.add(core_id.strip().rstrip(".:"))
                            
                            # Ensure all components for highlighting are present
                            if all(original_line_obj_from_map.get(k) is not None for k in ['text', 'x0', 'y0', 'x1', 'y1']):
                                final_highlight_boxes_list.append({
                                    'id': original_line_id, # Use the ID from the input line_obj
                                    'text': legend_text_to_use, 
                                    'x0': original_line_obj_from_map['x0'], 'y0': original_line_obj_from_map['y0'],
                                    'x1': original_line_obj_from_map['x1'], 'y1': original_line_obj_from_map['y1'],
                                    'type_from_llm': "legend_item", 'tag_from_llm': core_id
                                })
                                processed_legends_for_pass2_prompt_context.append(legend_data_llm)
                            else:
                                print(f"TIMER LOG: (get_fence_related_text_boxes) Warning: Original line object for ID '{original_line_id}' missing data for highlighting.")
                except Exception as e_p1:
                    print(f"TIMER LOG: (get_fence_related_text_boxes) Error in Pass 1 (LLM/Parsing): {type(e_p1).__name__}: {e_p1}")
                    if response_content_pass1: print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 1 Response before error: {response_content_pass1[:500]}")
            
            print(f"TIMER LOG: (get_fence_related_text_boxes) After Pass 1, unique core legend identifiers for Pass 2: {confirmed_legend_core_ids}")

            # --- PASS 2: Indicator Identification using Words, guided by Pass 1 IDs ---
            if confirmed_legend_core_ids and words: 
                print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Pass 2: Indicator Identification from {len(words)} Words.")
                pass2_logic_start_time = time.time() # Renamed to avoid conflict
                
                candidate_indicator_words_for_llm2 = []
                for idx, word_obj in enumerate(words):
                    text_val = word_obj['text'].strip()
                    if not text_val: continue
                    core_word_text_match = re.match(r"^\s*\(?\s*([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\s*\)?\.?\s*$", text_val)
                    core_word_text = core_word_text_match.group(1) if core_word_text_match else text_val

                    if core_word_text in confirmed_legend_core_ids:
                        if 0 < len(text_val) <= 5: 
                            is_already_legend_part = False
                            for leg_box_dict in final_highlight_boxes_list: # Check against legends already added
                                if leg_box_dict.get('type_from_llm') == 'legend_item':
                                    # Check if word is spatially within the legend box
                                    lx0, ly0, lx1, ly1 = leg_box_dict.get('x0'), leg_box_dict.get('y0'), leg_box_dict.get('x1'), leg_box_dict.get('y1')
                                    if all(v is not None for v in [lx0, ly0, lx1, ly1]): # Ensure coords exist
                                        word_center_x = (word_obj['x0'] + word_obj['x1']) / 2
                                        word_center_y = (word_obj['top'] + word_obj['bottom']) / 2
                                        if lx0 <= word_center_x <= lx1 and ly0 <= word_center_y <= ly1:
                                            is_already_legend_part = True
                                            break
                            if not is_already_legend_part:
                                candidate_indicator_words_for_llm2.append({
                                    "id": f"word_{idx}", "text": text_val, "core_text_matched": core_word_text,
                                    "x0": round(word_obj['x0'], 2), "y0": round(word_obj['top'], 2),
                                    "x1": round(word_obj['x1'], 2), "y1": round(word_obj['bottom'], 2)
                                })
                
                print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 2: Found {len(candidate_indicator_words_for_llm2)} candidate indicators for LLM.")

                if candidate_indicator_words_for_llm2:
                    indicators_json_str_pass2 = json.dumps(candidate_indicator_words_for_llm2, separators=(',',':'))
                    pass1_legend_context_for_pass2_prompt = []
                    for leg_detail in processed_legends_for_pass2_prompt_context:
                        leg_text_snippet = leg_detail.get("full_legend_text", "")[:70] + "..."
                        pass1_legend_context_for_pass2_prompt.append({
                            "identifier": leg_detail.get("core_identifier_text"),
                            "description_snippet": leg_text_snippet
                        })
                    confirmed_legends_context_str_for_prompt = json.dumps(pass1_legend_context_for_pass2_prompt, separators=(',',':'))

                    prompt_pass2 = f"""You are an engineering drawing analyst.
Context: These Fence Legend Items and their core identifiers were previously identified:
{confirmed_legends_context_str_for_prompt}

Now, you are given a new JSON list of CANDIDATE INDICATOR text elements. Their 'core_text_matched' value matches one of the legend identifiers above.
Your task is to determine if each candidate is TRULY acting as a standalone graphical callout/indicator.
A true indicator is usually: short, spatially distinct from legend blocks, and appears to label a part of the drawing.
It is NOT part of a longer sentence, dimension string, or title block.

Input Candidate Indicators:
{indicators_json_str_pass2}

Output ONLY a single valid JSON object with one key: "confirmed_indicators".
Value is a list of objects, each with "id", "matched_legend_identifier", "text_content".
Example: {{"confirmed_indicators": [{{"id": "word_102", "matched_legend_identifier": "F1", "text_content": "F1"}}]}}
If none, return {{"confirmed_indicators": []}}. Strictly JSON.
"""
                    response_content_pass2 = ""
                    try:
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Initiating Pass 2 Text LLM call for indicators ({len(candidate_indicator_words_for_llm2)} candidates).")
                        response_obj_pass2 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass2)])
                        response_content_pass2 = response_obj_pass2.content
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 2 LLM call completed (took {time.time() - pass2_logic_start_time:.4f}s).")
                        
                        clean_resp_pass2 = response_content_pass2.strip()
                        match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_pass2, re.IGNORECASE)
                        if match_json_block: clean_resp_pass2 = match_json_block.group(1).strip()
                        else:
                            first_brace = clean_resp_pass2.find('{'); last_brace = clean_resp_pass2.rfind('}')
                            if first_brace != -1 and last_brace > first_brace: clean_resp_pass2 = clean_resp_pass2[first_brace : last_brace+1]

                        parsed_data_pass2 = json.loads(clean_resp_pass2)
                        confirmed_indicators_from_llm = parsed_data_pass2.get("confirmed_indicators", [])
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 2 Parsing. Found {len(confirmed_indicators_from_llm)} confirmed indicators from LLM.")

                        original_pass2_candidates_map = {item['id']: item for item in candidate_indicator_words_for_llm2}
                        for ind_data in confirmed_indicators_from_llm:
                            if not isinstance(ind_data, dict): continue
                            el_id = ind_data.get("id")
                            if not el_id: # LLM MUST return the ID for indicators too
                                print("TIMER LOG: (get_fence_related_text_boxes) Warning: LLM Pass 2 indicator item missing 'id'. Skipping.")
                                continue

                            original_box_data = original_pass2_candidates_map.get(el_id)
                            if original_box_data and all(original_box_data.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({
                                    'id': el_id, # Use the ID from the candidate word
                                    'text': original_box_data['text'], 
                                    'x0': original_box_data['x0'], 'y0': original_box_data['y0'],
                                    'x1': original_box_data['x1'], 'y1': original_box_data['y1'],
                                    'type_from_llm': "indicator", 
                                    'tag_from_llm': ind_data.get("matched_legend_identifier")
                                })
                            else:
                                print(f"TIMER LOG: (get_fence_related_text_boxes) Warning: Original candidate word for ID '{el_id}' not found or missing data.")

                    except Exception as e_p2:
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Error in Pass 2 LLM/Parsing: {type(e_p2).__name__}: {e_p2}")
                        if response_content_pass2: print(f"TIMER LOG: (get_fence_related_text_boxes) Pass 2 Response before error: {response_content_pass2[:500]}")
        
        # Deduplicate final_highlight_boxes_list by 'id'
        dedup_map_final = {}
        for item in final_highlight_boxes_list:
            item_id = item.get('id') 
            if item_id: # Ensure item_id is not None and is a valid key
                dedup_map_final[item_id] = item 
        final_highlight_boxes_list = list(dedup_map_final.values())


        print(f"TIMER LOG: (get_fence_related_text_boxes) Finished. Returning {len(final_highlight_boxes_list)} boxes. Total time: {time.time() - overall_gfrtb_start_time:.4f}s.")
        return final_highlight_boxes_list, page_width, page_height

    except Exception as e_outer:
        print(f"TIMER LOG: (get_fence_related_text_boxes) CRITICAL Outer Error: {type(e_outer).__name__}: {e_outer}. Time: {time.time() - overall_gfrtb_start_time:.4f}s.")
        # Return the list collected so far, even if an error occurred.
        # Ensure deduplication if it hasn't run.
        if 'final_highlight_boxes_list' in locals() and isinstance(final_highlight_boxes_list, list):
            dedup_map_final_except = {item.get('id'): item for item in final_highlight_boxes_list if item.get('id')}
            return list(dedup_map_final_except.values()), page_width, page_height
        return [], page_width, page_height # Fallback