# utils.py 
# (This is the same as the V7 "Few-Shot Prompts" version previously provided)

import time
import functools
import fitz  # PyMuPDF
import base64
import re
import random 
from langchain_core.messages import HumanMessage
from openai import RateLimitError, APIError, APITimeoutError
import pdfplumber
from io import BytesIO
import json

# Custom Exception for unrecoverable rate limit
class UnrecoverableRateLimitError(Exception):
    pass

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
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (RateLimitError). Last error: {rle}")
                raise UnrecoverableRateLimitError(f"OpenAI API rate limit. Please try later. (Details: {rle})")
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: RateLimitError for '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {rle}")
            time.sleep(delay)
        except (APIError, APITimeoutError) as apie:
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries for '{func_name}' (APIError/Timeout). Last error: {apie}")
                raise # Re-raise original or a custom one summarizing this
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
def analyze_page(page_data_for_analysis, llm_text, llm_vision, fence_keywords):
    page_num = page_data_for_analysis.get('page_number', 'N/A')
    text_content = page_data_for_analysis.get("text", "")
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Starting analysis.")
    
    text_response_content = None
    text_snippet = extract_snippet(text_content, fence_keywords)
    text_found = False
    vision_found = False
    vision_response_content = None

    text_prompt = f"""Analyze the following text from an engineering drawing page.
Does this text contain any information, descriptions, or specifications related to fences, fencing, gates, barriers, or guardrails?
Consider any mention of these terms or types of these structures.
Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning. Text: {text_content}"""
    
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Preparing for text LLM call.")
    try:
        response_obj = retry_with_backoff(llm_text.invoke, [HumanMessage(content=text_prompt)])
        text_response_content = response_obj.content
        text_found = is_positive_response(text_response_content)
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Text LLM call successful.")
    except UnrecoverableRateLimitError: raise
    except Exception as e:
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Error during text analysis LLM call: {e}")
        text_response_content = f"Error in text analysis: {e}"

    image_b64_for_vision = page_data_for_analysis.get("image_b64")
    if llm_vision and image_b64_for_vision and (not text_found or page_data_for_analysis.get("force_vision", False)):
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Preparing for vision LLM call.")
        image_url = f"data:image/png;base64,{image_b64_for_vision}"
        vision_prompt_messages = [ HumanMessage(content=[ {"type": "text", "text": "Analyze this engineering drawing image. Does it visually depict any fences, fencing structures, gates, or barriers? Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning."}, {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}} ]) ]
        try:
            response_obj_vision = retry_with_backoff(llm_vision.invoke, vision_prompt_messages)
            vision_response_content = response_obj_vision.content
            vision_found = is_positive_response(vision_response_content)
            print(f"TIMER LOG: (analyze_page) Page {page_num} - Vision LLM call successful.")
        except UnrecoverableRateLimitError: raise
        except Exception as e_vision:
            print(f"TIMER LOG: (analyze_page) Page {page_num} - Error during vision analysis LLM call: {e_vision}")
            vision_response_content = f"Error in vision analysis: {e_vision}"

    fence_found_overall = text_found or vision_found
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Analysis complete. Fence found: {fence_found_overall}.")
    
    return {
        "page_number": page_num, "fence_found": fence_found_overall, "text_found": text_found,
        "vision_found": vision_found, "text_response": text_response_content,
        "vision_response": vision_response_content, "text_snippet": text_snippet
    }


@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app, selected_llm_model_name="gpt-3.5-turbo"):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting REFINED TWO-PASS - v7 (Few-Shot Prompts).")
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
                words = page_obj.extract_words(use_text_flow=True, split_at_punctuation=False, x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
            except Exception as e: print(f"TIMER LOG: (get_fence_related_text_boxes) Error page_obj.extract_words: {e}")
            extracted_lines = []
            try:
                extracted_lines = page_obj.extract_text_lines(layout=True, use_text_flow=True, strip=True, return_chars=False)
            except Exception as e_lines: print(f"TIMER LOG: (get_fence_related_text_boxes) Error page_obj.extract_text_lines: {e_lines}")
            candidate_legend_lines_for_llm1 = []
            if extracted_lines:
                for line_idx, line_obj_data in enumerate(extracted_lines):
                    line_text = line_obj_data.get('text', '').strip()
                    if not line_text: continue
                    line_text_lower = line_text.lower()
                    has_fence_keyword = any(kw in line_text_lower for kw in fence_keywords_from_app)
                    is_candidate = has_fence_keyword or \
                                   any(term in line_text_lower for term in ["detail", "type", "schedule", "item", "legend", "notes", "spec", "assy", "section", "elevation", "matl", "constr", "view", "plan", "typical", "standard", "description", "callout"]) or \
                                   bool(re.match(r"^\s*\(?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\)?[\s:.)\-]", line_text))
                    if is_candidate and (1 < len(line_text.split()) < 80) and len(line_text) < 500:
                        if all(k in line_obj_data for k in ['x0', 'top', 'x1', 'bottom']):
                            candidate_legend_lines_for_llm1.append({"id": f"line_{line_idx}", "text": line_text, "x0": round(line_obj_data['x0'],2), "y0": round(line_obj_data['top'],2), "x1": round(line_obj_data['x1'],2), "y1": round(line_obj_data['bottom'],2)})
            print(f"TIMER LOG: Pass 1: Found {len(candidate_legend_lines_for_llm1)} candidate lines.")
            identified_legends_from_pass1_llm_output, confirmed_legend_core_ids, processed_legends_for_pass2_prompt_context = [], set(), []
            if candidate_legend_lines_for_llm1:
                lines_json_str_pass1 = json.dumps(candidate_legend_lines_for_llm1, separators=(',',':'))
                pass1_examples = """
Example 1 Input Line: {"id": "line_23", "text": "1. 6' HIGH CHAIN LINK FENCE, TYP.", ...}
Example 1 Output: {"id": "line_23", "full_text": "1. 6' HIGH CHAIN LINK FENCE, TYP.", "core_identifier_text": "1", "type": "legend_item"}
Example 2 Input Line: {"id": "line_45", "text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS.", ...}
Example 2 Output: {"id": "line_45", "full_text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS.", "core_identifier_text": "FENCE_POST_FOOTING_NOTE", "type": "note"}
Example 3 Input Line: {"id": "line_67", "text": "TYPICAL WILDLIFE FENCE INSTALLATION", ...}
Example 3 Output: {"id": "line_67", "full_text": "TYPICAL WILDLIFE FENCE INSTALLATION", "core_identifier_text": "WILDLIFE_FENCE_INSTALL_DETAIL_TITLE", "type": "description"}
Example 4 Input Line: {"id": "line_88", "text": "FENCE TYPE F2: WROUGHT IRON, SEE DETAIL A/S-502", ...}
Example 4 Output: {"id": "line_88", "full_text": "FENCE TYPE F2: WROUGHT IRON, SEE DETAIL A/S-502", "core_identifier_text": "F2", "type": "legend_item"}"""
                prompt_pass1 = f"""You are an engineering drawing analyst. Provided JSON list of TEXT LINES (page width {page_width:.2f}, height {page_height:.2f}). Fence keywords: {', '.join(fence_keywords_from_app)}.
Task: 1. Identify lines for Fence Legend Items, Specs, Notes/Descriptions. Be inclusive. 2. Determine `core_identifier_text`: Priority 1: Clear prefix tag (e.g., "F1.", "1.", "NOTE 3:") -> extract tag ("F1", "1", "NOTE 3"), remove trailing punctuation. Priority 2: Descriptive text -> short, unique, ALL_CAPS_SNAKE_CASE summary (e.g., "FENCE_HEIGHT_SPEC"). Priority 3: Generic -> "N/A_DESC". 3. Provide `type`: "legend_item", "specification", "note", or "description".
Examples: {pass1_examples}
Output ONLY a single valid JSON: {{"identified_fences": [{{...output...}}]}}. 'id' must match input 'id'. 'full_text' original text. If none, {{"identified_fences": []}}. Strictly JSON. Input Lines: {lines_json_str_pass1}"""
                response_content_pass1 = ""
                try:
                    response_obj_pass1 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass1)])
                    response_content_pass1 = response_obj_pass1.content
                    clean_resp_pass1 = response_content_pass1.strip()
                    match_json_block = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_pass1,re.IGNORECASE)
                    if match_json_block: clean_resp_pass1 = match_json_block.group(1).strip()
                    else: first_brace=clean_resp_pass1.find('{'); last_brace=clean_resp_pass1.rfind('}'); clean_resp_pass1=clean_resp_pass1[first_brace:last_brace+1] if first_brace!=-1 and last_brace>first_brace else clean_resp_pass1
                    parsed_data_pass1 = json.loads(clean_resp_pass1)
                    identified_legends_from_pass1_llm_output = parsed_data_pass1.get("identified_fences", [])
                    print(f"TIMER LOG: Pass 1 LLM. Found {len(identified_legends_from_pass1_llm_output)} items.")
                    if identified_legends_from_pass1_llm_output:
                        temp_legend_map = {item['id']: item for item in candidate_legend_lines_for_llm1} 
                        for legend_data_llm in identified_legends_from_pass1_llm_output:
                            if not isinstance(legend_data_llm, dict): continue
                            original_line_id = legend_data_llm.get("id")
                            if not original_line_id: continue
                            original_line_obj_from_map = temp_legend_map.get(original_line_id)
                            if not original_line_obj_from_map: continue
                            item_text_to_use = legend_data_llm.get("full_text", original_line_obj_from_map.get("text","")) 
                            core_id_from_llm = legend_data_llm.get("core_identifier_text")
                            item_type = legend_data_llm.get("type", "unknown") 
                            core_id_processed = ""
                            if core_id_from_llm and isinstance(core_id_from_llm, str) and core_id_from_llm.strip() and core_id_from_llm != "N/A_DESC":
                                core_id_processed = core_id_from_llm.strip().upper(); core_id_processed = re.sub(r"[.:\s]+$", "", core_id_processed) 
                                confirmed_legend_core_ids.add(core_id_processed)
                            if all(original_line_obj_from_map.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({'id':original_line_id,'text':item_text_to_use,'x0':original_line_obj_from_map['x0'],'y0':original_line_obj_from_map['y0'],'x1':original_line_obj_from_map['x1'],'y1':original_line_obj_from_map['y1'],'type_from_llm':item_type,'tag_from_llm':core_id_processed if core_id_processed else core_id_from_llm})
                                processed_legends_for_pass2_prompt_context.append({"id":original_line_id,"core_identifier_text":core_id_processed if core_id_processed else core_id_from_llm,"full_text":item_text_to_use,"type":item_type})
                except UnrecoverableRateLimitError: raise
                except Exception as e_p1: print(f"TIMER LOG: Error Pass 1 LLM/Parse: {e_p1}"); print(f"TIMER LOG: Pass 1 Resp: {response_content_pass1[:500]}")
            print(f"TIMER LOG: After Pass 1, core_ids for Pass 2: {confirmed_legend_core_ids}")
            if confirmed_legend_core_ids and words:
                candidate_indicator_words_for_llm2 = []
                for idx, word_obj_data in enumerate(words):
                    text_val = word_obj_data['text'].strip()
                    if not text_val: continue
                    core_word_text_match = re.match(r"^\s*\(?\s*([A-Za-z0-9]+(?:[\.\-_][A-Za-z0-9]+)*)\s*\)?\.?\s*$", text_val)
                    raw_core_word_text = core_word_text_match.group(1) if core_word_text_match else text_val
                    core_word_text_for_match = raw_core_word_text.upper(); core_word_text_for_match = re.sub(r"[.:\s]+$", "", core_word_text_for_match)
                    if core_word_text_for_match in confirmed_legend_core_ids and 0 < len(text_val) <= 15:
                        is_already_part = False
                        for id_item_box in final_highlight_boxes_list:
                            if id_item_box.get('tag_from_llm','').upper()==core_word_text_for_match or text_val.upper() in id_item_box.get('text','').upper():
                                lx0,ly0,lx1,ly1 = id_item_box.get('x0'),id_item_box.get('y0'),id_item_box.get('x1'),id_item_box.get('y1')
                                if all(v is not None for v in [lx0,ly0,lx1,ly1]):
                                    wcx,wcy=(word_obj_data['x0']+word_obj_data['x1'])/2, (word_obj_data['top']+word_obj_data['bottom'])/2
                                    if (lx0-2)<=wcx<=(lx1+2) and (ly0-2)<=wcy<=(ly1+2): is_already_part=True; break
                        if not is_already_part: candidate_indicator_words_for_llm2.append({"id":f"word_{idx}","text":text_val,"core_text_matched":core_word_text_for_match,"x0":round(word_obj_data['x0'],2),"y0":round(word_obj_data['top'],2),"x1":round(word_obj_data['x1'],2),"y1":round(word_obj_data['bottom'],2)})
                print(f"TIMER LOG: Pass 2: Found {len(candidate_indicator_words_for_llm2)} candidate indicators.")
                if candidate_indicator_words_for_llm2:
                    indicators_json_str_pass2 = json.dumps(candidate_indicator_words_for_llm2, separators=(',',':'))
                    pass1_ctx_json_pass2 = [{'identifier':d.get('core_identifier_text'),'description_snippet':d.get('full_text','')[:70]+('...' if len(d.get('full_text',''))>70 else ''),'type':d.get('type')} for d in processed_legends_for_pass2_prompt_context]
                    ctx_str_prompt = json.dumps(pass1_ctx_json_pass2, separators=(',',':'))
                    pass2_examples = """
Example 1 Input: {"id":"word_102","text":"F1","core_text_matched":"F1",...}; Output: {"id":"word_102","matched_legend_identifier":"F1","text_content":"F1"}
Example 2 Input: {"id":"word_105","text":"1","core_text_matched":"1",...}; Output: {"id":"word_105","matched_legend_identifier":"1","text_content":"1"}
Example 3 Input: {"id":"word_10","text":"Fence","core_text_matched":"FENCE_POST_FOOTING_NOTE",...}; Output: [] (part of note, not indicator)
Example 4 Input: {"id":"word_500","text":"SEE","core_text_matched":"F2",...}; Output: [] (part of legend item desc, not indicator of "F2")"""
                    prompt_pass2 = f"""You are an engineering drawing analyst. Context: Fence-related items (legends, specs, notes) & core IDs previously ID'd: {ctx_str_prompt}.
Given CANDIDATE INDICATOR text elements (their 'core_text_matched' matches a context ID). Task: Determine if truly standalone graphical callout/indicator. True indicator: Short, spatially distinct, labels part of drawing, NOT part of longer sentence/dimension or main body of legend/spec from context.
Examples (output for 'confirmed_indicators' list): {pass2_examples}
Input Candidates: {indicators_json_str_pass2}
Output ONLY single valid JSON: {{"confirmed_indicators": [{{...}}]}}. If candidate NOT true indicator, DON'T include. If none, {{"confirmed_indicators": []}}. Strictly JSON."""
                    response_content_pass2 = ""
                    try:
                        response_obj_pass2 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass2)])
                        response_content_pass2 = response_obj_pass2.content
                        clean_resp_pass2=response_content_pass2.strip()
                        match_json_block=re.search(r"```json\s*([\s\S]*?)\s*```",clean_resp_pass2,re.IGNORECASE)
                        if match_json_block:clean_resp_pass2=match_json_block.group(1).strip()
                        else:first_brace=clean_resp_pass2.find('{');last_brace=clean_resp_pass2.rfind('}');clean_resp_pass2=clean_resp_pass2[first_brace:last_brace+1] if first_brace!=-1 and last_brace > first_brace else clean_resp_pass2
                        parsed_data_pass2 = json.loads(clean_resp_pass2)
                        confirmed_indicators_from_llm = parsed_data_pass2.get("confirmed_indicators", [])
                        print(f"TIMER LOG: Pass 2 LLM. Found {len(confirmed_indicators_from_llm)} confirmed indicators.")
                        original_pass2_candidates_map = {item['id']: item for item in candidate_indicator_words_for_llm2}
                        for ind_data in confirmed_indicators_from_llm:
                            if not isinstance(ind_data,dict):continue
                            el_id=ind_data.get("id")
                            if not el_id: continue
                            original_box_data = original_pass2_candidates_map.get(el_id)
                            if original_box_data and all(original_box_data.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({'id':el_id,'text':original_box_data['text'],'x0':original_box_data['x0'],'y0':original_box_data['y0'],'x1':original_box_data['x1'],'y1':original_box_data['y1'],'type_from_llm':"indicator",'tag_from_llm':ind_data.get("matched_legend_identifier")})
                    except UnrecoverableRateLimitError: raise
                    except Exception as e_p2: print(f"TIMER LOG: Error Pass 2 LLM/Parse: {e_p2}"); print(f"TIMER LOG: Pass 2 Resp: {response_content_pass2[:500]}")
        dedup_map_final = {}
        for item in final_highlight_boxes_list: item_id=item.get('id'); dedup_map_final[item_id]=item if item_id else dedup_map_final.get(str(item), item) # basic fallback for missing id
        final_highlight_boxes_list = list(dedup_map_final.values())
        print(f"TIMER LOG: (get_fence_related_text_boxes) Finished. Returning {len(final_highlight_boxes_list)} boxes. Total: {time.time()-overall_gfrtb_start_time:.4f}s.")
        return final_highlight_boxes_list, page_width, page_height
    except Exception as e_outer:
        print(f"TIMER LOG: (get_fence_related_text_boxes) CRITICAL Outer Error: {e_outer}. Time: {time.time()-overall_gfrtb_start_time:.4f}s.")
        if 'final_highlight_boxes_list' in locals() and isinstance(final_highlight_boxes_list,list):
            dedup_map_final_except={item.get('id'):item for item in final_highlight_boxes_list if item.get('id')}
            return list(dedup_map_final_except.values()),page_width,page_height
        return [],page_width,page_height