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
        if match: return match.group(0)
    return None

def is_positive_response(response: str) -> bool:
    response = response.strip().lower()
    return response.startswith("yes") or response.startswith('"yes')

@time_it
def retry_with_backoff(llm_invoke_method, messages_list, retries=5, base_delay=2):
    func_name = llm_invoke_method.__qualname__ if hasattr(llm_invoke_method, '__qualname__') else "llm_invoke"
    for attempt in range(retries):
        try: return llm_invoke_method(messages_list)
        except RateLimitError as rle:
            if attempt == retries - 1: print(f"TIMER LOG: Max retries '{func_name}' (RateLimitError): {rle}"); raise UnrecoverableRateLimitError(f"OpenAI API rate limit. Try later. (Details: {rle})")
            delay = base_delay * (2**attempt) + random.uniform(0,1); print(f"TIMER LOG: RateLimitError '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {rle}"); time.sleep(delay)
        except (APIError, APITimeoutError) as apie:
            if attempt == retries - 1: print(f"TIMER LOG: Max retries '{func_name}' (APIError/Timeout): {apie}"); raise
            delay = base_delay * (2**attempt) + random.uniform(0,1); print(f"TIMER LOG: API Error/Timeout '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {apie}"); time.sleep(delay)
        except Exception as e:
            if attempt == retries - 1: print(f"TIMER LOG: Max retries '{func_name}' (Unexpected): {e}"); raise
            delay = base_delay * (2**attempt) + random.uniform(0,1); print(f"TIMER LOG: Unexpected Error '{func_name}'. Retry {attempt+1}/{retries} in {delay:.2f}s. Error: {e}"); time.sleep(delay)
    raise RuntimeError(f"Exceeded max retries for {func_name}.")

@time_it
def analyze_page(page_data_for_analysis, llm_text, llm_vision, fence_keywords):
    page_num = page_data_for_analysis.get('page_number', 'N/A'); text_content = page_data_for_analysis.get("text", "")
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Starting analysis.")
    text_response_content, text_snippet, vision_response_content = None, extract_snippet(text_content, fence_keywords), None
    text_found, vision_found = False, False
    text_prompt = f"""Analyze text from an engineering drawing. Does it mention fences, fencing, gates, barriers, or guardrails? Answer 'Yes' or 'No', then explain. Text: {text_content}"""
    try:
        response_obj = retry_with_backoff(llm_text.invoke, [HumanMessage(content=text_prompt)])
        text_response_content = response_obj.content; text_found = is_positive_response(text_response_content)
    except UnrecoverableRateLimitError: raise
    except Exception as e: print(f"TIMER LOG: (analyze_page) Text LLM Error Pg {page_num}: {e}"); text_response_content = f"Error: {e}"
    image_b64 = page_data_for_analysis.get("image_b64")
    if llm_vision and image_b64 and (not text_found or page_data_for_analysis.get("force_vision", False)):
        vision_prompt_msg = [HumanMessage(content=[{"type":"text","text":"Analyze drawing image. Visual fences, gates, barriers? 'Yes'/'No', then explain."}, {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}","detail":"low"}}])]
        try:
            response_obj_vision = retry_with_backoff(llm_vision.invoke, vision_prompt_msg)
            vision_response_content = response_obj_vision.content; vision_found = is_positive_response(vision_response_content)
        except UnrecoverableRateLimitError: raise
        except Exception as e_vis: print(f"TIMER LOG: (analyze_page) Vision LLM Error Pg {page_num}: {e_vis}"); vision_response_content = f"Error: {e_vis}"
    fence_found_overall = text_found or vision_found
    return {"page_number":page_num, "fence_found":fence_found_overall, "text_found":text_found, "vision_found":vision_found, "text_response":text_response_content, "vision_response":vision_response_content, "text_snippet":text_snippet}

@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app, selected_llm_model_name="gpt-3.5-turbo"):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting TWO-PASS - Reverted to Precise Logic (like v4/v5).")
    overall_gfrtb_start_time = time.time()
    page_width, page_height = 0,0
    final_highlight_boxes_list, words, extracted_lines = [], [], []
    try:
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            if not pdf.pages: return [],0,0
            page_obj = pdf.pages[0]; page_width, page_height = page_obj.width, page_obj.height
            if not page_width or not page_height: return [],0,0
            try: words = page_obj.extract_words(use_text_flow=True,split_at_punctuation=False,x_tolerance=1.5,y_tolerance=1.5,keep_blank_chars=False)
            except Exception as e: print(f"TIMER LOG: Error extract_words: {e}")
            try: extracted_lines = page_obj.extract_text_lines(layout=True,use_text_flow=True,strip=True,return_chars=False)
            except Exception as e_l: print(f"TIMER LOG: Error extract_text_lines: {e_l}")
            
            candidate_legend_lines_for_llm1 = []
            if extracted_lines:
                for line_idx, line_data in enumerate(extracted_lines):
                    line_text = line_data.get('text','').strip()
                    if not line_text: continue
                    lt_lower = line_text.lower()
                    has_fk = any(kw in lt_lower for kw in fence_keywords_from_app)
                    has_dt = any(t in lt_lower for t in ["detail","type","schedule","item","legend","notes","spec","section","elevation","plan","view"])
                    starts_id = bool(re.match(r"^\s*\(?([A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*)\)?[\s:.)\-]", line_text))
                    if (has_fk or has_dt or starts_id) and (1 < len(line_text.split()) < 60) and len(line_text) < 400:
                        if all(k in line_data for k in ['x0','top','x1','bottom']):
                            candidate_legend_lines_for_llm1.append({"id":f"line_{line_idx}","text":line_text,"x0":round(line_data['x0'],2),"y0":round(line_data['top'],2),"x1":round(line_data['x1'],2),"y1":round(line_data['bottom'],2)})
            print(f"TIMER LOG: Pass 1: Found {len(candidate_legend_lines_for_llm1)} candidate legend lines.")
            
            identified_legends_pass1, confirmed_core_ids, processed_legends_ctx_pass2 = [], set(), []
            if candidate_legend_lines_for_llm1:
                lines_json_p1 = json.dumps(candidate_legend_lines_for_llm1, separators=(',',':'))
                # --- Reverted Pass 1 Prompt for higher precision ---
                prompt_pass1 = f"""You are an engineering drawing analyst.
Provided with a JSON list of TEXT LINES from a drawing page (width {page_width:.2f}, height {page_height:.2f}).
Fence-related keywords: {', '.join(fence_keywords_from_app)}.
Your task is to:
1. Identify lines that are clear **Fence Legend Items**. These usually have a distinct tag or identifier (e.g., "1.", "F-1", "TYPE A").
2. For each identified legend item, extract its **`core_identifier_text`** (e.g., "1" from "1. TYPE A FENCE", "F-1" from "DETAIL F-1 FENCE POST"). If no clear tag, use "N/A_LEGEND_ITEM".
Output ONLY a single valid JSON object: {{"identified_legends": [{{"id": "line_id_from_input", "full_legend_text": "original_text_of_line", "core_identifier_text": "extracted_tag"}}]}}.
If no such legend items are found, return {{"identified_legends": []}}. Strictly JSON.
Input Text Lines:
{lines_json_p1}"""
                resp_content_p1 = ""
                try:
                    resp_obj_p1 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass1)])
                    resp_content_p1 = resp_obj_p1.content
                    clean_resp_p1 = resp_content_p1.strip()
                    match_json = re.search(r"```json\s*([\s\S]*?)\s*```", clean_resp_p1, re.IGNORECASE)
                    if match_json: clean_resp_p1 = match_json.group(1).strip()
                    else: fb=clean_resp_p1.find('{'); lb=clean_resp_p1.rfind('}'); clean_resp_p1=clean_resp_p1[fb:lb+1] if fb!=-1 and lb>fb else clean_resp_p1
                    parsed_p1 = json.loads(clean_resp_p1)
                    identified_legends_pass1 = parsed_p1.get("identified_legends", [])
                    print(f"TIMER LOG: Pass 1 LLM. Found {len(identified_legends_pass1)} legend items.")
                    if identified_legends_pass1:
                        map_p1_candidates = {item['id']:item for item in candidate_legend_lines_for_llm1}
                        for legend_llm in identified_legends_pass1:
                            if not isinstance(legend_llm,dict): continue
                            orig_id = legend_llm.get("id")
                            if not orig_id: continue
                            orig_line_obj = map_p1_candidates.get(orig_id)
                            if not orig_line_obj: continue
                            text_use = legend_llm.get("full_legend_text", orig_line_obj.get("text",""))
                            core_id = legend_llm.get("core_identifier_text")
                            core_id_clean = ""
                            if core_id and isinstance(core_id,str) and core_id.strip() and core_id != "N/A_LEGEND_ITEM":
                                core_id_clean = core_id.strip().upper()
                                core_id_clean = re.sub(r"[.:\s]+$","",core_id_clean)
                                confirmed_core_ids.add(core_id_clean)
                            if all(orig_line_obj.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({'id':orig_id,'text':text_use,'x0':orig_line_obj['x0'],'y0':orig_line_obj['y0'],'x1':orig_line_obj['x1'],'y1':orig_line_obj['y1'],'type_from_llm':"legend_item",'tag_from_llm':core_id_clean if core_id_clean else core_id})
                                processed_legends_ctx_pass2.append({"identifier":core_id_clean if core_id_clean else core_id,"description_snippet":text_use[:70]})
                except UnrecoverableRateLimitError: raise
                except Exception as e_p1_proc: print(f"TIMER LOG: Error Pass 1 Proc: {e_p1_proc}"); print(f"TIMER LOG: Pass 1 Resp: {resp_content_p1[:500]}")
            print(f"TIMER LOG: After Pass 1, core_ids for Pass 2: {confirmed_core_ids}")
            
            if confirmed_core_ids and words:
                candidate_indicators_p2 = []
                for idx, word_data in enumerate(words):
                    text_val = word_data['text'].strip()
                    if not text_val: continue
                    match_core = re.match(r"^\s*\(?\s*([A-Za-z0-9]+(?:[\.\-_][A-Za-z0-9]+)*)\s*\)?\.?\s*$", text_val)
                    raw_core = match_core.group(1) if match_core else text_val
                    core_match = raw_core.upper(); core_match = re.sub(r"[.:\s]+$","",core_match)
                    if core_match in confirmed_core_ids and 0 < len(text_val) <= 10: # Max length for indicator text
                        is_part = False
                        for item_box in final_highlight_boxes_list:
                            if item_box.get('tag_from_llm','').upper()==core_match or text_val.upper() in item_box.get('text','').upper():
                                lx0,ly0,lx1,ly1=item_box.get('x0'),item_box.get('y0'),item_box.get('x1'),item_box.get('y1')
                                if all(v is not None for v in [lx0,ly0,lx1,ly1]):
                                    wcx,wcy=(word_data['x0']+word_data['x1'])/2,(word_data['top']+word_data['bottom'])/2
                                    if (lx0-2)<=wcx<=(lx1+2) and (ly0-2)<=wcy<=(ly1+2):is_part=True;break
                        if not is_part:candidate_indicators_p2.append({"id":f"word_{idx}","text":text_val,"core_text_matched":core_match,"x0":round(word_data['x0'],2),"y0":round(word_data['top'],2),"x1":round(word_data['x1'],2),"y1":round(word_data['bottom'],2)})
                print(f"TIMER LOG: Pass 2: Found {len(candidate_indicators_p2)} candidate indicators.")
                if candidate_indicators_p2:
                    indicators_json_p2 = json.dumps(candidate_indicators_p2, separators=(',',':'))
                    ctx_json_p2 = json.dumps(processed_legends_ctx_pass2, separators=(',',':')) # Use the simplified context
                    # --- Pass 2 Prompt - kept simpler, relying on Pass 1 being more precise ---
                    prompt_pass2 = f"""You are an engineering drawing analyst.
Context: These Fence Legend Items were previously identified: {ctx_json_p2}
Now, analyze CANDIDATE INDICATOR text elements. Their 'core_text_matched' matches an 'identifier' from the context.
Determine if each candidate is TRULY a standalone graphical callout/indicator.
A true indicator is short (e.g. "F1", "1"), spatially distinct, and NOT part of the legend item's full description text.
Input Candidate Indicators: {indicators_json_p2}
Output ONLY a single valid JSON: {{"confirmed_indicators": [{{"id": "word_id", "matched_legend_identifier": "core_match_val", "text_content": "original_text"}}]}}.
If none confirmed, return {{"confirmed_indicators": []}}. Strictly JSON."""
                    resp_content_p2 = ""
                    try:
                        resp_obj_p2 = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_pass2)])
                        resp_content_p2 = resp_obj_p2.content
                        clean_resp_p2=resp_content_p2.strip()
                        match_json_p2=re.search(r"```json\s*([\s\S]*?)\s*```",clean_resp_p2,re.IGNORECASE)
                        if match_json_p2:clean_resp_p2=match_json_p2.group(1).strip()
                        else:fb_p2=clean_resp_p2.find('{');lb_p2=clean_resp_p2.rfind('}');clean_resp_p2=clean_resp_p2[fb_p2:lb_p2+1] if fb_p2!=-1 and lb_p2>fb_p2 else clean_resp_p2
                        parsed_p2=json.loads(clean_resp_p2)
                        confirmed_indicators_llm = parsed_p2.get("confirmed_indicators",[])
                        print(f"TIMER LOG: Pass 2 LLM. Found {len(confirmed_indicators_llm)} confirmed indicators.")
                        map_p2_candidates = {item['id']:item for item in candidate_indicators_p2}
                        for ind_llm in confirmed_indicators_llm:
                            if not isinstance(ind_llm,dict):continue
                            el_id=ind_llm.get("id")
                            if not el_id:continue
                            orig_box_data=map_p2_candidates.get(el_id)
                            if orig_box_data and all(orig_box_data.get(k) is not None for k in ['text','x0','y0','x1','y1']):
                                final_highlight_boxes_list.append({'id':el_id,'text':orig_box_data['text'],'x0':orig_box_data['x0'],'y0':orig_box_data['y0'],'x1':orig_box_data['x1'],'y1':orig_box_data['y1'],'type_from_llm':"indicator",'tag_from_llm':ind_llm.get("matched_legend_identifier")})
                    except UnrecoverableRateLimitError: raise
                    except Exception as e_p2_proc: print(f"TIMER LOG: Error Pass 2 Proc: {e_p2_proc}"); print(f"TIMER LOG: Pass 2 Resp: {resp_content_p2[:500]}")
        
        dedup_map = {item.get('id',str(item)):item for item in final_highlight_boxes_list} # Ensure key exists
        final_highlight_boxes_list = list(dedup_map.values())
        print(f"TIMER LOG: (get_fence_related_text_boxes) Finished. Returning {len(final_highlight_boxes_list)} boxes. Total: {time.time()-overall_gfrtb_start_time:.4f}s.")
        return final_highlight_boxes_list, page_width, page_height
    except Exception as e_outer:
        print(f"TIMER LOG: (get_fence_related_text_boxes) CRITICAL Outer Error: {e_outer}. Time: {time.time()-overall_gfrtb_start_time:.4f}s.")
        # Fallback if final_highlight_boxes_list isn't defined or not a list
        if 'final_highlight_boxes_list' in locals() and isinstance(final_highlight_boxes_list, list):
            dedup_map_exc = {item.get('id', str(item)): item for item in final_highlight_boxes_list}
            return list(dedup_map_exc.values()), page_width, page_height
        return [], page_width, page_height