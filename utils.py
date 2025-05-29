# utils.py
# (Keep all other functions like time_it, retry_with_backoff, analyze_page, etc., as they were in V8)
# The main change is in get_fence_related_text_boxes -> Pass 2 prompt and examples

# ... (imports and other functions as in the previous full utils.py V8) ...

@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app, selected_llm_model_name="gpt-3.5-turbo"):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting REFINED TWO-PASS - v9 (Stricter Numerical Indicator Filtering in Pass 2).")
    # ... (overall_gfrtb_start_time, page_width, page_height, final_highlight_boxes_list, words initialization) ...
    # ... (pdfplumber parsing to get page_obj, words, extracted_lines - same as before) ...
    # ... (Pass 1 logic, including its prompt and examples, remains the same as V8 - it should correctly identify fence-specific items and their core IDs) ...
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

            # --- PASS 1 Logic (Identical to V8) ---
            print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Pass 1: Legend/Description Identification.")
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
Example 1 Input Line: {"id": "line_23", "text": "1. 6' HIGH INTERIOR COURT FENCE, COLOR: BLACK VINYL", "x0": 50.0, "y0": 100.0, "x1": 450.0, "y1": 110.0}
Example 1 Output for this line (within the 'identified_fences' list): {"id": "line_23", "full_text": "1. 6' HIGH INTERIOR COURT FENCE, COLOR: BLACK VINYL", "core_identifier_text": "1", "type": "legend_item"}
Example 2 Input Line: {"id": "line_45", "text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS PER DETAIL 3/CD-2.", "x0": 50.0, "y0": 120.0, "x1": 480.0, "y1": 130.0}
Example 2 Output for this line: {"id": "line_45", "full_text": "ALL FENCE POSTS TO BE SET IN CONCRETE FOOTINGS PER DETAIL 3/CD-2.", "core_identifier_text": "FENCE_POST_FOOTING_NOTE", "type": "note"}
Example 3 Input Line: {"id": "line_67", "text": "6. POST TENSION SLAB - REFER TO DETAIL 3, CD-1", "x0": 50.0, "y0": 150.0, "x1": 400.0, "y1": 160.0}
Example 3 Output for this line: {} (OMIT this non-fence item)
Example 4 Input Line: {"id": "line_88", "text": "FENCE TYPE F2A: CHAIN LINK W/ PRIVACY SLATS", "x0": 50.0, "y0": 180.0, "x1": 350.0, "y1": 190.0}
Example 4 Output for this line: {"id": "line_88", "full_text": "FENCE TYPE F2A: CHAIN LINK W/ PRIVACY SLATS", "core_identifier_text": "F2A", "type": "legend_item"}
Example 5 Input Line: {"id": "line_92", "text": "PROJECT BOUNDARY LINE", "x0": 50.0, "y0": 200.0, "x1": 200.0, "y1": 210.0}
Example 5 Output for this line: {} (OMIT this non-fence item)"""
                prompt_pass1 = f"""You are an engineering drawing analyst. Your primary goal is to identify text elements that are **SPECIFICALLY about fences, gates, or their components/installation.**
You are provided with a JSON list of TEXT LINES from a drawing page. Fence-related keywords: {', '.join(fence_keywords_from_app)}.
Your task is:
1.  Analyze each input line. Identify ONLY lines **explicitly describing FENCES, GATES, BARRIERS, GUARDRAILS, or their direct attributes.** Ignore lines about unrelated items (e.g., "LIGHT POLE", "POST TENSION SLAB"), even if they are in a list with fence items.
2.  For each **fence-specific** line:
    a.  Determine **`core_identifier_text`**: **Priority 1:** Exact tag (e.g., "F1", "1", "NOTE 3"), removing trailing punctuation. **Priority 2:** ALL_CAPS_SNAKE_CASE summary (e.g., "FENCE_HEIGHT_SPEC"). **Priority 3:** "GENERAL_FENCE_NOTE" or "N/A_DESC".
    b.  Provide **`type`**: "legend_item", "specification", "note", or "description".
Examples: {pass1_examples}
Output ONLY a single valid JSON object: {{"identified_fences": [{{...output for fence-related line 1...}}, ...]}}. 'id' and 'full_text' must match input. If NO fence-specific items, return {{"identified_fences": []}}. Strictly JSON.
Input Text Lines: {lines_json_str_pass1}"""
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
                                core_id_processed = core_id_from_llm.strip().upper()
                                core_id_processed = re.sub(r"[.:\s]+$", "", core_id_processed) 
                                confirmed_legend_core_ids.add(core_id_processed)
                            if all(original_line_obj_from_map.get(k) is not None for k in ['text', 'x0', 'y0', 'x1', 'y1']):
                                final_highlight_boxes_list.append({'id': original_line_id, 'text': item_text_to_use, 'x0': original_line_obj_from_map['x0'], 'y0': original_line_obj_from_map['y0'],'x1': original_line_obj_from_map['x1'], 'y1': original_line_obj_from_map['y1'],'type_from_llm': item_type, 'tag_from_llm': core_id_processed if core_id_processed else core_id_from_llm })
                                processed_legends_for_pass2_prompt_context.append({"id": original_line_id, "core_identifier_text": core_id_processed if core_id_processed else core_id_from_llm, "full_text": item_text_to_use, "type": item_type})
                except UnrecoverableRateLimitError: raise 
                except Exception as e_p1: print(f"TIMER LOG: Error in Pass 1: {e_p1}"); print(f"TIMER LOG: Pass 1 Resp: {response_content_pass1[:500] if response_content_pass1 else 'None'}")
            print(f"TIMER LOG: After Pass 1, core_ids for Pass 2: {confirmed_legend_core_ids}")

            # --- PASS 2: Indicator Identification (Focus on Numerical Specificity) ---
            if confirmed_legend_core_ids and words: 
                print(f"TIMER LOG: Starting Pass 2: Indicator Identification ({len(words)} Words).")
                # ... (candidate_indicator_words_for_llm2 selection logic remains the same as V7/V8) ...
                candidate_indicator_words_for_llm2 = []
                for idx, word_obj_data in enumerate(words):
                    text_val = word_obj_data['text'].strip()
                    if not text_val: continue
                    core_word_text_match = re.match(r"^\s*\(?\s*([A-Za-z0-9]+(?:[\.\-_][A-Za-z0-9]+)*)\s*\)?\.?\s*$", text_val)
                    raw_core_word_text = core_word_text_match.group(1) if core_word_text_match else text_val
                    core_word_text_for_match = raw_core_word_text.upper()
                    core_word_text_for_match = re.sub(r"[.:\s]+$", "", core_word_text_for_match)
                    if core_word_text_for_match in confirmed_legend_core_ids:
                        if 0 < len(text_val) <= 15: # Length check for candidate text
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
                                    "core_text_matched": core_word_text_for_match, # This is the ID it matched from Pass 1
                                    "x0": round(word_obj_data['x0'], 2), "y0": round(word_obj_data['top'], 2),
                                    "x1": round(word_obj_data['x1'], 2), "y1": round(word_obj_data['bottom'], 2)
                                })
                print(f"TIMER LOG: Pass 2: Found {len(candidate_indicator_words_for_llm2)} candidate indicators.")

                if candidate_indicator_words_for_llm2:
                    indicators_json_str_pass2 = json.dumps(candidate_indicator_words_for_llm2, separators=(',',':'))
                    # ... (pass1_context_json_for_pass2 preparation same as V7/V8) ...
                    pass1_context_json_for_pass2 = []
                    for leg_detail in processed_legends_for_pass2_prompt_context:
                        leg_text_snippet = leg_detail.get("full_text", "")[:70]; 
                        if len(leg_detail.get("full_text", "")) > 70: leg_text_snippet += "..."
                        pass1_context_json_for_pass2.append({"identifier": leg_detail.get("core_identifier_text"),"description_snippet": leg_text_snippet, "type" : leg_detail.get("type")})
                    confirmed_legends_context_str_for_prompt = json.dumps(pass1_context_json_for_pass2, separators=(',',':'))
                    
                    # --- Updated Few-Shot Examples and Instructions for Pass 2 (v9) ---
                    pass2_examples = """
Example 1 (Good Indicator):
Input Candidate: {"id": "word_105", "text": "1", "core_text_matched": "1", "x0": 350.0, "y0": 250.0, "x1": 355.0, "y1": 258.0}
(Context has: {"identifier": "1", "description_snippet": "1. 6' HIGH INTERIOR COURT FENCE...", "type": "legend_item"})
Output for this candidate: {"id": "word_105", "matched_legend_identifier": "1", "text_content": "1"}
Reasoning: This "1" is likely a callout pointing to the fence type 1.

Example 2 (Bad Indicator - Dimension):
Input Candidate: {"id": "word_200", "text": "10'", "core_text_matched": "10", "x0": 400.0, "y0": 300.0, "x1": 415.0, "y1": 308.0} 
(Assume "10" is NOT a core_identifier from Pass 1. If "10" *was* a core_id like "POST_10", this example would change. For now, assume it matched "10" from a legend item like "10. Ground Rod" which IS fence-related, but this "10'" is clearly a dimension.)
Output for this candidate: [] 
Reasoning: "10'" looks like a dimension (10 feet), not a callout for a legend item "10".

Example 3 (Bad Indicator - Part of text):
Input Candidate: {"id": "word_10", "text": "Fence", "core_text_matched": "FENCE_POST_FOOTING_NOTE", "x0": 60.0, "y0": 121.0, "x1": 90.0, "y1": 129.0}
(Context has: {"identifier": "FENCE_POST_FOOTING_NOTE", "description_snippet": "ALL FENCE POSTS TO BE SET...", "type": "note"})
Output for this candidate: []
Reasoning: This word "Fence" is part of the note text itself, not a standalone indicator of the entire note.

Example 4 (Good Alphanumeric Indicator):
Input Candidate: {"id": "word_77", "text": "F2A", "core_text_matched": "F2A", "x0": 500.0, "y0": 550.0, "x1": 520.0, "y1": 558.0}
(Context has: {"identifier": "F2A", "description_snippet": "FENCE TYPE F2A: CHAIN LINK...", "type": "legend_item"})
Output for this candidate: {"id": "word_77", "matched_legend_identifier": "F2A", "text_content": "F2A"}

Example 5 (Bad Numerical Indicator - Looks like a quantity or part of a measurement in descriptive text):
Input Candidate: {"id": "word_301", "text": "2", "core_text_matched": "2", "x0": 150.0, "y0": 400.0, "x1": 155.0, "y1": 408.0}
(Context has: {"identifier": "2", "description_snippet": "2. GATE HARDWARE SET (HINGES, LATCH)", "type": "legend_item"})
(Imagine this "2" on the drawing is next to text like "PROVIDE 2 ANCHOR BOLTS")
Output for this candidate: []
Reasoning: This "2" appears to be a quantity within a descriptive text or dimension, not a callout pointing to legend item 2.
"""

                    prompt_pass2 = f"""You are an engineering drawing analyst.
Context: These Fence-related items (legends, specs, notes) and their core identifiers were previously identified:
{confirmed_legends_context_str_for_prompt}

You are given a JSON list of CANDIDATE INDICATOR text elements. Their 'core_text_matched' value matches a `core_identifier_text` from the context.
Your task is to determine if each candidate is TRULY acting as a standalone graphical callout/indicator on the drawing.
A true indicator:
- Is usually very short (e.g., "F1", "1", "A", "N3").
- Is spatially distinct and appears to label a specific part of the drawing, often near leader lines or symbols.
- Is NOT part of a longer sentence, dimension string (e.g., "10'-0\"", "1:20", values on dimension lines), title block, or the main body of a legend/specification item already detailed in the context.
- **Crucially for numbers:** A number is an indicator if it's clearly tagging an item corresponding to a legend entry (e.g., a "1" in a circle with a leader line pointing to a fence). It is NOT an indicator if it's just a measurement, a quantity (e.g., "2 POSTS"), part of a scale, or a page number.

Examples (output shows items for 'confirmed_indicators'):
{pass2_examples}

Input Candidate Indicators:
{indicators_json_str_pass2}

Output ONLY a single valid JSON object: {{"confirmed_indicators": [{{...}}]}}.
If a candidate is NOT a true indicator, OMIT it. If none, return {{"confirmed_indicators": []}}. Strictly JSON.
"""
                    response_content_pass2 = ""
                    # ... (The rest of Pass 2 LLM call, JSON parsing, and processing logic remains the same as V7/V8) ...
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
        
        # ... (Deduplication and return logic remains the same as V7/V8) ...
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