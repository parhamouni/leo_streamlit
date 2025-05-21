import time
import functools
import fitz  # PyMuPDF
import base64
import re
import random
from langchain_core.messages import HumanMessage
from openai import RateLimitError # Ensure this is the correct import for your OpenAI library version
import pdfplumber
from io import BytesIO
import json

# --- Timing Decorator ---
def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        call_context = ""
        # Basic context detection (can be made more sophisticated)
        if "prompt_legend" in str(args) or "prompt_legend" in str(kwargs) or ("legend item" in str(args).lower()):
            call_context = " (Legend Analysis Call)"
        elif "prompt_indicator" in str(args) or "prompt_indicator" in str(kwargs) or ("indicator" in str(args).lower() and "legend" in str(args).lower()):
            call_context = " (Indicator Analysis Call)"
        
        print(f"TIMER LOG: ---> Entering {func_name}{call_context}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"TIMER LOG: <--- Exiting  {func_name}{call_context} (Duration: {duration:.4f} seconds)")
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
            # This is the actual call to the LLM's invoke method
            return llm_invoke_method(messages_list)
        except RateLimitError as rle:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"TIMER LOG: RateLimitError for '{func_name}'. Retrying in {delay:.2f}s (Attempt {attempt+1}/{retries}). Error: {rle}")
            time.sleep(delay)
        except Exception as e: # Catching a broader range of API errors (e.g., connection issues, server errors)
            delay = base_delay * (2 ** attempt) + random.uniform(0,1) # Use backoff for other errors too
            print(f"TIMER LOG: API Error for '{func_name}' (Attempt {attempt+1}/{retries}). Retrying in {delay:.2f}s. Error: {type(e).__name__}: {e}")
            time.sleep(delay)
            if attempt == retries - 1:
                print(f"TIMER LOG: Max retries exceeded for '{func_name}'. Last error: {type(e).__name__}: {e}")
                raise # Re-raise the last exception if all retries fail
    # This part should ideally not be reached if retries are handled correctly
    raise RuntimeError(f"Max retries exceeded for {func_name} after multiple attempts and an unknown issue.")


@time_it
def analyze_page(page, llm_text, llm_vision, fence_keywords):
    page_num = page.get('page_number', 'N/A')
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Starting analysis.")
    text = page["text"]
    text_response_content = None
    text_snippet = extract_snippet(text, fence_keywords) # Timed
    text_found = False
    vision_found = False
    vision_response_content = None

    text_prompt = f"""Analyze the following text from an engineering drawing page.
Does this text contain any information, descriptions, or specifications related to fences, fencing, gates, barriers, or guardrails?
Consider any mention of these terms or types of these structures.
Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning. Text: {text}"""
    
    print(f"TIMER LOG: (analyze_page) Page {page_num} - Preparing for text LLM call.")
    try:
        response_obj = retry_with_backoff(llm_text.invoke, [HumanMessage(content=text_prompt)]) # Timed
        text_response_content = response_obj.content
        text_found = is_positive_response(text_response_content)
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Text LLM call successful.")
    except Exception as e:
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Error during text analysis LLM call: {type(e).__name__}: {e}")
        text_response_content = f"Error in text analysis: {e}"

    if not text_found and llm_vision:
        print(f"TIMER LOG: (analyze_page) Page {page_num} - Preparing for vision LLM call.")
        image_url = f"data:image/png;base64,{page['image_b64']}"
        vision_prompt_messages = [ HumanMessage(content=[ {"type": "text", "text": "Analyze this engineering drawing image. Does it visually depict any fences, fencing structures, gates, or barriers? Start your answer with 'Yes' or 'No'. Then, briefly explain your reasoning."}, {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}} ]) ]
        try:
            response_obj = retry_with_backoff(llm_vision.invoke, vision_prompt_messages) # Timed
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

@time_it
def get_fence_related_text_boxes(page_bytes, llm, fence_keywords_from_app):
    print(f"TIMER LOG: (get_fence_related_text_boxes) Starting text box extraction.")
    overall_gfrtb_start_time = time.time()

    # Initialize return variables at the very top
    highlight_boxes = []
    found_legend_details = []
    page_width, page_height = 0, 0
    unique_highlight_boxes = [] # MOVED INITIALIZATION HERE

    try:
        step_start_time = time.time()
        with pdfplumber.open(BytesIO(page_bytes)) as pdf:
            print(f"TIMER LOG: (get_fence_related_text_boxes) pdfplumber.open took {time.time() - step_start_time:.4f}s.")
            if not pdf.pages:
                print("TIMER LOG: (get_fence_related_text_boxes) Warning: PDF document contains no pages.")
                return [], 0, 0 # Return initialized empty list
            page = pdf.pages[0]
            page_width = page.width
            page_height = page.height

            if page_width == 0 or page_height == 0:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Warning: PDF page has zero dimensions ({page_width}x{page_height}).")
                return [], page_width, page_height # Return initialized empty list
            
            step_start_time = time.time()
            words = []
            try:
                # Consider making extract_words settings configurable or testing alternatives if this is slow
                words = page.extract_words(
                    use_text_flow=True, # Generally better for reading order
                    split_at_punctuation=False, # Keep punctuation attached for context
                    x_tolerance=1.5, # Default is 3. Smaller might be more precise but slower.
                    y_tolerance=1.5, # Default is 3.
                    keep_blank_chars=False
                )
                print(f"TIMER LOG: (get_fence_related_text_boxes) page.extract_words found {len(words)} words (took {time.time() - step_start_time:.4f}s).")
            except Exception as e: # Catch any error during word extraction
                print(f"TIMER LOG: (get_fence_related_text_boxes) Error during page.extract_words: {type(e).__name__}: {e} (took {time.time() - step_start_time:.4f}s).")
                # Depending on severity, you might want to return early or continue with empty 'words'

            # --- Part 1: Identify Fence-Related Legend Items & Their Identifiers ---
            print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Part 1: Legend Item Identification.")
            part1_start_time = time.time()
            lines_for_legend_analysis = []
            processed_line_texts_for_legend = set()

            if words: # Only proceed if words were extracted
                # More Robust Line Reconstruction Logic
                sorted_words = sorted(words, key=lambda w: (round(w['top'],0), round(w['x0'],0))) # Sort by y then x
                current_line_buffer = []
                y_tolerance_for_line = 5  # Max vertical distance (pixels) for words to be on the same line
                max_horizontal_gap_factor = 2.5  # Allow gap up to X times the width of the last char of previous word
                min_words_for_line = 2 # Lines for legend analysis should have at least 2 words

                for word_obj in sorted_words:
                    if not current_line_buffer:
                        current_line_buffer.append(word_obj)
                    else:
                        last_word_in_line = current_line_buffer[-1]
                        # Check vertical alignment (top of words should be close)
                        vertical_match = abs(word_obj['top'] - last_word_in_line['top']) < y_tolerance_for_line
                        
                        # Check horizontal continuity
                        # Gap should not be excessively large.
                        # Last char width can be approximated or use average char width if available.
                        # A simpler fixed max_horizontal_gap can also be used.
                        # max_gap = (last_word_in_line['x1'] - last_word_in_line['x0']) / (len(last_word_in_line['text']) if last_word_in_line['text'] else 1) * max_horizontal_gap_factor
                        # Using a simpler fixed gap for now:
                        max_gap = 30 # pixels
                        horizontal_match = (word_obj['x0'] > last_word_in_line['x0'] and \
                                            word_obj['x0'] - last_word_in_line['x1'] < max_gap)

                        if vertical_match and horizontal_match:
                            current_line_buffer.append(word_obj)
                        else: # New line detected
                            if len(current_line_buffer) >= min_words_for_line:
                                line_text = " ".join(w['text'] for w in current_line_buffer).strip()
                                if line_text and line_text not in processed_line_texts_for_legend:
                                    lines_for_legend_analysis.append({
                                        'text': line_text,
                                        'x0': min(w['x0'] for w in current_line_buffer),
                                        'top': min(w['top'] for w in current_line_buffer),
                                        'x1': max(w['x1'] for w in current_line_buffer),
                                        'bottom': max(w['bottom'] for w in current_line_buffer),
                                    })
                                    processed_line_texts_for_legend.add(line_text)
                            current_line_buffer = [word_obj] # Start new line with current word
                
                # Add the last processed line
                if len(current_line_buffer) >= min_words_for_line:
                    line_text = " ".join(w['text'] for w in current_line_buffer).strip()
                    if line_text and line_text not in processed_line_texts_for_legend:
                        lines_for_legend_analysis.append({
                            'text': line_text,
                            'x0': min(w['x0'] for w in current_line_buffer),
                            'top': min(w['top'] for w in current_line_buffer),
                            'x1': max(w['x1'] for w in current_line_buffer),
                            'bottom': max(w['bottom'] for w in current_line_buffer),
                        })
            print(f"TIMER LOG: (get_fence_related_text_boxes) Line reconstruction for legend analysis found {len(lines_for_legend_analysis)} potential lines (took {time.time() - part1_start_time:.4f}s).")
            
            # --- Legend LLM Calls ---
            total_legend_llm_time = 0
            num_legend_llm_calls = 0
            if lines_for_legend_analysis: # Only proceed if there are lines to analyze
                for i, line_data in enumerate(lines_for_legend_analysis):
                    line_text = line_data['text']
                    if len(line_text.split()) < 2 or len(line_text) > 300 : continue # Filter candidates

                    print(f"TIMER LOG: (get_fence_related_text_boxes) Legend LLM Call #{i+1}/{len(lines_for_legend_analysis)} for text: '{line_text[:70]}...'")
                    llm_call_start_time = time.time()
                    prompt_legend = f"""Analyze the text: "{line_text}"
Is it a fence-related legend item, note, or specification from an engineering drawing (describing fences, gates, posts, barriers, keywords: {', '.join(fence_keywords_from_app)})?
If yes, what is its identifier or tag (e.g., "1", "A", "F-1", "Type A")? The identifier is usually at the beginning.
Respond ONLY with a JSON object: {{"is_fence_related_annotation": true/false, "identifier": "ID_HERE_OR_null"}}
Ensure "identifier" is null if not fence-related OR if no clear identifier is present.
"""
                    try:
                        response_obj = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_legend)])
                        response_content = response_obj.content
                        num_legend_llm_calls +=1
                        llm_call_duration = time.time() - llm_call_start_time
                        total_legend_llm_time += llm_call_duration
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Legend LLM Call #{i+1} response (took {llm_call_duration:.4f}s): {response_content[:120]}")
                        
                        data = json.loads(response_content.strip())
                        if data.get("is_fence_related_annotation") and data.get("identifier"):
                            identifier = str(data["identifier"]).strip().rstrip('.').strip()
                            if not identifier or identifier.lower() == "null": continue
                            
                            print(f"TIMER LOG: (get_fence_related_text_boxes) Identified fence legend: '{line_text[:50]}' with ID: '{identifier}'")
                            box_coords = {'text': line_text, 'x0': line_data['x0'], 'y0': line_data['top'], 'x1': line_data['x1'], 'y1': line_data['bottom']}
                            highlight_boxes.append(box_coords)
                            found_legend_details.append({'identifier': identifier, 'description': line_text, 'box': box_coords})
                    except json.JSONDecodeError:
                        print(f"TIMER LOG: (get_fence_related_text_boxes) JSONDecodeError for legend: '{response_content[:120]}' on text '{line_text[:50]}'")
                    except Exception as e_llm_leg:
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Error processing legend LLM response: {type(e_llm_leg).__name__}: {e_llm_leg} for text '{line_text[:50]}'")

            if num_legend_llm_calls > 0:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Part 1 (Legend LLMs) - {num_legend_llm_calls} calls, total time: {total_legend_llm_time:.4f}s, avg: {total_legend_llm_time/num_legend_llm_calls:.4f}s.")
            else:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Part 1 (Legend LLMs) - No LLM calls made (or no lines qualified).")
            print(f"TIMER LOG: (get_fence_related_text_boxes) Part 1: Legend Item Identification finished (total duration: {time.time() - part1_start_time:.4f}s). Found {len(found_legend_details)} fence legends.")

            # --- Part 2 & 3: Indicator Identification ---
            print(f"TIMER LOG: (get_fence_related_text_boxes) Starting Part 2: Indicator Identification. Words: {len(words)}, Found Legends: {len(found_legend_details)}")
            part2_start_time = time.time()
            total_indicator_llm_time = 0
            num_indicator_llm_calls = 0
            if found_legend_details and words: # Only proceed if legends were found and words exist
                legend_ids_set = {legend['identifier'] for legend in found_legend_details}
                # Consider further filtering 'words' here to only those that look like potential IDs (e.g., short, numeric, alphanumeric)
                
                for i, word_data in enumerate(words):
                    word_text_cleaned = word_data['text'].strip().rstrip('.').strip()
                    # Heuristic: indicators are often short. Adjust length as needed.
                    if not word_text_cleaned or len(word_text_cleaned) > 7 or len(word_text_cleaned) == 0: continue 

                    if word_text_cleaned in legend_ids_set:
                        relevant_legend = next((leg for leg in found_legend_details if leg['identifier'] == word_text_cleaned), None)
                        if not relevant_legend: continue
                        
                        # Avoid re-highlighting the ID if it's part of the already highlighted legend item text.
                        is_part_of_legend_text = any(
                            leg_detail['identifier'] == word_text_cleaned and # Check if it's the ID of this legend
                            word_data['x0'] >= leg_detail['box']['x0'] and word_data['x1'] <= leg_detail['box']['x1'] and
                            word_data['top'] >= leg_detail['box']['y0'] and word_data['bottom'] <= leg_detail['box']['y1']
                            for leg_detail in found_legend_details
                        )
                        if is_part_of_legend_text:
                            # print(f"TIMER LOG: (get_fence_related_text_boxes) Skipping '{word_text_cleaned}' as it's part of its own legend box.")
                            continue
                        
                        print(f"TIMER LOG: (get_fence_related_text_boxes) Indicator LLM Call for word #{i+1} '{word_text_cleaned}' (Legend: '{relevant_legend['identifier']}: {relevant_legend['description'][:30]}...')")
                        llm_call_start_time = time.time()
                        prompt_indicator = f"""Text element "{word_text_cleaned}" was found on a drawing.
A known fence-related legend item is "{relevant_legend['identifier']}: {relevant_legend['description']}".
Is the standalone text element "{word_text_cleaned}" acting as a callout, tag, or specific indicator on the drawing that directly references THIS fence item?
It should NOT be confirmed if it's part of a larger unrelated text, a dimension, a page number, title block info, or just an incidental occurrence.
Respond with ONLY "YES" or "NO"."""
                        try:
                            response_obj = retry_with_backoff(llm.invoke, [HumanMessage(content=prompt_indicator)])
                            response_content = response_obj.content
                            num_indicator_llm_calls += 1
                            llm_call_duration = time.time() - llm_call_start_time
                            total_indicator_llm_time += llm_call_duration
                            print(f"TIMER LOG: (get_fence_related_text_boxes) Indicator LLM Call #{i+1} response: '{response_content.strip()}' (took {llm_call_duration:.4f}s).")

                            if "YES" in response_content.upper():
                                print(f"TIMER LOG: (get_fence_related_text_boxes) Confirmed indicator: '{word_text_cleaned}' for legend ID '{relevant_legend['identifier']}'")
                                # Add to highlight_boxes, ensure it's not an exact duplicate of an existing box
                                new_box = {'text': word_text_cleaned, 'x0': word_data['x0'], 'y0': word_data['top'], 'x1': word_data['x1'], 'y1': word_data['bottom']}
                                if not any(b['x0'] == new_box['x0'] and b['y0'] == new_box['y0'] and b['text'] == new_box['text'] for b in highlight_boxes):
                                     highlight_boxes.append(new_box)
                                # break # Potentially break if one word matches one legend, or allow multiple?
                        except Exception as e_llm_ind:
                            print(f"TIMER LOG: (get_fence_related_text_boxes) Error during indicator LLM call for '{word_text_cleaned}': {type(e_llm_ind).__name__}: {e_llm_ind}")
            
            if num_indicator_llm_calls > 0:
                 print(f"TIMER LOG: (get_fence_related_text_boxes) Part 2 (Indicator LLMs) - {num_indicator_llm_calls} calls, total time: {total_indicator_llm_time:.4f}s, avg: {total_indicator_llm_time/num_indicator_llm_calls:.4f}s.")
            else:
                print(f"TIMER LOG: (get_fence_related_text_boxes) Part 2 (Indicator LLMs) - No LLM calls made (or no qualifying words/legends).")
            print(f"TIMER LOG: (get_fence_related_text_boxes) Part 2: Indicator Identification finished (total duration: {time.time() - part2_start_time:.4f}s).")

            # --- Deduplication ---
            dedup_start_time = time.time()
            # unique_highlight_boxes is already initialized
            seen_box_signatures = set()
            for box in highlight_boxes: # highlight_boxes contains both legend texts and indicator texts
                sig = (round(box['x0'],1), round(box['y0'],1), round(box['x1'],1), round(box['y1'],1), box['text'])
                if sig not in seen_box_signatures:
                    unique_highlight_boxes.append(box)
                    seen_box_signatures.add(sig)
            print(f"TIMER LOG: (get_fence_related_text_boxes) Deduplication took {time.time() - dedup_start_time:.4f}s. Input boxes: {len(highlight_boxes)}, Unique boxes: {len(unique_highlight_boxes)}.")
            
            print(f"TIMER LOG: (get_fence_related_text_boxes) Successfully finished. Returning {len(unique_highlight_boxes)} unique boxes. Total function time: {time.time() - overall_gfrtb_start_time:.4f}s.")
            return unique_highlight_boxes, page_width, page_height

    except Exception as e_outer: # Catch any other unexpected error within the main try block
        print(f"TIMER LOG: (get_fence_related_text_boxes) CRITICAL Error during processing: {type(e_outer).__name__}: {e_outer}. Total function time until error: {time.time() - overall_gfrtb_start_time:.4f}s.")
        # Return what we have, which might be empty lists or partially filled
        return unique_highlight_boxes, page_width, page_height