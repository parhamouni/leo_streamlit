# llm_utils.py
from typing import List, Optional
from langchain_core.messages import HumanMessage

# ---------- JSON-invoking helper ----------

def _supports_bind(llm) -> bool:
    return hasattr(llm, "bind")

def json_invoke(llm, messages: List[HumanMessage], max_retries: int = 2):
    """
    Invoke LLM forcing JSON-object outputs when backend supports it.
    Falls back to normal .invoke if .bind/response_format isn't supported.
    """
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            if _supports_bind(llm):
                bound = llm.bind(response_format={"type": "json_object"})
                return bound.invoke(messages)
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err

# ---------- Prompt builders (safe) ----------

def make_analyze_page_prompt(context_info: str, page_text: str) -> str:
    # NOTE: Double braces {{ }} are used to keep literal JSON in an f-string.
    return f"""
You are a careful engineering-drawing analyst.
Return STRICT JSON with the schema:
{{"answer":"yes|no","confidence":0..1,"signals":["..."],"reason":"..."}}
Rules:
- Decide if this page is ABOUT fences/gates/guardrails/barriers.
- Favor RECALL over precision. When uncertain but plausible, answer "yes" with lower confidence.
- Extract short key SIGNALS (words/phrases) that influenced your decision.
- DO NOT include markdown fences. Output one JSON object only.

Few-shots (positive & sparse cases):
1) Text: "F-2 8' CL FENCE – SEE DETAIL 3/L2-01"
   -> {{"answer":"yes","confidence":0.9,"signals":["F-2","FENCE","DETAIL REFERENCE"],"reason":"Fence legend with explicit callout"}}
2) Text: "GENERAL NOTES: Contractor to install perimeter security as per schedule."
   -> {{"answer":"yes","confidence":0.55,"signals":["perimeter security","schedule"],"reason":"Weak but plausible reference to fences/gates context"}}
3) Text: "TYPICAL SECTION – LIGHTING LAYOUT"
   -> {{"answer":"no","confidence":0.85,"signals":["lighting"],"reason":"Unrelated electrical plan"}}

Context Info:
{context_info}
-----
Page Combined Text:
<<<TEXT
{page_text}
TEXT>>>
""".strip()

def make_legend_discovery_prompt(page_num: int, page_text: str) -> str:
    return f"""
You extract FENCE/GATE/RAILING legend items from a page. Favor RECALL.
Return STRICT JSON: {{"legend_items":[{{"identifier":"...","title":"... or null","description":"... or null","page":{page_num},"confidence":0..1}}]]}}
Guidelines:
- Include probable identifiers even if unsure (lower confidence). Examples: F1, F-2A, 1, G-03.
- Identify a short title if present (e.g., "8' CL FENCE").
- description may include notes like material/height/detail refs.
- DO NOT include markdown fences.

Page Text:
<<<TEXT
{page_text}
TEXT>>>
""".strip()

def make_legend_merge_prompt(candidates_json: str) -> str:
    return f"""
You are merging candidate legend items across pages.
Return STRICT JSON: {{"index":[{{"identifier":"canonical","synonyms":["..."],"definition_pages":[int,...],"title":"... or null","short_description":"... or null","overall_confidence":0..1}}]]}}
Rules:
- Normalize variations (e.g., F-1, F1, (F1)) to one canonical identifier.
- Merge data from duplicates (pages, titles, descriptions); compute overall confidence.
- Keep only FENCE/GATE/RAILING-related items.
- DO NOT include markdown fences.

Candidates (JSON list):
<<<JSON
{candidates_json}
JSON>>>
""".strip()

def make_page_ref_prompt(id_list_preview: str, page_text: str) -> str:
    return f"""
You detect references to known legend identifiers on this page. Favor RECALL.
Return STRICT JSON: {{"references":[{{"identifier":"...","evidence":"short snippet","confidence":0..1}}]]}}
Guidelines:
- If context suggests an identifier is used on this page (even with weak evidence), include it with lower confidence.
- Provide a short textual evidence (few words) from the page.
- Consider nearby context if present. DO NOT include markdown fences.

Known identifiers: {id_list_preview}

Page Text:
<<<TEXT
{page_text}
TEXT>>>
""".strip()
