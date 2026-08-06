"""Stage-4 bounded per-document agentic resolver.

WHY: the deterministic passes (page-local layer→category matching, then the
Stage-2 document-level legend registry) still leave residue on real plan sets:
layers that sit in `unmatched_layers` on some pages, and pages that detected
fence layers but ended up with an empty `layer_to_category`. Those gaps need
*looking at the drawing* — reading a legend flag near the linework, comparing
line styles between sheets — which is exactly what a small tool-using agent
can do and a batch prompt cannot.

This module runs ONE bounded agent per DOCUMENT (not per page). It is given
the dispute list up front, a handful of read-only tools over the persisted
results + source PDF (page listings, the legend registry, per-layer line
stats, text search, page-crop rendering), and two write tools:
`record_assignment` and `finish`. Hard rules:

- It NEVER writes geometry or measurements. The only output is layer→category
  assignments with provenance (`decided_by: "agentic-resolver"`).
- Categories must validate against the known vocabulary (registry entries +
  categories already mapped anywhere in the document); anything else is
  dropped with a note.
- Budgets (tool calls, images, wall clock) are enforced by the LOOP, not
  trusted to the model. When a budget trips we keep whatever was recorded.
- `run_resolver` never raises; failures degrade to notes.

Application (`apply_resolutions`) is strictly non-destructive: it only fills
mappings for layers that are currently unmatched on a page and stamps
`measurements.category_source[layer] = "agentic-resolver"` so the UI can show
where every mapping came from.

OpenAI Responses API shapes used (verified against openai==2.4.0 in venv):
- request: `client.responses.create(model=..., input=[...], tools=[...],
  tool_choice="auto")` where tools are
  `{"type": "function", "name", "description", "parameters", "strict": False}`.
- response: `response.output` is a list of items; `.type == "function_call"`
  items carry `.name` / `.arguments` (JSON string) / `.call_id`;
  `.type == "message"` items carry `.content` (list of output_text parts).
- tool results go back as input items
  `{"type": "function_call_output", "call_id", "output"}` where `output` is a
  string OR a content-part list (SDK 2.4.0: FunctionCallOutput.output is
  Union[str, ResponseFunctionCallOutputItemListParam]) — so page crops are
  attached inline as `{"type": "input_image", "image_url": "data:image/png;
  base64,..."}` parts inside the function_call_output itself; no separate
  user message is needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from utils_vector import transform_coords_for_rotation

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.1"
_CONFIDENCE_LEVELS = ("high", "medium", "low")
_MAX_BBOXES_PER_PAGE = 5
_MAX_RENDER_LONG_SIDE_PX = 1600
_MIN_DPI, _MAX_DPI = 50.0, 200.0


# ---------------------------------------------------------------------------
# Dispute detection + category vocabulary
# ---------------------------------------------------------------------------

def find_disputes(results: dict) -> List[dict]:
    """Pages the deterministic passes could not fully resolve.

    Returns [{page_num, kind, layers}] with kind:
    - "no_mapping": the page has fence_layers but an EMPTY layer_to_category
      (nothing matched at all — the whole page is in dispute);
    - "unmatched_layers": the page has some mappings but layers remain in
      unmatched_layers.
    One entry per disputed page; empty list means nothing to do.
    """
    disputes: List[dict] = []
    for fp in results.get("fence_pages") or []:
        m = fp.get("measurements") or {}
        fence_layers = m.get("fence_layers") or []
        l2c = m.get("layer_to_category") or {}
        unmatched = m.get("unmatched_layers") or []
        if fence_layers and not l2c:
            layers = list(dict.fromkeys(list(fence_layers) + list(unmatched)))
            disputes.append({"page_num": fp.get("page_num"),
                             "kind": "no_mapping", "layers": layers})
        elif unmatched:
            disputes.append({"page_num": fp.get("page_num"),
                             "kind": "unmatched_layers",
                             "layers": list(unmatched)})
    return disputes


def _category_vocabulary(results: dict) -> Dict[str, str]:
    """lowercased name -> canonical name for every KNOWN category.

    Union of (a) legend-registry entries formatted the same way the matching
    LLM names categories (`"{indicator}: {keyword}"`, keyword alone when the
    indicator is blank) and (b) every category value already present in any
    page's layer_to_category. The resolver may only assign these.
    """
    vocab: Dict[str, str] = {}
    entries = (results.get("legend_registry") or {}).get("entries") or []
    for e in entries:
        ind = str(e.get("indicator") or "").strip()
        kw = str(e.get("keyword") or "").strip()
        cat = f"{ind}: {kw}" if ind else kw
        if cat:
            vocab.setdefault(cat.lower(), cat)
    for fp in results.get("fence_pages") or []:
        m = fp.get("measurements") or {}
        for cat in (m.get("layer_to_category") or {}).values():
            cat = str(cat or "").strip()
            if cat:
                vocab.setdefault(cat.lower(), cat)
    return vocab


# ---------------------------------------------------------------------------
# Tool handlers (plain functions over results/pdf, held by a session object)
# ---------------------------------------------------------------------------

class _ResolverSession:
    """Per-run state: results, open PDF, vocabulary, recorded output."""

    def __init__(self, results: dict, pdf_path: str, notes: List[str]):
        self.results = results
        self.notes = notes
        self.vocab = _category_vocabulary(results)
        self.assignments: List[dict] = []
        self.done = False
        self.tool_calls_used = 0
        self.images_used = 0
        self.doc: Optional[fitz.Document] = None
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:  # PDF is optional: text/registry tools still work
            self.notes.append(f"could not open PDF {pdf_path!r}: {e}")
        self._handlers = {
            "list_pages": self.tool_list_pages,
            "get_legend_registry": self.tool_get_legend_registry,
            "list_layers": self.tool_list_layers,
            "search_text": self.tool_search_text,
            "render_page_crop": self.tool_render_page_crop,
            "record_assignment": self.tool_record_assignment,
            "finish": self.tool_finish,
        }

    def close(self) -> None:
        if self.doc is not None:
            try:
                self.doc.close()
            except Exception:
                pass
            self.doc = None

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, name: str, arguments_json: str) -> dict:
        """Run one tool call; always returns a JSON-serializable dict."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(args, dict):
                args = {}
        except Exception as e:
            return {"error": f"arguments were not valid JSON: {e}"}
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}",
                    "available": sorted(self._handlers)}
        try:
            return handler(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            log.warning("tool %s failed: %s", name, e)
            return {"error": f"{name} failed: {e}"}

    # -- read tools --------------------------------------------------------

    def tool_list_pages(self) -> dict:
        pages = []
        for fp in self.results.get("fence_pages") or []:
            m = fp.get("measurements") or {}
            pages.append({
                "page_num": fp.get("page_num"),
                "fence_layers": m.get("fence_layers") or [],
                "unmatched_layers": m.get("unmatched_layers") or [],
                "layer_to_category": m.get("layer_to_category") or {},
                "legend_entry_count": len(fp.get("legend_entries") or []),
                "measurement_method": m.get("measurement_method"),
            })
        return {"fence_pages": pages,
                "pdf_page_count": len(self.doc) if self.doc is not None else None}

    def tool_get_legend_registry(self) -> dict:
        reg = self.results.get("legend_registry") or {}
        entries = reg.get("entries") or []
        if not entries:
            return {"entries": [],
                    "note": "no document-level legend registry is present"}
        return {"entries": entries, "consolidated": reg.get("consolidated")}

    def tool_list_layers(self, page_num: Any) -> dict:
        fp = self._find_page(page_num)
        if fp is None:
            return {"error": f"page {page_num} is not a fence page"}
        m = fp.get("measurements") or {}
        stats: Dict[str, dict] = {
            layer: {"line_count": 0, "total_length_pts": 0.0}
            for layer in (m.get("fence_layers") or [])
        }
        for line in m.get("all_fence_lines") or []:
            layer = line.get("layer") or ""
            s = stats.setdefault(layer, {"line_count": 0, "total_length_pts": 0.0})
            s["line_count"] += 1
            s["total_length_pts"] += float(line.get("length_pts") or 0.0)
        for s in stats.values():
            s["total_length_pts"] = round(s["total_length_pts"], 1)
        return {"page_num": fp.get("page_num"), "layers": stats,
                "layer_to_category": m.get("layer_to_category") or {},
                "unmatched_layers": m.get("unmatched_layers") or []}

    def tool_search_text(self, query: str) -> dict:
        q = str(query or "").strip().lower()
        if not q:
            return {"error": "query must be a non-empty string"}
        if self.doc is None:
            return {"error": "source PDF is unavailable"}
        pages_out = []
        for pidx in range(len(self.doc)):
            page = self.doc[pidx]
            words = page.get_text("words")  # mediabox-space (x0,y0,x1,y1,word,block,line,word_no)
            rot = page.rotation
            mw, mh = page.mediabox.width, page.mediabox.height
            hits: List[tuple] = []
            for w in words:
                if q in str(w[4]).lower():
                    hits.append((w[0], w[1], w[2], w[3]))
            # Joined bigrams so multi-word queries ("chain link") can hit even
            # though get_text("words") tokenizes on whitespace.
            for a, b in zip(words, words[1:]):
                if a[5] == b[5] and a[6] == b[6]:  # same block + line
                    if q in f"{a[4]} {b[4]}".lower():
                        hits.append((min(a[0], b[0]), min(a[1], b[1]),
                                     max(a[2], b[2]), max(a[3], b[3])))
            if not hits:
                continue
            bboxes = []
            for (x0, y0, x1, y1) in hits[:_MAX_BBOXES_PER_PAGE]:
                ax, ay = transform_coords_for_rotation(x0, y0, rot, mw, mh)
                bx, by = transform_coords_for_rotation(x1, y1, rot, mw, mh)
                bboxes.append([round(min(ax, bx), 1), round(min(ay, by), 1),
                               round(max(ax, bx), 1), round(max(ay, by), 1)])
            pages_out.append({"page_num": pidx + 1, "count": len(hits),
                              "bboxes_display": bboxes})
        return {"query": query, "pages": pages_out,
                "pages_searched": len(self.doc),
                "note": "bboxes are display-space (rotation applied), like fence line coords"}

    def tool_render_page_crop(self, page_num: Any, x0: Any, y0: Any,
                              x1: Any, y1: Any, dpi: Any = 110) -> dict:
        if self.doc is None:
            return {"error": "source PDF is unavailable"}
        try:
            pidx = int(page_num) - 1
        except (TypeError, ValueError):
            return {"error": f"bad page_num {page_num!r}"}
        if not (0 <= pidx < len(self.doc)):
            return {"error": f"page {page_num} out of range 1..{len(self.doc)}"}
        try:
            coords = [max(0.0, min(1.0, float(v))) for v in (x0, y0, x1, y1)]
        except (TypeError, ValueError):
            return {"error": "x0/y0/x1/y1 must be numbers in [0, 1]"}
        rx0, rx1 = sorted((coords[0], coords[2]))
        ry0, ry1 = sorted((coords[1], coords[3]))
        if rx1 - rx0 < 1e-3 or ry1 - ry0 < 1e-3:
            return {"error": "crop rectangle has (near) zero area"}
        try:
            dpi_f = float(dpi)
        except (TypeError, ValueError):
            dpi_f = 110.0
        dpi_f = max(_MIN_DPI, min(_MAX_DPI, dpi_f))

        page = self.doc[pidx]
        rect = page.rect  # display-space rect (rotation applied), same space as clip
        clip = fitz.Rect(rect.x0 + rx0 * rect.width, rect.y0 + ry0 * rect.height,
                         rect.x0 + rx1 * rect.width, rect.y0 + ry1 * rect.height)
        zoom = dpi_f / 72.0
        long_px = max(clip.width, clip.height) * zoom
        if long_px > _MAX_RENDER_LONG_SIDE_PX:
            zoom *= _MAX_RENDER_LONG_SIDE_PX / long_px
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        # The loop pops "_png_base64" and attaches it as an input_image content
        # part; only the marker metadata is echoed as text.
        return {"image": True, "page_num": pidx + 1,
                "clip_display": [round(clip.x0, 1), round(clip.y0, 1),
                                 round(clip.x1, 1), round(clip.y1, 1)],
                "px_width": pix.width, "px_height": pix.height,
                "effective_dpi": round(zoom * 72.0, 1),
                "_png_base64": png_b64}

    # -- write tools -------------------------------------------------------

    def tool_record_assignment(self, layer: str, category: str, pages: Any,
                               evidence: str = "",
                               confidence: str = "medium") -> dict:
        layer = str(layer or "").strip()
        if not layer:
            return {"recorded": False, "error": "layer is required"}
        canonical = self.vocab.get(str(category or "").strip().lower())
        if canonical is None:
            note = (f"dropped assignment {layer!r} -> {category!r}: "
                    "category not in known vocabulary")
            self.notes.append(note)
            return {"recorded": False, "error": note,
                    "known_categories": sorted(self.vocab.values())}
        if pages != "all":
            if (not isinstance(pages, list) or not pages
                    or not all(isinstance(p, (int, float))
                               and float(p).is_integer() for p in pages)):
                return {"recorded": False,
                        "error": 'pages must be a non-empty list of ints or "all"'}
            pages = [int(p) for p in pages]
        conf = str(confidence or "").strip().lower()
        if conf not in _CONFIDENCE_LEVELS:
            conf = "low"
        self.assignments.append({
            "layer": layer,
            "category": canonical,
            "pages": pages,
            "evidence": str(evidence or ""),
            "confidence": conf,
            "decided_by": "agentic-resolver",
        })
        return {"recorded": True, "layer": layer, "category": canonical,
                "pages": pages}

    def tool_finish(self, summary: str = "") -> dict:
        self.done = True
        if summary:
            self.notes.append(f"agent summary: {summary}")
        return {"ok": True, "assignments_recorded": len(self.assignments)}

    # -- helpers -----------------------------------------------------------

    def _find_page(self, page_num: Any) -> Optional[dict]:
        try:
            target = int(page_num)
        except (TypeError, ValueError):
            return None
        for fp in self.results.get("fence_pages") or []:
            if fp.get("page_num") == target:
                return fp
        return None


# ---------------------------------------------------------------------------
# Responses-API plumbing
# ---------------------------------------------------------------------------

def _tool_schemas() -> List[dict]:
    def fn(name: str, description: str, properties: dict,
           required: List[str]) -> dict:
        return {"type": "function", "name": name, "description": description,
                "strict": False,
                "parameters": {"type": "object", "properties": properties,
                               "required": required,
                               "additionalProperties": False}}

    num = {"type": "number"}
    return [
        fn("list_pages",
           "List every detected fence page: page_num, fence_layers, "
           "unmatched_layers, current layer_to_category, legend entry count, "
           "measurement_method.", {}, []),
        fn("get_legend_registry",
           "The document-level legend registry (indicator/keyword/description/"
           "source_page entries), or a note that it is absent.", {}, []),
        fn("list_layers",
           "Per-layer vector stats for one fence page: line count and total "
           "length in points from the measured fence lines.",
           {"page_num": {"type": "integer"}}, ["page_num"]),
        fn("search_text",
           "Case-insensitive text search across ALL pdf pages (single words "
           "and adjacent-word bigrams). Returns per page: hit count and up to "
           "5 display-space bboxes.",
           {"query": {"type": "string"}}, ["query"]),
        fn("render_page_crop",
           "Render a crop of a PDF page as an image. x0/y0/x1/y1 are RELATIVE "
           "0-1 coordinates of the display page rect (0,0 = top-left). dpi is "
           "clamped to [50, 200]. Counts against your image budget.",
           {"page_num": {"type": "integer"}, "x0": num, "y0": num,
            "x1": num, "y1": num, "dpi": num},
           ["page_num", "x0", "y0", "x1", "y1"]),
        fn("record_assignment",
           "Record one layer->category assignment. category MUST be one of "
           "the known categories; pages is a list of page numbers or \"all\"; "
           "evidence must cite the page and what you saw/read there.",
           {"layer": {"type": "string"}, "category": {"type": "string"},
            "pages": {"anyOf": [{"type": "array",
                                 "items": {"type": "integer"}},
                                {"type": "string", "enum": ["all"]}]},
            "evidence": {"type": "string"},
            "confidence": {"type": "string",
                           "enum": list(_CONFIDENCE_LEVELS)}},
           ["layer", "category", "pages", "evidence", "confidence"]),
        fn("finish",
           "End the session with a short summary of what was resolved and "
           "what was left unresolved (and why).",
           {"summary": {"type": "string"}}, ["summary"]),
    ]


_SYSTEM_PROMPT = """\
You are the agentic resolver for a fencing takeoff pipeline. Deterministic
passes already extracted fence linework per PDF layer and mapped most layers
to fence categories from the plan legends. You are handed the residue: a
DISPUTE list of pages whose layers are still unmatched (kind
"unmatched_layers") or that have fence layers but no mappings at all (kind
"no_mapping").

Your ONLY job is to decide layer -> category assignments you can justify from
the document itself. You never touch geometry or measurements.

Tools: list_pages, get_legend_registry, list_layers, search_text,
render_page_crop (images), record_assignment, finish.

Budgets (enforced by the harness — it will cut you off): at most
{max_tool_calls} tool calls, at most {max_images} rendered images, and
{max_wall_seconds:.0f} seconds of wall clock. Spend images on the pages where
a look will actually decide something; record assignments as soon as you are
confident so nothing is lost if a budget trips.

Hard rules:
- Only record an assignment you can visually or textually justify. The
  evidence string must cite the page and what you saw or read there (e.g.
  'page 4: crop shows the dashed line labeled "6\\' CHAIN LINK FENCE" running
  on layer V-FENC-A').
- category must be one of the known categories you were given; invented
  categories are rejected.
- Leaving a layer unresolved is perfectly fine — an honest "could not
  determine" beats a guess. Say so in finish(summary).
Call finish when done."""


def _initial_input(session: _ResolverSession, disputes: List[dict],
                   max_tool_calls: int, max_images: int,
                   max_wall_seconds: float) -> List[dict]:
    brief = {
        "disputes": disputes,
        "known_categories": sorted(session.vocab.values()),
        "fence_page_nums": [fp.get("page_num")
                            for fp in session.results.get("fence_pages") or []],
        "pdf_page_count": len(session.doc) if session.doc is not None else None,
    }
    return [
        {"role": "developer",
         "content": _SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls,
                                          max_images=max_images,
                                          max_wall_seconds=max_wall_seconds)},
        {"role": "user",
         "content": ("Resolve what you can from this dispute list.\n"
                     + json.dumps(brief, indent=2))},
    ]


def _message_text(item: Any) -> str:
    """Best-effort text of a response 'message' output item."""
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content or []:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def run_resolver(results: dict, pdf_path: str, client=None,
                 model: str = DEFAULT_MODEL, max_tool_calls: int = 40,
                 max_images: int = 12,
                 max_wall_seconds: float = 300.0) -> dict:
    """Run the bounded per-document agent. Never raises."""
    t0 = time.monotonic()
    notes: List[str] = []
    disputes: List[dict] = []
    session: Optional[_ResolverSession] = None
    budget_exhausted = False

    try:
        disputes = find_disputes(results)
        if not disputes:
            notes.append("no disputes: every fence layer already mapped")
        elif client is None:
            notes.append("no LLM client provided; agentic resolution skipped")
        else:
            session = _ResolverSession(results, pdf_path, notes)
            budget_exhausted = _run_loop(
                session, client, model, disputes, t0,
                max_tool_calls=max_tool_calls, max_images=max_images,
                max_wall_seconds=max_wall_seconds)
    except Exception as e:  # contract: never raise out of run_resolver
        log.exception("agentic resolver failed")
        notes.append(f"resolver error: {e}")
    finally:
        if session is not None:
            session.close()

    return {
        "assignments": list(session.assignments) if session else [],
        "notes": "; ".join(notes),
        "tool_calls_used": session.tool_calls_used if session else 0,
        "images_used": session.images_used if session else 0,
        "wall_s": round(time.monotonic() - t0, 2),
        "budget_exhausted": budget_exhausted,
        "disputes": disputes,
    }


def _run_loop(session: _ResolverSession, client, model: str,
              disputes: List[dict], t0: float, *, max_tool_calls: int,
              max_images: int, max_wall_seconds: float) -> bool:
    """The agent loop proper. Returns True iff a budget was exhausted."""
    input_items = _initial_input(session, disputes, max_tool_calls,
                                 max_images, max_wall_seconds)
    tools = _tool_schemas()
    retried = False

    while not session.done:
        if time.monotonic() - t0 > max_wall_seconds:
            session.notes.append("wall-clock budget exhausted")
            return True
        try:
            response = client.responses.create(
                model=model, input=input_items, tools=tools,
                tool_choice="auto")
        except Exception as e:
            if not retried:
                retried = True
                session.notes.append(f"API error, retrying once: {e}")
                continue
            session.notes.append(f"API error after retry, finishing: {e}")
            return False

        calls = []
        for item in response.output or []:
            itype = getattr(item, "type", None)
            if itype == "function_call":
                calls.append(item)
                # Echo the call into history so the API sees the pairing with
                # its function_call_output below.
                input_items.append({"type": "function_call",
                                    "call_id": item.call_id,
                                    "name": item.name,
                                    "arguments": item.arguments})
            elif itype == "message":
                text = _message_text(item)
                if text:
                    input_items.append({"role": "assistant", "content": text})
            # other item types (reasoning, ...) are ignored

        if not calls:
            # Plain-text answer without finish(): treat as done.
            session.notes.append("model stopped calling tools; loop ended")
            return False

        for call in calls:
            if time.monotonic() - t0 > max_wall_seconds:
                session.notes.append("wall-clock budget exhausted")
                return True
            if session.tool_calls_used >= max_tool_calls:
                session.notes.append("tool-call budget exhausted")
                return True
            if (call.name == "render_page_crop"
                    and session.images_used >= max_images):
                session.notes.append("image budget exhausted")
                return True

            result = session.dispatch(call.name, call.arguments)
            session.tool_calls_used += 1
            log.debug("tool %s -> %s", call.name,
                      "image" if result.get("image") else result)

            png_b64 = (result.pop("_png_base64", None)
                       if isinstance(result, dict) else None)
            if png_b64 is not None:
                session.images_used += 1
                # SDK 2.4.0 accepts a content-part list for
                # function_call_output.output, so the crop rides along with
                # its metadata inside the SAME tool output item.
                output: Any = [
                    {"type": "input_text", "text": json.dumps(result)},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{png_b64}",
                     "detail": "high"},
                ]
            else:
                output = json.dumps(result)
            input_items.append({"type": "function_call_output",
                                "call_id": call.call_id, "output": output})
            if session.done:
                break
    return False


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_resolutions(fence_results: List[dict], resolution: dict) -> int:
    """Apply resolver assignments to fence pages, strictly non-destructively.

    For each assignment and each of its pages ("all" = every fence page):
    fill measurements.layer_to_category[layer] ONLY IF the layer is currently
    unmatched there (an existing mapping is never overridden), refresh
    unmatched_layers, and stamp category_source[layer] = "agentic-resolver".
    Returns the number of (page, layer) mappings applied.
    """
    applied = 0
    for a in (resolution or {}).get("assignments") or []:
        layer = a.get("layer")
        category = a.get("category")
        pages = a.get("pages")
        if not layer or not category:
            continue
        for fp in fence_results:
            page_num = fp.get("page_num")
            if pages != "all":
                if not isinstance(pages, list) or page_num not in pages:
                    continue
            m = fp.get("measurements")
            if not isinstance(m, dict):
                continue
            fence_layers = m.get("fence_layers") or []
            l2c = m.get("layer_to_category")
            if not isinstance(l2c, dict):
                l2c = {}
            unmatched = m.get("unmatched_layers") or []
            if layer in l2c:  # never override an existing mapping
                continue
            if layer not in fence_layers and layer not in unmatched:
                continue  # layer not present on this page
            l2c[layer] = category
            m["layer_to_category"] = l2c
            base = fence_layers if fence_layers else list(unmatched)
            m["unmatched_layers"] = [l for l in base if l not in l2c]
            m.setdefault("category_source", {})[layer] = "agentic-resolver"
            applied += 1
    return applied


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_client():
    """OpenAI client from repo secrets; None (with a stderr note) on failure."""
    try:
        from secrets_loader import load_api_keys
        keys = load_api_keys()
        api_key = keys.get("openai_key") or ""
        if not api_key:
            print("[DEBUG] no openai_key in secrets; running without client",
                  file=sys.stderr)
            return None
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[DEBUG] could not build OpenAI client: {e}", file=sys.stderr)
        return None


def _main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m agentic.resolver",
        description="Bounded per-document agentic resolver for unmatched "
                    "fence layers.")
    ap.add_argument("results_json", help="path to a job's results.json")
    ap.add_argument("pdf_path", help="path to the source PDF")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--apply", action="store_true",
                    help="apply assignments to the results and write to --out")
    ap.add_argument("--out", default=None,
                    help="output path for --apply (default: "
                         "<results>.resolved.json; never the input itself)")
    args = ap.parse_args(argv)

    results_path = Path(args.results_json)
    results = json.loads(results_path.read_text())

    resolution = run_resolver(results, args.pdf_path,
                              client=_build_client(), model=args.model)
    print(json.dumps(resolution, indent=2))

    if args.apply:
        out_path = (Path(args.out) if args.out
                    else results_path.with_suffix(".resolved.json"))
        if out_path.resolve() == results_path.resolve():
            raise SystemExit(
                "--out must differ from the input results.json "
                "(never overwrite the input in place)")
        applied = apply_resolutions(results.get("fence_pages") or [],
                                    resolution)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[DEBUG] applied {applied} (page, layer) mappings -> {out_path}",
              file=sys.stderr)


if __name__ == "__main__":
    _main()
