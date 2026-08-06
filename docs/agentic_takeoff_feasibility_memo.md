# Agentic takeoff — feasibility memo + gold-set experiment (2026-08-06)

## Part 1 — Empirical: what actually happened on subset_gold

`subset_gold/selected_pages.pdf` = 5 real sheets from 3 projects, marked up by two human
estimators (sladams, Christina Adams-Moody). Discovered semantics (the 2026-05 evaluation
attempts scored 0.0 because they assumed line-trace gold and used bbox-IoU):

- The markup is **scope identification**, not measured runs: struck keynote rows, highlighted
  keynote codes/flags, struck detail titles. Page 1's highlights are the architect's stamp
  (contact info — excluded). Page 4's KEY NOTES text layer is CID-mojibake (native text
  unreadable — the OCR-fallback production condition).
- Gold sets: p2 {0113,0401,0402,0403,3219,3301} · p3 {+0508} · p4 struck rows
  {4,6,7,10,14,17,28,32,35,40,45,46} · p5 four struck detail titles + chain-link callouts.

Three systems scored on "identify the fence-scope items" (P/R vs estimator's set):

| System | p2 | p3 | p4 (mojibake text) | p5 |
|---|---|---|---|---|
| Current pipeline (text path, as deployed) | R 0.67 / P 0.80 | R 0.71 / P 1.0 | ids only, **no descriptions → cannot decide scope** | titles extracted, no scope decision |
| gpt-5.1 vision, one-shot (find table → read) | R 0.17 | R 0.14 | **R 0.92 / P 0.92** | R 0 |
| + one generic verify loop (cross-check plan codes vs read items, re-crop, re-read) | **R 0.83 / P 0.71** | **R 0.86 / P 1.0** | R 0.75 / P 0.69 (run variance — see below) | partial (bollard + chainlink callout) |

Reading of the evidence:
1. **A VLM matched a human estimator 11/12 on the sheet whose text layer is garbage** — the
   exact case the current pipeline is structurally blind on. The misses are judgment-call
   items (transformer pad 3301), not reading failures.
2. **The verify loop is what makes it work** (p2: 0.17→0.83, p3: 0.14→0.86). One-shot vision
   is brittle at region-finding; iteration + deterministic cross-checking recovers.
3. **Run-to-run variance is real** (p4 dropped 45/46 on the second run while gaining noise).
   Production shape needs 3-vote self-consistency on scope decisions.
4. Cost of the full 4-sheet agentic run: ~15 gpt-5.1 vision calls ≈ cents. Latency ~4 min
   serial, parallelizable per sheet.
5. The harness lives in `evals/gold_harness.py` (polyline-buffer + mark→text mapping;
   visual overlay sanity check) and `evals/agentic_scope.py` (the prototype).

**Sharpest single finding** (from the codebase survey, verified by hand): the live pipeline
has never actually run its layer→category matcher. `pipeline.py:1015-1017` passes raw ADE
chunks positionally where `fence_definitions`/`fence_instances` belong
(`utils_ade/measure.py:192-206`); raw chunks lack indicator/keyword keys, so
`llm_match_layers_to_definitions` (utils_ade/llm.py:766) returns {} on every page — including
pages whose legend IS on the same sheet (confirmed live: gold p1 run has 12 proper legend
entries and layer_to_category={}). Any judgment that "the LLM can't categorize" is
contaminated by this bug. Spawned fix task: "Fix legend-starvation bug in live measurement path".

## Part 2 — Architecture memo (multi-agent survey of the live codebase)

# Memo: Can an agentic LLM system solve fence takeoff? Honest assessment + staged plan

**To:** Parham · **Re:** "Can't we have an agentic system to solve this systematically?" · **Basis:** code survey of the live fence-api-v2 path, 5 real production jobs, current guardrails/economics.

---

## 1. Direct answer

**Yes for the semantic layer, no for the geometric layer, and — critically — you don't yet have evidence that the current LLMs are failing.**

The strongest finding from the survey: **the live pipeline has never actually run the layer→category matcher with real inputs.** `pipeline.py:1015-1017` passes raw ADE chunks positionally as `fence_definitions`/`fence_instances`; the chunks lack `indicator`/`keyword` keys, so `llm_match_layers_to_definitions` returns `{}` at `llm.py:793-794` on **every page of every job** (confirmed empty on 229/229 fence pages across 5 real jobs). The non-live workers pass the correct objects (`ops/analysis_worker.py:960-961`). So "auto-categorization is weak" is currently 90% a plumbing bug and 10% a model-capability question. Any verdict on LLM capability before fixing that line is premature.

With that caveat, here is the honest capability split:

**LLMs (current frontier, especially with vision) are already reliable at:**
- **Legend reading.** "CL = 6' chain link fence, see detail 3/A-501" from a legend crop is a solved problem for a VLM. Your text-only extraction already mostly works (89 legend entries found on ADASP).
- **Layer-name semantics.** `V-FENC-CHLK` → chain link, `A-BLDG-WALL-TILE-PATT` → *not* a fence — this is exactly the discrimination LLMs are good at, *when prompted to discriminate*. Today's prompts are recall-biased by design: "If unsure, include" (`llm.py:730`), "map it to the most likely one" (`llm.py:818`), force-map to `cat_names[0]` (`llm.py:844-847`). The over-match is instructed, not emergent.
- **Cross-sheet association.** "The legend on A-001 defines symbols used on A-101" is a reasoning task LLMs handle well — but your architecture never gives them the chance: legend extraction sees only same-page chunks (`llm.py:208`), and the one cross-page call is dead (`pipeline.py:1093` reads a `"text"` key legend entries don't have).
- **Visual verification of candidates.** "Is this highlighted polyline a fence or a tile hatch?" from a rendered crop — high accuracy, and you already have every piece needed to build it (see Stage 3).

**LLMs are NOT reliable at, and should never own:**
- **Precise vector geometry.** Counting, measuring, deduplicating 56,980 line segments on one page. No model does this; your deterministic PyMuPDF extraction + connected-run decomposition is the right tool and is battle-tested against 274k-line pages.
- **Exhaustive coverage at scale.** An agent asked to "check all 793 pages" will skim, drift, and hallucinate completeness. Coverage must be a deterministic loop that *invokes* models, never a model deciding coverage.
- **Arithmetic under scale factors.** Length × scale → feet stays in code.
- **Consistency across hundreds of decisions.** Per-page independent calls will categorize the same layer differently on page 12 vs page 40. The fix is architectural (decide per-layer once per document, not per-page), not a better model.

**So: "true good results" is achievable, but the path is not "add an agent."** It is: fix the bug, give the semantic layer document-level context and vision, make it precision-biased, verify with a VLM, and keep geometry/measurement/caching deterministic. The human canvas stays as the final mile (§5).

One more blocker: **you cannot claim "true good" without measurement.** There is no active eval harness — all evaluators are in `archive/`, the only gold is 112 highlight bboxes on 5 pages from Oct 2024, and zero customer corrections have been persisted (`umt_state.json` count on the live box: 0). Today you literally cannot tell whether a change helps.

---

## 2. Staged proposal (each independently shippable, ROI order)

### Stage 0 — Eval harness (1 week; do first, everything else is flying blind without it)
- **Assets:** `results.json` already persists everything needed per line (geometry, layer, length_pts, scale+method, legend bboxes). Missing only truth.
- **Build gold with your own UI:** run the 5 real jobs' fence pages through the UMT canvas yourself, correct them, and persist. Fix the durability bug first: `umt_state` `line_assignments` are keyed by vector-extraction index valid only against the exact source PDF, and source PDFs live in `/tmp` which gets reaped (`api_server.py:1377-1395`). Copy the source PDF into the results dir alongside `results.json`, or re-key assignments by a geometry hash (start/end/layer). ~20-30 labeled pages spanning the failure taxonomy (ADASP dense pages, cross-sheet legends, the `V-FENC-CHLK` easy case, the tile-hatch WALL case) is enough to rank changes.
- **Metrics:** per-page line-level **precision/recall** of fence-line selection (match by geometry); **category accuracy** per layer; **footage error** = |detected_ft − gold_ft| / gold_ft per category per page (the metric customers feel); **scale correctness** as a separate binary (ADASP's accepted `SCALE: 1:1` regex hit → 4.53 ft for 1,274 segments shows why); page-classification P/R against a hand list.
- **Harness:** resurrect the IoU matcher from `archive/evaluation/fence_evaluator.py:34` as a script comparing `results.json` vs gold `umt_state`; add `subset_gold/df_annotations_sub.csv` as a second track. Runs offline, no pipeline changes, zero risk.
- Also wire minimal **LLM spend metering** into the v2 pipeline now (`spend_tracker` exists but is only wired into legacy `app_ade_prod.py:5377`) — Stages 2-4 need cost visibility.

### Stage 1 — Fix the starvation bug + de-bias prompts (2-3 days; highest ROI in the whole plan)
- One-line fix at `pipeline.py:1015-1017`: pass `legend_entries`/`figure_instances` (computed 90 lines earlier) like the workers do. This alone turns on `layer_to_category` and `indicator_measurements` for every same-sheet-legend page — likely a large fraction of pages.
- Kill the recall bias: delete "If unsure, include" (`llm.py:730`); replace force-map-to-first-category (`llm.py:844-847`, `854-855`) with an explicit `"unmatched"` bucket surfaced in the UI; drop `'WALL'`/`'RAIL'` from the substring fallback (`measure.py:317`) or require word-boundary + length match; fix the substring expansion (`llm.py:752`).
- Adopt **structured outputs / JSON mode** everywhere (currently regex `r'\{.*?\}'` at `llm.py:828` breaks on nested JSON). Free reliability.
- Port the 16k-char classification context from the unbatched path (`llm.py:520-522`) to the batch path that actually runs (`llm.py:631` truncates to 1,200 chars — legends live at page end and get cut).
- Cache note: prompt text is not in any cache key, but scope is per-job (`api_server.py:247`) so nothing needs bumping; bump `phase3_measure`'s params anyway for in-flight requeues.
- **Re-run the eval after this stage.** This is the new baseline the agentic question gets judged against.

### Stage 2 — Document-level legend registry (1-2 weeks; fixes the cross-sheet starvation)
- **Slot:** after Phase 2 completes (`pipeline.py:860`), when `ade_chunks_by_page` holds all fence pages' chunks — the survey identified this as *the* insertion point. Build one doc-level pass: collect all legend entries from all pages (the per-chunk extraction already exists), one gpt-5.1 consolidation call to dedupe/normalize into a **document legend registry** (indicator → category → source page), then pass it read-only into `_process_fence_page` (`pipeline.py:872`; workers close over shared dicts, read-only is thread-safe). Per-page matching then always has definitions, whether or not the legend is on that sheet. Relax `FENCE_MEASURE_REQUIRE_INSTANCES` (`measure.py:264-268`) when registry definitions exist — 236/238 ADASP pages currently skip measurement at that guard.
- **Consistency win:** decide layer→category **once per (layer, document)**, not per page — one call with the full layer census + registry, results fanned out. Eliminates cross-page inconsistency and cuts call count.
- **Model:** gpt-5.1 for consolidation (one call), gpt-5-mini for fan-out. **Cost:** ~$0.05-0.50/doc, <60s added. **Cache:** new phase key `phase3_legend_index_v1`, trade-specific params tier; include a prompt-version string in the params hash this time. **Containment:** pure addition inside the background job; on failure, fall back to today's per-page behavior; cap registry at ~200 entries.

### Stage 3 — Vision verification of candidates (2-3 weeks; kills the tile-hatch/WALL false-positive class)
- **What:** after measurement, render each fence page with candidate lines colored per category and ask a VLM one question per page: "which of these numbered/colored layer groups are actually fencing vs. hatch/signage/curb?" Verify at the **(page, layer)** level — never per line.
- **Reuse, don't build:** `exports.py:191-528` already draws rotation-correct, category-colored, numbered overlays in O(n) on dense pages; `_PAGE_RENDER_SCRIPT` (`api_server.py:1188-1205`) already rasterizes in a crash-isolated 30s subprocess; `utils_vector.py:665` is the existing production VLM precedent (Gemini-style full-page PNG at 100 dpi). The one missing primitive is crop rendering — a small `clip=` extension to the render script (nothing in active code uses `clip=` today). Crops matter: ADASP sheets are 2448×1584 pt; a legend or dense-area crop at 150 dpi is far cheaper and more legible than a full ARCH-D page.
- **Output:** per-layer verdicts with confidence; low-confidence → flag in UI, don't silently drop (precision bias with human backstop).
- **Model:** vision-capable analysis tier. **Cost:** 1 image call per fence page → ADASP-class (238 fence pages) ≈ 238 calls × ~2-4k tokens ≈ **$2-8/doc**; typical 60-fence-page plan set ≈ $0.50-2. **Latency:** +3-6 min at 8 workers within the existing Phase 3 pool. **Containment:** all rendering via the subprocess pattern (MuPDF C-crashes can't kill the API); respect `MAX_FENCE_LINES=5000` skips — skipped pages get no overlay, so no verification, honestly reported; memory budget: render→encode→free per page, never hold 238 PNGs (the 2026-07-07 OOM lesson); hard per-job call budget + spend metering from Stage 0.

### Stage 4 — Bounded agentic cross-sheet resolver (3-4 weeks; only after Stages 1-3 measured)
- **Scope discipline:** ONE agent per **document** (not per page), invoked only for what deterministic passes couldn't resolve: unmatched layers, missing legends, unverified scales, conflicting categories. Tools: `render_page_crop(page, bbox, dpi)`, `search_text(query) → pages`, `list_layers(page)`, `get_legend_registry()`, `read_sheet_index()`. Hard budget: ~40-60 tool calls, wall-clock cap, token cap — then emit partial results with provenance.
- **Slot:** new reconciliation phase after Phase 3 (`pipeline.py:1071-1074`); re-emission via `page_cb` upsert by (document_id, page_number) is an established pattern (`api_server.py:294-324`). Runs inside the background job — nginx 504s anything synchronous at 600s. Carve a progress band out of the hard-coded percents.
- **Output contract:** the agent never writes measurements. It writes *layer→category assignments, scale corrections, and legend links*, each with `decided_by`/evidence provenance (Smart Assign at `api_server.py:1596` already returns exactly this shape — copy it). Deterministic code recomputes footage.
- **Model:** analysis tier with vision. **Cost:** ~$1-10/doc, +3-10 min. **Cache:** results keyed at doc level (`agentic_resolve_v1`), include tool-schema + prompt version in the hash. **Containment:** per-tool subprocess rendering, deterministic tool implementations, budget exhaustion = graceful degradation to Stage 2 output, never job failure.
- **Precedent warning:** `archive/evaluation/fence_detector_agentic.py` is a 508-line prior attempt at LLM-guided line clustering that was abandoned. It failed because it put the agent in the geometry loop. This design keeps the agent strictly in the semantic/dispute-resolution loop.

---

## 3. Where the human stays, and why

Keep UMT + Smart Assign as the **final authority**, permanently:
- **Takeoffs feed bids.** A 15% footage error is a mispriced contract; liability argues for *verify-not-replace*. The credible pitch is "review in 10 minutes instead of takeoff in 3 hours," not "zero-touch."
- Every automated decision should carry provenance (which legend, which sheet, which vision verdict) so the human verifies claims rather than redoing work — the `decided_by`/`votes` shape already exists.
- **Corrections are your moat.** Persisted `umt_state` diffs against auto-output are labeled training/eval data no competitor has. Today that corpus is empty and structurally leaky (Stage 0 fixes durability). Every stage should widen this flywheel: better auto → more usage → more corrections → better eval → better auto.

---

## 4. What NOT to do

- **No full agentic rewrite.** The deterministic pipeline embodies ~2 years of survivorship: 20s SIGKILL subprocess for MuPDF GIL-holding C-loops (`pipeline.py:229`), OOM guards, dense-page caps, poison-pill requeue limits, crash-isolated rendering. An agent framework re-learns each of these as a production incident. The 32GB cgroup and 2-job concurrency leave no headroom for exploratory agent memory behavior.
- **No per-page agent on big docs — the math:** 800 pages × ~10 agent turns × ~8k tokens (vision-bearing) ≈ 64M tokens ≈ **$80-250 per document** at current frontier pricing, and at ~5-15s/turn ≈ **11-30 hours** wall-clock against today's 35-minute ADASP run and no per-job timeout. Non-starter on cost, latency, and consistency (800 independent agents will not agree with each other).
- **Don't let a model touch geometry.** 157,184 lines on ADASP; 292k on real pages. Extraction, dedup, run decomposition, length math: deterministic forever.
- **Don't skip Stage 0/1 to jump to Stage 4.** The current "LLMs can't categorize" evidence is contaminated by the `pipeline.py:1015` bug. Measure post-fix before buying agent complexity.
- **Don't ship any new LLM phase without metering** — v2 records zero spend today and `MAX_DAILY_SPEND_USD=0` means unlimited (`config.py:222`).

---

## 5. Rough cost per document

Pricing caveat: I don't have verified current OpenAI list prices; ranges below assume roughly $1-2.5/M input, $8-12/M output for gpt-5.1-class and ~5-10× cheaper for gpt-5-mini-class — **verify against the current price sheet before budgeting.** "Today" figures derive from measured call volumes (≈15-16 gpt-5.1 calls per fence page for legend extraction; ADASP took ~476 vision-bearing scale calls).

| Document | Today (est.) | + Stage 2 registry | + Stage 3 vision verify | + Stage 4 agent | New total (est.) | Wall-clock delta |
|---|---|---|---|---|---|---|
| Small plan set (8 pg, ~4 fence) | $0.10-0.50 | +$0.02-0.05 | +$0.05-0.20 | rarely triggers | **$0.2-0.8** | +~1 min |
| Typical plans (80-120 pg, ~60-80 fence) | $2-8 | +$0.05-0.30 | +$0.50-2 | +$1-5 | **$4-15** | +5-10 min |
| ADASP-class (793 pg, ~240 fence) | $8-30 | +$0.20-0.50 | +$2-8 | +$3-10 | **$13-50** | +10-20 min |

Two structural notes: (a) the fence cache is per-job scoped (`api_server.py:247`, `SHARED_PHASES` empty) so every re-analysis re-pays the full bill — the `reanalyze` docstring's "no new LLM spend" claim (`api_server.py:738-740`) is currently false; content-addressed cross-job sharing of deterministic phases is a real cost lever once you trust the cache. (b) Per-chunk legend extraction (~15 calls/page) is today's dominant spend and is batchable (the batch variant exists at `llm.py:96-191` but only non-live workers use it) — batching could cut current cost 3-5× and fund Stages 3-4 outright.

**Bottom line:** current LLMs plus this architecture can plausibly reach "review-grade" auto-takeoff (high precision, honest flagging of the rest). They cannot reach "unreviewed-bid-grade," and you shouldn't sell that. The fastest path to visibly better results is not agentic at all: it's `pipeline.py:1015`, prompt de-biasing, and a legend registry — with an eval harness so we can prove it.