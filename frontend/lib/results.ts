/**
 * Shared types matching the pipeline result shape produced by
 * pipeline.run_analysis() / saved by job_registry.save_results().
 *
 * Fields are mostly optional because the upstream Python is loosely
 * typed and a value can legitimately be missing (e.g. when measurement
 * was skipped or LLM extraction returned no specs).
 */

export type ScaleInfo = {
  success?: boolean;
  verified_scale?: number;
  scale_text?: string;
  confidence?: "low" | "medium" | "high" | string;
  message?: string;
  method?: string;
  raw_response?: string;
  page_size?: {
    width_pts?: number;
    height_pts?: number;
    width_inches?: number;
    height_inches?: number;
    detected_size?: string;
    scale_factor?: number;
  };
};

export type LayerMeasurement = {
  segment_count?: number;
  total_length_pts?: number;
  total_length_feet?: number;
  connected_runs?: number;
  layer_name?: string;
};

export type DimensionMeasurement = {
  text?: string;
  page_num?: number;
  measured_length_feet?: number;
  measured_length_pts?: number;
  bbox?: number[];
  [k: string]: unknown;
};

export type IndicatorMeasurement = {
  segment_count?: number;
  instance_count?: number;
  total_length_feet?: number;
  total_length_pts?: number;
  [k: string]: unknown;
};

export type Measurements = {
  measurement_method?: string;
  skip_reason?: string;
  page_info?: {
    width?: number;
    height?: number;
    width_pts?: number;
    height_pts?: number;
    rotation?: number;
    scale_factor?: number;
    scale_detected?: boolean;
  };
  fence_layers?: unknown[];
  layer_to_category?: Record<string, string>;
  layer_measurements?: Record<string, LayerMeasurement>;
  indicator_measurements?: Record<string, IndicatorMeasurement>;
  dimension_measurements?: DimensionMeasurement[];
  proximity_totals?: {
    total_segments?: number;
    total_length_feet?: number;
    total_length_pts?: number;
  };
  totals?: {
    total_layers?: number;
    total_segments?: number;
    total_length_feet?: number;
  };
  all_fence_lines?: unknown[];
};

export type LegendEntry = {
  indicator?: string;
  keyword?: string;
  description?: string;
  source?: string;
  extraction_pass?: string;
  bbox?: number[];
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  [k: string]: unknown;
};

export type Instance = {
  id?: string;
  indicator?: string;
  type?: string;
  text?: string;
  bbox?: number[];
  page_num?: number;
  [k: string]: unknown;
};

export type AdeChunk = {
  id?: string;
  type?: string;
  text?: string;
  [k: string]: unknown;
};

export type FencePage = {
  page_idx: number;
  page_num: number;
  width?: number;
  height?: number;
  rotation?: number;
  fence_text?: string;
  ade_chunks?: AdeChunk[];
  definitions?: AdeChunk[]; // legend-type chunks
  instances?: Instance[]; // figure-type chunks
  keyword_matches?: Array<{
    keyword?: string;
    text?: string;
    page_num?: number;
    [k: string]: unknown;
  }>;
  legend_entries?: LegendEntry[];
  scale_info?: ScaleInfo;
  measurements?: Measurements;
  // LLM classification details when keyword fallback was used
  classification?: {
    confidence?: number;
    reasoning?: string;
    [k: string]: unknown;
  };
};

export type NonFencePage = {
  page_idx: number;
  page_num: number;
  fence_text?: string;
  // Why was this classified non-fence?
  classification?: {
    method?: string;
    confidence?: number;
    reasoning?: string;
    [k: string]: unknown;
  };
  // Legacy / top-level field names emitted by pipeline.py.
  method?: string;
  reason?: string;
  confidence?: number;
  signals?: string[];
  keyword_count?: number;
  keywords_found?: string[];
};

export type ElementSpec = {
  height?: string;
  post_type?: string;
  post_spacing?: string;
  top_rail?: string;
  bottom_rail?: string;
  material?: string;
  gauge?: string;
  mesh_size?: string;
  foundation?: string;
  gate_info?: string;
  detail_page?: string;
  full_details?: string;
  notes?: string;
  [k: string]: unknown;
};

export type PipelineResults = {
  fence_pages?: FencePage[];
  non_fence_pages?: NonFencePage[];
  element_details?: Record<string, ElementSpec>;
  per_page_scale_info?: Record<string, ScaleInfo>;
  unified_measurements?: Record<string, Measurements>;
  page_categories?: Record<string, string>;
  total_pages?: number;
  timings?: Record<string, number>;
  error?: string | null;
};

// --- helpers ---

export type DetectionMethod =
  | "ade"
  | "keyword_llm"
  | "keyword"
  | "none";

/**
 * Determine which detection path produced the fence-page result, mirroring
 * the badge logic in app_ade_prod.py around lines 5009-5093.
 */
export function detectionMethod(p: FencePage): DetectionMethod {
  const hasAde = (p.definitions?.length ?? 0) > 0 || (p.instances?.length ?? 0) > 0;
  if (hasAde) return "ade";
  const kw = p.keyword_matches?.length ?? 0;
  if (kw > 0 && p.classification) return "keyword_llm";
  if (kw > 0) return "keyword";
  return "none";
}

export function detectionLabel(m: DetectionMethod): {
  label: string;
  className: string;
  icon: string;
} {
  switch (m) {
    case "ade":
      return {
        label: "ADE Detection",
        className: "bg-emerald-100 text-emerald-800",
        icon: "🎯",
      };
    case "keyword_llm":
      return {
        label: "Keyword + LLM",
        className: "bg-amber-100 text-amber-800",
        icon: "🔍",
      };
    case "keyword":
      return {
        label: "Keyword-based",
        className: "bg-amber-100 text-amber-800",
        icon: "🔤",
      };
    case "none":
      return {
        label: "No detection",
        className: "bg-gray-100 text-gray-700",
        icon: "❌",
      };
  }
}

/** Truthy spec value? Used to filter empty rows from element-spec tables. */
export function specHasContent(spec: ElementSpec): boolean {
  const fields: (keyof ElementSpec)[] = [
    "height",
    "post_type",
    "post_spacing",
    "top_rail",
    "bottom_rail",
    "material",
    "gauge",
    "mesh_size",
    "foundation",
    "gate_info",
    "detail_page",
    "full_details",
    "notes",
  ];
  return fields.some((f) => {
    const v = spec[f];
    return typeof v === "string" && v.trim().length > 0;
  });
}

/**
 * The element_details dict is keyed by the verbatim chunk text including
 * a leading <a id='…'></a> anchor. We strip that wrapper for display
 * and truncate.
 */
export function cleanElementKey(key: string, max = 80): string {
  // Strip leading anchor tag like `<a id='...'></a>\n\n` if present.
  const m = key.match(/^<a\s+id=['"][^'"]+['"]>\s*<\/a>\s*\n*\s*([\s\S]*)$/);
  const text = (m ? m[1] : key).trim();
  if (text.length <= max) return text;
  return text.slice(0, max) + "…";
}
