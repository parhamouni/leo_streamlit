"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Stage, Layer, Image as KImage, Line as KLine } from "react-konva";
import { apiFetch, apiJson } from "@/lib/api";

type VectorLine = {
  idx: number;
  start: [number, number];
  end: [number, number];
  length_pts: number;
  layer: string;
};

type VectorLinesResponse = {
  page_num: number;
  page_idx: number;
  rotation: number;
  pdf_width: number;
  pdf_height: number;
  verified_scale?: number | null;
  lines: VectorLine[];
  auto_categories: Record<string, CategoryInfo>;
  auto_assignments: Record<string, string>;
  source_missing?: boolean;
};

type UserDrawnLine = {
  start: [number, number];
  end: [number, number];
  category: string;
};

type CategoryInfo = {
  indicator?: string;
  keyword?: string;
  color: [number, number, number];
};

type PageState = {
  categories: Record<string, CategoryInfo>;
  line_assignments: Record<string, string>;
  user_drawn_lines: UserDrawnLine[];
  scale_override?: number;
  min_line_pts?: number;
};

type UmtState = {
  version: number;
  pages: Record<string, PageState>;
};

const PALETTE: [number, number, number][] = [
  [0, 255, 0],
  [255, 165, 0],
  [0, 191, 255],
  [255, 0, 255],
  [255, 255, 0],
  [0, 255, 255],
  [255, 105, 180],
  [173, 255, 47],
];

const UNASSIGNED_RGB: [number, number, number] = [220, 38, 38]; // red-600

function rgb([r, g, b]: [number, number, number], alpha = 1) {
  return `rgba(${r},${g},${b},${alpha})`;
}

function scaleInchesToPointsPerFoot(scaleInches?: number | null) {
  const scale = Number(scaleInches);
  if (!Number.isFinite(scale) || scale <= 0) return 864 / 360;
  return 864 / scale;
}

// Drop the pipeline's "indicator code" placeholders (where the keyword is
// just the indicator number, e.g. "11: 11"). Mirrors `_clean_legend_entries`
// on the backend so the chip list matches what `auto_categories` returns.
function isPlaceholderCategory(name: string, info: CategoryInfo): boolean {
  const ind = (info.indicator ?? "").trim();
  const kw = (info.keyword ?? "").trim();
  if (ind && kw && ind === kw) return true;
  // Defensive fallback when only the name is available.
  const m = name.match(/^([^:]+):\s*(.+)$/);
  if (m && m[1].trim() === m[2].trim()) return true;
  return false;
}

function cleanCategoryMap(
  cats: Record<string, CategoryInfo>,
): Record<string, CategoryInfo> {
  const out: Record<string, CategoryInfo> = {};
  for (const [name, info] of Object.entries(cats)) {
    if (!isPlaceholderCategory(name, info)) out[name] = info;
  }
  return out;
}

function defaultCategoriesFromLegend(
  legend: Array<{ indicator?: string; keyword?: string }>,
): Record<string, CategoryInfo> {
  const cats: Record<string, CategoryInfo> = {};
  for (const le of legend) {
    const indicator = (le.indicator ?? "").trim();
    const keyword = (le.keyword ?? "").trim();
    if (!keyword) continue;
    if (indicator && indicator === keyword) continue;
    const name = indicator ? `${indicator}: ${keyword}` : keyword;
    if (cats[name]) continue;
    cats[name] = {
      indicator,
      keyword,
      color: PALETTE[Object.keys(cats).length % PALETTE.length],
    };
  }
  return cats;
}

export default function UMTCanvasInner({
  jobId,
  pageNum,
  legendEntries,
  initiallyOpen = false,
  skipReason = null,
  scaleOverride,
  onSaved,
}: {
  jobId: string;
  pageNum: number;
  legendEntries: Array<{ indicator?: string; keyword?: string }>;
  initiallyOpen?: boolean;
  skipReason?: string | null;
  scaleOverride?: number | null;
  onSaved?: () => void;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [vectorData, setVectorData] = useState<VectorLinesResponse | null>(null);
  const [pageState, setPageState] = useState<PageState>({
    categories: {},
    line_assignments: {},
    user_drawn_lines: [],
  });
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">(
    "idle",
  );
  const [drawMode, setDrawMode] = useState(false);
  const [pendingLine, setPendingLine] = useState<{
    start: [number, number];
    end: [number, number];
  } | null>(null);
  const [highlightedLayer, setHighlightedLayer] = useState<string | null>(null);
  const [linePopover, setLinePopover] = useState<
    | { kind: "vector"; idx: number; x: number; y: number }
    | { kind: "drawn"; idx: number; x: number; y: number }
    | null
  >(null);
  const [zoom, setZoom] = useState(1.0);
  const [panMode, setPanMode] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const panStateRef = useRef<{
    startX: number;
    startY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  // Debounced save: schedule a single trailing PUT for the latest state.
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestStateRef = useRef<PageState>(pageState);
  latestStateRef.current = pageState;

  const scheduleSave = useCallback((nextState?: PageState) => {
    if (nextState) latestStateRef.current = nextState;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    setSaveStatus("saving");
    saveTimerRef.current = setTimeout(async () => {
      try {
        await apiFetch(`/api/jobs/${jobId}/umt-state/${pageNum}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(latestStateRef.current),
        });
        setSaveStatus("saved");
        onSaved?.();
      } catch (e) {
        console.error("UMT save failed", e);
        setSaveStatus("error");
      }
    }, 500);
  }, [jobId, onSaved, pageNum]);

  useEffect(() => {
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, []);

  // Read latest zoom from a ref so the wheel handler doesn't have to
  // re-bind on every zoom change.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  // Ctrl/Cmd + wheel zooms around the cursor; plain wheel scrolls
  // natively. We attach manually with passive: false so we can call
  // preventDefault to suppress the browser's page-zoom fallback.
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el || !open) return;
    const handler = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const oldZoom = zoomRef.current;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const newZoom = Math.max(
        0.25,
        Math.min(4, +(oldZoom * factor).toFixed(3)),
      );
      if (newZoom === oldZoom) return;
      // Keep the point under the cursor anchored: scrollLeft and
      // scrollTop adjust so cursorPdfPoint stays at the same screen pos
      // after zoom changes the content size.
      const oldContentX = el.scrollLeft + cx;
      const oldContentY = el.scrollTop + cy;
      const ratio = newZoom / oldZoom;
      setZoom(newZoom);
      requestAnimationFrame(() => {
        el.scrollLeft = oldContentX * ratio - cx;
        el.scrollTop = oldContentY * ratio - cy;
      });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [open, vectorData, bgImage]);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  useEffect(() => {
    if (scaleOverride === undefined) return;
    setPageState((prev) => {
      const current = prev.scale_override ?? null;
      const nextScale = scaleOverride ?? null;
      if (current === nextScale) return prev;
      const next = { ...prev };
      if (nextScale == null) delete next.scale_override;
      else next.scale_override = nextScale;
      latestStateRef.current = next;
      return next;
    });
  }, [scaleOverride]);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [open]);

  async function load() {
    if (!open && imageUrl) {
      setOpen(true);
      return;
    }
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    setLoading(true);
    setError(null);
    try {
      const [imgResp, vecData, stateData] = await Promise.all([
        apiFetch(`/api/jobs/${jobId}/page-image/${pageNum}?dpi=110`),
        apiJson<VectorLinesResponse>(
          `/api/jobs/${jobId}/page-vector-lines/${pageNum}`,
        ),
        apiJson<UmtState>(`/api/jobs/${jobId}/umt-state`),
      ]);
      const blob = await imgResp.blob();
      const url = URL.createObjectURL(blob);
      setImageUrl(url);

      const img = new window.Image();
      img.onload = () => setBgImage(img);
      img.src = url;

      setVectorData(vecData);

      const pageKey = `page_${pageNum}`;
      const existing = stateData.pages?.[pageKey];
      // Categories: saved wins when non-empty; otherwise prefer the
      // server-built palette (legend + auto-detected fallback bucket),
      // falling back to legend-only as a last resort.
      const savedCats = cleanCategoryMap(existing?.categories ?? {});
      const autoCats = cleanCategoryMap(vecData.auto_categories ?? {});
      const cats =
        Object.keys(savedCats).length > 0
          ? savedCats
          : Object.keys(autoCats).length > 0
            ? autoCats
            : defaultCategoriesFromLegend(legendEntries);
      // Line assignments: only suppress auto-seed when the user has
      // *actually assigned* something (or drawn lines). Saved categories
      // alone shouldn't block the auto-seed — that was the previous bug.
      const savedAssignments = existing?.line_assignments ?? {};
      const savedDrawn = existing?.user_drawn_lines ?? [];
      const rawAssignments =
        Object.keys(savedAssignments).length > 0 || savedDrawn.length > 0
          ? savedAssignments
          : (vecData.auto_assignments ?? {});
      // Drop assignments whose target category no longer exists (most
      // commonly because we just filtered out an indicator-code placeholder).
      const assignments: Record<string, string> = {};
      for (const [k, v] of Object.entries(rawAssignments)) {
        if (cats[v]) assignments[k] = v;
      }
      setPageState({
        categories: cats,
        line_assignments: assignments,
        user_drawn_lines: savedDrawn,
        scale_override: existing?.scale_override,
        min_line_pts: existing?.min_line_pts ?? 20,
      });
      const firstCat = Object.keys(cats)[0];
      setActiveCategory(firstCat ?? null);
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e);
      // Common case: backend 404 for pages with no vector data — surface
      // a friendlier message than "API 404 for /api/jobs/.../page-vector-lines/X".
      const friendly = /\b404\b/.test(raw)
        ? "No measurement data is available for this page (the pipeline may have skipped it)."
        : raw;
      setError(friendly);
    } finally {
      setLoading(false);
    }
  }

  const baseScale = useMemo(() => {
    if (!vectorData) return 1;
    const targetW = Math.min(containerWidth || 800, 1400);
    return targetW / vectorData.pdf_width;
  }, [vectorData, containerWidth]);
  const scale = baseScale * zoom;

  const stageWidth = vectorData ? vectorData.pdf_width * scale : 0;
  const stageHeight = vectorData ? vectorData.pdf_height * scale : 0;

  const minLen = pageState.min_line_pts ?? 20;
  const detectedScaleInches = vectorData?.verified_scale ?? null;
  const activeScaleInches = pageState.scale_override ?? detectedScaleInches ?? 360;
  const measurementScale = scaleInchesToPointsPerFoot(activeScaleInches);

  const layerSummary = useMemo(() => {
    if (!vectorData)
      return [] as Array<{
        name: string;
        count: number;
        lengthPts: number;
        dominantCat: string | null;
      }>;
    const map = new Map<
      string,
      { count: number; lengthPts: number; catCounts: Map<string, number> }
    >();
    for (const ln of vectorData.lines) {
      const k = ln.layer || "(no layer)";
      const cur = map.get(k) ?? {
        count: 0,
        lengthPts: 0,
        catCounts: new Map<string, number>(),
      };
      cur.count += 1;
      cur.lengthPts += ln.length_pts;
      const cat = pageState.line_assignments[String(ln.idx)];
      if (cat) cur.catCounts.set(cat, (cur.catCounts.get(cat) ?? 0) + 1);
      map.set(k, cur);
    }
    return Array.from(map.entries())
      .map(([name, v]) => {
        let dominantCat: string | null = null;
        let bestCount = 0;
        v.catCounts.forEach((c, cat) => {
          if (c > bestCount) {
            bestCount = c;
            dominantCat = cat;
          }
        });
        return {
          name,
          count: v.count,
          lengthPts: v.lengthPts,
          dominantCat,
        };
      })
      .sort((a, b) => b.lengthPts - a.lengthPts);
  }, [vectorData, pageState.line_assignments]);

  // Explicit assign — used by per-line popover and per-layer dropdown.
  // Pass `null` as `cat` to unassign.
  function assignLineTo(lineIdx: number, cat: string | null) {
    setError(null);
    const next = { ...pageState, line_assignments: { ...pageState.line_assignments } };
    const k = String(lineIdx);
    if (cat === null) delete next.line_assignments[k];
    else next.line_assignments[k] = cat;
    setPageState(next);
    scheduleSave(next);
  }

  function assignLayerTo(layerName: string, cat: string | null) {
    if (!vectorData) return;
    setError(null);
    const matchingLines = vectorData.lines.filter(
      (ln) => (ln.layer || "(no layer)") === layerName,
    );
    if (matchingLines.length === 0) {
      setError(`No lines on layer "${layerName}".`);
      return;
    }
    const next = {
      ...pageState,
      line_assignments: { ...pageState.line_assignments },
    };
    for (const ln of matchingLines) {
      const k = String(ln.idx);
      if (cat === null) delete next.line_assignments[k];
      else next.line_assignments[k] = cat;
    }
    setPageState(next);
    scheduleSave(next);
  }

  function toggleAssignment(lineIdx: number) {
    if (!activeCategory) {
      setError("Pick or add a category first.");
      return;
    }
    setError(null);
    const key = String(lineIdx);
    const next = { ...pageState, line_assignments: { ...pageState.line_assignments } };
    if (next.line_assignments[key] === activeCategory) {
      delete next.line_assignments[key];
    } else {
      next.line_assignments[key] = activeCategory;
    }
    setPageState(next);
    scheduleSave(next);
  }

  function addCategory(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    if (pageState.categories[trimmed]) return;
    const idx = Object.keys(pageState.categories).length;
    const next = {
      ...pageState,
      categories: {
        ...pageState.categories,
        [trimmed]: {
          indicator: "",
          keyword: trimmed,
          color: PALETTE[idx % PALETTE.length],
        },
      },
    };
    setPageState(next);
    setActiveCategory(trimmed);
    scheduleSave(next);
  }

  function removeCategory(name: string) {
    const cats = { ...pageState.categories };
    delete cats[name];
    const assignments = { ...pageState.line_assignments };
    for (const k of Object.keys(assignments)) {
      if (assignments[k] === name) delete assignments[k];
    }
    const drawn = (pageState.user_drawn_lines ?? []).filter(
      (l) => l.category !== name,
    );
    const next = {
      ...pageState,
      categories: cats,
      line_assignments: assignments,
      user_drawn_lines: drawn,
    };
    setPageState(next);
    if (activeCategory === name) {
      setActiveCategory((prev) => {
        const remaining = Object.keys(cats);
        return remaining[0] ?? null;
      });
    }
    scheduleSave(next);
  }

  function setMinLen(v: number) {
    const next = { ...pageState, min_line_pts: v };
    setPageState(next);
    scheduleSave(next);
  }

  function clearAssignments() {
    if (!confirm("Clear all line assignments on this page?")) return;
    const next = { ...pageState, line_assignments: {} };
    setPageState(next);
    scheduleSave(next);
  }

  function assignLayer(layerName: string) {
    if (!activeCategory) {
      setError("Pick a target category first.");
      return;
    }
    if (!vectorData) return;
    setError(null);
    const matchingLines = vectorData.lines.filter(
      (ln) => (ln.layer || "(no layer)") === layerName,
    );
    if (matchingLines.length === 0) {
      setError(`No lines on layer "${layerName}".`);
      return;
    }
    const next = {
      ...pageState,
      line_assignments: { ...pageState.line_assignments },
    };
    for (const ln of matchingLines) {
      next.line_assignments[String(ln.idx)] = activeCategory;
    }
    setPageState(next);
    scheduleSave(next);
  }

  async function smartAutoAssign() {
    const existing = Object.keys(pageState.line_assignments).length;
    if (existing > 0) {
      if (
        !confirm(
          `Smart-assign will replace your ${existing} existing assignment(s) on this page. Continue?`,
        )
      ) {
        return;
      }
    }
    setError(null);
    try {
      const data = await apiJson<{
        assignments: Record<string, string>;
        layer_assignments: Record<
          string,
          { category: string | null; line_count: number }
        >;
        stats: {
          by_indicator: number;
          by_layer: number;
          unassigned: number;
          total: number;
          instance_count: number;
          layer_count: number;
          layers_assigned: number;
        };
      }>(
        `/api/jobs/${jobId}/page-vector-lines/${pageNum}/smart-assign`,
      );
      const newAssignments = data.assignments ?? {};
      // Make sure every assigned-to category exists in our category map —
      // smart-assign only references legend categories, but the user may
      // have deleted some locally.
      const cats = { ...pageState.categories };
      const distinctCats = Array.from(new Set(Object.values(newAssignments)));
      for (const c of distinctCats) {
        if (!cats[c]) {
          const idx = Object.keys(cats).length;
          cats[c] = {
            indicator: c.split(":")[0]?.trim() || "",
            keyword: c.split(":").slice(1).join(":").trim() || c,
            color: PALETTE[idx % PALETTE.length],
          };
        }
      }
      const next = {
        ...pageState,
        categories: cats,
        line_assignments: newAssignments,
      };
      setPageState(next);
      const s = data.stats;
      setError(
        `Smart-assigned ${s.layers_assigned} of ${s.layer_count} layers (${s.by_indicator} lines) using indicator-bbox proximity · ${s.unassigned} lines left unassigned (low-confidence layers + layers with no nearby indicator). Click a line to override manually.`,
      );
      scheduleSave(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function reassignAutoDetected() {
    if (!activeCategory) {
      setError("Pick the target category first.");
      return;
    }
    if (activeCategory === "Auto-detected") {
      setError(`"Auto-detected" is already the target — pick a different category.`);
      return;
    }
    setError(null);
    let moved = 0;
    const next = { ...pageState, line_assignments: { ...pageState.line_assignments } };
    for (const k of Object.keys(next.line_assignments)) {
      if (next.line_assignments[k] === "Auto-detected") {
        next.line_assignments[k] = activeCategory;
        moved += 1;
      }
    }
    if (moved === 0) {
      setError("No Auto-detected lines to reassign on this page.");
    } else {
      setPageState(next);
      scheduleSave(next);
    }
  }

  function pdfPointFromStage(
    e: { target: { getStage: () => unknown } },
  ): [number, number] | null {
    const stage = (e.target.getStage() as null | {
      getPointerPosition: () => { x: number; y: number } | null;
    });
    if (!stage) return null;
    const ptr = stage.getPointerPosition();
    if (!ptr || !vectorData) return null;
    const x = Math.max(0, Math.min(vectorData.pdf_width, ptr.x / scale));
    const y = Math.max(0, Math.min(vectorData.pdf_height, ptr.y / scale));
    return [x, y];
  }

  function startDrawAt(pt: [number, number]) {
    setPendingLine({ start: pt, end: pt });
  }

  function updateDrawTo(pt: [number, number]) {
    setPendingLine((prev) => (prev ? { start: prev.start, end: pt } : prev));
  }

  function commitPendingLine() {
    if (!pendingLine) return;
    const dx = pendingLine.end[0] - pendingLine.start[0];
    const dy = pendingLine.end[1] - pendingLine.start[1];
    const lenPts = Math.hypot(dx, dy);
    if (lenPts < 4) {
      // Treat tiny strokes as a click-cancel — discard.
      setPendingLine(null);
      return;
    }
    if (!activeCategory) {
      setPendingLine(null);
      setError("Pick a category before drawing.");
      return;
    }
    const start = pendingLine.start;
    const end = pendingLine.end;
    const cat = activeCategory;
    const next = {
      ...pageState,
      user_drawn_lines: [
        ...(pageState.user_drawn_lines ?? []),
        { start, end, category: cat },
      ],
    };
    setPageState(next);
    setPendingLine(null);
    scheduleSave(next);
  }

  function removeUserDrawnLine(idx: number) {
    const drawn = [...(pageState.user_drawn_lines ?? [])];
    drawn.splice(idx, 1);
    const next = { ...pageState, user_drawn_lines: drawn };
    setPageState(next);
    scheduleSave(next);
  }

  function reassignUserDrawnLine(idx: number, cat: string) {
    const arr = [...(pageState.user_drawn_lines ?? [])];
    if (idx < 0 || idx >= arr.length) return;
    arr[idx] = { ...arr[idx], category: cat };
    const next = { ...pageState, user_drawn_lines: arr };
    setPageState(next);
    scheduleSave(next);
  }

  function resetToAuto() {
    if (!vectorData) return;
    if (
      !confirm(
        "Replace your assignments + categories with the auto-detected ones for this page?",
      )
    )
      return;
    const autoCats = cleanCategoryMap(vectorData.auto_categories ?? {});
    const cats =
      Object.keys(autoCats).length > 0
        ? autoCats
        : defaultCategoriesFromLegend(legendEntries);
    const rawAuto = vectorData.auto_assignments ?? {};
    const filteredAuto: Record<string, string> = {};
    for (const [k, v] of Object.entries(rawAuto)) {
      if (cats[v]) filteredAuto[k] = v;
    }
    const next = {
      ...pageState,
      categories: cats,
      line_assignments: filteredAuto,
    };
    setPageState(next);
    const firstCat = Object.keys(cats)[0];
    setActiveCategory(firstCat ?? null);
    scheduleSave(next);
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={load}
        className="text-xs text-blue-600 hover:underline"
      >
        {loading
          ? "Loading measurement canvas…"
          : open
            ? "Hide measurement canvas"
            : "📐 Open measurement canvas"}
      </button>
      {error && <div className="text-xs text-red-600 break-all">{error}</div>}
      {open && (
        <div
          ref={containerRef}
          className="border rounded bg-gray-50 overflow-hidden"
        >
          {!vectorData || !bgImage ? (
            <div className="p-8 text-center text-xs text-gray-500">
              {loading ? "Loading…" : "No data"}
            </div>
          ) : (
            <>
              {(vectorData.source_missing || skipReason) && (
                <div className="px-3 py-2 text-xs bg-yellow-50 border-b border-yellow-200 text-yellow-800">
                  {vectorData.source_missing
                    ? "Original PDF is no longer on disk — reading vector lines from the highlighted PDF instead. Saved assignments still render; auto-overlay is suppressed (re-upload the document to refresh)."
                    : `No auto-detected lines on this page${skipReason ? ` (${skipReason})` : ""}. You can still add a category and draw lines manually.`}
                </div>
              )}
              <CategoryPanel
                categories={pageState.categories}
                active={activeCategory}
                setActive={setActiveCategory}
                addCategory={addCategory}
                removeCategory={removeCategory}
              />
              <div className="px-3 py-2 text-xs text-gray-700 border-y bg-white flex flex-wrap items-center gap-x-4 gap-y-1">
                <span className="font-mono">Page {pageNum}</span>
                <span>{vectorData.lines.length} vector lines</span>
                <span>
                  {Object.keys(pageState.line_assignments).length} assigned
                </span>
                <label className="flex items-center gap-1">
                  Min line (pts):
                  <input
                    type="number"
                    min={0}
                    step={1}
                    value={minLen}
                    onChange={(e) => setMinLen(Number(e.target.value) || 0)}
                    className="w-16 border rounded px-1 py-0.5 text-xs"
                  />
                </label>
                <div
                  className="flex items-center gap-1"
                  title="Tip: Ctrl/Cmd + mouse-wheel zooms around the cursor"
                >
                  <span>Zoom:</span>
                  <button
                    type="button"
                    onClick={() =>
                      setZoom((z) => Math.max(0.25, +(z * 0.8).toFixed(2)))
                    }
                    className="px-1.5 border rounded hover:bg-gray-50"
                    title="Zoom out"
                  >
                    −
                  </button>
                  <input
                    type="range"
                    min={0.25}
                    max={4}
                    step={0.05}
                    value={zoom}
                    onChange={(e) => setZoom(Number(e.target.value))}
                    className="w-24"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setZoom((z) => Math.min(4, +(z * 1.25).toFixed(2)))
                    }
                    className="px-1.5 border rounded hover:bg-gray-50"
                    title="Zoom in"
                  >
                    +
                  </button>
                  <span className="font-mono w-12 text-right">
                    {Math.round(zoom * 100)}%
                  </span>
                  {zoom !== 1 && (
                    <button
                      type="button"
                      onClick={() => setZoom(1)}
                      className="text-blue-600 hover:underline"
                      title="Reset zoom to 100%"
                    >
                      reset
                    </button>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setDrawMode((m) => !m);
                    setPanMode(false);
                    setPendingLine(null);
                    setError(null);
                  }}
                  className={`px-2 py-0.5 rounded border text-xs ${
                    drawMode
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 hover:bg-gray-50"
                  }`}
                  title="Click-drag on the canvas to draw a new line in the active category"
                >
                  {drawMode ? "✏️ Draw mode (on)" : "✏️ Draw mode"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPanMode((m) => !m);
                    setDrawMode(false);
                    setPendingLine(null);
                    setError(null);
                  }}
                  className={`px-2 py-0.5 rounded border text-xs ${
                    panMode
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 hover:bg-gray-50"
                  }`}
                  title="Click-drag on the canvas to pan around (useful at high zoom)"
                >
                  {panMode ? "✋ Pan mode (on)" : "✋ Pan mode"}
                </button>
                <button
                  type="button"
                  onClick={smartAutoAssign}
                  className="text-blue-700 hover:underline"
                  title="Match each line to a category using indicator-bbox proximity, then layer-name tokens. Replaces existing assignments on this page."
                >
                  🎯 Smart auto-assign (indicator + layer)
                </button>
                <button
                  type="button"
                  onClick={reassignAutoDetected}
                  className="text-blue-600 hover:underline"
                  title="Move every line currently in Auto-detected into the active category"
                >
                  Reassign Auto-detected → {activeCategory ?? "(pick category)"}
                </button>
                <button
                  type="button"
                  onClick={resetToAuto}
                  className="text-blue-600 hover:underline"
                  title="Restore auto-detected assignments + categories"
                >
                  Reset to auto
                </button>
                <button
                  type="button"
                  onClick={clearAssignments}
                  className="text-red-600 hover:underline"
                >
                  Clear assignments
                </button>
                <span className="ml-auto text-gray-500">
                  {saveStatus === "saving"
                    ? "Saving…"
                    : saveStatus === "saved"
                      ? "Saved"
                      : saveStatus === "error"
                        ? "Save failed"
                        : ""}
                </span>
              </div>
              {layerSummary.length > 0 && (
                <details className="px-3 py-2 text-xs bg-white border-b">
                  <summary className="cursor-pointer text-gray-700 select-none">
                    Layers ({layerSummary.length}) — click a row to highlight
                    its lines on the canvas; use the dropdown to assign the
                    whole layer
                    {highlightedLayer ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          setHighlightedLayer(null);
                        }}
                        className="ml-3 text-blue-600 hover:underline"
                      >
                        clear highlight
                      </button>
                    ) : null}
                  </summary>
                  <div className="mt-2 max-h-64 overflow-auto border rounded">
                    <table className="w-full text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr className="text-left text-gray-600">
                          <th className="px-2 py-1 font-medium">Layer</th>
                          <th className="px-2 py-1 font-medium text-right">
                            Lines
                          </th>
                          <th className="px-2 py-1 font-medium text-right">
                            Length (ft)
                          </th>
                          <th className="px-2 py-1 font-medium">Assigned to</th>
                        </tr>
                      </thead>
                      <tbody>
                        {layerSummary.map((l) => {
                          const isHi = highlightedLayer === l.name;
                          return (
                            <tr
                              key={l.name}
                              className={`border-t cursor-pointer ${
                                isHi
                                  ? "bg-yellow-100"
                                  : "hover:bg-blue-50/40"
                              }`}
                              onClick={() =>
                                setHighlightedLayer((prev) =>
                                  prev === l.name ? null : l.name,
                                )
                              }
                            >
                              <td className="px-2 py-1 font-mono break-all">
                                {l.name}
                              </td>
                              <td className="px-2 py-1 text-right tabular-nums">
                                {l.count}
                              </td>
                              <td className="px-2 py-1 text-right tabular-nums">
                                {(l.lengthPts / measurementScale).toFixed(1)}
                              </td>
                              <td
                                className="px-2 py-1"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <select
                                  value={l.dominantCat ?? ""}
                                  onChange={(e) =>
                                    assignLayerTo(
                                      l.name,
                                      e.target.value || null,
                                    )
                                  }
                                  className="border rounded px-1 py-0.5 text-xs max-w-[14rem]"
                                  title="Assign all lines on this layer to the chosen category (or unassign)"
                                >
                                  <option value="">— unassigned —</option>
                                  {Object.keys(pageState.categories).map(
                                    (c) => (
                                      <option key={c} value={c}>
                                        {c.length > 60
                                          ? c.slice(0, 57) + "…"
                                          : c}
                                      </option>
                                    ),
                                  )}
                                </select>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </details>
              )}
              <div
                ref={scrollContainerRef}
                className="overflow-auto max-h-[80vh] select-none"
                style={{
                  cursor: panMode
                    ? panStateRef.current
                      ? "grabbing"
                      : "grab"
                    : "default",
                }}
                onMouseDown={(e) => {
                  if (!panMode || !scrollContainerRef.current) return;
                  panStateRef.current = {
                    startX: e.clientX,
                    startY: e.clientY,
                    scrollLeft: scrollContainerRef.current.scrollLeft,
                    scrollTop: scrollContainerRef.current.scrollTop,
                  };
                  scrollContainerRef.current.style.cursor = "grabbing";
                  e.preventDefault();
                }}
                onMouseMove={(e) => {
                  const ps = panStateRef.current;
                  if (!panMode || !ps || !scrollContainerRef.current) return;
                  scrollContainerRef.current.scrollLeft =
                    ps.scrollLeft - (e.clientX - ps.startX);
                  scrollContainerRef.current.scrollTop =
                    ps.scrollTop - (e.clientY - ps.startY);
                }}
                onMouseUp={() => {
                  panStateRef.current = null;
                  if (scrollContainerRef.current) {
                    scrollContainerRef.current.style.cursor = panMode
                      ? "grab"
                      : "default";
                  }
                }}
                onMouseLeave={() => {
                  panStateRef.current = null;
                }}
              >
                <Stage
                  width={stageWidth}
                  height={stageHeight}
                  onMouseDown={(e) => {
                    if (panMode) return;
                    if (!drawMode) return;
                    // Only start drawing when the click lands on empty
                    // stage. If it landed on a line, that line's own
                    // onClick will open the reassignment popover.
                    if (e.target !== e.target.getStage()) return;
                    if (!activeCategory) {
                      setError("Pick a category before drawing.");
                      return;
                    }
                    const pt = pdfPointFromStage(e);
                    if (pt) startDrawAt(pt);
                  }}
                  onMouseMove={(e) => {
                    if (!drawMode || !pendingLine) return;
                    const pt = pdfPointFromStage(e);
                    if (pt) updateDrawTo(pt);
                  }}
                  onMouseUp={() => {
                    if (drawMode) commitPendingLine();
                  }}
                  onMouseLeave={() => {
                    if (drawMode) commitPendingLine();
                  }}
                  style={{
                    cursor: panMode
                      ? "inherit"
                      : drawMode
                        ? "crosshair"
                        : "default",
                  }}
                >
                  <Layer listening={false}>
                    <KImage
                      image={bgImage}
                      x={0}
                      y={0}
                      width={stageWidth}
                      height={stageHeight}
                    />
                  </Layer>
                  <Layer listening={!panMode}>
                    {vectorData.lines.map((ln) => {
                      if (ln.length_pts < minLen) return null;
                      const cat = pageState.line_assignments[String(ln.idx)];
                      const catInfo = cat
                        ? pageState.categories[cat]
                        : undefined;
                      const color = catInfo?.color ?? UNASSIGNED_RGB;
                      const assigned = !!cat;
                      const onLayer =
                        highlightedLayer !== null &&
                        (ln.layer || "(no layer)") === highlightedLayer;
                      const stroke = onLayer
                        ? "rgba(255,235,0,1)"
                        : rgb(color, assigned ? 0.95 : 0.55);
                      const strokeWidth = onLayer ? 4 : assigned ? 3 : 1.4;
                      return (
                        <KLine
                          key={`v-${ln.idx}`}
                          points={[
                            ln.start[0] * scale,
                            ln.start[1] * scale,
                            ln.end[0] * scale,
                            ln.end[1] * scale,
                          ]}
                          stroke={stroke}
                          strokeWidth={strokeWidth}
                          hitStrokeWidth={10}
                          perfectDrawEnabled={false}
                          shadowForStrokeEnabled={false}
                          onClick={(e) => {
                            const evt = e.evt as MouseEvent;
                            if (evt.shiftKey) {
                              toggleAssignment(ln.idx);
                              return;
                            }
                            setLinePopover({
                              kind: "vector",
                              idx: ln.idx,
                              x: evt.clientX,
                              y: evt.clientY,
                            });
                          }}
                          onTap={(e) => {
                            const stage = e.target.getStage();
                            const ptr = stage?.getPointerPosition();
                            const rect = stage?.container().getBoundingClientRect();
                            if (ptr && rect) {
                              setLinePopover({
                                kind: "vector",
                                idx: ln.idx,
                                x: rect.left + ptr.x,
                                y: rect.top + ptr.y,
                              });
                            }
                          }}
                          onMouseEnter={(e) => {
                            const stage = e.target.getStage();
                            if (stage) stage.container().style.cursor = "pointer";
                          }}
                          onMouseLeave={(e) => {
                            const stage = e.target.getStage();
                            if (stage) stage.container().style.cursor = "default";
                          }}
                        />
                      );
                    })}
                    {pageState.user_drawn_lines.map((ln, i) => {
                      const catInfo = pageState.categories[ln.category];
                      const color = catInfo?.color ?? UNASSIGNED_RGB;
                      return (
                        <KLine
                          key={`u-${i}`}
                          points={[
                            ln.start[0] * scale,
                            ln.start[1] * scale,
                            ln.end[0] * scale,
                            ln.end[1] * scale,
                          ]}
                          stroke={rgb(color, 1)}
                          strokeWidth={3}
                          hitStrokeWidth={10}
                          listening={!panMode}
                          onClick={(e) => {
                            const evt = e.evt as MouseEvent;
                            setLinePopover({
                              kind: "drawn",
                              idx: i,
                              x: evt.clientX,
                              y: evt.clientY,
                            });
                          }}
                          onTap={(e) => {
                            const stage = e.target.getStage();
                            const ptr = stage?.getPointerPosition();
                            const rect = stage?.container().getBoundingClientRect();
                            if (ptr && rect) {
                              setLinePopover({
                                kind: "drawn",
                                idx: i,
                                x: rect.left + ptr.x,
                                y: rect.top + ptr.y,
                              });
                            }
                          }}
                          onMouseEnter={(e) => {
                            if (drawMode || panMode) return;
                            const stage = e.target.getStage();
                            if (stage) stage.container().style.cursor = "pointer";
                          }}
                          onMouseLeave={(e) => {
                            if (drawMode || panMode) return;
                            const stage = e.target.getStage();
                            if (stage) stage.container().style.cursor = "default";
                          }}
                        />
                      );
                    })}
                    {pendingLine && (
                      <KLine
                        points={[
                          pendingLine.start[0] * scale,
                          pendingLine.start[1] * scale,
                          pendingLine.end[0] * scale,
                          pendingLine.end[1] * scale,
                        ]}
                        stroke={rgb(
                          (activeCategory &&
                            pageState.categories[activeCategory]?.color) ||
                            UNASSIGNED_RGB,
                          0.85,
                        )}
                        strokeWidth={2.5}
                        dash={[6, 4]}
                        listening={false}
                      />
                    )}
                  </Layer>
                </Stage>
              </div>
            </>
          )}
        </div>
      )}
      {linePopover && vectorData && linePopover.kind === "vector" && (() => {
        const ln = vectorData.lines[linePopover.idx];
        const current =
          pageState.line_assignments[String(linePopover.idx)] ?? null;
        return (
          <LinePopover
            x={linePopover.x}
            y={linePopover.y}
            header={
              <>
                <div className="font-medium text-gray-800">
                  Line #{ln?.idx ?? "?"}
                </div>
                <div className="font-mono break-all">
                  layer: {ln?.layer || "(no layer)"}
                </div>
                <div>
                  length: {ln?.length_pts.toFixed(1) ?? "?"} pts · current:{" "}
                  {current ?? "— unassigned —"}
                </div>
              </>
            }
            currentCategory={current}
            categories={pageState.categories}
            onPick={(cat) => {
              assignLineTo(linePopover.idx, cat);
              setLinePopover(null);
            }}
            onRemove={() => {
              assignLineTo(linePopover.idx, null);
              setLinePopover(null);
            }}
            removeLabel="✕ Unassign"
            onClose={() => setLinePopover(null)}
          />
        );
      })()}
      {linePopover && linePopover.kind === "drawn" && (() => {
        const arr = pageState.user_drawn_lines ?? [];
        const dl = arr[linePopover.idx];
        if (!dl) return null;
        const dx = dl.end[0] - dl.start[0];
        const dy = dl.end[1] - dl.start[1];
        const lenPts = Math.hypot(dx, dy);
        return (
          <LinePopover
            x={linePopover.x}
            y={linePopover.y}
            header={
              <>
                <div className="font-medium text-gray-800">
                  Drawn line #{linePopover.idx + 1}
                </div>
                <div>
                  length: {lenPts.toFixed(1)} pts · current: {dl.category}
                </div>
              </>
            }
            currentCategory={dl.category}
            categories={pageState.categories}
            onPick={(cat) => {
              if (cat) reassignUserDrawnLine(linePopover.idx, cat);
              setLinePopover(null);
            }}
            onRemove={() => {
              removeUserDrawnLine(linePopover.idx);
              setLinePopover(null);
            }}
            removeLabel="🗑 Delete drawn line"
            onClose={() => setLinePopover(null)}
          />
        );
      })()}
    </div>
  );
}

function LinePopover({
  x,
  y,
  header,
  currentCategory,
  categories,
  onPick,
  onRemove,
  removeLabel,
  onClose,
}: {
  x: number;
  y: number;
  header: React.ReactNode;
  currentCategory: string | null;
  categories: Record<string, CategoryInfo>;
  onPick: (cat: string) => void;
  onRemove: () => void;
  removeLabel: string;
  onClose: () => void;
}) {
  const W = 320;
  const left = Math.min(Math.max(8, x + 8), window.innerWidth - W - 8);
  const top = Math.min(y + 8, window.innerHeight - 240);
  return (
    <>
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
        onContextMenu={(e) => {
          e.preventDefault();
          onClose();
        }}
      />
      <div
        role="dialog"
        className="fixed z-50 bg-white border rounded shadow-lg p-2 text-xs"
        style={{ left, top, width: W }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-1 pb-1 border-b mb-1 text-gray-600">{header}</div>
        <div className="max-h-56 overflow-auto">
          {Object.entries(categories).map(([name, info]) => (
            <button
              key={name}
              type="button"
              onClick={() => onPick(name)}
              className={`flex items-center gap-2 w-full px-2 py-1 rounded text-left hover:bg-blue-50 ${
                currentCategory === name ? "bg-blue-100" : ""
              }`}
            >
              <span
                className="inline-block w-3 h-3 rounded-sm border border-gray-300 shrink-0"
                style={{
                  backgroundColor: `rgb(${info.color[0]},${info.color[1]},${info.color[2]})`,
                }}
              />
              <span className="truncate">{name}</span>
            </button>
          ))}
          <button
            type="button"
            onClick={onRemove}
            className="w-full px-2 py-1 mt-1 rounded text-left text-red-600 hover:bg-red-50"
          >
            {removeLabel}
          </button>
        </div>
      </div>
    </>
  );
}

function CategoryPanel({
  categories,
  active,
  setActive,
  addCategory,
  removeCategory,
}: {
  categories: Record<string, CategoryInfo>;
  active: string | null;
  setActive: (c: string | null) => void;
  addCategory: (name: string) => void;
  removeCategory: (name: string) => void;
}) {
  const [newName, setNewName] = useState("");
  const names = Object.keys(categories);

  return (
    <div className="px-3 py-2 bg-white border-b flex flex-wrap items-center gap-2">
      <span className="text-xs font-medium text-gray-700">Categories:</span>
      {names.length === 0 && (
        <span className="text-xs text-gray-500 italic">
          none — add one to start assigning lines
        </span>
      )}
      {names.map((name) => {
        const info = categories[name];
        const isActive = active === name;
        return (
          <div
            key={name}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer border ${
              isActive
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:bg-gray-50"
            }`}
            onClick={() => setActive(name)}
          >
            <span
              className="inline-block w-3 h-3 rounded-sm border border-gray-300"
              style={{
                backgroundColor: `rgb(${info.color[0]},${info.color[1]},${info.color[2]})`,
              }}
            />
            <span>{name}</span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                if (confirm(`Delete category "${name}" and unassign its lines?`)) {
                  removeCategory(name);
                }
              }}
              className="text-gray-400 hover:text-red-600 ml-1"
              title="Delete"
            >
              ×
            </button>
          </div>
        );
      })}
      <form
        className="flex items-center gap-1 ml-2"
        onSubmit={(e) => {
          e.preventDefault();
          addCategory(newName);
          setNewName("");
        }}
      >
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="new category"
          className="border rounded px-2 py-0.5 text-xs w-32"
        />
        <button
          type="submit"
          className="text-xs px-2 py-0.5 border rounded hover:bg-gray-50"
        >
          Add
        </button>
      </form>
    </div>
  );
}
