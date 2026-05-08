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
  lines: VectorLine[];
  auto_categories: Record<string, CategoryInfo>;
  auto_assignments: Record<string, string>;
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

function defaultCategoriesFromLegend(
  legend: Array<{ indicator?: string; keyword?: string }>,
): Record<string, CategoryInfo> {
  const cats: Record<string, CategoryInfo> = {};
  for (const le of legend) {
    const indicator = (le.indicator ?? "").trim();
    const keyword = (le.keyword ?? "").trim();
    if (!keyword) continue;
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
}: {
  jobId: string;
  pageNum: number;
  legendEntries: Array<{ indicator?: string; keyword?: string }>;
  initiallyOpen?: boolean;
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

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  // Debounced save: schedule a single trailing PUT for the latest state.
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestStateRef = useRef<PageState>(pageState);
  latestStateRef.current = pageState;

  const scheduleSave = useCallback(() => {
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
      } catch (e) {
        console.error("UMT save failed", e);
        setSaveStatus("error");
      }
    }, 500);
  }, [jobId, pageNum]);

  useEffect(() => {
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

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
      const savedCats = existing?.categories ?? {};
      const autoCats = vecData.auto_categories ?? {};
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
      const assignments =
        Object.keys(savedAssignments).length > 0 || savedDrawn.length > 0
          ? savedAssignments
          : (vecData.auto_assignments ?? {});
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
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const scale = useMemo(() => {
    if (!vectorData) return 1;
    const targetW = Math.min(containerWidth || 800, 1400);
    return targetW / vectorData.pdf_width;
  }, [vectorData, containerWidth]);

  const stageWidth = vectorData ? vectorData.pdf_width * scale : 0;
  const stageHeight = vectorData ? vectorData.pdf_height * scale : 0;

  const minLen = pageState.min_line_pts ?? 20;

  function toggleAssignment(lineIdx: number) {
    if (!activeCategory) {
      setError("Pick or add a category first.");
      return;
    }
    setError(null);
    setPageState((prev) => {
      const key = String(lineIdx);
      const next = { ...prev, line_assignments: { ...prev.line_assignments } };
      if (next.line_assignments[key] === activeCategory) {
        delete next.line_assignments[key];
      } else {
        next.line_assignments[key] = activeCategory;
      }
      return next;
    });
    scheduleSave();
  }

  function addCategory(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    setPageState((prev) => {
      if (prev.categories[trimmed]) return prev;
      const idx = Object.keys(prev.categories).length;
      return {
        ...prev,
        categories: {
          ...prev.categories,
          [trimmed]: {
            indicator: "",
            keyword: trimmed,
            color: PALETTE[idx % PALETTE.length],
          },
        },
      };
    });
    setActiveCategory(trimmed);
    scheduleSave();
  }

  function removeCategory(name: string) {
    setPageState((prev) => {
      const cats = { ...prev.categories };
      delete cats[name];
      const assignments = { ...prev.line_assignments };
      for (const k of Object.keys(assignments)) {
        if (assignments[k] === name) delete assignments[k];
      }
      const drawn = (prev.user_drawn_lines ?? []).filter(
        (l) => l.category !== name,
      );
      return {
        ...prev,
        categories: cats,
        line_assignments: assignments,
        user_drawn_lines: drawn,
      };
    });
    if (activeCategory === name) {
      setActiveCategory((prev) => {
        const remaining = Object.keys(pageState.categories).filter(
          (c) => c !== name,
        );
        return remaining[0] ?? null;
      });
    }
    scheduleSave();
  }

  function setMinLen(v: number) {
    setPageState((prev) => ({ ...prev, min_line_pts: v }));
    scheduleSave();
  }

  function clearAssignments() {
    if (!confirm("Clear all line assignments on this page?")) return;
    setPageState((prev) => ({ ...prev, line_assignments: {} }));
    scheduleSave();
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
    setPageState((prev) => {
      const next = { ...prev, line_assignments: { ...prev.line_assignments } };
      for (const k of Object.keys(next.line_assignments)) {
        if (next.line_assignments[k] === "Auto-detected") {
          next.line_assignments[k] = activeCategory;
          moved += 1;
        }
      }
      return next;
    });
    if (moved === 0) {
      setError("No Auto-detected lines to reassign on this page.");
    } else {
      scheduleSave();
    }
  }

  function resetToAuto() {
    if (!vectorData) return;
    if (
      !confirm(
        "Replace your assignments + categories with the auto-detected ones for this page?",
      )
    )
      return;
    const autoCats = vectorData.auto_categories ?? {};
    const cats =
      Object.keys(autoCats).length > 0
        ? autoCats
        : defaultCategoriesFromLegend(legendEntries);
    setPageState((prev) => ({
      ...prev,
      categories: cats,
      line_assignments: vectorData.auto_assignments ?? {},
    }));
    const firstCat = Object.keys(cats)[0];
    setActiveCategory(firstCat ?? null);
    scheduleSave();
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
              <div className="overflow-auto max-h-[80vh]">
                <Stage width={stageWidth} height={stageHeight}>
                  <Layer listening={false}>
                    <KImage
                      image={bgImage}
                      x={0}
                      y={0}
                      width={stageWidth}
                      height={stageHeight}
                    />
                  </Layer>
                  <Layer>
                    {vectorData.lines.map((ln) => {
                      if (ln.length_pts < minLen) return null;
                      const cat = pageState.line_assignments[String(ln.idx)];
                      const catInfo = cat
                        ? pageState.categories[cat]
                        : undefined;
                      const color = catInfo?.color ?? UNASSIGNED_RGB;
                      const assigned = !!cat;
                      return (
                        <KLine
                          key={`v-${ln.idx}`}
                          points={[
                            ln.start[0] * scale,
                            ln.start[1] * scale,
                            ln.end[0] * scale,
                            ln.end[1] * scale,
                          ]}
                          stroke={rgb(color, assigned ? 0.95 : 0.55)}
                          strokeWidth={assigned ? 3 : 1.4}
                          hitStrokeWidth={10}
                          perfectDrawEnabled={false}
                          shadowForStrokeEnabled={false}
                          onClick={() => toggleAssignment(ln.idx)}
                          onTap={() => toggleAssignment(ln.idx)}
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
                          listening={false}
                        />
                      );
                    })}
                  </Layer>
                </Stage>
              </div>
            </>
          )}
        </div>
      )}
    </div>
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
