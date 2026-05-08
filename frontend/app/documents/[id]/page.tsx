"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { apiFetch, apiJson, ApiError } from "@/lib/api";

// --- Types ---------------------------------------------------------------

type DashboardDoc = {
  id: string;
  user_id: string;
  original_filename: string;
  storage_path: string;
  document_status: string;
  total_pages: number | null;
  created_at: string;
  latest_job_id: string | null;
  job_status: string | null;
  current_phase: string | null;
  progress_percent: number | null;
  error_message: string | null;
};

type ScaleInfo = {
  success?: boolean;
  verified_scale?: number;
  scale_text?: string;
  confidence?: "low" | "medium" | "high";
  message?: string;
  method?: string;
};

type Measurements = {
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
  measurement_method?: string;
  fence_layers?: unknown[];
  layer_measurements?: Record<string, unknown>;
};

type LegendEntry = {
  indicator?: string;
  description?: string;
  bbox?: number[];
  [k: string]: unknown;
};

type Instance = {
  indicator?: string;
  bbox?: number[];
  page_num?: number;
  [k: string]: unknown;
};

type FencePage = {
  page_idx: number;
  page_num: number;
  width?: number;
  height?: number;
  rotation?: number;
  fence_text?: string;
  ade_chunks?: unknown[];
  definitions?: LegendEntry[];
  instances?: Instance[];
  keyword_matches?: unknown[];
  legend_entries?: LegendEntry[];
  scale_info?: ScaleInfo;
  measurements?: Measurements;
};

type NonFencePage = {
  page_idx: number;
  page_num: number;
  fence_text?: string;
};

type PipelineResults = {
  fence_pages?: FencePage[];
  non_fence_pages?: NonFencePage[];
  total_pages?: number;
  timings?: Record<string, number>;
  error?: string | null;
};

const POLL_MS = 3000;

// --- Helpers -------------------------------------------------------------

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatDuration(timings?: Record<string, number>): string | null {
  if (!timings) return null;
  const total = Object.values(timings).reduce((a, b) => a + b, 0);
  if (!total) return null;
  if (total < 60) return `${total.toFixed(1)}s`;
  const m = Math.floor(total / 60);
  const s = Math.round(total - m * 60);
  return `${m}m ${s}s`;
}

const STATUS_CLASSES: Record<string, string> = {
  queued: "bg-gray-100 text-gray-800",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  cancelled: "bg-yellow-100 text-yellow-800",
};

function StatusBadge({ status }: { status: string | null }) {
  const label = status ?? "—";
  const cls = STATUS_CLASSES[label] ?? "bg-gray-100 text-gray-800";
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}
    >
      {label}
    </span>
  );
}

// --- Page ----------------------------------------------------------------

export default function DocumentDetailPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const docId = params.id;

  const [authReady, setAuthReady] = useState(false);
  const [doc, setDoc] = useState<DashboardDoc | null>(null);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"fence" | "all">("fence");
  const [downloading, setDownloading] = useState(false);

  // Auth gate
  useEffect(() => {
    let active = true;
    supabase()
      .auth.getSession()
      .then(({ data }) => {
        if (!active) return;
        if (!data.session) {
          router.replace("/login");
          return;
        }
        setAuthReady(true);
      });
    return () => {
      active = false;
    };
  }, [router]);

  // Fetch document + (if completed) results
  const refresh = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        const found = await apiJson<DashboardDoc>(
          `/api/documents/${docId}`,
        );
        setDoc(found);
        setError(null);

        if (found.job_status === "completed" && found.latest_job_id) {
          const r = await apiJson<PipelineResults>(
            `/api/jobs/${found.latest_job_id}/results`,
          );
          setResults(r);
        } else {
          setResults(null);
        }
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          setError("Document not found or you don't have access");
        } else {
          const msg =
            e instanceof ApiError
              ? `API ${e.status}: ${typeof e.body === "string" ? e.body : JSON.stringify(e.body)}`
              : e instanceof Error
                ? e.message
                : String(e);
          setError(msg);
        }
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [docId],
  );

  useEffect(() => {
    if (authReady) refresh(false);
  }, [authReady, refresh]);

  // Poll while running/queued
  const docRef = useRef<DashboardDoc | null>(null);
  docRef.current = doc;
  useEffect(() => {
    if (!authReady || !doc) return;
    if (doc.job_status !== "queued" && doc.job_status !== "running") return;

    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = () => {
      if (!alive) return;
      if (typeof document !== "undefined" && document.hidden) return;
      refresh(true).finally(() => {
        const stillActive =
          docRef.current?.job_status === "queued" ||
          docRef.current?.job_status === "running";
        if (alive && stillActive) timer = setTimeout(tick, POLL_MS);
      });
    };

    timer = setTimeout(tick, POLL_MS);

    const onVisibility = () => {
      if (!alive) return;
      if (!document.hidden) {
        if (timer) clearTimeout(timer);
        tick();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [authReady, doc, refresh]);

  // Fetch the highlighted PDF blob once, when job completes
  useEffect(() => {
    if (!doc || doc.job_status !== "completed" || !doc.latest_job_id) {
      return;
    }
    if (pdfBlobUrl) return; // already loaded

    let active = true;
    let url: string | null = null;
    apiFetch(`/api/jobs/${doc.latest_job_id}/highlighted-pdf`)
      .then((r) => r.blob())
      .then((blob) => {
        if (!active) return;
        url = URL.createObjectURL(blob);
        setPdfBlobUrl(url);
      })
      .catch(() => {
        // Highlighted PDF may not exist if pipeline didn't generate one
      });

    return () => {
      active = false;
      if (url) URL.revokeObjectURL(url);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc?.job_status, doc?.latest_job_id]);

  async function onDownload() {
    if (!doc?.latest_job_id) return;
    setDownloading(true);
    try {
      const resp = await apiFetch(`/api/jobs/${doc.latest_job_id}/highlighted-pdf`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `fence_${doc.original_filename}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(`Download failed: ${msg}`);
    } finally {
      setDownloading(false);
    }
  }

  // --- Render ---

  if (!authReady || (loading && !doc)) {
    return (
      <main className="min-h-screen flex items-center justify-center text-gray-500">
        Loading…
      </main>
    );
  }

  if (error && !doc) {
    return (
      <main className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-3xl mx-auto bg-white rounded-lg shadow p-8 space-y-3">
          <Link
            href="/dashboard"
            className="text-sm text-blue-600 hover:underline"
          >
            ← Back to dashboard
          </Link>
          <div className="text-red-700">{error}</div>
        </div>
      </main>
    );
  }

  if (!doc) return null;

  const fencePages = (results?.fence_pages ?? []).filter(
    (p): p is FencePage => p && typeof p === "object",
  );
  const nonFencePages = results?.non_fence_pages ?? [];
  const totalCount = fencePages.length + nonFencePages.length;

  // Sum total fence length across all fence pages
  const totalLengthFt = fencePages.reduce((acc, p) => {
    const ft =
      p.measurements?.proximity_totals?.total_length_feet ??
      p.measurements?.totals?.total_length_feet ??
      0;
    return acc + (ft || 0);
  }, 0);

  const isActive =
    doc.job_status === "queued" || doc.job_status === "running";
  const isComplete = doc.job_status === "completed";

  return (
    <main className="min-h-screen bg-gray-50 p-6 sm:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Back link */}
        <Link
          href="/dashboard"
          className="inline-block text-sm text-blue-600 hover:underline"
        >
          ← Back to dashboard
        </Link>

        {/* Header card */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <h1
                className="text-xl font-semibold truncate"
                title={doc.original_filename}
              >
                {doc.original_filename}
              </h1>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-gray-600">
                <StatusBadge status={doc.job_status} />
                <span>{doc.total_pages ?? "—"} pages</span>
                <span>•</span>
                <span>uploaded {formatDate(doc.created_at)}</span>
                {results?.timings && formatDuration(results.timings) && (
                  <>
                    <span>•</span>
                    <span>analysed in {formatDuration(results.timings)}</span>
                  </>
                )}
              </div>
              {isActive && (
                <div className="mt-3">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    {doc.current_phase && (
                      <span className="font-mono">{doc.current_phase}</span>
                    )}
                    <span className="font-mono">{doc.progress_percent ?? 0}%</span>
                  </div>
                  <div className="mt-1 h-1.5 w-full max-w-md bg-gray-200 rounded overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-700 ease-out"
                      style={{ width: `${doc.progress_percent ?? 0}%` }}
                    />
                  </div>
                </div>
              )}
              {doc.error_message && (
                <div className="mt-3 text-sm text-red-700 break-all">
                  {doc.error_message}
                </div>
              )}
            </div>
            {isComplete && (
              <button
                onClick={onDownload}
                disabled={downloading}
                className="rounded bg-black text-white px-4 py-2 text-sm hover:bg-gray-800 disabled:opacity-60 whitespace-nowrap"
              >
                {downloading ? "Preparing…" : "Download highlighted PDF"}
              </button>
            )}
          </div>
        </section>

        {/* Summary stats */}
        {isComplete && results && (
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Total pages" value={String(totalCount || results.total_pages || doc.total_pages || "—")} />
            <StatCard label="Fence pages" value={String(fencePages.length)} accent="green" />
            <StatCard label="Non-fence pages" value={String(nonFencePages.length)} accent="gray" />
            <StatCard label="Total fence length" value={totalLengthFt > 0 ? `${totalLengthFt.toFixed(1)} ft` : "—"} accent={totalLengthFt > 0 ? "green" : "gray"} />
          </section>
        )}

        {/* Embedded highlighted PDF viewer */}
        {isComplete && pdfBlobUrl && (
          <section className="bg-white rounded-lg shadow overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">Highlighted PDF</h2>
              <span className="text-xs text-gray-500">
                green = legend definitions • purple = fence indicators • orange = keyword matches • cyan = measured fence lines
              </span>
            </div>
            <iframe
              src={pdfBlobUrl}
              className="w-full h-[80vh] border-0"
              title="Highlighted fence PDF"
            />
          </section>
        )}

        {/* Per-page details */}
        {isComplete && results && (
          <section className="bg-white rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">Pages</h2>
              <div className="flex gap-1 text-xs">
                <FilterChip
                  active={filter === "fence"}
                  onClick={() => setFilter("fence")}
                >
                  Fence ({fencePages.length})
                </FilterChip>
                <FilterChip
                  active={filter === "all"}
                  onClick={() => setFilter("all")}
                >
                  All ({totalCount})
                </FilterChip>
              </div>
            </div>

            <div className="divide-y">
              {filter === "fence" ? (
                fencePages.length === 0 ? (
                  <div className="p-6 text-sm text-gray-500 text-center">
                    No fence pages detected.
                  </div>
                ) : (
                  fencePages.map((p) => <FencePageCard key={p.page_num} page={p} />)
                )
              ) : (
                <NonFenceList nonFence={nonFencePages} fence={fencePages} />
              )}
            </div>
          </section>
        )}

        {/* Status messages while incomplete */}
        {!isComplete && !doc.error_message && (
          <section className="bg-white rounded-lg shadow p-6 text-sm text-gray-600">
            {isActive
              ? "Analyzing… this page auto-refreshes."
              : `Job is ${doc.job_status}. No results to show.`}
          </section>
        )}
      </div>
    </main>
  );
}

// --- Sub-components ------------------------------------------------------

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "green" | "gray";
}) {
  const valueCls =
    accent === "green"
      ? "text-green-700"
      : accent === "gray"
        ? "text-gray-700"
        : "text-gray-900";
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-xs uppercase text-gray-500">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${valueCls}`}>{value}</div>
    </div>
  );
}

function FilterChip({
  children,
  active,
  onClick,
}: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-2.5 py-1 rounded ${
        active
          ? "bg-blue-100 text-blue-700"
          : "text-gray-600 hover:bg-gray-100"
      }`}
    >
      {children}
    </button>
  );
}

function FencePageCard({ page }: { page: FencePage }) {
  const totalFt =
    page.measurements?.proximity_totals?.total_length_feet ??
    page.measurements?.totals?.total_length_feet ??
    null;
  const totalSeg =
    page.measurements?.proximity_totals?.total_segments ??
    page.measurements?.totals?.total_segments ??
    null;
  const totalPts = page.measurements?.proximity_totals?.total_length_pts;
  const scale = page.scale_info;
  const definitions = page.definitions ?? page.legend_entries ?? [];
  const instances = page.instances ?? [];

  return (
    <details className="group" open>
      <summary className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50">
        <div className="flex items-center gap-3 min-w-0">
          <span className="font-mono text-gray-700">Page {page.page_num}</span>
          <span className="inline-flex items-center gap-1 text-xs text-green-700">
            ✓ fence
          </span>
          {scale?.verified_scale && (
            <span className="text-xs text-gray-500">
              scale 1:{scale.verified_scale}
              {scale.confidence ? ` (${scale.confidence})` : ""}
            </span>
          )}
        </div>
        <div className="text-xs text-gray-600 flex items-center gap-3">
          {totalFt != null && totalFt > 0 && (
            <span className="font-mono">{totalFt.toFixed(1)} ft</span>
          )}
          {definitions.length > 0 && (
            <span>{definitions.length} legend</span>
          )}
          {instances.length > 0 && (
            <span>{instances.length} instances</span>
          )}
        </div>
      </summary>

      <div className="px-4 pb-4 space-y-4">
        {/* Fence text */}
        {page.fence_text && (
          <div>
            <div className="text-xs uppercase text-gray-500 mb-1">
              Detected text
            </div>
            <div className="text-sm bg-gray-50 border rounded p-3 whitespace-pre-wrap font-mono text-xs">
              {page.fence_text.slice(0, 800)}
              {page.fence_text.length > 800 ? "…" : ""}
            </div>
          </div>
        )}

        {/* Measurement totals */}
        {(totalFt != null || totalSeg != null || totalPts != null) && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {totalFt != null && (
              <Metric label="Length (scaled)" value={`${totalFt.toFixed(1)} ft`} />
            )}
            {totalPts != null && (
              <Metric label="Length (points)" value={totalPts.toLocaleString()} />
            )}
            {totalSeg != null && (
              <Metric label="Segments" value={String(totalSeg)} />
            )}
          </div>
        )}

        {/* Scale details */}
        {scale && (scale.scale_text || scale.message) && (
          <div className="text-xs text-gray-600 bg-gray-50 border rounded p-2">
            <div className="font-medium text-gray-700">Scale: {scale.scale_text ?? "—"}</div>
            {scale.method && <div>method: {scale.method}</div>}
            {scale.message && (
              <div className="mt-1 italic">{scale.message}</div>
            )}
          </div>
        )}

        {/* Legend definitions */}
        {definitions.length > 0 && (
          <DataTable
            title="Legend definitions"
            rows={definitions.map((d) => ({
              indicator: String(d.indicator ?? ""),
              description: String(d.description ?? ""),
            }))}
            columns={["indicator", "description"]}
          />
        )}

        {/* Instances */}
        {instances.length > 0 && (
          <DataTable
            title="Detected instances"
            rows={instances.map((i) => ({
              indicator: String(i.indicator ?? ""),
              location: i.bbox
                ? `[${i.bbox.map((n) => Math.round(n)).join(", ")}]`
                : "—",
            }))}
            columns={["indicator", "location"]}
          />
        )}
      </div>
    </details>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 border rounded p-2">
      <div className="text-[10px] uppercase text-gray-500">{label}</div>
      <div className="font-mono font-medium text-gray-900">{value}</div>
    </div>
  );
}

function DataTable({
  title,
  rows,
  columns,
}: {
  title: string;
  rows: Record<string, string>[];
  columns: string[];
}) {
  return (
    <div>
      <div className="text-xs uppercase text-gray-500 mb-1">{title}</div>
      <div className="overflow-x-auto border rounded">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
            <tr>
              {columns.map((c) => (
                <th key={c} className="text-left px-3 py-1.5 font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-gray-50">
                {columns.map((c) => (
                  <td key={c} className="px-3 py-1.5 text-gray-800 align-top">
                    {r[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function NonFenceList({
  nonFence,
  fence,
}: {
  nonFence: NonFencePage[];
  fence: FencePage[];
}) {
  const fenceNums = new Set(fence.map((f) => f.page_num));
  const all = [
    ...fence.map((f) => ({ page_num: f.page_num, isFence: true })),
    ...nonFence.map((n) => ({ page_num: n.page_num, isFence: false })),
  ].sort((a, b) => a.page_num - b.page_num);

  return (
    <table className="w-full text-sm">
      <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
        <tr>
          <th className="text-left px-4 py-2 font-medium w-20">Page</th>
          <th className="text-left px-4 py-2 font-medium">Classification</th>
        </tr>
      </thead>
      <tbody className="divide-y">
        {all.map((p) => (
          <tr key={p.page_num} className="hover:bg-gray-50">
            <td className="px-4 py-2 font-mono">{p.page_num}</td>
            <td className="px-4 py-2">
              {fenceNums.has(p.page_num) ? (
                <span className="text-green-700">✓ fence</span>
              ) : (
                <span className="text-gray-500">non-fence</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
