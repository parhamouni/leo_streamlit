"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { apiFetch, apiJson, ApiError } from "@/lib/api";

// --- Types ---------------------------------------------------------------

type Document = {
  id: string;
  user_id: string;
  original_filename: string;
  storage_path: string;
  status: string;
  total_pages: number | null;
  created_at: string;
};

type DashboardDoc = Document & {
  latest_job_id: string | null;
  job_status: string | null;
  current_phase: string | null;
  progress_percent: number | null;
  error_message: string | null;
};

type PipelineResults = {
  fence_pages?: number[];
  non_fence_pages?: number[];
  element_details?: Record<string, unknown>;
  per_page_scale_info?: Record<string, unknown>;
  unified_measurements?: Record<string, unknown>;
  page_categories?: Record<string, unknown>;
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

async function downloadHighlightedPDF(jobId: string, filename: string) {
  const resp = await apiFetch(`/api/jobs/${jobId}/highlighted-pdf`);
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `fence_${filename}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Page ----------------------------------------------------------------

export default function DocumentDetailPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const docId = params.id;

  const [authReady, setAuthReady] = useState(false);
  const [doc, setDoc] = useState<DashboardDoc | null>(null);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "fence">("all");
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

  // Poll while job is active
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

  async function onDownload() {
    if (!doc?.latest_job_id) return;
    setDownloading(true);
    try {
      await downloadHighlightedPDF(doc.latest_job_id, doc.original_filename);
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

  const fencePages = results?.fence_pages ?? [];
  const nonFencePages = results?.non_fence_pages ?? [];
  const allPagesSorted = [...fencePages, ...nonFencePages].sort((a, b) => a - b);
  const visiblePages = filter === "fence" ? fencePages : allPagesSorted;
  const fenceSet = new Set(fencePages);

  const isActive =
    doc.job_status === "queued" || doc.job_status === "running";
  const isComplete = doc.job_status === "completed";

  return (
    <main className="min-h-screen bg-gray-50 p-6 sm:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
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
                    <span>took {formatDuration(results.timings)}</span>
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
        {results && (
          <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatCard
              label="Total pages"
              value={String(results.total_pages ?? doc.total_pages ?? "—")}
            />
            <StatCard
              label="Fence pages"
              value={String(fencePages.length)}
              accent="green"
            />
            <StatCard
              label="Non-fence pages"
              value={String(nonFencePages.length)}
              accent="gray"
            />
          </section>
        )}

        {/* Page-level results */}
        {isComplete && results && (
          <section className="bg-white rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">Pages</h2>
              <div className="flex gap-1 text-xs">
                <FilterChip
                  active={filter === "all"}
                  onClick={() => setFilter("all")}
                >
                  All ({allPagesSorted.length})
                </FilterChip>
                <FilterChip
                  active={filter === "fence"}
                  onClick={() => setFilter("fence")}
                >
                  Fence only ({fencePages.length})
                </FilterChip>
              </div>
            </div>
            {visiblePages.length === 0 ? (
              <div className="p-6 text-sm text-gray-500 text-center">
                {filter === "fence"
                  ? "No fence pages detected in this document."
                  : "No pages."}
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium w-20">Page</th>
                    <th className="text-left px-4 py-2 font-medium">Classification</th>
                    <th className="text-left px-4 py-2 font-medium">Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {visiblePages.map((p) => (
                    <PageRow
                      key={p}
                      pageNumber={p}
                      isFence={fenceSet.has(p)}
                      perPageScaleInfo={results.per_page_scale_info}
                    />
                  ))}
                </tbody>
              </table>
            )}
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

function PageRow({
  pageNumber,
  isFence,
  perPageScaleInfo,
}: {
  pageNumber: number;
  isFence: boolean;
  perPageScaleInfo?: Record<string, unknown>;
}) {
  const scaleEntry = perPageScaleInfo?.[String(pageNumber)] as
    | { scale?: number; unit?: string; method?: string }
    | undefined;
  const scaleNote = scaleEntry?.scale
    ? `scale 1:${scaleEntry.scale}${
        scaleEntry.unit ? ` (${scaleEntry.unit})` : ""
      }`
    : null;

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-2 font-mono text-gray-700">{pageNumber}</td>
      <td className="px-4 py-2">
        {isFence ? (
          <span className="inline-flex items-center gap-1 text-green-700">
            <span>✓</span>
            <span>fence</span>
          </span>
        ) : (
          <span className="text-gray-500">non-fence</span>
        )}
      </td>
      <td className="px-4 py-2 text-gray-500 text-xs">
        {scaleNote ?? "—"}
      </td>
    </tr>
  );
}
