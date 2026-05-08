"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Link from "next/link";
import { supabase } from "@/lib/supabase";
import { apiFetch, apiJson, ApiError } from "@/lib/api";
import {
  etaSeconds,
  formatEta,
  phaseRange,
  withinPhasePct,
} from "@/lib/eta";
import { RowActions } from "@/components/RowActions";
import { UMTCanvas } from "@/components/UMTCanvas";
import {
  cleanElementKey,
  detectionLabel,
  detectionMethod,
  specHasContent,
  type DimensionMeasurement,
  type ElementSpec,
  type FencePage,
  type LayerMeasurement,
  type LegendEntry,
  type NonFencePage,
  type PipelineResults,
} from "@/lib/results";

// ---------------------------------------------------------------------------
// Types specific to this page
// ---------------------------------------------------------------------------

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
  job_started_at: string | null;
  phase_started_at: string | null;
};

type PageRow = {
  page_number: number;
  is_fence_page: boolean;
  result_json: (Partial<FencePage> & { phase?: string }) | null;
  created_at: string;
};

const POLL_MS = 3000;

function isPhase1cStub(row: PageRow): boolean {
  const r = row.result_json;
  if (!r) return true;
  if (r.phase === "phase1c") return true;
  // Heuristic for older rows: nothing rich populated yet.
  return !r.measurements && !r.definitions && !r.scale_info && !r.fence_text;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatDuration(timings?: Record<string, number>): string | null {
  if (!timings) return null;
  const total = timings.total ?? Object.values(timings).reduce((a, b) => a + b, 0);
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
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function DocumentDetailPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const docId = params.id;

  const [authReady, setAuthReady] = useState(false);
  const [doc, setDoc] = useState<DashboardDoc | null>(null);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"fence" | "all" | "nonfence">("fence");
  const [downloading, setDownloading] = useState(false);
  const [pagesSoFar, setPagesSoFar] = useState<PageRow[]>([]);

  // Auth
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

  // Fetch document + (when completed) results
  const refresh = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        const found = await apiJson<DashboardDoc>(`/api/documents/${docId}`);
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

  // Live per-page rows (Sprint 2 / A7). Polls the new pages endpoint while
  // a job is queued/running so pages appear in the detail view as the
  // worker writes them. Stops polling once the job leaves an active state;
  // the existing results-JSON render then takes over.
  useEffect(() => {
    if (!authReady || !doc) return;
    const active = doc.job_status === "queued" || doc.job_status === "running";
    if (!active) return;

    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const fetchPages = async () => {
      try {
        const resp = await apiJson<{ pages: PageRow[] }>(
          `/api/documents/${docId}/pages`,
        );
        if (alive) setPagesSoFar(resp.pages ?? []);
      } catch {
        // best-effort; the next tick may succeed
      }
    };

    const tick = () => {
      if (!alive) return;
      if (typeof document !== "undefined" && document.hidden) return;
      fetchPages().finally(() => {
        const stillActive =
          docRef.current?.job_status === "queued" ||
          docRef.current?.job_status === "running";
        if (alive && stillActive) timer = setTimeout(tick, POLL_MS);
      });
    };

    fetchPages();
    timer = setTimeout(tick, POLL_MS);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [authReady, doc, docId]);

  async function downloadFromJob(
    pathSegment: string,
    filenameFallback: string,
  ) {
    if (!doc?.latest_job_id) return;
    setDownloading(true);
    try {
      const resp = await apiFetch(
        `/api/jobs/${doc.latest_job_id}/${pathSegment}`,
      );
      const blob = await resp.blob();
      // Honour Content-Disposition filename when present.
      const cd = resp.headers.get("Content-Disposition") || "";
      const m = /filename="?([^"]+)"?/.exec(cd);
      const filename = m?.[1] || filenameFallback;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
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

  const onDownloadHighlighted = () =>
    downloadFromJob("highlighted-pdf", `fence_${doc?.original_filename ?? "document.pdf"}`);
  const onDownloadMeasurementPdf = () =>
    downloadFromJob(
      "measurement-pdf",
      `${(doc?.original_filename ?? "document").replace(/\.[^.]+$/, "")}_measurements.pdf`,
    );
  const onDownloadMeasurementExcel = () =>
    downloadFromJob(
      "measurement-excel",
      `${(doc?.original_filename ?? "document").replace(/\.[^.]+$/, "")}_measurements.xlsx`,
    );

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
          <Link href="/dashboard" className="text-sm text-blue-600 hover:underline">
            ← Back to dashboard
          </Link>
          <div className="text-red-700">{error}</div>
        </div>
      </main>
    );
  }
  if (!doc) return null;

  // ---- derived ----
  const fencePages = (results?.fence_pages ?? []).filter(
    (p): p is FencePage => p && typeof p === "object",
  );
  const nonFencePages = (results?.non_fence_pages ?? []).filter(
    (p): p is NonFencePage => p && typeof p === "object",
  );
  const totalCount = fencePages.length + nonFencePages.length;
  const totalLengthFt = fencePages.reduce((acc, p) => {
    const ft =
      p.measurements?.proximity_totals?.total_length_feet ??
      p.measurements?.totals?.total_length_feet ??
      0;
    return acc + (ft || 0);
  }, 0);
  const elementDetails = results?.element_details ?? {};
  const populatedSpecs = Object.entries(elementDetails).filter(([, spec]) =>
    specHasContent(spec),
  );

  const isActive = doc.job_status === "queued" || doc.job_status === "running";
  const isComplete = doc.job_status === "completed";

  return (
    <main className="min-h-screen bg-gray-50 p-6 sm:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <Link href="/dashboard" className="inline-block text-sm text-blue-600 hover:underline">
          ← Back to dashboard
        </Link>

        {/* ---------- Header ---------- */}
        <section className="bg-white rounded-lg shadow p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-semibold truncate" title={doc.original_filename}>
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
                <ProgressBlock doc={doc} />
              )}
              {doc.error_message && (
                <div className="mt-3 text-sm text-red-700 break-all">
                  {doc.error_message}
                </div>
              )}
            </div>
            <div className="flex items-center gap-3 whitespace-nowrap">
              <RowActions
                jobId={doc.latest_job_id}
                jobStatus={doc.job_status}
                filename={doc.original_filename}
                onChanged={() => {
                  if (doc.job_status !== "queued" && doc.job_status !== "running") {
                    router.replace("/dashboard");
                  } else {
                    refresh(false);
                  }
                }}
              />
              {isComplete && (
                <div className="flex flex-col gap-1.5 items-stretch">
                  <button
                    onClick={onDownloadHighlighted}
                    disabled={downloading}
                    className="rounded bg-black text-white px-4 py-2 text-sm hover:bg-gray-800 disabled:opacity-60"
                  >
                    {downloading ? "Preparing…" : "Download highlighted PDF"}
                  </button>
                  <div className="flex gap-1.5">
                    <button
                      onClick={onDownloadMeasurementPdf}
                      disabled={downloading}
                      className="flex-1 rounded border border-gray-300 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-100 disabled:opacity-60"
                      title="Auto-detected measurement lines (UMT edits not yet supported)"
                    >
                      Measurement PDF
                    </button>
                    <button
                      onClick={onDownloadMeasurementExcel}
                      disabled={downloading}
                      className="flex-1 rounded border border-gray-300 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-100 disabled:opacity-60"
                      title="Auto-detected measurement rows as XLSX"
                    >
                      Measurement Excel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ---------- Summary stats ---------- */}
        {isComplete && results && (
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard
              label="Total pages"
              value={String(totalCount || results.total_pages || doc.total_pages || "—")}
            />
            <StatCard label="Fence pages" value={String(fencePages.length)} accent="green" />
            <StatCard label="Non-fence pages" value={String(nonFencePages.length)} accent="gray" />
            <StatCard
              label="Total fence length"
              value={totalLengthFt > 0 ? `${totalLengthFt.toFixed(1)} ft` : "—"}
              accent={totalLengthFt > 0 ? "green" : "gray"}
            />
          </section>
        )}

        {/* ---------- Phase timings ---------- */}
        {isComplete && results?.timings && (
          <section className="bg-white rounded-lg shadow p-4">
            <h2 className="font-medium mb-2 text-sm">Phase timings</h2>
            <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 text-xs">
              {Object.entries(results.timings).map(([phase, secs]) => (
                <div key={phase} className="bg-gray-50 rounded p-2">
                  <div className="text-[10px] uppercase text-gray-500">{phase}</div>
                  <div className="font-mono font-medium">{secs.toFixed(1)}s</div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ---------- Element specifications (Sprint 1: A1, A2) ---------- */}
        {isComplete && populatedSpecs.length > 0 && (
          <section className="bg-white rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">
                Element specifications ({populatedSpecs.length})
              </h2>
              <span className="text-xs text-gray-500">
                LLM-extracted material / construction details per legend entry
              </span>
            </div>
            <ElementSpecsTable specs={populatedSpecs} />
          </section>
        )}
        {isComplete && populatedSpecs.length === 0 && Object.keys(elementDetails).length > 0 && (
          <section className="bg-white rounded-lg shadow p-4 text-sm text-gray-500">
            No populated element specifications were extracted. ({Object.keys(elementDetails).length}{" "}
            chunks examined; all fields empty.)
          </section>
        )}

        {/* ---------- Pages list ---------- */}
        {isComplete && results && (
          <section className="bg-white rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">Pages</h2>
              <div className="flex gap-1 text-xs">
                <FilterChip active={filter === "fence"} onClick={() => setFilter("fence")}>
                  Fence ({fencePages.length})
                </FilterChip>
                <FilterChip active={filter === "nonfence"} onClick={() => setFilter("nonfence")}>
                  Non-fence ({nonFencePages.length})
                </FilterChip>
                <FilterChip active={filter === "all"} onClick={() => setFilter("all")}>
                  All ({totalCount})
                </FilterChip>
              </div>
            </div>

            <div className="divide-y">
              {filter === "fence" &&
                (fencePages.length === 0 ? (
                  <Empty msg="No fence pages detected." />
                ) : (
                  fencePages.map((p) => (
                    <FencePageCard
                      key={p.page_num}
                      page={p}
                      jobId={doc.latest_job_id}
                    />
                  ))
                ))}
              {filter === "nonfence" &&
                (nonFencePages.length === 0 ? (
                  <Empty msg="No non-fence pages." />
                ) : (
                  nonFencePages.map((p) => <NonFencePageCard key={p.page_num} page={p} />)
                ))}
              {filter === "all" && (
                <>
                  {fencePages.map((p) => (
                    <FencePageCard
                      key={`f-${p.page_num}`}
                      page={p}
                      jobId={doc.latest_job_id}
                    />
                  ))}
                  {nonFencePages.map((p) => (
                    <NonFencePageCard key={`n-${p.page_num}`} page={p} />
                  ))}
                </>
              )}
            </div>
          </section>
        )}

        {/* Live "Pages so far" while a job runs (Sprint 2 / A7) */}
        {isActive && (
          <section className="bg-white rounded-lg shadow">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-medium">
                Pages so far ({pagesSoFar.length}
                {doc.total_pages ? ` / ${doc.total_pages}` : ""})
              </h2>
              <span className="text-xs text-gray-500">
                Auto-refreshes every {POLL_MS / 1000}s
              </span>
            </div>
            {pagesSoFar.length === 0 ? (
              <div className="p-6 text-sm text-gray-500">
                Classifying pages… results appear here as each page finishes.
              </div>
            ) : (
              <div className="divide-y">
                {pagesSoFar.map((row) => (
                  // jobId intentionally omitted while running — the
                  // highlighted PDF doesn't exist yet, so the "Load page
                  // image" button would be misleading. It comes back as
                  // soon as the job completes and the regular render
                  // path takes over below.
                  <LivePageRow key={row.page_number} row={row} />
                ))}
              </div>
            )}
          </section>
        )}

        {!isComplete && !isActive && !doc.error_message && (
          <section className="bg-white rounded-lg shadow p-6 text-sm text-gray-600">
            Job is {doc.job_status}. No results to show.
          </section>
        )}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Empty({ msg }: { msg: string }) {
  return <div className="p-6 text-sm text-gray-500 text-center">{msg}</div>;
}

function ProgressBlock({ doc }: { doc: DashboardDoc }) {
  const overall = doc.progress_percent ?? 0;
  const range = phaseRange(doc.current_phase);
  const within = range ? withinPhasePct(doc.current_phase, overall) : null;
  const overallEta = formatEta(etaSeconds(doc.job_started_at, overall));
  const phaseEta =
    range && within != null
      ? formatEta(etaSeconds(doc.phase_started_at, within))
      : null;

  return (
    <div className="mt-3 space-y-2">
      {/* Overall */}
      <div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <span className="font-mono text-xs uppercase tracking-wide text-gray-500">
            Overall
          </span>
          <span className="font-mono">{overall}%</span>
          {overallEta && (
            <span className="text-xs text-gray-500">· {overallEta} left</span>
          )}
        </div>
        <div className="mt-1 h-1.5 w-full max-w-md bg-gray-200 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-700 ease-out"
            style={{ width: `${overall}%` }}
          />
        </div>
      </div>

      {/* Within current phase */}
      {range && within != null && (
        <div>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span className="font-mono text-xs uppercase tracking-wide text-gray-500">
              {range.label}
            </span>
            <span className="font-mono">{Math.round(within)}%</span>
            {phaseEta && (
              <span className="text-xs text-gray-500">· {phaseEta} for this phase</span>
            )}
          </div>
          <div className="mt-1 h-1.5 w-full max-w-md bg-gray-100 rounded overflow-hidden">
            <div
              className="h-full bg-emerald-500 transition-all duration-700 ease-out"
              style={{ width: `${within}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

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
        active ? "bg-blue-100 text-blue-700" : "text-gray-600 hover:bg-gray-100"
      }`}
    >
      {children}
    </button>
  );
}

function LivePageRow({ row, jobId }: { row: PageRow; jobId?: string | null }) {
  const stub = isPhase1cStub(row);
  if (!stub && row.is_fence_page && row.result_json) {
    // Phase 3 enrichment is in — render with the existing rich card.
    return (
      <FencePageCard page={row.result_json as FencePage} jobId={jobId} />
    );
  }

  // Phase 1c stub — minimal placeholder.
  const label = row.is_fence_page ? "fence" : "non-fence";
  const labelCls = row.is_fence_page
    ? "bg-blue-100 text-blue-700"
    : "bg-gray-100 text-gray-600";
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-3 min-w-0">
        <span className="font-mono text-gray-700">Page {row.page_number}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${labelCls}`}>{label}</span>
      </div>
      {row.is_fence_page && (
        <span className="text-xs text-gray-400 italic">analyzing…</span>
      )}
    </div>
  );
}

function PageImage({ jobId, pageNum }: { jobId: string; pageNum: number }) {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [url]);

  async function load() {
    if (url) {
      setOpen(!open);
      return;
    }
    setOpen(true);
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch(
        `/api/jobs/${jobId}/page-image/${pageNum}?dpi=110`,
      );
      const blob = await resp.blob();
      setUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <button
        type="button"
        onClick={load}
        className="text-xs text-blue-600 hover:underline"
      >
        {loading
          ? "Loading image…"
          : url
            ? open
              ? "Hide page image"
              : "Show page image"
            : "🖼️ Load page image"}
      </button>
      {error && (
        <div className="mt-1 text-xs text-red-600 break-all">{error}</div>
      )}
      {open && url && (
        <div className="mt-2 border rounded overflow-hidden bg-gray-50">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={url}
            alt={`Page ${pageNum} with fence highlights`}
            className="w-full h-auto block"
          />
        </div>
      )}
    </div>
  );
}

function FencePageCard({
  page,
  jobId,
}: {
  page: FencePage;
  jobId?: string | null;
}) {
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
  const definitions = page.definitions ?? [];
  const instances = page.instances ?? [];
  const legend = page.legend_entries ?? [];
  const adeChunkCount = (definitions.length ?? 0) + (instances.length ?? 0);
  const skipReason = page.measurements?.skip_reason;

  const method = detectionMethod(page);
  const m = detectionLabel(method);

  return (
    <details className="group" open>
      <summary className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50">
        <div className="flex items-center gap-3 min-w-0">
          <span className="font-mono text-gray-700">Page {page.page_num}</span>
          <span
            className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded ${m.className}`}
            title={m.label}
          >
            <span>{m.icon}</span>
            <span>{m.label}</span>
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
          {legend.length > 0 && <span>{legend.length} legend</span>}
          {instances.length > 0 && <span>{instances.length} instances</span>}
        </div>
      </summary>

      <div className="px-4 pb-4 space-y-4">
        {/* Page image with fence overlays (lazy-loaded) */}
        {jobId && <PageImage jobId={jobId} pageNum={page.page_num} />}

        {/* Sprint 4 / C1: interactive measurement canvas (read-only for
            now — checkpoint 4a.3). Click handlers + drawing land in 4a.4. */}
        {jobId && (
          <UMTCanvas
            jobId={jobId}
            pageNum={page.page_num}
            legendEntries={legend}
          />
        )}

        {/* ADE chunks metrics (Sprint 1: A6) */}
        {adeChunkCount > 0 && (
          <div className="grid grid-cols-3 gap-3">
            <Metric label="ADE chunks" value={String(adeChunkCount)} />
            <Metric label="Legend" value={String(definitions.length)} />
            <Metric label="Figures" value={String(instances.length)} />
          </div>
        )}

        {/* Skip-reason warning */}
        {skipReason && (
          <div className="text-xs bg-yellow-50 border border-yellow-200 rounded p-2 text-yellow-800">
            <span className="font-medium">Measurement skipped:</span> {skipReason}
          </div>
        )}

        {/* Page text intentionally not rendered — app_ade_prod.py never
            showed it; fence_text is an LLM-classifier input, not a
            user-facing artifact. */}

        {/* Measurement totals */}
        {(totalFt != null || totalSeg != null || totalPts != null) && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {totalFt != null && (
              <Metric label="Length (scaled)" value={`${totalFt.toFixed(1)} ft`} />
            )}
            {totalPts != null && (
              <Metric label="Length (points)" value={totalPts.toLocaleString()} />
            )}
            {totalSeg != null && <Metric label="Segments" value={String(totalSeg)} />}
          </div>
        )}

        {/* Scale details with debug expander (Sprint 1: B3) */}
        {scale && (scale.scale_text || scale.message || scale.method) && (
          <details className="text-xs bg-gray-50 border rounded">
            <summary className="cursor-pointer px-3 py-2 font-medium text-gray-700 hover:bg-gray-100">
              Scale: {scale.scale_text ?? "—"}
              {scale.method && (
                <span className="text-gray-500"> · method: {scale.method}</span>
              )}
              {scale.confidence && (
                <span className="text-gray-500"> · {scale.confidence} confidence</span>
              )}
            </summary>
            <div className="px-3 pb-3 pt-1 space-y-1 text-gray-600">
              {scale.message && <div className="italic">{scale.message}</div>}
              {scale.page_size && (
                <div>
                  Page size:{" "}
                  {scale.page_size.detected_size ??
                    `${scale.page_size.width_pts}×${scale.page_size.height_pts} pts`}
                </div>
              )}
              {scale.raw_response && (
                <details>
                  <summary className="cursor-pointer hover:underline">
                    Raw LLM response
                  </summary>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-[10px] max-h-32 overflow-y-auto">
                    {scale.raw_response}
                  </pre>
                </details>
              )}
            </div>
          </details>
        )}

        {/* Layer-Based Breakdown (Sprint 1: B1) */}
        <LayerBreakdown layerMeasurements={page.measurements?.layer_measurements} />

        {/* Dimension Line Measurements (Sprint 1: B2) */}
        <DimensionLines dims={page.measurements?.dimension_measurements} />

        {/* Legend definitions (rich, with descriptions) */}
        {legend.length > 0 && <LegendTable rows={legend} />}

        {/* Classification reasoning (when keyword fallback) */}
        {page.classification?.reasoning && (
          <div className="text-xs bg-blue-50 border border-blue-200 rounded p-2">
            <div className="font-medium text-blue-900">
              Classification ({page.classification.confidence != null
                ? `${Math.round((page.classification.confidence ?? 0) * 100)}%`
                : "—"}{" "}
              confidence)
            </div>
            <div className="italic text-blue-800 mt-1">{page.classification.reasoning}</div>
          </div>
        )}
      </div>
    </details>
  );
}

function NonFencePageCard({ page }: { page: NonFencePage }) {
  const reasoning = page.classification?.reasoning ?? page.reason;
  const method = page.classification?.method ?? page.method;
  const conf = page.classification?.confidence;
  return (
    <details className="group">
      <summary className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50">
        <div className="flex items-center gap-3">
          <span className="font-mono text-gray-700">Page {page.page_num}</span>
          <span className="text-gray-500 text-xs">non-fence</span>
          {method && <span className="text-xs text-gray-400">via {method}</span>}
          {conf != null && (
            <span className="text-xs text-gray-400">
              {Math.round(conf * 100)}% confidence
            </span>
          )}
        </div>
      </summary>
      <div className="px-4 pb-4 space-y-2">
        {reasoning && (
          <div className="text-sm text-gray-700">
            <div className="text-xs uppercase text-gray-500 mb-1">Why excluded</div>
            <div className="bg-gray-50 border rounded p-3 italic">{reasoning}</div>
          </div>
        )}
        {page.keywords_found && page.keywords_found.length > 0 && (
          <div className="text-xs text-gray-600">
            Keywords found: {page.keywords_found.join(", ")}
          </div>
        )}
        {page.fence_text && (
          <div>
            <div className="text-xs uppercase text-gray-500 mb-1">Page text</div>
            <div className="bg-gray-50 border rounded p-3 whitespace-pre-wrap font-mono text-xs max-h-32 overflow-y-auto">
              {page.fence_text}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}

function LegendTable({ rows }: { rows: LegendEntry[] }) {
  return (
    <div>
      <div className="text-xs uppercase text-gray-500 mb-1">
        Legend definitions ({rows.length})
      </div>
      <div className="overflow-x-auto border rounded">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
            <tr>
              <th className="text-left px-3 py-1.5 font-medium w-24">Indicator</th>
              <th className="text-left px-3 py-1.5 font-medium w-40">Keyword</th>
              <th className="text-left px-3 py-1.5 font-medium">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-3 py-1.5 align-top font-mono text-xs">
                  {r.indicator || "—"}
                </td>
                <td className="px-3 py-1.5 align-top text-gray-700">{r.keyword || "—"}</td>
                <td className="px-3 py-1.5 align-top text-gray-700">{r.description || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LayerBreakdown({
  layerMeasurements,
}: {
  layerMeasurements?: Record<string, LayerMeasurement>;
}) {
  if (!layerMeasurements) return null;
  const entries = Object.entries(layerMeasurements);
  if (entries.length === 0) return null;

  return (
    <details className="text-xs bg-gray-50 border rounded">
      <summary className="cursor-pointer px-3 py-2 font-medium text-gray-700 hover:bg-gray-100">
        Layer-based breakdown ({entries.length} layer{entries.length === 1 ? "" : "s"})
      </summary>
      <div className="px-3 pb-3 pt-1 overflow-x-auto">
        <table className="w-full">
          <thead className="text-gray-500 text-[10px] uppercase">
            <tr>
              <th className="text-left py-1 font-medium">Layer</th>
              <th className="text-right py-1 font-medium">Segments</th>
              <th className="text-right py-1 font-medium">Length (ft)</th>
              <th className="text-right py-1 font-medium">Connected runs</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {entries.map(([layerName, lm]) => (
              <tr key={layerName}>
                <td className="py-1 font-mono">{lm.layer_name ?? layerName}</td>
                <td className="py-1 text-right">{lm.segment_count ?? 0}</td>
                <td className="py-1 text-right">
                  {lm.total_length_feet != null
                    ? lm.total_length_feet.toFixed(1)
                    : "—"}
                </td>
                <td className="py-1 text-right">{lm.connected_runs ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function DimensionLines({ dims }: { dims?: DimensionMeasurement[] }) {
  if (!dims || dims.length === 0) return null;
  const top = dims.slice(0, 10);
  return (
    <details className="text-xs bg-gray-50 border rounded">
      <summary className="cursor-pointer px-3 py-2 font-medium text-gray-700 hover:bg-gray-100">
        Dimension lines ({dims.length}
        {dims.length > 10 ? `, top 10 shown` : ""})
      </summary>
      <div className="px-3 pb-3 pt-1 overflow-x-auto">
        <table className="w-full">
          <thead className="text-gray-500 text-[10px] uppercase">
            <tr>
              <th className="text-left py-1 font-medium">Text</th>
              <th className="text-right py-1 font-medium">Measured (ft)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {top.map((d, idx) => (
              <tr key={idx}>
                <td className="py-1 max-w-md truncate" title={d.text ?? ""}>
                  {d.text ?? "—"}
                </td>
                <td className="py-1 text-right font-mono">
                  {d.measured_length_feet != null
                    ? d.measured_length_feet.toFixed(1)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function ElementSpecsTable({ specs }: { specs: [string, ElementSpec][] }) {
  const cols: Array<{ key: keyof ElementSpec; label: string }> = [
    { key: "height", label: "Height" },
    { key: "post_type", label: "Post type" },
    { key: "post_spacing", label: "Spacing" },
    { key: "material", label: "Material" },
    { key: "gauge", label: "Gauge" },
    { key: "mesh_size", label: "Mesh" },
    { key: "detail_page", label: "Detail page" },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
          <tr>
            <th className="text-left px-4 py-2 font-medium w-72">Element</th>
            {cols.map((c) => (
              <th key={String(c.key)} className="text-left px-3 py-2 font-medium">
                {c.label}
              </th>
            ))}
            <th className="px-3 py-2 font-medium text-right w-20">Details</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {specs.map(([key, spec], idx) => {
            const fullDetails = spec.full_details || spec.notes;
            return (
              <tr key={idx} className="hover:bg-gray-50 align-top">
                <td className="px-4 py-2 text-xs text-gray-700 max-w-xs">
                  <span className="font-mono break-all" title={key}>
                    {cleanElementKey(key, 100)}
                  </span>
                </td>
                {cols.map((c) => (
                  <td key={String(c.key)} className="px-3 py-2 text-gray-700">
                    {(spec[c.key] as string) || "—"}
                  </td>
                ))}
                <td className="px-3 py-2 text-right">
                  {fullDetails ? (
                    <details className="group">
                      <summary className="text-xs text-blue-600 hover:underline cursor-pointer">
                        view
                      </summary>
                      <div className="absolute right-4 mt-1 max-w-md bg-white border rounded shadow-lg p-3 text-xs whitespace-pre-wrap text-gray-700 z-10">
                        {fullDetails}
                      </div>
                    </details>
                  ) : (
                    <span className="text-gray-300 text-xs">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
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
