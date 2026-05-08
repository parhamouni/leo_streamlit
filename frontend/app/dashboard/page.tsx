"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { apiJson, ApiError } from "@/lib/api";
import { UploadButton } from "@/components/UploadButton";

type Document = {
  id: string;
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

type DocumentList = { documents: Document[] };

const POLL_MS = 3000;

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

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function isActive(d: Document): boolean {
  return d.job_status === "queued" || d.job_status === "running";
}

function ProgressCell({ doc }: { doc: Document }) {
  if (doc.job_status === "running" || doc.job_status === "queued") {
    const pct = doc.progress_percent ?? 0;
    return (
      <div className="flex items-center gap-2 justify-end">
        <div className="w-24 h-1.5 bg-gray-200 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-700 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 font-mono w-9 text-right">
          {pct}%
        </span>
      </div>
    );
  }
  if (doc.job_status === "completed") {
    return <span className="text-xs text-green-700">done</span>;
  }
  if (doc.job_status === "failed") {
    return <span className="text-xs text-red-600">failed</span>;
  }
  if (doc.job_status === "cancelled") {
    return <span className="text-xs text-yellow-700">cancelled</span>;
  }
  return <span className="text-xs text-gray-400">—</span>;
}

function LiveIndicator() {
  return (
    <span className="flex items-center gap-1.5 text-xs text-gray-500">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
      </span>
      Live
    </span>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [authReady, setAuthReady] = useState(false);

  const [docs, setDocs] = useState<Document[] | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);

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
        setEmail(data.session.user.email ?? null);
        setUserId(data.session.user.id);
        setAuthReady(true);

        if (window.location.hash.includes("access_token")) {
          window.history.replaceState(
            null,
            "",
            window.location.pathname + window.location.search,
          );
        }
      });
    return () => {
      active = false;
    };
  }, [router]);

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setInitialLoading(true);
    try {
      const data = await apiJson<DocumentList>("/api/documents");
      setDocs(data.documents);
      setError(null);
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? `API ${e.status}: ${typeof e.body === "string" ? e.body : JSON.stringify(e.body)}`
          : e instanceof Error
            ? e.message
            : String(e);
      setError(msg);
    } finally {
      if (!silent) setInitialLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    if (!authReady) return;
    refresh(false);
  }, [authReady, refresh]);

  // Live polling — runs while any job is queued or running. Pauses when
  // the tab is hidden, resumes immediately when it becomes visible again.
  const docsRef = useRef<Document[] | null>(null);
  docsRef.current = docs;

  useEffect(() => {
    if (!authReady || !docs) return;
    const anyActive = docs.some(isActive);
    if (!anyActive) {
      setPolling(false);
      return;
    }
    setPolling(true);

    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = () => {
      if (!alive) return;
      if (typeof document !== "undefined" && document.hidden) {
        // Don't poll while hidden; we'll resume on visibilitychange.
        return;
      }
      refresh(true).finally(() => {
        const stillActive =
          docsRef.current?.some(isActive) ?? false;
        if (alive && stillActive) {
          timer = setTimeout(tick, POLL_MS);
        }
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
      setPolling(false);
    };
  }, [authReady, docs, refresh]);

  async function onLogout() {
    await supabase().auth.signOut();
    router.replace("/login");
  }

  if (!authReady) {
    return (
      <main className="min-h-screen flex items-center justify-center text-gray-500">
        Loading…
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-50 p-6 sm:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Leo Fence</h1>
            <p className="text-sm text-gray-500">{email}</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => refresh(false)}
              disabled={initialLoading}
              className="text-sm px-3 py-1.5 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50"
            >
              {initialLoading ? "Refreshing…" : "Refresh"}
            </button>
            <button
              onClick={onLogout}
              className="text-sm text-gray-700 hover:underline"
            >
              Sign out
            </button>
          </div>
        </div>

        {/* Upload */}
        <section className="bg-white rounded-lg shadow p-4">
          <UploadButton onUploaded={() => refresh(false)} />
        </section>

        {/* Documents card */}
        <section className="bg-white rounded-lg shadow">
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-3">
              <h2 className="font-medium">Your documents</h2>
              {polling && <LiveIndicator />}
            </div>
            <span className="text-xs text-gray-400 font-mono">{userId}</span>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border-b border-red-200 text-sm text-red-700">
              <div className="font-medium">Failed to load documents</div>
              <div className="font-mono text-xs mt-1 break-all">{error}</div>
              <button
                onClick={() => refresh(false)}
                className="mt-2 text-xs underline"
              >
                Try again
              </button>
            </div>
          )}

          {initialLoading && docs === null && (
            <div className="p-8 text-center text-sm text-gray-500">
              Loading documents…
            </div>
          )}

          {!initialLoading && docs !== null && docs.length === 0 && !error && (
            <div className="p-8 text-center text-sm text-gray-500">
              No documents yet. Upload a PDF to get started.
            </div>
          )}

          {docs !== null && docs.length > 0 && (
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Filename</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-left px-4 py-2 font-medium">Pages</th>
                  <th className="text-left px-4 py-2 font-medium">Uploaded</th>
                  <th className="text-right px-4 py-2 font-medium">Progress</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {docs.map((d) => (
                  <tr key={d.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <div
                        className="font-medium truncate max-w-md"
                        title={d.original_filename}
                      >
                        {d.original_filename}
                      </div>
                      <div className="text-xs text-gray-400 font-mono">
                        {d.id.slice(0, 8)}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={d.job_status} />
                      {d.job_status === "running" && d.current_phase && (
                        <div className="text-xs text-gray-500 mt-1">
                          {d.current_phase}
                        </div>
                      )}
                      {d.error_message && (
                        <div
                          className="text-xs text-red-600 mt-1 truncate max-w-xs"
                          title={d.error_message}
                        >
                          {d.error_message}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-gray-700">
                      {d.total_pages ?? "—"}
                    </td>
                    <td className="px-4 py-3 text-gray-500">
                      {formatDate(d.created_at)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <ProgressCell doc={d} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </main>
  );
}
