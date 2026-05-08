"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { apiJson, ApiError } from "@/lib/api";

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

export default function DashboardPage() {
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [authReady, setAuthReady] = useState(false);

  const [docs, setDocs] = useState<Document[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      });
    return () => {
      active = false;
    };
  }, [router]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiJson<DocumentList>("/api/documents");
      setDocs(data.documents);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(`API ${e.status}: ${typeof e.body === "string" ? e.body : JSON.stringify(e.body)}`);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch documents once auth is ready
  useEffect(() => {
    if (!authReady) return;
    refresh();
  }, [authReady, refresh]);

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
              onClick={refresh}
              disabled={loading}
              className="text-sm px-3 py-1.5 rounded border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50"
            >
              {loading ? "Refreshing…" : "Refresh"}
            </button>
            <button
              onClick={onLogout}
              className="text-sm text-gray-700 hover:underline"
            >
              Sign out
            </button>
          </div>
        </div>

        {/* Documents card */}
        <section className="bg-white rounded-lg shadow">
          <div className="flex items-center justify-between p-4 border-b">
            <h2 className="font-medium">Your documents</h2>
            <span className="text-xs text-gray-400 font-mono">{userId}</span>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border-b border-red-200 text-sm text-red-700">
              <div className="font-medium">Failed to load documents</div>
              <div className="font-mono text-xs mt-1 break-all">{error}</div>
              <button
                onClick={refresh}
                className="mt-2 text-xs underline"
              >
                Try again
              </button>
            </div>
          )}

          {loading && docs === null && (
            <div className="p-8 text-center text-sm text-gray-500">
              Loading documents…
            </div>
          )}

          {!loading && docs !== null && docs.length === 0 && !error && (
            <div className="p-8 text-center text-sm text-gray-500">
              No documents yet. Upload a PDF to get started.
              <br />
              <span className="text-xs">(Upload UI coming in the next checkpoint.)</span>
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
                      <div className="font-medium truncate max-w-md" title={d.original_filename}>
                        {d.original_filename}
                      </div>
                      <div className="text-xs text-gray-400 font-mono">{d.id.slice(0, 8)}</div>
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={d.job_status} />
                      {d.error_message && (
                        <div className="text-xs text-red-600 mt-1 truncate max-w-xs" title={d.error_message}>
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
                    <td className="px-4 py-3 text-right text-gray-500 font-mono text-xs">
                      {d.job_status === "running" || d.job_status === "queued"
                        ? `${d.progress_percent ?? 0}%`
                        : d.current_phase ?? "—"}
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
