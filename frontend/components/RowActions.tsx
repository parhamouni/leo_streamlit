"use client";

import { useState } from "react";
import { apiFetch } from "@/lib/api";

type Props = {
  jobId: string | null;
  jobStatus: string | null;
  filename: string;
  onChanged: () => void;
};

/**
 * Per-row action button. Lives in the dashboard's far-right column.
 *
 * - jobStatus in {completed, failed, cancelled}  → "Delete" button (confirms)
 * - jobStatus in {queued, running}  → nothing here. Cancel is rendered
 *   inline with the progress bar (see ProgressCell in dashboard/page.tsx)
 *   so it's visually paired with the work it stops.
 * - no job (e.g. dedup edge case)  → nothing
 */
export function RowActions({ jobId, jobStatus, filename, onChanged }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!jobId || !jobStatus) return null;

  const isTerminal =
    jobStatus === "completed" ||
    jobStatus === "failed" ||
    jobStatus === "cancelled";
  if (!isTerminal) return null;

  async function run(verb: "cancel" | "delete") {
    if (verb === "delete") {
      const ok = window.confirm(
        `Delete "${filename}"? This removes the document, its job history, and any generated artifacts. This cannot be undone.`,
      );
      if (!ok) return;
    }
    setBusy(true);
    setError(null);
    try {
      await apiFetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="flex items-center justify-end gap-2"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        onClick={() => run("delete")}
        disabled={busy}
        className="text-xs text-red-700 hover:underline disabled:opacity-50"
        title="Remove the document and all artifacts permanently"
      >
        {busy ? "…" : "Delete"}
      </button>
      {error && (
        <span className="text-xs text-red-600 ml-2" title={error}>
          ⚠
        </span>
      )}
    </div>
  );
}
