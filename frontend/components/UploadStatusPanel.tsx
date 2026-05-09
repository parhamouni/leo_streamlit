"use client";

/**
 * Floating upload-status panel — bottom-right of the viewport, mounted in
 * the root layout so it's visible on every route. Reads the same upload
 * queue as `UploadButton`'s inline display via the shared `UploadProvider`.
 *
 * Hides itself when there's nothing to show. Collapsible (header click)
 * so it doesn't get in the way once uploads are done.
 */
import { useEffect, useState } from "react";
import { fmtBytes, useUpload, type UploadEntry } from "@/contexts/UploadContext";

export function UploadStatusPanel() {
  const { queue, abort, clearFinished } = useUpload();
  const [open, setOpen] = useState(true);

  const remaining = queue.filter(
    (x) => x.status === "queued" || x.status === "uploading",
  ).length;
  const finished = queue.length - remaining;

  // Re-open the panel automatically when a new upload starts after the
  // user collapsed it.
  useEffect(() => {
    if (remaining > 0) setOpen(true);
  }, [remaining]);

  // The whole point of this floating widget is to show progress while the
  // user is on a different route from the dashboard. Once nothing is
  // actively uploading, it has no job — hide it. Done/deduped/failed
  // entries still live in the queue and are visible inline on the
  // dashboard's UploadButton; we just don't take up screen real estate
  // here for them.
  if (remaining === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 max-w-[calc(100vw-2rem)] bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-3 px-3 py-2 text-sm border-b bg-gray-50 hover:bg-gray-100"
      >
        <span className="font-medium text-gray-800">
          {remaining > 0
            ? `Uploading… (${finished}/${queue.length})`
            : `Uploads (${queue.length})`}
        </span>
        <span className="text-xs text-gray-500">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <>
          <div className="max-h-72 overflow-y-auto divide-y">
            {queue.map((e) => (
              <Row key={e.id} entry={e} onAbort={() => abort(e.id)} />
            ))}
          </div>
          {finished > 0 && (
            <div className="px-3 py-1.5 text-right border-t bg-gray-50">
              <button
                onClick={clearFinished}
                className="text-xs text-blue-600 hover:underline"
              >
                Clear finished
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Row({
  entry,
  onAbort,
}: {
  entry: UploadEntry;
  onAbort: () => void;
}) {
  return (
    <div className="px-3 py-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium text-gray-800" title={entry.fileName}>
            {entry.fileName}
          </div>
          <div className="text-[11px] text-gray-400">
            {fmtBytes(entry.fileSize)}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <StatusPill entry={entry} />
          {(entry.status === "queued" || entry.status === "uploading") && (
            <button
              onClick={onAbort}
              className="text-gray-400 hover:text-red-600 leading-none"
              title="Cancel this upload"
            >
              ✕
            </button>
          )}
        </div>
      </div>
      {entry.status === "uploading" && (
        <div className="mt-1 h-1 w-full bg-gray-200 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${entry.progress}%` }}
          />
        </div>
      )}
      {entry.status === "failed" && entry.error && (
        <div className="mt-1 text-[11px] text-red-600 break-all">
          {entry.error}
        </div>
      )}
    </div>
  );
}

function StatusPill({ entry }: { entry: UploadEntry }) {
  switch (entry.status) {
    case "queued":
      return <span className="text-gray-500 whitespace-nowrap">queued</span>;
    case "uploading":
      return (
        <span className="text-blue-700 whitespace-nowrap font-mono">
          {entry.progress}%
        </span>
      );
    case "done":
      return <span className="text-green-700 whitespace-nowrap">✓</span>;
    case "deduped":
      return (
        <span
          className="text-yellow-700 whitespace-nowrap"
          title="Already uploaded earlier"
        >
          ⊘
        </span>
      );
    case "cancelled":
      return <span className="text-gray-500 whitespace-nowrap">cancelled</span>;
    case "failed":
      return <span className="text-red-700 whitespace-nowrap">✗</span>;
  }
}
