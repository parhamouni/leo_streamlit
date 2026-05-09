"use client";

/**
 * UploadButton — drop-zone + inline queue display for the dashboard.
 * State + XHR lifecycle is owned by `<UploadProvider>` in app/layout.tsx
 * so navigating away mid-upload doesn't kill the request. This component
 * is only the *view*.
 */
import { useEffect, useRef, useState } from "react";
import {
  fmtBytes,
  useUpload,
  type UploadEntry,
} from "@/contexts/UploadContext";

export function UploadButton({
  onUploaded,
  configJson = "{}",
}: {
  onUploaded?: () => void;
  configJson?: string;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const { queue, enqueue, abort, clearFinished, successCount } = useUpload();

  // Fire onUploaded when an upload succeeds. Backwards compatible with the
  // dashboard's pre-context refresh hook.
  const lastSuccessRef = useRef(successCount);
  useEffect(() => {
    if (successCount !== lastSuccessRef.current) {
      lastSuccessRef.current = successCount;
      onUploaded?.();
    }
  }, [successCount, onUploaded]);

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      enqueue(e.target.files, configJson);
      e.target.value = "";
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      enqueue(e.dataTransfer.files, configJson);
    }
  }

  const remaining = queue.filter(
    (x) => x.status === "queued" || x.status === "uploading",
  ).length;
  const finished = queue.length - remaining;

  return (
    <div className="space-y-3">
      {/* Drop zone — always interactive, even while uploading */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        role="button"
        tabIndex={0}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          dragOver
            ? "border-blue-400 bg-blue-50"
            : "border-gray-300 bg-gray-50 hover:bg-gray-100"
        }`}
      >
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={onPick}
          className="hidden"
        />
        <div className="flex flex-col items-center gap-1">
          <svg
            width="32"
            height="32"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-gray-400"
          >
            <path
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-7.5-9V18m0-10.5L9 12m4.5-4.5L18 12"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div className="text-sm">
            <span className="font-medium text-blue-600">Click to upload</span>{" "}
            <span className="text-gray-500">or drag PDFs here</span>
          </div>
          <div className="text-xs text-gray-400">
            {remaining > 0
              ? `${remaining} file${remaining === 1 ? "" : "s"} in progress — they keep going if you navigate away`
              : "Drop more anytime — they'll queue up"}
          </div>
        </div>
      </div>

      {/* Queue display (same data as the floating panel; kept inline for
          the dashboard view). */}
      {queue.length > 0 && (
        <div className="border rounded-lg divide-y">
          <div className="flex items-center justify-between px-3 py-2 text-xs text-gray-500 bg-gray-50">
            <span>
              {finished}/{queue.length} done
            </span>
            {finished > 0 && (
              <button
                onClick={clearFinished}
                className="text-blue-600 hover:underline"
              >
                Clear finished
              </button>
            )}
          </div>
          {queue.map((e) => (
            <div key={e.id} className="px-3 py-2 text-sm">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium" title={e.fileName}>
                    {e.fileName}
                  </div>
                  <div className="text-xs text-gray-400">
                    {fmtBytes(e.fileSize)}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <StatusPill entry={e} />
                  {(e.status === "queued" || e.status === "uploading") && (
                    <button
                      onClick={() => abort(e.id)}
                      className="text-xs text-gray-400 hover:text-red-600"
                      title="Cancel this upload"
                    >
                      ✕
                    </button>
                  )}
                </div>
              </div>
              {e.status === "uploading" && (
                <div className="mt-1.5 h-1.5 w-full bg-gray-200 rounded overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${e.progress}%` }}
                  />
                </div>
              )}
              {e.status === "failed" && e.error && (
                <div className="mt-1 text-xs text-red-600 break-all">
                  {e.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusPill({ entry }: { entry: UploadEntry }) {
  switch (entry.status) {
    case "queued":
      return (
        <span className="text-xs text-gray-500 whitespace-nowrap">queued</span>
      );
    case "uploading":
      return (
        <span className="text-xs text-blue-700 whitespace-nowrap font-mono">
          {entry.progress}%
        </span>
      );
    case "done":
      return (
        <span className="text-xs text-green-700 whitespace-nowrap">✓ done</span>
      );
    case "deduped":
      return (
        <span
          className="text-xs text-yellow-700 whitespace-nowrap"
          title={`Already uploaded — existing job is ${entry.dedupedExistingStatus ?? "present"}`}
        >
          ⊘ already uploaded
        </span>
      );
    case "cancelled":
      return (
        <span className="text-xs text-gray-500 whitespace-nowrap">
          cancelled
        </span>
      );
    case "failed":
      return (
        <span className="text-xs text-red-700 whitespace-nowrap">✗ failed</span>
      );
  }
}
