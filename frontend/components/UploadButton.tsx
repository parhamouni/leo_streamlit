"use client";

import { useEffect, useRef, useState } from "react";
import { supabase } from "@/lib/supabase";

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

type EntryStatus = "queued" | "uploading" | "done" | "deduped" | "failed";

type Entry = {
  id: string;
  file: File;
  status: EntryStatus;
  progress: number; // 0..100
  error?: string;
  dedupedExistingStatus?: string;
};

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

async function authToken(): Promise<string | null> {
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}

type UploadResponse = {
  job_id?: string | null;
  document_id?: string | null;
  status?: string;
  existing_status?: string;
  queue_position?: number | null;
  running_jobs?: number | null;
};

/**
 * Upload one file via XMLHttpRequest so we get real bytes-uploaded
 * progress events. Resolves with the server's parsed response on 2xx;
 * rejects with an Error whose message is the server-reported reason.
 */
function uploadOne(
  file: File,
  onProgress: (pct: number) => void,
  token: string | null,
): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `${apiBase}/api/jobs`;
    xhr.open("POST", url);
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(100);
        let parsed: UploadResponse = {};
        try {
          parsed = JSON.parse(xhr.responseText);
        } catch {}
        resolve(parsed);
      } else {
        let detail = xhr.responseText;
        try {
          const j = JSON.parse(xhr.responseText);
          detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j);
        } catch {}
        reject(new Error(`HTTP ${xhr.status} — ${detail.slice(0, 200)}`));
      }
    });

    xhr.addEventListener("error", () =>
      reject(new Error("Network error during upload")),
    );
    xhr.addEventListener("abort", () =>
      reject(new Error("Upload aborted")),
    );

    const form = new FormData();
    form.append("pdf", file);
    xhr.send(form);
  });
}

export function UploadButton({ onUploaded }: { onUploaded: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [queue, setQueue] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // Drive the queue: when not busy, pick the next queued entry and upload it.
  useEffect(() => {
    if (busy) return;
    const next = queue.find((e) => e.status === "queued");
    if (!next) return;

    setBusy(true);
    setQueue((q) =>
      q.map((x) =>
        x.id === next.id ? { ...x, status: "uploading", progress: 0 } : x,
      ),
    );

    (async () => {
      try {
        const token = await authToken();
        const resp = await uploadOne(
          next.file,
          (pct) =>
            setQueue((q) =>
              q.map((x) =>
                x.id === next.id ? { ...x, progress: pct } : x,
              ),
            ),
          token,
        );
        const wasDeduped = resp.status === "deduped";
        setQueue((q) =>
          q.map((x) =>
            x.id === next.id
              ? {
                  ...x,
                  status: wasDeduped ? "deduped" : "done",
                  progress: 100,
                  dedupedExistingStatus: resp.existing_status,
                }
              : x,
          ),
        );
        onUploaded();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setQueue((q) =>
          q.map((x) =>
            x.id === next.id ? { ...x, status: "failed", error: msg } : x,
          ),
        );
        // Refresh in case earlier successes need rendering
        onUploaded();
      } finally {
        setBusy(false);
      }
    })();
  }, [queue, busy, onUploaded]);

  function enqueue(filesIn: FileList | File[]) {
    const accepted = Array.from(filesIn).filter(
      (f) =>
        f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"),
    );
    if (accepted.length === 0) return;
    setQueue((q) => [
      ...q,
      ...accepted.map<Entry>((f) => ({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${f.name}`,
        file: f,
        status: "queued",
        progress: 0,
      })),
    ]);
  }

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      enqueue(e.target.files);
      e.target.value = "";
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      enqueue(e.dataTransfer.files);
    }
  }

  function clearFinished() {
    setQueue((q) =>
      q.filter(
        (x) =>
          x.status !== "done" &&
          x.status !== "failed" &&
          x.status !== "deduped",
      ),
    );
  }

  const remaining = queue.filter((x) => x.status === "queued" || x.status === "uploading").length;
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
              ? `${remaining} file${remaining === 1 ? "" : "s"} in progress…`
              : "Drop more anytime — they'll queue up"}
          </div>
        </div>
      </div>

      {/* Queue display */}
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
                  <div className="truncate font-medium" title={e.file.name}>
                    {e.file.name}
                  </div>
                  <div className="text-xs text-gray-400">
                    {fmtBytes(e.file.size)}
                  </div>
                </div>
                <StatusPill entry={e} />
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

function StatusPill({ entry }: { entry: Entry }) {
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
    case "failed":
      return (
        <span className="text-xs text-red-700 whitespace-nowrap">✗ failed</span>
      );
  }
}
