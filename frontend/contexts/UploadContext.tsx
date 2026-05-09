"use client";

/**
 * UploadProvider — keeps in-flight uploads alive across route changes.
 *
 * Before this existed, the upload UI lived inside `app/dashboard/page.tsx`,
 * and the XHR object used to send the file was held in that component's
 * state. Navigating to another page (e.g. clicking on a document)
 * unmounted the dashboard, garbage-collected the XHR, and aborted the
 * upload mid-stream.
 *
 * This provider lives in `app/layout.tsx` (which is *not* unmounted on
 * intra-app navigation) and owns the queue + XHR references. Components
 * that mount/unmount with the route just observe state via `useUpload()`.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { supabase } from "@/lib/supabase";

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export type UploadStatus =
  | "queued"
  | "uploading"
  | "done"
  | "deduped"
  | "failed"
  | "cancelled";

export type UploadEntry = {
  id: string;
  fileName: string;
  fileSize: number;
  status: UploadStatus;
  progress: number; // 0..100
  error?: string;
  dedupedExistingStatus?: string;
  jobId?: string | null;
  documentId?: string | null;
  // Bumped each time this entry transitions to a terminal success state, so
  // listeners can refresh their data without setting up event subscribers.
  completedAt?: number;
};

type UploadResponse = {
  job_id?: string | null;
  document_id?: string | null;
  status?: string;
  existing_status?: string;
  queue_position?: number | null;
  running_jobs?: number | null;
};

export type UploadContextValue = {
  queue: UploadEntry[];
  enqueue: (files: FileList | File[], configJson?: string) => void;
  abort: (id: string) => void;
  clearFinished: () => void;
  /** Monotonic count of successful uploads — useful as a useEffect dep
   * to drive document-list refreshes without registering callbacks. */
  successCount: number;
};

const UploadCtx = createContext<UploadContextValue | null>(null);

export function useUpload(): UploadContextValue {
  const ctx = useContext(UploadCtx);
  if (!ctx) {
    throw new Error(
      "useUpload() must be used inside <UploadProvider>. Mount it in app/layout.tsx.",
    );
  }
  return ctx;
}

async function authToken(): Promise<string | null> {
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}

export function UploadProvider({ children }: { children: React.ReactNode }) {
  const [queue, setQueue] = useState<UploadEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [successCount, setSuccessCount] = useState(0);

  // The XHR objects can't live in React state — they're not serializable
  // and we don't want re-renders to recreate them. A ref keyed by entry id
  // is the natural fit. `configByIdRef` keeps the config snapshot at
  // enqueue time so a settings change mid-upload doesn't apply
  // retroactively.
  const xhrByIdRef = useRef(new Map<string, XMLHttpRequest>());
  const configByIdRef = useRef(new Map<string, string>());

  // Drive the queue: when not busy, pick the next queued entry and upload
  // it. Same single-flight model as the old UploadButton, just hoisted.
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

    let cancelled = false;
    (async () => {
      try {
        const token = await authToken();
        const configJson = configByIdRef.current.get(next.id) ?? "{}";
        const resp = await uploadOne(
          next.id,
          next,
          token,
          configJson,
          (pct) =>
            setQueue((q) =>
              q.map((x) =>
                x.id === next.id ? { ...x, progress: pct } : x,
              ),
            ),
          xhrByIdRef.current,
        );
        if (cancelled) return;
        const wasDeduped = resp.status === "deduped";
        setQueue((q) =>
          q.map((x) =>
            x.id === next.id
              ? {
                  ...x,
                  status: wasDeduped ? "deduped" : "done",
                  progress: 100,
                  jobId: resp.job_id ?? null,
                  documentId: resp.document_id ?? null,
                  dedupedExistingStatus: resp.existing_status,
                  completedAt: Date.now(),
                }
              : x,
          ),
        );
        setSuccessCount((n) => n + 1);
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        const aborted = msg === "Upload aborted";
        setQueue((q) =>
          q.map((x) =>
            x.id === next.id
              ? {
                  ...x,
                  status: aborted ? "cancelled" : "failed",
                  error: aborted ? undefined : msg,
                }
              : x,
          ),
        );
      } finally {
        xhrByIdRef.current.delete(next.id);
        configByIdRef.current.delete(next.id);
        setBusy(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [queue, busy]);

  const enqueue = useCallback(
    (filesIn: FileList | File[], configJson: string = "{}") => {
      const accepted = Array.from(filesIn).filter(
        (f) =>
          f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"),
      );
      if (accepted.length === 0) return;

      const newEntries: Array<{ entry: UploadEntry; file: File; cfg: string }> =
        accepted.map((f) => {
          const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${f.name}`;
          return {
            entry: {
              id,
              fileName: f.name,
              fileSize: f.size,
              status: "queued",
              progress: 0,
            },
            file: f,
            cfg: configJson,
          };
        });

      // Stash the actual File + config separately from the React state
      // (File objects survive across renders by reference; we just need
      // a place keyed by id).
      for (const { entry, file, cfg } of newEntries) {
        fileByIdRef.current.set(entry.id, file);
        configByIdRef.current.set(entry.id, cfg);
      }
      setQueue((q) => [...q, ...newEntries.map((n) => n.entry)]);
    },
    [],
  );

  const abort = useCallback((id: string) => {
    const xhr = xhrByIdRef.current.get(id);
    if (xhr) xhr.abort();
    // If the entry is still queued (no XHR yet), just mark it cancelled.
    setQueue((q) =>
      q.map((x) =>
        x.id === id && x.status === "queued"
          ? { ...x, status: "cancelled" }
          : x,
      ),
    );
  }, []);

  const clearFinished = useCallback(() => {
    setQueue((q) =>
      q.filter(
        (x) =>
          x.status === "queued" ||
          x.status === "uploading",
      ),
    );
  }, []);

  // We need real File objects to feed XHR's FormData, but File is not
  // serializable into our React state. Stash by id in a ref.
  const fileByIdRef = useRef(new Map<string, File>());

  // Read fileByIdRef inside the upload helper.
  function uploadOne(
    id: string,
    entry: UploadEntry,
    token: string | null,
    configJson: string,
    onProgress: (pct: number) => void,
    xhrMap: Map<string, XMLHttpRequest>,
  ): Promise<UploadResponse> {
    return new Promise((resolve, reject) => {
      const file = fileByIdRef.current.get(id);
      if (!file) {
        reject(new Error("File reference lost (was the upload re-enqueued?)"));
        return;
      }
      const xhr = new XMLHttpRequest();
      xhrMap.set(id, xhr);
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
      xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));

      const form = new FormData();
      form.append("pdf", file);
      form.append("config", configJson);
      xhr.send(form);
    });
  }

  // Cleanup file refs whenever an entry leaves the queue.
  useEffect(() => {
    const ids = new Set(queue.map((q) => q.id));
    for (const id of Array.from(fileByIdRef.current.keys())) {
      if (!ids.has(id)) fileByIdRef.current.delete(id);
    }
  }, [queue]);

  const value = useMemo<UploadContextValue>(
    () => ({ queue, enqueue, abort, clearFinished, successCount }),
    [queue, enqueue, abort, clearFinished, successCount],
  );

  return <UploadCtx.Provider value={value}>{children}</UploadCtx.Provider>;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
