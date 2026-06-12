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
  const [successCount, setSuccessCount] = useState(0);

  // The XHR objects can't live in React state — they're not serializable
  // and we don't want re-renders to recreate them. A ref keyed by entry id
  // is the natural fit. `configByIdRef` keeps the config snapshot at
  // enqueue time so a settings change mid-upload doesn't apply
  // retroactively. `fileByIdRef` stashes the actual File objects (not
  // safely serializable into React state).
  const xhrByIdRef = useRef(new Map<string, XMLHttpRequest>());
  const configByIdRef = useRef(new Map<string, string>());
  const fileByIdRef = useRef(new Map<string, File>());

  // Single-flight queue driver. Critical to avoid a `useEffect([queue,busy])`
  // pattern: `setQueue` to mark "uploading" would re-fire the effect,
  // its cleanup would set cancelled=true, and the async block would bail
  // out *before* marking the entry done — leaving it stuck at "uploading"
  // forever. Refs are immune to React's render lifecycle.
  const busyRef = useRef(false);
  const queueRef = useRef<UploadEntry[]>([]);
  queueRef.current = queue;

  const drive = useCallback(async () => {
    if (busyRef.current) return;
    const next = queueRef.current.find((e) => e.status === "queued");
    if (!next) return;
    busyRef.current = true;

    setQueue((q) =>
      q.map((x) =>
        x.id === next.id ? { ...x, status: "uploading", progress: 0 } : x,
      ),
    );

    try {
      const token = await authToken();
      const configJson = configByIdRef.current.get(next.id) ?? "{}";
      const resp = await uploadOne(
        next.id,
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
      busyRef.current = false;
      // After this upload finishes, see if anything else queued up while
      // we were busy.
      drive();
    }
  }, []);

  // Kick the driver whenever the queue changes — covers `enqueue` adding
  // new items. The driver itself is idempotent thanks to busyRef.
  useEffect(() => {
    drive();
  }, [queue, drive]);

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

  // Auto-remove successful uploads from the queue ~1s after they
  // complete. The dashboard's document list already shows the file
  // (the same successCount bump that fires this useEffect drives the
  // list refresh), so leaving "✓ done" rows in the upload widget is
  // just dead duplicate info. Failed/cancelled entries are kept
  // indefinitely so the user can read the error.
  useEffect(() => {
    const terminal = queue.filter(
      (x) => x.status === "done" || x.status === "deduped",
    );
    if (terminal.length === 0) return;
    const ids = new Set(terminal.map((x) => x.id));
    const t = setTimeout(() => {
      setQueue((q) => q.filter((x) => !ids.has(x.id)));
    }, 1000);
    return () => clearTimeout(t);
  }, [queue]);

  // Read fileByIdRef (declared above) inside the upload helper.
  function uploadOne(
    id: string,
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
          detail = detail.slice(0, 200);
          // 4xx carry a user-facing reason (file too large, too many pages,
          // unreadable PDF) — show it verbatim. 5xx are server faults where
          // the status code is the useful signal, so keep it prefixed.
          const isClientError = xhr.status >= 400 && xhr.status < 500;
          reject(new Error(isClientError ? detail : `HTTP ${xhr.status} — ${detail}`));
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
