"use client";

import { useRef, useState } from "react";
import { apiFetch, ApiError } from "@/lib/api";

type UploadState =
  | { kind: "idle" }
  | { kind: "uploading"; current: number; total: number; filename: string }
  | { kind: "error"; message: string };

export function UploadButton({ onUploaded }: { onUploaded: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<UploadState>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);

  async function uploadFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) =>
      f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"),
    );
    if (list.length === 0) {
      setState({ kind: "error", message: "Only PDF files are accepted." });
      return;
    }

    for (let i = 0; i < list.length; i++) {
      const file = list[i];
      setState({
        kind: "uploading",
        current: i + 1,
        total: list.length,
        filename: file.name,
      });
      try {
        const form = new FormData();
        form.append("pdf", file);
        await apiFetch("/api/jobs", { method: "POST", body: form });
      } catch (e) {
        const msg =
          e instanceof ApiError
            ? `${file.name}: API ${e.status} — ${
                typeof e.body === "string" ? e.body : JSON.stringify(e.body)
              }`
            : e instanceof Error
              ? `${file.name}: ${e.message}`
              : `${file.name}: upload failed`;
        setState({ kind: "error", message: msg });
        onUploaded(); // refresh whatever did succeed
        return;
      }
    }

    setState({ kind: "idle" });
    onUploaded();
  }

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      uploadFiles(e.target.files);
      e.target.value = "";
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  }

  const busy = state.kind === "uploading";

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
        dragOver
          ? "border-blue-400 bg-blue-50"
          : "border-gray-300 bg-gray-50 hover:bg-gray-100"
      } ${busy ? "opacity-70" : ""}`}
    >
      <input
        ref={fileRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        onChange={onPick}
        className="hidden"
        disabled={busy}
      />
      <div className="flex flex-col items-center gap-2">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-gray-400">
          <path d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-7.5-9V18m0-10.5L9 12m4.5-4.5L18 12" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="text-sm">
          {busy ? (
            <span className="text-blue-700">
              Uploading {state.current}/{state.total} — {state.filename}…
            </span>
          ) : (
            <>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="font-medium text-blue-600 hover:underline"
              >
                Click to upload
              </button>{" "}
              <span className="text-gray-500">or drag PDFs here</span>
            </>
          )}
        </div>
        {state.kind === "error" && (
          <div className="text-xs text-red-600 mt-1 break-all max-w-md">
            {state.message}
          </div>
        )}
      </div>
    </div>
  );
}
