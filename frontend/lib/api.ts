"use client";

import { supabase } from "./supabase";

// Empty means same-origin (Next.js rewrites /api/* to the backend).
// Set NEXT_PUBLIC_API_BASE_URL to absolute URL when the backend lives
// on a different host (e.g. production: https://api.adamsfence.net).
const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown, message: string) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

// Status codes that are usually transient (Postgres pool timeout under
// burst load, momentary upstream blip, Vercel cold start, etc.) — worth
// silently retrying once or twice before surfacing the error to the user.
// 500/502/503/504 fit that profile; 4xx do not (those are real client
// errors and we don't want to mask them).
const TRANSIENT_STATUSES = new Set([500, 502, 503, 504]);
const RETRY_METHODS = new Set(["GET", "HEAD"]);

function sleep(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

/**
 * Fetch wrapper that auto-attaches the current Supabase access token as
 * `Authorization: Bearer …`. Throws ApiError on non-2xx.
 *
 * For idempotent reads (GET/HEAD) it transparently retries on 5xx with
 * exponential backoff (300 ms, 800 ms, 2 s = 4 attempts total). The
 * backend's Postgres pool sometimes 500s under burst load and recovers
 * within a second; without retries the dashboard would flash a scary
 * "API 500: Internal Server Error" banner during normal operation.
 */
export async function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const { data } = await supabase().auth.getSession();
  const token = data.session?.access_token;

  const headers = new Headers(init.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const url = path.startsWith("http") ? path : `${apiBase}${path}`;
  const method = (init.method ?? "GET").toUpperCase();
  const canRetry = RETRY_METHODS.has(method);
  const backoffsMs = canRetry ? [300, 800, 2000] : [];

  let lastResp: Response | null = null;
  for (let attempt = 0; attempt <= backoffsMs.length; attempt++) {
    const resp = await fetch(url, { ...init, headers });
    if (resp.ok) return resp;
    lastResp = resp;
    if (!canRetry || !TRANSIENT_STATUSES.has(resp.status)) break;
    if (attempt < backoffsMs.length) {
      // Add up to +50% random jitter so a burst of clients that all 500ed at
      // the same instant don't retry in lockstep and re-stampede the pool.
      const base = backoffsMs[attempt];
      await sleep(base + Math.floor(Math.random() * base * 0.5));
      continue;
    }
  }

  // Out of retries — surface the most recent error.
  const resp = lastResp!;
  let body: unknown = null;
  try {
    body = await resp.clone().json();
  } catch {
    try {
      body = await resp.clone().text();
    } catch {}
  }
  throw new ApiError(resp.status, body, `API ${resp.status} for ${path}`);
}

export async function apiJson<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const resp = await apiFetch(path, init);
  return resp.json() as Promise<T>;
}

// Some client-side security software (AV "web shields", filtering proxies)
// silently kills large streamed responses mid-body — observed in prod as
// downloads consistently dying ~14 MB in with fetch() throwing
// "Failed to fetch" while small exports from the same session succeed.
// Requesting the file in ranged chunks keeps every response comfortably
// under that kind of ceiling, and a chunk that does get cut is retried
// from the byte where it failed instead of restarting the whole file.
const DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024;
const CHUNK_RETRIES = 3;

/**
 * Download a (possibly large) binary response as a Blob.
 *
 * Probes Range support with the first chunk: static artifacts
 * (FileResponse) answer 206 and are fetched chunk-by-chunk; dynamically
 * generated exports answer 200 and fall back to a single-shot body.
 * Returns the first response too so callers can read headers
 * (Content-Disposition filename).
 */
export async function apiFetchBlob(
  path: string,
  onProgress?: (receivedBytes: number, totalBytes: number) => void,
): Promise<{ blob: Blob; response: Response }> {
  const first = await apiFetch(path, {
    headers: { Range: `bytes=0-${DOWNLOAD_CHUNK_BYTES - 1}` },
    cache: "no-store",
  });
  if (first.status !== 206) {
    return { blob: await first.blob(), response: first };
  }

  const total = Number(
    /\/(\d+)\s*$/.exec(first.headers.get("Content-Range") ?? "")?.[1] ?? NaN,
  );
  const parts: Blob[] = [await first.blob()];
  let received = parts[0].size;
  if (!Number.isFinite(total) || received >= total) {
    return { blob: parts[0], response: first };
  }
  onProgress?.(received, total);

  while (received < total) {
    const end = Math.min(received + DOWNLOAD_CHUNK_BYTES, total) - 1;
    let attempt = 0;
    for (;;) {
      try {
        const resp = await apiFetch(path, {
          headers: { Range: `bytes=${received}-${end}` },
          cache: "no-store",
        });
        if (resp.status !== 206) {
          throw new ApiError(
            resp.status,
            null,
            `expected 206 for ranged chunk, got ${resp.status}`,
          );
        }
        const part = await resp.blob();
        parts.push(part);
        received += part.size;
        break;
      } catch (e) {
        // A mid-chunk network cut lands here; back off briefly and re-request
        // the same range. `received` only advances on success.
        attempt += 1;
        if (attempt > CHUNK_RETRIES) throw e;
        await sleep(400 * attempt);
      }
    }
    onProgress?.(received, total);
  }

  return {
    blob: new Blob(parts, {
      type: first.headers.get("Content-Type") ?? "application/octet-stream",
    }),
    response: first,
  };
}
