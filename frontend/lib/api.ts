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
      await sleep(backoffsMs[attempt]);
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
