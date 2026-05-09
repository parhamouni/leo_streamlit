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

/**
 * Fetch wrapper that auto-attaches the current Supabase access token as
 * `Authorization: Bearer …`. Throws ApiError on non-2xx.
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
  const resp = await fetch(url, { ...init, headers });

  if (!resp.ok) {
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
  return resp;
}

export async function apiJson<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const resp = await apiFetch(path, init);
  return resp.json() as Promise<T>;
}
