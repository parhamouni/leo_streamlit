# Vercel deploy — `frontend/`

## One-time setup

1. **Create the Vercel project** pointing at this repo:
   - Project root → `frontend/` (the Next.js app lives there; `vercel.json` is at the project root)
   - Framework preset → Next.js (auto-detected)
   - Build command → `next build` (default)
   - Install command → `npm install` (default)

2. **Environment variables** (Vercel project → Settings → Environment Variables). Set on **Production** + **Preview** scopes:

   | Name | Value | Notes |
   |---|---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | `https://<project>.supabase.co` | from Supabase Studio → Project Settings → API |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJ...` (anon, public, ES256) | same page; this is safe to ship to the browser |
   | `NEXT_PUBLIC_API_BASE_URL` | `https://api.<your-domain>` | the AWS backend's public URL — see [deployment_notes.md](deployment_notes.md) |
   | `FENCE_API_INTERNAL_URL` | (empty in prod) | only used by the dev rewrite proxy in `next.config.mjs`; in prod the browser hits `NEXT_PUBLIC_API_BASE_URL` directly |

3. **Branch deploys**:
   - `main` → Production (custom domain `app.<your-domain>`)
   - everything else → Preview (e.g. `feat/web-app-migration` PR previews)

4. **Custom domain** (optional but recommended):
   - Add `app.<your-domain>` in Vercel → Settings → Domains
   - Point a CNAME at `cname.vercel-dns.com`

## What the rewrite proxy does

`next.config.mjs` rewrites `/api/*` → `FENCE_API_INTERNAL_URL` (default `http://127.0.0.1:8513`). This is a **dev-only** convenience so the frontend can call the backend on `localhost:3000` without CORS. In Vercel production:

- Leave `FENCE_API_INTERNAL_URL` unset.
- The frontend talks to `NEXT_PUBLIC_API_BASE_URL` directly via CORS (the backend's `FENCE_CORS_ORIGINS` env var must list the Vercel domain — see Phase 11.3).

## Acceptance checklist

- [ ] Production build succeeds on Vercel (`vercel --prod` or `git push origin main`).
- [ ] `https://app.<your-domain>/login` loads.
- [ ] Email/password signup → redirected to `/dashboard`.
- [ ] `/dashboard` calls `/api/me` and `/api/documents` against the AWS backend without CORS errors.
- [ ] Upload a small PDF; the document appears in the list and the job progresses.

## Rollback

Vercel keeps every previous build under Project → Deployments. Click → **Promote to Production** to roll back instantly. No DB or backend impact.
