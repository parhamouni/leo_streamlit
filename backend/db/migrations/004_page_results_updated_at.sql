-- 004_page_results_updated_at.sql
--
-- Sprint 2 / A7 follow-up: when the Phase 1c stub upserts a row and Phase 3
-- later upserts the rich enrichment, the existing `created_at` doesn't move
-- (ON CONFLICT preserves it), so it's impossible to tell from a snapshot
-- alone whether per-page enrichment ever streamed in or all rows landed in
-- one batch. `updated_at` lets the frontend (and us) observe live progress.
--
-- Idempotent. Run in Supabase SQL Editor.

alter table page_results
  add column if not exists updated_at timestamptz not null default now();

-- ---------------------------------------------------------------------------
-- ROLLBACK
-- ---------------------------------------------------------------------------
--   alter table page_results drop column if exists updated_at;
