-- 003_phase_timing.sql
--
-- Sprint 2 / A8: per-phase progress and ETAs.
--
-- Adds a `phase_started_at` column to `jobs` so the frontend can compute a
-- rate-based ETA both for the overall run (using `started_at`) and for the
-- current phase (using `phase_started_at`). The worker writes this column
-- via the existing `db.update_job_progress` helper, which sets it to now()
-- whenever `current_phase` actually changes.
--
-- Idempotent. Run in Supabase SQL Editor.

alter table jobs add column if not exists phase_started_at timestamptz;

-- ---------------------------------------------------------------------------
-- ROLLBACK
-- ---------------------------------------------------------------------------
--   alter table jobs drop column if exists phase_started_at;
