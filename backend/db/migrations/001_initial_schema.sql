-- 001_initial_schema.sql
--
-- Multi-user web-app schema for the Leo Fence Detector migration.
-- Run manually in Supabase SQL Editor (see ../README.md).
--
-- Tables:
--   documents     — one row per uploaded PDF (owned by a Supabase user)
--   jobs          — one row per processing run against a document
--   page_results  — per-page output from the pipeline (joined to documents)
--   artifacts     — generated files in S3 (highlighted PDF, exports, etc.)
--
-- Ownership:
--   documents.user_id  → auth.users.id  (the source of truth for ownership)
--   jobs.user_id       → auth.users.id  (denormalized copy for fast queries)
--   page_results / artifacts inherit ownership via document_id → documents.user_id
--
-- This migration is idempotent (safe to re-run).

-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------
create table if not exists documents (
  id                uuid primary key default gen_random_uuid(),
  user_id           uuid not null references auth.users(id) on delete cascade,
  original_filename text not null,
  storage_path      text not null,
  status            text not null default 'uploaded',
  total_pages       int,
  created_at        timestamptz not null default now()
);

create index if not exists idx_documents_user_id    on documents(user_id);
create index if not exists idx_documents_created_at on documents(created_at desc);

-- ---------------------------------------------------------------------------
-- jobs
-- ---------------------------------------------------------------------------
create table if not exists jobs (
  id               uuid primary key default gen_random_uuid(),
  document_id      uuid not null references documents(id) on delete cascade,
  user_id          uuid not null references auth.users(id) on delete cascade,
  status           text not null default 'queued',
  current_phase    text,
  progress_percent int  not null default 0,
  error_message    text,
  started_at       timestamptz,
  finished_at      timestamptz,
  created_at       timestamptz not null default now()
);

create index if not exists idx_jobs_user_id     on jobs(user_id);
create index if not exists idx_jobs_document_id on jobs(document_id);
create index if not exists idx_jobs_status      on jobs(status);

-- ---------------------------------------------------------------------------
-- page_results
-- ---------------------------------------------------------------------------
create table if not exists page_results (
  id            uuid primary key default gen_random_uuid(),
  document_id   uuid not null references documents(id) on delete cascade,
  page_number   int  not null,
  is_fence_page boolean not null default false,
  result_json   jsonb,
  created_at    timestamptz not null default now(),
  unique (document_id, page_number)
);

create index if not exists idx_page_results_document_id on page_results(document_id);

-- ---------------------------------------------------------------------------
-- artifacts
-- ---------------------------------------------------------------------------
create table if not exists artifacts (
  id           uuid primary key default gen_random_uuid(),
  document_id  uuid not null references documents(id) on delete cascade,
  job_id       uuid references jobs(id) on delete cascade,
  type         text not null,
  storage_path text not null,
  created_at   timestamptz not null default now()
);

create index if not exists idx_artifacts_document_id on artifacts(document_id);
create index if not exists idx_artifacts_job_id      on artifacts(job_id);

-- ---------------------------------------------------------------------------
-- ROLLBACK
-- ---------------------------------------------------------------------------
-- Run this block in the SQL Editor to revert this migration.
-- Order matters: drop child tables first.
--
--   drop table if exists artifacts;
--   drop table if exists page_results;
--   drop table if exists jobs;
--   drop table if exists documents;
