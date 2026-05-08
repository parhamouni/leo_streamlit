-- 002_dedup_pdf_hash.sql
--
-- Adds a pdf_hash column to documents so re-uploads of the same file
-- are detected and short-circuited (no second analysis run).
--
-- Idempotent. Run in Supabase SQL Editor.

alter table documents add column if not exists pdf_hash text;
create index if not exists idx_documents_user_hash
  on documents(user_id, pdf_hash);

-- ---------------------------------------------------------------------------
-- ROLLBACK
-- ---------------------------------------------------------------------------
--   drop index if exists idx_documents_user_hash;
--   alter table documents drop column if exists pdf_hash;
