-- 000_local_pre.sql
--
-- LOCAL-POSTGRES ONLY. Not run against Supabase.
--
-- Stubs Supabase-isms that 001+ depend on:
--   * `auth` schema with a minimal `auth.users(id uuid pk)` so the FKs
--     in documents.user_id / jobs.user_id resolve. The real source of
--     truth for identity is still Supabase Auth (JWT-verified at the
--     API layer); this table just exists to satisfy the FK constraint.
--   * `pgcrypto` for `gen_random_uuid()` (PG16 has it in pg_catalog
--     too, but enabling pgcrypto keeps this portable to older PG).
--
-- The companion file `999_local_post.sql` adds triggers (must run after
-- the documents/jobs tables exist).
--
-- This file is idempotent (safe to re-run).

create extension if not exists pgcrypto;

create schema if not exists auth;

create table if not exists auth.users (
  id uuid primary key
);
