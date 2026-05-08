# `backend/db/`

Supabase Postgres schema for the multi-user web app.

## Layout

- `migrations/` — versioned SQL migration files. Run **manually** in the Supabase SQL Editor; we don't auto-apply migrations from code.

## Running a migration

1. Open the project in [Supabase Studio](https://app.supabase.com/).
2. **SQL Editor → New query**.
3. Paste the contents of the next pending migration file (e.g. `migrations/001_initial_schema.sql`).
4. Run. All migration files are idempotent (`if not exists` / `create or replace`), so re-running is safe.
5. Verify in **Table Editor** that the expected tables/indexes exist.

## Rollback

Each migration documents its own rollback in a `-- ROLLBACK` comment block at the bottom. Copy that block into the SQL Editor to revert.

## Schema invariants

- Every user-facing row carries a `user_id uuid` column referencing `auth.users(id)`.
- Every backend query for user data **must** filter by `user_id`. Row-Level Security policies in a later migration will enforce this at the DB layer too.
