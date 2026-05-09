-- 999_local_post.sql
--
-- LOCAL-POSTGRES ONLY. Not run against Supabase.
--
-- Lazy-creates auth.users rows when documents/jobs are inserted, so the
-- FK constraint in 001 resolves without the API needing to know it's
-- talking to local Postgres. The Supabase JWT middleware has already
-- verified the user_id at the API layer; we just shadow it here so the
-- FK doesn't reject the insert.
--
-- See 000_local_pre.sql for the auth schema + table stub. This file
-- runs AFTER 001 because the triggers attach to documents/jobs.
--
-- This file is idempotent (safe to re-run).

create or replace function auth._ensure_user() returns trigger as $$
begin
  insert into auth.users (id) values (new.user_id) on conflict do nothing;
  return new;
end;
$$ language plpgsql;

drop trigger if exists ensure_auth_user on documents;
create trigger ensure_auth_user
  before insert on documents
  for each row execute function auth._ensure_user();

drop trigger if exists ensure_auth_user on jobs;
create trigger ensure_auth_user
  before insert on jobs
  for each row execute function auth._ensure_user();
