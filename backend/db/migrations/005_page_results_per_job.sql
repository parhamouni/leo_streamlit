-- 005_page_results_per_job.sql
--
-- Let a document hold per-job page_results so a document analysed in more
-- than one mode (e.g. fence AND electrical) keeps BOTH sets instead of the
-- latest run overwriting the previous one. Also records the trade on each
-- job so the frontend can offer Fence/Electrical tabs.
--
-- Safe to re-run (idempotent guards). Wrap in a transaction when applying.

begin;

-- 1. Trade (mode) each job analysed in.
alter table jobs add column if not exists trade text;

-- 2. Owning job for each page_results row.
alter table page_results add column if not exists job_id uuid references jobs(id) on delete cascade;

-- 3. Backfill job_id to each document's latest job — the existing single set
--    of rows reflects the most recent run (overwrite semantics pre-migration).
update page_results pr
set job_id = (
  select j.id from jobs j
  where j.document_id = pr.document_id
  order by j.created_at desc
  limit 1
)
where pr.job_id is null;

-- 4. Backfill jobs.trade: prefer the _trade stamped into this job's page rows,
--    else default to fence. (After step 3 only the latest job per doc has
--    rows, so older jobs fall back to fence — their trade is unrecoverable.)
update jobs j
set trade = coalesce(
  (select pr.result_json->>'_trade'
   from page_results pr
   where pr.job_id = j.id and pr.result_json ? '_trade'
   limit 1),
  'fence'
)
where j.trade is null;

-- 5. Swap uniqueness from (document, page) to (document, job, page) so
--    multiple jobs' pages coexist.
alter table page_results drop constraint if exists page_results_document_id_page_number_key;
alter table page_results add constraint page_results_doc_job_page_key
  unique (document_id, job_id, page_number);

create index if not exists idx_page_results_job_id on page_results(job_id);

commit;

-- Rollback (manual):
--   alter table page_results drop constraint if exists page_results_doc_job_page_key;
--   alter table page_results add constraint page_results_document_id_page_number_key unique (document_id, page_number);
--   alter table page_results drop column if exists job_id;
--   alter table jobs drop column if exists trade;
