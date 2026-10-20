-- Aperture on Supabase: let the job worker find work under row-level security.
--
-- Apply after 0001 (same command, same order rule: `alembic upgrade head` first). Idempotent.
--
-- The worker is not a request: it carries no API key and so has no tenant. Under RLS a query
-- with no tenant sees zero rows, so `select ... from jobs where status = 'queued'` always came
-- back empty and every queued ingest and cluster recompute sat at `queued` forever, while the
-- API happily kept enqueuing more.
--
-- Same shape as the identity lookups in 0001: one definer-rights function, as narrow as it can
-- be. It answers "which org owns the oldest queued job?" and nothing else — one org id, no
-- payloads, no other columns. The worker binds that org to its transaction and then claims the
-- job through the ordinary org_isolation policy, so the claim itself is still tenant-checked.

begin;

create or replace function next_queued_job_org(job_kinds text[]) returns text
language sql security definer stable
set search_path = public
as $$
  select org_id
    from jobs
   where status = 'queued'
     and (job_kinds is null or kind = any(job_kinds))
   order by created_at, id
   limit 1;
$$;

revoke all on function next_queued_job_org(text[]) from public;
grant execute on function next_queued_job_org(text[]) to aperture_api;

commit;
