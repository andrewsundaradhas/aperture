-- Aperture on Supabase: take Aperture's tables and functions off Supabase's public Data API.
--
-- Apply after 0002 (same command: `python -m aperture.dbsetup`). Idempotent, and a no-op on plain
-- Postgres, where the `anon` / `authenticated` roles do not exist.
--
-- Supabase's default privileges grant `anon` and `authenticated` — the roles behind the
-- publishable key that ships in browser bundles — full access to every new table and EXECUTE on
-- every new function in `public`. Found on a fresh project:
--
--   * `POST /rest/v1/rpc/resolve_org_slug`, `/rpc/resolve_api_key` and `/rpc/next_queued_job_org`
--     answered 200 to anyone holding the publishable key. 0001/0002 revoke these from PUBLIC,
--     but Supabase grants them to `anon` directly, so that revoke did not reach it. The slug
--     lookup turns into an org-id oracle for the whole internet.
--   * All 14 tables were granted to `anon`. Row-level security still returned no rows, but that
--     made RLS the only barrier between the internet and fleet data.
--
-- Aperture never uses the Data API — the dashboard goes through the FastAPI service, which
-- connects as `aperture_api`. So these roles get nothing.

begin;

do $$
declare
  r text;
begin
  foreach r in array array['anon', 'authenticated'] loop
    if exists (select 1 from pg_roles where rolname = r) then
      execute format('revoke all on all tables in schema public from %I', r);
      execute format('revoke all on all sequences in schema public from %I', r);
      execute format('revoke execute on function resolve_api_key(text) from %I', r);
      execute format('revoke execute on function resolve_org_slug(text) from %I', r);
      execute format('revoke execute on function next_queued_job_org(text[]) from %I', r);
      execute format('revoke execute on function current_org_id() from %I', r);
      -- Tables and functions created later (future Alembic revisions) must not reopen this.
      execute format('alter default privileges in schema public revoke all on tables from %I', r);
      execute format('alter default privileges in schema public revoke all on sequences from %I', r);
      execute format('alter default privileges in schema public revoke execute on functions from %I', r);
    end if;
  end loop;

  -- Alembic's bookkeeping table is the owner's business. 0001 grants the app role every table
  -- in `public`, which included this one; the API never touches it, and a role that could
  -- rewrite the recorded schema version could make the next deploy skip or replay migrations.
  if to_regclass('public.alembic_version') is not null then
    revoke all on table alembic_version from aperture_api;
  end if;
end
$$;

commit;
