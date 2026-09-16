-- Aperture on Supabase: row-level security, the API's database role, and the pgvector columns.
--
-- **This file does not create tables.** Alembic owns the schema — `apps/api/alembic/versions/`
-- is what `docker compose up` runs, what Render runs, and what the tests run. An earlier version
-- of this directory declared the tables a second time, in SQL, and the two definitions had
-- silently drifted: Alembic makes `failure_clusters.embedding` a `json` column and ids
-- `varchar`, while the SQL made them `vector(512)` and `uuid`, and then dropped `embedding`
-- outright. Applying both, as DEPLOY.md told you to, removed the column the ORM writes to and
-- broke cluster recompute. One schema owner, and this is not it.
--
-- Run order:
--     cd apps/api && alembic upgrade head        # tables
--     psql "$DATABASE_URL" -f infra/supabase/migrations/0001_rls_and_vectors.sql
--
-- Everything here is idempotent, so re-running after a later Alembic revision is safe and is
-- how you pick up policies for any table added since.

begin;

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- 1. pgvector columns for the production clustering path
-- ---------------------------------------------------------------------------
-- Added alongside the ORM's `embedding` (json), never in place of it. Nothing writes these yet:
-- `aperture/clustering/embed.py` still stores JSON, which works identically on SQLite and
-- Postgres. They exist so the switch to real vector search is a code change against a column
-- that is already there, and they are correctly dimensioned for the two embedding paths —
-- unlike the `vector(512)` this replaces, which matched neither.
alter table failure_clusters add column if not exists embedding_signature vector(10);   -- EMBED_DIM
alter table failure_clusters add column if not exists embedding_visual    vector(384);  -- ViT-S pooled

comment on column failure_clusters.embedding_signature is
  'Deterministic failure-signature vector (aperture/clustering/embed.py, EMBED_DIM=10). Unused until clustering moves off the json column.';
comment on column failure_clusters.embedding_visual is
  'Pooled ViT-Small embedding, 384-dim, learned path. Unused until clustering moves off the json column.';

-- ---------------------------------------------------------------------------
-- 2. A non-owner role for the API
-- ---------------------------------------------------------------------------
-- RLS does not apply to a table's owner. Connecting as `postgres` — which is what the Supabase
-- connection string gives you by default — silently bypasses every policy below, which is how
-- an unprotected deployment can pass a careless review. Point APERTURE_DATABASE_URL at this
-- role instead, and give it a password out of band.

do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'aperture_api') then
    create role aperture_api login;
  end if;
end
$$;

grant usage on schema public to aperture_api;
grant select, insert, update, delete on all tables in schema public to aperture_api;
grant usage, select on all sequences in schema public to aperture_api;
alter default privileges in schema public
  grant select, insert, update, delete on tables to aperture_api;

-- ---------------------------------------------------------------------------
-- 3. Which organization is this request for?
-- ---------------------------------------------------------------------------
-- Two sources, because there are two kinds of caller: a Supabase Auth JWT (dashboard users) and
-- the API's own pooled connection, which resolves an API key and then sets a per-transaction
-- GUC:  set local aperture.org_id = '<uuid>';
--
-- `set local` is scoped to the transaction, so a pooled connection handed to the next request
-- cannot inherit the previous tenant's identity.

create or replace function current_org_id() returns text
language plpgsql stable as $$
declare
  claim text;
begin
  begin
    claim := current_setting('request.jwt.claims', true)::json ->> 'org_id';
  exception when others then
    claim := null;
  end;
  if claim is null or claim = '' then
    claim := nullif(current_setting('aperture.org_id', true), '');
  end if;
  return claim;
end
$$;

-- ---------------------------------------------------------------------------
-- 4. Policies
-- ---------------------------------------------------------------------------
-- Tables carrying org_id directly.

do $$
declare t text;
begin
  foreach t in array array['organizations','robots','episodes','failure_clusters',
                           'attribution_jobs','api_keys','jobs']
  loop
    execute format('alter table %I enable row level security', t);
    execute format('alter table %I force row level security', t);
    execute format('drop policy if exists org_isolation on %I', t);
  end loop;
end
$$;

create policy org_isolation on organizations
  using (id = current_org_id()) with check (id = current_org_id());
create policy org_isolation on robots
  using (org_id = current_org_id()) with check (org_id = current_org_id());
create policy org_isolation on episodes
  using (org_id = current_org_id()) with check (org_id = current_org_id());
create policy org_isolation on failure_clusters
  using (org_id = current_org_id()) with check (org_id = current_org_id());
create policy org_isolation on attribution_jobs
  using (org_id = current_org_id()) with check (org_id = current_org_id());
create policy org_isolation on api_keys
  using (org_id = current_org_id()) with check (org_id = current_org_id());
create policy org_isolation on jobs
  using (org_id = current_org_id()) with check (org_id = current_org_id());

-- Child tables reach the organization through their parent, so isolation holds without
-- denormalising org_id onto every row.

do $$
declare t text;
begin
  foreach t in array array['episode_frames','failure_classifications','attributions',
                           'cluster_episodes','dataset_exports','verification_runs']
  loop
    execute format('alter table %I enable row level security', t);
    execute format('alter table %I force row level security', t);
    execute format('drop policy if exists org_isolation on %I', t);
  end loop;
end
$$;

create policy org_isolation on episode_frames
  using (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()))
  with check (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()));

create policy org_isolation on failure_classifications
  using (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()))
  with check (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()));

create policy org_isolation on attributions
  using (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()))
  with check (exists (select 1 from episodes e where e.id = episode_id and e.org_id = current_org_id()));

create policy org_isolation on cluster_episodes
  using (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()))
  with check (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()));

create policy org_isolation on dataset_exports
  using (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()))
  with check (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()));

create policy org_isolation on verification_runs
  using (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()))
  with check (exists (select 1 from failure_clusters c where c.id = cluster_id and c.org_id = current_org_id()));

-- ---------------------------------------------------------------------------
-- 5. Bootstrapping: resolving identity before RLS can be satisfied
-- ---------------------------------------------------------------------------
-- A chicken-and-egg problem, and the reason a careful RLS rollout still locks itself out: every
-- policy needs `aperture.org_id`, but the request only carries an API key. Working out which
-- org that key belongs to means reading `api_keys` (or `organizations`) — both of which are
-- behind the very policy we cannot satisfy yet. Without a way in, authentication finds nothing
-- and every request 403s.
--
-- These two definer-rights functions are that way in, and they are deliberately the narrowest
-- one available: each takes a credential and returns a single org id. They expose no rows, no
-- key hashes, and no way to enumerate — passing a wrong credential returns null, which the
-- caller already knew.

create or replace function resolve_api_key(candidate_hash text) returns text
language sql security definer stable
set search_path = public
as $$
  select org_id from api_keys where key_hash = candidate_hash and revoked_at is null limit 1;
$$;

-- For the static `APERTURE_API_KEYS` map, which names an org by slug rather than by key hash.
-- Needed to bootstrap the very first key onto a fresh deployment, before any api_keys row exists.
create or replace function resolve_org_slug(candidate_slug text) returns text
language sql security definer stable
set search_path = public
as $$
  select id from organizations where slug = candidate_slug limit 1;
$$;

revoke all on function resolve_api_key(text) from public;
revoke all on function resolve_org_slug(text) from public;
grant execute on function resolve_api_key(text) to aperture_api;
grant execute on function resolve_org_slug(text) to aperture_api;

commit;
