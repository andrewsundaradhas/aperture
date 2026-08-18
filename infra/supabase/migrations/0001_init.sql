-- Aperture production schema for Supabase Postgres (build doc §5).
-- Single-tenant isolation via row-level security; pgvector for embedding-space clustering.
-- Locally the app runs on SQLite via SQLAlchemy create_all; this file is the production DB.

create extension if not exists "uuid-ossp";
create extension if not exists vector;   -- pgvector, bundled with Supabase (build doc §3)

create table organizations (
  id   uuid primary key default uuid_generate_v4(),
  slug text unique not null,
  name text not null,
  tier text not null default 'pilot'   -- pilot|growth|enterprise
);

create table robots (
  id              uuid primary key default uuid_generate_v4(),
  org_id          uuid not null references organizations(id) on delete cascade,
  embodiment_type text not null,
  policy_name     text not null
);

create table episodes (
  id            uuid primary key default uuid_generate_v4(),
  robot_id      uuid not null references robots(id) on delete cascade,
  org_id        uuid not null references organizations(id) on delete cascade,
  started_at    timestamptz not null default now(),
  outcome       text not null,          -- success|fail
  source_format text not null,          -- rlds|lerobot
  instruction   text,
  rlds_uri      text                    -- raw blob in R2
);

create table episode_frames (
  id                uuid primary key default uuid_generate_v4(),
  episode_id        uuid not null references episodes(id) on delete cascade,
  t                 int not null,
  action_confidence double precision,
  contact_force     double precision,
  subgoal           text
);

create table failure_classifications (
  id         uuid primary key default uuid_generate_v4(),
  episode_id uuid not null unique references episodes(id) on delete cascade,
  surface    text not null,             -- perception|grounding|motor
  method     text not null default 'heuristic',
  confidence double precision not null,
  details    jsonb not null default '{}'
);

create table attributions (
  id                    uuid primary key default uuid_generate_v4(),
  episode_id            uuid not null unique references episodes(id) on delete cascade,
  attention_map_uri     text,
  counterfactual_result jsonb
);

create table attribution_jobs (
  id                uuid primary key default uuid_generate_v4(),
  episode_id        uuid not null references episodes(id) on delete cascade,
  org_id            uuid not null references organizations(id) on delete cascade,
  status            text not null default 'queued',   -- queued|running|done|failed
  attention_map_uri text,
  error             text,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create table failure_clusters (
  id               uuid primary key default uuid_generate_v4(),
  org_id           uuid not null references organizations(id) on delete cascade,
  label            text not null,
  dominant_surface text,
  embedding        vector(512),          -- CLIP ViT-B/32 centroid in production
  episode_count    int not null default 0,
  created_at       timestamptz not null default now()
);

create table cluster_episodes (
  id         uuid primary key default uuid_generate_v4(),
  cluster_id uuid not null references failure_clusters(id) on delete cascade,
  episode_id uuid not null references episodes(id) on delete cascade,
  unique (cluster_id, episode_id)
);

create table dataset_exports (
  id            uuid primary key default uuid_generate_v4(),
  cluster_id    uuid not null references failure_clusters(id) on delete cascade,
  format        text not null,           -- rlds|lerobot
  export_uri    text not null,
  episode_count int not null default 0,
  created_at    timestamptz not null default now()
);

create table verification_runs (
  id                uuid primary key default uuid_generate_v4(),
  cluster_id        uuid not null references failure_clusters(id) on delete cascade,
  pre_success_rate  double precision not null,
  post_success_rate double precision not null,
  pre_n             int not null default 0,
  post_n            int not null default 0,
  verified_at       timestamptz not null default now()
);

-- Indexes
create index idx_episodes_org on episodes(org_id);
create index idx_episodes_robot on episodes(robot_id);
create index idx_frames_episode on episode_frames(episode_id);
create index idx_clusters_org on failure_clusters(org_id);

-- ---------------------------------------------------------------------------
-- Single-tenant isolation (build doc Phase 0 / PRD §9.3): one org can never read
-- another org's rows. Policies key off a per-request JWT claim `org_id` set by Supabase
-- Auth. Enable RLS on every tenant-scoped table.
-- ---------------------------------------------------------------------------
alter table organizations         enable row level security;
alter table robots                enable row level security;
alter table episodes              enable row level security;
alter table episode_frames        enable row level security;
alter table failure_classifications enable row level security;
alter table attributions          enable row level security;
alter table attribution_jobs      enable row level security;
alter table failure_clusters      enable row level security;
alter table cluster_episodes      enable row level security;
alter table dataset_exports       enable row level security;
alter table verification_runs     enable row level security;

create policy org_isolation_robots on robots
  using (org_id = (auth.jwt() ->> 'org_id')::uuid);
create policy org_isolation_episodes on episodes
  using (org_id = (auth.jwt() ->> 'org_id')::uuid);
create policy org_isolation_clusters on failure_clusters
  using (org_id = (auth.jwt() ->> 'org_id')::uuid);
create policy org_isolation_jobs on attribution_jobs
  using (org_id = (auth.jwt() ->> 'org_id')::uuid);
-- Child tables inherit isolation through their parent's org_id via joins in the API layer;
-- add equivalent policies keyed on a joined org_id for defense-in-depth before GA.
