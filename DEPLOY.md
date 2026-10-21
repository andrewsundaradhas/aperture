# Deploying Aperture

Two supported shapes: **self-hosted** (Docker Compose, everything inside your network) and
**managed free-tier** (Render + Supabase + Cloudflare R2 + Vercel).

> **Status:** Docker was not available on the machine where the Compose stack was written, so
> `docker compose up` itself has not been run. Its startup sequence has been: the same
> migrations, security SQL, bootstrap and API/worker commands, run in that order against
> Postgres 17 + pgvector and MinIO, ingested a real 800-episode LeRobot dataset through a
> presigned upload as the restricted database role. Treat the first `docker compose up` as a
> smoke test of the container wiring.

## Verified on

Supabase Postgres 17.6 (2026-09-16): all five Alembic migrations apply, RLS is enforced on all
13 tables, and cross-tenant reads, inserts, updates and deletes are all blocked at the database
for the `aperture_api` role. A query that omits its `where org_id` clause entirely returns only
the caller's rows — which is the point of having the layer at all.

Postgres 17.11 + pgvector (2026-10-20, local): signup, every write path, the job worker and the
admin CLI all run as `aperture_api` with RLS forced — `apps/api/tests/test_postgres_rls.py`,
which CI now runs against a pgvector service container on every push.

**Connecting from IPv4.** Supabase's direct host (`db.<ref>.supabase.co`) is IPv6-only. On an
IPv4-only network use the pooler instead, in **session mode** (port 5432, not 6543 — migrations
and `SET LOCAL` want a real session):

```
postgresql+psycopg2://<role>.<project-ref>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

Note the username is `role.project-ref`; the pooler routes on it, and omitting the ref fails with
`no tenant identifier provided`.

## Self-hosted (recommended for robot fleet data)

Most robotics teams will not ship proprietary policy rollouts to a vendor's cloud. Nothing in
Aperture requires it — the API is stateless and needs only Postgres and an S3-compatible
bucket, both of which the stack provides.

```bash
cp .env.compose.example .env     # set the passwords and the bootstrap key
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:3000
# MinIO     http://localhost:9001
```

Services, in start order: `db` (Postgres 16 + pgvector) → `migrate` (`alembic upgrade head` as
the owner, then exits) → `db-security` (applies `infra/supabase/migrations/*.sql` — row-level
security, the restricted `aperture_api` role and its password — then exits) → `api` and
`worker`, both connected **as `aperture_api`**, so RLS is enforced. `api` first runs
`python -m aperture.admin ensure-org` so the org named by `APERTURE_BOOTSTRAP_ORG` exists.
`storage` (MinIO) + `storage-init` (creates the bucket) run alongside; MinIO's S3 port 9000 is
published because uploaders PUT archives to it directly — set `APERTURE_PUBLIC_STORAGE_URL` to
the address they use. `web` is the dashboard.

Onboarding a customer: `docker compose exec api python -m aperture.admin create-org --slug
<slug> --name "<Name>"` prints their API key once.

The `worker` is separate from the `api` on purpose: a job that exhausts memory should take down
the worker, not the process serving requests, and the two scale differently — ingestion is
bursty, serving is not. Running the API alone still works for a single machine
(`APERTURE_INLINE_WORKER=true`, the default) — it drains the queue on a background thread, which
is what makes the README quick-start work without a second process.

### Minimum external dependencies

| Need | Version | Why |
|---|---|---|
| Postgres | **16+, with `pgvector`** | `failure_clusters.embedding_*` are vector columns; RLS policies assume 9.5+ syntax |
| S3-compatible storage | any | episode blobs and frame images (MinIO, S3, R2, Ceph) |
| CPU | 2 cores, 4 GB RAM | the default heuristic path has no ML dependencies at all |
| GPU | optional | only for the learned path (`APERTURE_USE_LEARNED_MODELS=true`) and for training |

The learned path is **off by default** and the whole product works without it. A GPU is needed
only to train checkpoints (`ml/training/`) or to run the attention-rollout worker.

### Not yet containerised

The external attention-rollout worker is still a notebook (`ml/notebooks/`), because it wants a
GPU and OpenVLA weights that do not belong in this image. It pulls work over HTTP from
`/v1/attribution_jobs`, so it can run anywhere with a GPU and network access to the API.

## Managed free-tier

| Concern | Service | Config |
|---|---|---|
| API | Render | `infra/render.yaml` |
| Dashboard | Vercel | `apps/web/vercel.json`, root directory `apps/web` |
| Database | Supabase Postgres | `alembic upgrade head`, **then** `python -m aperture.dbsetup` — both as the owner; `render.yaml` runs them on every deploy via `APERTURE_MIGRATION_DATABASE_URL` |
| Storage | Cloudflare R2 | `APERTURE_R2_*` |

See [docs/LOCAL_VS_PRODUCTION.md](docs/LOCAL_VS_PRODUCTION.md) for the full mapping.

### Vercel

Import the repo and set **Root Directory to `apps/web`** — that is where `vercel.json` and the
Next app live, and it is a dashboard setting, not something `vercel.json` can express.

The dashboard has two sign-in modes, chosen by whether `APERTURE_API_KEY` is set. None of these
may be `NEXT_PUBLIC_*`: that prefix inlines the value into the browser bundle.

**Per-org (recommended for a hosted product)** — leave `APERTURE_API_KEY` unset. Each person
signs in with their organization's API key; it is sealed with AES-GCM into an HttpOnly cookie
and the proxy attaches it, so one deployment serves every customer and each sees only their own
fleet.

| Variable | Required | Without it |
|---|---|---|
| `APERTURE_API_BASE_URL` | **yes** | Defaults to `http://localhost:8000`, which on a serverless function is itself — every request 502s. |
| `APERTURE_SESSION_SECRET` | **yes** | **Every route returns 503** (fails closed). The build still succeeds, so set it before the first deploy. |

**Single-org** — set `APERTURE_API_KEY`; one shared password gates the dashboard.

| Variable | Required | Without it |
|---|---|---|
| `APERTURE_API_BASE_URL` | **yes** | as above |
| `APERTURE_API_KEY` | **yes** | Selects this mode; every request uses it. |
| `APERTURE_DASHBOARD_PASSWORD` | **yes** | **Every route returns 503.** |
| `APERTURE_SESSION_SECRET` | no | Falls back to the password, so rotating the password also invalidates live sessions. |

Failed sign-ins are throttled per client (10 per 15 minutes, then 429), in memory per instance.

**Uploads from the dashboard.** On R2/S3 the browser PUTs archives straight to the bucket's
presigned URL, so the bucket needs a CORS rule allowing `PUT` from the dashboard's origin. On
the local filesystem backend uploads stream through the dashboard's API proxy.

`NEXT_PUBLIC_SENTRY_DSN` / `SENTRY_DSN` are optional; the build and runtime both no-op without
them. `SENTRY_AUTH_TOKEN`, `SENTRY_ORG` and `SENTRY_PROJECT` only affect source-map upload.

## Security checklist before real data

- [ ] **Rotate any key that was ever set as `NEXT_PUBLIC_API_KEY`.** That prefix inlines the
      value into the browser bundle; assume every such key is public. The dashboard now proxies
      through `app/api/aperture/[...path]/route.ts` and holds the key server-side.
- [ ] Run migrations **in order**, as the database owner: `alembic upgrade head` creates the
      tables, then `python -m aperture.dbsetup` applies RLS, the vector columns and the job-queue
      lookup from `infra/supabase/migrations/`. Alembic owns the schema; the SQL is additive and
      idempotent. Compose and `render.yaml` both do this for you.
- [ ] Point `APERTURE_DATABASE_URL` at the **non-owner** `aperture_api` role, with a password you
      set yourself. RLS does not apply to a table's owner, and a Supabase connection string gives
      you `postgres` (the owner) by default — connecting as it silently disables every policy
      while everything still appears to work.
      Set its password **once**, after the first migration:
      `APERTURE_DATABASE_URL=<owner url> APERTURE_DB_API_PASSWORD=<new> python -m aperture.dbsetup --set-api-password ../../infra/supabase/migrations`.
      Not on every deploy: re-setting it makes Supabase's pooler refuse the role for minutes.
      Through the pooler the username is `aperture_api.<project-ref>`.
- [ ] Set a real `APERTURE_BOOTSTRAP_KEY`, mint per-org keys via `/v1/onboarding/keys` or
      `python -m aperture.admin issue-key`, then remove the bootstrap key. Issued keys are stored
      as hashes; the bootstrap key is not.
- [ ] Decide on signup. It is **closed in production** unless `APERTURE_SIGNUP_TOKEN` is set;
      with the token, `POST /v1/onboarding/signup` needs it in `X-Signup-Token`. Otherwise
      provision orgs with `python -m aperture.admin create-org`.
- [ ] Set `APERTURE_SESSION_SECRET` (per-org mode) or `APERTURE_DASHBOARD_PASSWORD` (single-org).
      With `NODE_ENV=production` and neither, the dashboard refuses every request, which is
      deliberate — it fails closed rather than serving fleet data to anyone with the URL.
- [ ] Put TLS in front of both services. Neither container terminates TLS. The session cookie
      is issued `Secure` in production, so **sign-in will not work over plain HTTP.**
- [ ] Set `APERTURE_SENTRY_DSN` if you want error reporting; it stays inert otherwise.

### Known gaps

Tracked honestly rather than hidden:

- **No per-person identity yet.** Per-org sign-in uses the org's API key, so everyone in an org
  shares one identity: the audit trail cannot say *who* did something, and revoking one person
  means rotating a key. SSO (SAML/OIDC) with per-user accounts is the real answer;
  `apps/web/lib/session.ts` is the contained place to add it.
- **Rate limits are per-process.** `aperture/core/ratelimit.py` (API) and
  `apps/web/lib/loginLimit.ts` (dashboard sign-in) keep counters in memory, so N replicas grant
  N times the budget. Move them to Redis/KV before scaling out.
- **Classifier quality on real fleets.** The heuristics need per-frame signals most public
  datasets do not log (action confidence, contact force, sub-goal tokens); the commanded-action
  jerk signal covers part of motor. On `lerobot/xarm_lift_medium` most failures honestly come back
  `unknown`. The trained failure head was supervised on synthetic corruptions — retrain it on
  labelled fleet failures before selling `method="learned"` as a diagnosis.
