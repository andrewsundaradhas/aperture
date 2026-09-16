# Deploying Aperture

Two supported shapes: **self-hosted** (Docker Compose, everything inside your network) and
**managed free-tier** (Render + Supabase + Cloudflare R2 + Vercel).

> **Status:** the Compose stack below is written but has **not been run end to end** — Docker
> was not available on the machine where it was authored. Treat the first `docker compose up`
> as a smoke test, not a proven path. The API and its migrations are covered by tests; the
> container wiring is not.

## Verified on

Supabase Postgres 17.6 (2026-09-16): all five Alembic migrations apply, RLS is enforced on all
13 tables, and cross-tenant reads, inserts, updates and deletes are all blocked at the database
for the `aperture_api` role. A query that omits its `where org_id` clause entirely returns only
the caller's rows — which is the point of having the layer at all.

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
cp .env.compose.example .env     # set the three passwords
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:3000
# MinIO     http://localhost:9001
```

Services: `db` (Postgres 16 + pgvector), `storage` (MinIO), `storage-init` (creates the bucket,
then exits), `api` (runs `alembic upgrade head`, then uvicorn), `worker` (drains the job queue),
`web` (Next.js).

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
| Database | Supabase Postgres | `alembic upgrade head`, **then** apply `infra/supabase/migrations/` — in that order |
| Storage | Cloudflare R2 | `APERTURE_R2_*` |

See [docs/LOCAL_VS_PRODUCTION.md](docs/LOCAL_VS_PRODUCTION.md) for the full mapping.

### Vercel

Import the repo and set **Root Directory to `apps/web`** — that is where `vercel.json` and the
Next app live, and it is a dashboard setting, not something `vercel.json` can express.

Set these four in Project Settings → Environment Variables. None of them may be `NEXT_PUBLIC_*`:
that prefix inlines the value into the browser bundle, and two of these are credentials.

| Variable | Required | Without it |
|---|---|---|
| `APERTURE_DASHBOARD_PASSWORD` | **yes** | **Every route returns 503.** The build still succeeds, so the deploy looks green and the site is dead. |
| `APERTURE_API_KEY` | **yes** | The proxy sends an empty key and the API answers 401 on every call. |
| `APERTURE_API_BASE_URL` | **yes** | Defaults to `http://localhost:8000`, which on a serverless function is itself — every request 502s. |
| `APERTURE_SESSION_SECRET` | no | Falls back to the password, so rotating the password also invalidates live sessions. |

The 503 is deliberate — the dashboard fails closed rather than serving fleet data to anyone who
finds the URL — but it is the one failure mode that a green build hides, so set the password
before the first deploy rather than after.

`NEXT_PUBLIC_SENTRY_DSN` / `SENTRY_DSN` are optional; the build and runtime both no-op without
them. `SENTRY_AUTH_TOKEN`, `SENTRY_ORG` and `SENTRY_PROJECT` only affect source-map upload.

## Security checklist before real data

- [ ] **Rotate any key that was ever set as `NEXT_PUBLIC_API_KEY`.** That prefix inlines the
      value into the browser bundle; assume every such key is public. The dashboard now proxies
      through `app/api/aperture/[...path]/route.ts` and holds the key server-side.
- [ ] Run migrations **in order**: `cd apps/api && alembic upgrade head` creates the tables,
      then `psql "$DATABASE_URL" -f infra/supabase/migrations/0001_rls_and_vectors.sql` adds RLS
      and the vector columns. Alembic owns the schema; the SQL file is additive and idempotent.
- [ ] Point `APERTURE_DATABASE_URL` at the **non-owner** `aperture_api` role, with a password you
      set yourself. RLS does not apply to a table's owner, and a Supabase connection string gives
      you `postgres` (the owner) by default — connecting as it silently disables every policy
      while everything still appears to work.
- [ ] Set a real `APERTURE_BOOTSTRAP_KEY`, mint per-org keys via `/v1/onboarding/keys`, then
      remove the bootstrap key. Issued keys are stored as hashes; the bootstrap key is not.
- [ ] Set `APERTURE_DASHBOARD_PASSWORD` (and ideally a separate `APERTURE_SESSION_SECRET`).
      With `NODE_ENV=production` and no password the dashboard refuses every request, which is
      deliberate — it fails closed rather than serving fleet data to anyone with the URL.
- [ ] Put TLS in front of both services. Neither container terminates TLS. The session cookie
      is issued `Secure` in production, so **sign-in will not work over plain HTTP.**
- [ ] Set `APERTURE_SENTRY_DSN` if you want error reporting; it stays inert otherwise.

### Known gaps

Tracked honestly rather than hidden:

- **Dashboard sign-in is a shared password, not identity.** One `APERTURE_DASHBOARD_PASSWORD`
  gates everyone, so the audit trail cannot say *who* did something, and revoking one person
  means rotating for all. It is a pilot-grade gate behind a VPN. SSO (SAML/OIDC) is the real
  answer and replacing `apps/web/lib/session.ts` is the contained change that gets you there.
- **Rate limits are per-process.** `aperture/core/ratelimit.py` keeps buckets in memory, so N
  API replicas grant N times the budget. Move `_BUCKETS` to Redis before scaling out.
- **Ingestion and cluster recompute are synchronous.** Large uploads and big fleets will hit
  request timeouts until the job queue exists.
