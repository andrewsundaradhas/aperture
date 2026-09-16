# Aperture — Design Partner Onboarding

Follow this end to end without a call. ~5 minutes to a working, seeded instance.

## 1. Run the stack locally

```bash
# Backend
cd apps/api
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m aperture.seed                                # loads demo org + synthetic fixtures
uvicorn aperture.main:app --reload                     # http://localhost:8000/docs

# Frontend (separate terminal)
cd apps/web
npm install
npm run dev                                            # http://localhost:3000
```

Open http://localhost:3000 — you should see classified episodes and a failure cluster already.

## 2. Get your API key

For local development, set a generated API key in `APERTURE_API_KEYS`. For your own org:

```bash
curl -X POST http://localhost:8000/v1/onboarding/signup \
  -H "Content-Type: application/json" \
  -d '{"slug":"your-co","name":"Your Co"}'
# → returns {"api_key": "ak_...", ...}   save this
```

Only a hash of the key is stored, so it is shown exactly once. Manage keys with
`GET /v1/onboarding/keys` (prefixes only, never the secret), `POST /v1/onboarding/keys` to mint
another, and `DELETE /v1/onboarding/keys/{id}` to revoke. Rotate without downtime by minting the
new key, moving traffic across, then revoking the old one.

Register a robot:

```bash
curl -X POST http://localhost:8000/v1/onboarding/robots \
  -H "X-API-Key: ak_..." -H "Content-Type: application/json" \
  -d '{"embodiment_type":"franka","policy_name":"openvla-7b"}'
```

## 3. Upload your episodes

Aperture accepts **RLDS** (`.tfrecord`, or shards in an archive) and **LeRobot v3** (a
`.zip`/`.tar.gz` of the parquet + mp4 dataset directory). A single dataset upload creates one
episode per episode in it. Batch upload:

```bash
curl -X POST http://localhost:8000/v1/episodes/upload \
  -H "X-API-Key: ak_..." \
  -F "files=@episode1.rlds.json" \
  -F "files=@episode2.lerobot.json"
```

See [DATA_FORMATS.md](DATA_FORMATS.md) for every accepted format and the export schema, and
`apps/api/aperture/fixtures.py` for the legacy JSON shapes.

## 4. Run the loop

0. **Large datasets** upload straight to object storage via
   `POST /v1/uploads/presign` → `PUT` → `POST /v1/uploads/ingest`, which returns a job id to
   poll at `GET /v1/jobs/{id}`. See [DATA_FORMATS.md](DATA_FORMATS.md).
1. **Classify** each failed episode → `POST /v1/episodes/{id}/classify`
2. **Attribute** → `POST /v1/episodes/{id}/attribution` (attention heatmap + confidence trace + counterfactual)
3. **Cluster** the fleet → `POST /v1/clusters/recompute` (returns `202` + a job id; poll
   `GET /v1/jobs/{id}`, or pass `?wait=true` to block), then `GET /v1/clusters`
4. **Export** a scoped fine-tune dataset → `POST /v1/clusters/{id}/dataset-export`
5. Retrain on your side, then **verify** → `POST /v1/clusters/{id}/verify` with the new batch

The dashboard drives steps 1–5 with buttons; the API supports full programmatic use.

## 5. Deploy to production (optional)

The stack runs fully locally with zero external accounts. To deploy onto the free-tier
production services, supply the credentials below as env vars — no code changes are needed.
See `docs/LOCAL_VS_PRODUCTION.md` for the local ↔ production mapping.

- **Supabase (Postgres + pgvector)** — create a project, run `infra/supabase/migrations/0001_init.sql`
  (creates tables, enables `pgvector`, sets row-level-security isolation), then set
  `APERTURE_DATABASE_URL=postgresql+psycopg2://...` on the API. The app also enforces org
  isolation at the application layer.
- **Cloudflare R2 (object storage)** — create a bucket `aperture-blobs` and an S3 API token, then set
  `APERTURE_R2_ENDPOINT_URL`, `APERTURE_R2_ACCESS_KEY_ID`, `APERTURE_R2_SECRET_ACCESS_KEY`,
  `APERTURE_R2_BUCKET`. Storage auto-switches from the local filesystem to R2 when these are present.
- **Render (backend)** — New → Blueprint pointed at this repo (`infra/render.yaml` is ready); fill the
  `sync: false` env vars in the dashboard; verify `GET /health` returns 200.
- **Vercel (frontend)** — import the repo with root directory `apps/web` (`apps/web/vercel.json` is ready);
  set `APERTURE_API_BASE_URL` (your Render URL) and `APERTURE_API_KEY`. Both are **server-side
  only** — never prefix an API key with `NEXT_PUBLIC_`, which inlines it into the browser bundle
  for every visitor to read.
- **Sentry (optional)** — set `APERTURE_SENTRY_DSN` (API) and `NEXT_PUBLIC_SENTRY_DSN`/`SENTRY_DSN` (web).
  Both stay inert until set.
- **Real attention rollout (GPU)** — open `ml/notebooks/attention_rollout.ipynb` on Colab/Kaggle,
  set `APERTURE_API_BASE`, `APERTURE_API_KEY`, and the `APERTURE_R2_*` vars as notebook secrets. It loads
  open OpenVLA weights and writes real heatmaps to R2. Until then the API returns a deterministic
  simulated heatmap (flagged `simulated: true`) so the pipeline works without a GPU.

CI (`.github/workflows/ci.yml`) runs the backend tests and frontend build on every push automatically.

## Troubleshooting

- **401** — missing/invalid `X-API-Key`.
- **404 on an episode/cluster** — it belongs to a different org (tenant isolation), or the id is wrong.
- **422 on upload** — the file isn't valid RLDS/LeRobot JSON; the error names the file and reason.
- **413 on upload** — a file exceeds 50 MB or the batch exceeds 50 files (configurable).
