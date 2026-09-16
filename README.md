# Aperture

The evaluation and interpretability layer for teams running Vision-Language-Action (VLA)
policies on deployed robots. Aperture classifies **why** a policy failed, attributes the
failure with interpretability techniques, clusters similar failures across a fleet,
auto-assembles a scoped fine-tuning dataset, and verifies the fix worked after retraining.

It is **runnable locally with zero paid infrastructure** and is wired so the production
free-tier services (Supabase Postgres, Cloudflare R2, Render, Vercel, Colab GPU) drop in
via env vars.

## Layers (closing a loop)

1. **Ingestion** — accepts real RLDS (`.tfrecord`) and LeRobot v3 (parquet + mp4) datasets, and
   normalizes both into one internal representation. One upload can carry many episodes.
2. **Evaluation** — 3-heuristic classifier buckets each failure into perception /
   grounding / motor.
3. **Interpretability** — attention rollout, action-confidence traces, counterfactual probing.
4. **Clustering + Loop closure** — embedding-space clustering, scoped dataset export, and
   before/after verification after a retrain.

## Repository layout

```
aperture/
├── apps/
│   ├── api/                 # FastAPI backend
│   │   └── aperture/
│   │       ├── ingestion/       # RLDS TFRecord + LeRobot v3 readers, upload endpoints
│   │       ├── evaluation/      # heuristic failure classifier
│   │       ├── interpretability/# attention rollout, confidence trace, counterfactual probe
│   │       ├── clustering/      # embedding-space clustering job
│   │       ├── datasets/        # scoped dataset export service
│   │       ├── loop/            # before/after verification
│   │       ├── jobs/            # background queue + worker (clustering, ingestion)
│   │       ├── ml/              # model defs + inference runtime for the learned path (optional)
│   │       └── core/            # db models, auth, config, storage
│   └── web/                 # Next.js dashboard
├── ml/
│   ├── training/            # reproducible training for policy.pt / failure_head.pt
│   ├── notebooks/           # GPU wrappers around the same code
│   └── models/              # checkpoint provenance + metrics (weights gitignored)
├── infra/                   # render.yaml, vercel.json, supabase migrations
└── docs/
```

## Quick start (backend)

```bash
cd apps/api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m aperture.seed          # loads synthetic fixtures into a local SQLite db
uvicorn aperture.main:app --reload
# → http://localhost:8000/health  and  http://localhost:8000/docs
```

The seed prints the demo API key (`demo-key`); every endpoint takes it as `X-API-Key`.

Slow work — clustering and dataset ingestion — runs on a job queue. The API drains it on a
background thread by default, so the command above is all you need; production runs
`python -m aperture.jobs.worker` (or `make worker`) as its own process instead.

To ingest **real LeRobot v3 datasets** (parquet + mp4) add the `datasets` extra:

```bash
pip install -e ".[dev,datasets]"
```

Real RLDS `.tfrecord` ingestion needs nothing extra — that reader is pure stdlib. See
[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md).

Run the tests:

```bash
cd apps/api && pytest -q
```

## Quick start (dashboard)

```bash
cd apps/web
npm install
npm run dev            # → http://localhost:3000  (talks to the API at :8000)
```

## Optional: the learned path

By default Aperture runs with **zero ML dependencies** — the classifier is three deterministic
heuristics, attention heatmaps are simulated, and clustering uses a failure-signature embedding.
Everything above works in that mode.

A trained `AperturePolicy` + `FailureHead` can take over the classify / attribution / clustering
paths for episodes that carry frame images. Train them yourself — the checkpoints are not
published anywhere:

```bash
cd ml
pip install -e "../apps/api[ml]" pyarrow av
python -m training.train --download --device auto --epochs 10
```

That writes `policy.pt`, `failure_head.pt` and `metrics.json` into `ml/models/`, which is where
the API looks by default. Then:

```bash
cd apps/api && pip install -e ".[ml]"
export APERTURE_USE_LEARNED_MODELS=true
```

If the extra isn't installed or the checkpoints can't load, the flag is a no-op and the app
falls back to the heuristic path — nothing breaks.

## Deploying

`docker compose up --build` brings up the whole stack — API, dashboard, Postgres + pgvector, and
MinIO — entirely inside your own network. See [DEPLOY.md](DEPLOY.md), which also covers the
managed free-tier path and the security checklist.

## Docs

- [docs/PILOT_ONBOARDING.md](docs/PILOT_ONBOARDING.md) — the design-partner flow, end to end.
- [docs/LOCAL_VS_PRODUCTION.md](docs/LOCAL_VS_PRODUCTION.md) — how the local stack maps onto the
  free-tier production services.
- [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) — enabling the trained models over the
  heuristic/simulated default.
- [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) — what Aperture ingests (real RLDS TFRecord and
  LeRobot v3, plus the legacy JSON) and the exact export schema.
- [docs/classifier_eval.md](docs/classifier_eval.md) — how well the 3-heuristic classifier
  actually does, against a majority-class baseline, and where it breaks down.
- [ml/models/README.md](ml/models/README.md) — checkpoint provenance, training data, and the
  honest limits of what the failure head actually learned.
