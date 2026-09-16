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

