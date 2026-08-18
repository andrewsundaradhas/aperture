# Aperture

The evaluation and interpretability layer for teams running Vision-Language-Action (VLA)
policies on deployed robots. Aperture classifies **why** a policy failed, attributes the
failure with interpretability techniques, clusters similar failures across a fleet,
auto-assembles a scoped fine-tuning dataset, and verifies the fix worked after retraining.

It is **runnable locally with zero paid infrastructure** and is wired so the production
free-tier services (Supabase Postgres, Cloudflare R2, Render, Vercel, Colab GPU) drop in
via env vars.

## Layers (closing a loop)

1. **Ingestion** — accepts episode logs in RLDS / LeRobot format, normalizes both into one
   internal representation.
2. **Evaluation** — 3-heuristic classifier buckets each failure into perception /
   grounding / motor.
3. **Interpretability** — attention rollout, action-confidence traces, counterfactual probing.
4. **Clustering + Loop closure** — embedding-space clustering, scoped dataset export, and
   before/after verification after a retrain.

## Repository layout

```
aperture-platform/
├── apps/
│   ├── api/                 # FastAPI backend
│   │   └── aperture/
│   │       ├── ingestion/       # RLDS/LeRobot parsers, upload endpoints
│   │       ├── evaluation/      # heuristic failure classifier
│   │       ├── interpretability/# attention rollout, confidence trace, counterfactual probe
│   │       ├── clustering/      # embedding-space clustering job
│   │       ├── datasets/        # scoped dataset export service
│   │       ├── loop/            # before/after verification
│   │       ├── core/            # db models, auth, config, storage
│   │       └── main.py
│   └── web/                 # Next.js dashboard
├── ml/
│   └── notebooks/           # Colab-run training/eval notebooks
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

Run the tests:

```bash
cd apps/api && pytest -q
```

## Quick start (dashboard)

```bash
cd apps/web
npm install
npm run dev            # → http://localhost:3000  (proxies to the API at :8000)
```

See [docs/PILOT_ONBOARDING.md](docs/PILOT_ONBOARDING.md) for the design-partner flow and
[docs/LOCAL_VS_PRODUCTION.md](docs/LOCAL_VS_PRODUCTION.md) for how the local stack maps onto
the free-tier production services.
