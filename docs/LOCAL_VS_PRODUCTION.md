# Local build ↔ Free-tier production

This repo runs fully locally with zero paid infra, and swaps onto the free-tier production
services by setting env vars — no code changes.

| Concern | Local default | Production (free tier) | How to switch |
|---|---|---|---|
| Backend API | uvicorn on localhost | FastAPI on **Render** free web service | `infra/render.yaml` |
| Frontend | `next dev` on :3000 | Next.js on **Vercel** Hobby | `infra/vercel.json` |
| Database | file-backed **SQLite** | **Supabase** Postgres + RLS | set `APERTURE_DATABASE_URL` |
| Vector store | JSON embedding column + sklearn HDBSCAN | **pgvector** in Supabase | `infra/supabase/migrations/0001_init.sql` |
| Object storage | filesystem `.aperture_data/blobs` | **Cloudflare R2** (S3-compatible) | set `APERTURE_R2_*` |
| ML compute | simulated heatmaps, deterministic embeddings | **Colab/Kaggle** free GPU | `ml/notebooks/attention_rollout.ipynb` |
| VLA / vision weights | none needed | open **OpenVLA** / CLIP checkpoints | `ml/models/README.md` |
| Auth | static API-key map (`APERTURE_API_KEYS`) | **Supabase Auth** + per-org API keys | JWT `org_id` claim drives RLS |
| Error tracking | off | **Sentry** free tier | set `APERTURE_SENTRY_DSN` |

## What is real vs. substituted locally

**Real, identical to production:** ingestion + RLDS/LeRobot normalization, the 3-heuristic
classifier and its combination rule, clustering (HDBSCAN over embeddings), scoped dataset
export, loop-closure verification math, tenant isolation logic, all API contracts.

**Substituted locally by default, with a real learned path available** (enable with the `[ml]`
extra + `APERTURE_USE_LEARNED_MODELS=true` — see [ML_PIPELINE.md](ML_PIPELINE.md)):
- **Failure classifier** — the 3 heuristics run by default; the trained `FailureHead` runs when
  enabled and the episode has a frame image (`method="learned"`).
- **Attention rollout** — `interpretability/rollout.py` emits a deterministic simulated heatmap
  flagged `simulated: true`; the learned path computes a real language→patch cross-attention
  heatmap in-process, and the Colab notebook path writes OpenVLA rollouts to R2.
- **Clustering embedding** — `clustering/embed.py` computes a failure-signature vector by default;
  the learned path uses 384-dim pooled visual embeddings when every episode has an image.
- **Counterfactual policy** — `interpretability/counterfactual.py` uses `MockPolicyProbe`; the
  learned path plugs the trained policy behind the same `PolicyProbe` interface (still no paid LLM
  calls — it probes the policy, not Claude/GPT).

## Upgrade triggers

Move off free tiers only when real data volume forces it: Render→Railway/Fly for always-on,
Supabase Pro past ~500 MB, R2 pay-as-you-go past 10 GB, managed vector DB past ~1M embeddings,
Modal/Lambda GPU once jobs must run unattended.
