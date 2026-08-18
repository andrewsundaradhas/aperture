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
   before/after verification after a retrain.git add README.md