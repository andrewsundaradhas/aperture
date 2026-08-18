# Model checkpoints

Checkpoint **configs** live here; weights are gitignored (never commit `.bin`/`.safetensors`/`.pt`).

MVP uses open weights, run locally / on free Colab/Kaggle GPU:

- **VLA policy / attention rollout:** `openvla/openvla-7b` (or any HF ViT/VLA with
  `output_attentions=True`). Used by `ml/notebooks/attention_rollout.ipynb`.
- **Clustering encoder:** CLIP `ViT-B/32` — frozen vision-language encoder for failure-episode
  embeddings. The local build substitutes a deterministic failure-signature
  embedding (`apps/api/aperture/clustering/embed.py`) so clustering runs without weights.

No paid inference API is used anywhere in the MVP.
