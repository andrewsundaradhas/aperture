# ML pipeline — learned models

Aperture runs two ways:

- **Heuristic / simulated (default).** Zero ML dependencies. The failure classifier is three
  deterministic heuristics over per-frame signals, attention heatmaps are simulated, and
  clustering uses a failure-signature embedding. Runs anywhere.
- **Learned.** The trained `AperturePolicy` + `FailureHead` (see
  [`ml/models/README.md`](../ml/models/README.md)) run over episode **frame images**. Enabled
  explicitly; falls back to the heuristic/simulated path per-episode whenever an image is missing
  or the models can't load.

## Enabling the learned path

```bash
cd apps/api
pip install -e ".[ml]"                 # torch, torchvision, timm, sentence-transformers, hub, pillow
export APERTURE_USE_LEARNED_MODELS=true
# optional:
export APERTURE_MODEL_DEVICE=cuda            # default "cpu"
export APERTURE_LOCAL_MODEL_DIR=/path/to/dir # load policy.pt/failure_head.pt from disk (offline)
```

Weights download and cache from `KavinandHobbes/aperture-reference-policy` on first use. If the
extra isn't installed or the checkpoints can't load, `use_learned_models=true` is a no-op and the
app quietly uses the heuristic/simulated path — nothing breaks.

## Feeding images

The learned models need a `3×224×224` RGB frame. Uploads carry it as a base64 image per frame:

- **RLDS JSON:** `steps[i].observation.image` — base64 (a `data:` URI prefix is tolerated).
- **LeRobot JSON:** `frames[i]["observation.image"]` — base64.

Ingestion decodes each image to an object-storage blob and records `episode_frames.image_uri`
(nullable — episodes without images still ingest and classify heuristically). Preprocessing is
resize-to-224 with **no** normalization, matching training exactly.

## What each layer does when learned models are on

| Layer | Endpoint | Learned behavior | Fallback |
|---|---|---|---|
| Evaluation | `POST /v1/episodes/{id}/classify` | `FailureHead` over the first frame image → `method="learned"`, `details.probs` | 3-heuristic classifier |
| Interpretability | `POST /v1/episodes/{id}/attribution` | Real cross-attention (language→patch) heatmap per frame image, `simulated:false` | Simulated heatmap, or external GPU worker if R2 is configured |
| Interpretability | (grounding CF) | `RealPolicyProbe`: re-run the same image under instruction rephrasings, compare quantized action | `MockPolicyProbe` |
| Clustering | `POST /v1/clusters/recompute` | 384-dim pooled visual embeddings (only when **every** failed episode has an image) | Failure-signature embedding |

## Code map

- Model defs: [`apps/api/aperture/ml/model.py`](../apps/api/aperture/ml/model.py)
- Preprocessing: [`apps/api/aperture/ml/preprocess.py`](../apps/api/aperture/ml/preprocess.py)
- Runtime (HF loader, inference, graceful `is_available`): [`apps/api/aperture/ml/runtime.py`](../apps/api/aperture/ml/runtime.py)
- API glue (enable check, frame-image loading): [`apps/api/aperture/ml/gateway.py`](../apps/api/aperture/ml/gateway.py)
- Training: [`ml/notebooks/train_aperture.ipynb`](../ml/notebooks/train_aperture.ipynb)

## Caveats (be honest with pilots)

1. **`failure_head.pt` is trained on placeholder data** (constant label) — retrain on labeled
   failures before trusting learned classifications. The policy (Run A) is real.
2. Preprocessing must never add ImageNet normalization or the checkpoint degrades.
