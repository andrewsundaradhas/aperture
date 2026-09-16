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

First produce the checkpoints — they are not published anywhere, so there is nothing to
download (see [`../ml/models/README.md`](../ml/models/README.md)):

```bash
cd ml
pip install -e "../apps/api[ml]" pyarrow av
python -m training.train --download --device auto --epochs 10
```

That writes `policy.pt` / `failure_head.pt` / `metrics.json` into `ml/models/`. Then enable the
path:

```bash
cd apps/api
pip install -e ".[ml]"                 # torch, torchvision, timm, sentence-transformers, hub, pillow
export APERTURE_USE_LEARNED_MODELS=true
# optional:
export APERTURE_MODEL_DEVICE=cuda            # default "cpu"; "mps" on Apple silicon
export APERTURE_LOCAL_MODEL_DIR=/path/to/dir # default: the checkout's ml/models
```

`APERTURE_LOCAL_MODEL_DIR` already defaults to the checkout's `ml/models`, so a local train is
picked up with no further configuration. When a checkpoint is missing there the runtime falls
back to `APERTURE_HF_MODEL_REPO` on the Hugging Face Hub — **there is no public reference repo**,
so point that at one you own if you want hosted weights. If the extra isn't installed or the
checkpoints can't load, `use_learned_models=true` is a no-op and the app quietly uses the
heuristic/simulated path — nothing breaks.

## Feeding images

The learned models need a `3×224×224` RGB frame. Where that comes from depends on the upload
(see [DATA_FORMATS.md](DATA_FORMATS.md)):

- **LeRobot v3 archive:** decoded from the episode's mp4, one image per frame.
- **RLDS `.tfrecord`:** the `steps/observation/image` bytes feature.
- **Legacy RLDS JSON:** `steps[i].observation.image` — base64 (a `data:` URI prefix is tolerated).
- **Legacy LeRobot JSON:** `frames[i]["observation.image"]` — base64.

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

## The external GPU worker

The in-process learned path needs the model to fit on the API host. A 7B VLA (OpenVLA) does not
fit on a free dyno, so that work is handed to a worker on borrowed GPU — the notebooks in
`ml/notebooks/attention_rollout*.ipynb`. `POST /v1/episodes/{id}/attribution` always enqueues an
`AttributionJob`; when the API filled the heatmap itself the job is already `done`, and when it
could not the job stays `queued` for a worker:

| Endpoint | Purpose |
|---|---|
| `GET /v1/attribution_jobs?status=queued` | what is waiting, with each job's `instruction` and `rlds_uri` |
| `POST /v1/attribution_jobs/{id}/claim` | `queued` → `running`; returns 409 if another worker got there first |
| `POST /v1/attribution_jobs/{id}/done` | `running` → `done` with `{"attention_map_uri": "..."}`, and publishes it onto the episode |
| `POST /v1/attribution_jobs/{id}/failed` | `running` → `failed` with `{"error": "..."}` |

Every endpoint is scoped to the API key's organization: a worker holding one org's key can
neither see nor complete another org's jobs. Frame images come from the episode's raw uploaded
blob via `/v1/blobs/{key}` — `GET /v1/episodes/{id}` returns per-frame signals, not pixels.

## Code map

- Model defs: [`apps/api/aperture/ml/model.py`](../apps/api/aperture/ml/model.py)
- Preprocessing: [`apps/api/aperture/ml/preprocess.py`](../apps/api/aperture/ml/preprocess.py)
- Runtime (HF loader, inference, graceful `is_available`): [`apps/api/aperture/ml/runtime.py`](../apps/api/aperture/ml/runtime.py)
- API glue (enable check, frame-image loading): [`apps/api/aperture/ml/gateway.py`](../apps/api/aperture/ml/gateway.py)
- Worker queue: [`apps/api/aperture/interpretability/jobs.py`](../apps/api/aperture/interpretability/jobs.py)
- Training: [`ml/training/train.py`](../ml/training/train.py) (and the GPU wrapper
  [`ml/notebooks/train_aperture.ipynb`](../ml/notebooks/train_aperture.ipynb))

## Caveats (be honest with pilots)

1. **The vision tower is frozen during training, on purpose.** Fine-tuning it on one task's
   behaviour cloning collapsed the general visual representation and dropped failure-head
   accuracy from 0.993 to 0.691 (motor recall 0.982 to 0.354). Aperture's policy is an
   interpretability instrument, not a controller, so keeping general features serves both the
   attention map and the classifier from one tower. See `ml/models/README.md` for the
   measurement.
2. **`failure_head.pt` is trained under programmatic (weak) supervision, not labeled fleet
   failures.** Each surface is realised on real frames as the visual condition that
   distinguishes it — a degraded visual channel, a referentially ambiguous scene, or neither.
   That is a real, held-out-measurable task, but it classifies what a *single frame* can show,
   not your fleet's failure modes. Read `ml/models/metrics.json` (per-class precision/recall,
   confusion matrix, accuracy per corruption variant) rather than the headline accuracy, and
   retrain on your own labels before treating `method="learned"` as ground truth.
3. **One task, one embodiment, simulated.** Training uses `lerobot/xarm_lift_medium` — a single
   sim task. The policy is a working reference for the interpretability path, not a general
   manipulation policy.
4. Preprocessing must never add ImageNet normalization or the checkpoint degrades.
