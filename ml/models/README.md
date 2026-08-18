# Model checkpoints

Checkpoint **configs and provenance** live here; weights are gitignored (never commit
`.bin`/`.safetensors`/`.pt`).

## Trained reference models

Aperture ships a small trained VLA policy + failure classifier, produced by
[`ml/notebooks/train_aperture.ipynb`](../notebooks/train_aperture.ipynb) on a free Colab GPU and
published to the Hugging Face repo **`KavinandHobbes/aperture-reference-policy`**:

| File | Class | Size | What it is |
|---|---|---|---|
| `policy.pt` | `AperturePolicy` | ~181 MB | ViT-Small vision + frozen MiniLM language, cross-attention fusion, Gaussian action head (mean + logvar over a 7-dim action) |
| `failure_head.pt` | `FailureHead` | ~0.4 MB | 3-class MLP (perception / grounding / motor) over the pooled visual embedding |

The Python definitions that these `state_dict`s load into live in
[`apps/api/aperture/ml/model.py`](../../apps/api/aperture/ml/model.py) and **must** stay in
lockstep with the notebook.

### Architecture (as trained)

- **Vision:** `timm` `vit_small_patch16_224` (`num_features = 384`), 14×14 = 196 patch tokens (+1 cls).
- **Language:** `sentence-transformers` `all-MiniLM-L6-v2` (384-dim), **frozen**.
- **Fusion:** the projected instruction embedding is the *query* into an 8-head
  `MultiheadAttention` over the patch tokens. The returned attention weights (language → patches)
  are the interpretability signal, reshaped to the 14×14 grid for the attention heatmap.
- **Action head:** two MLPs → `action_mean` and `action_logvar`; trained with a Gaussian NLL.
- **Preprocessing:** images were only `Resize((224,224))`d — **no ImageNet normalization**. Inference
  must match exactly (see `apps/api/aperture/ml/preprocess.py`).

### Training data

- **Run A (policy):** `lerobot/xarm_lift_medium` (LeRobot / Open X-Embodiment), 10 epochs. Actions
  padded/truncated to 7 dims.
- **Run B (failure head):** RoboMIND (`x-humanoid-robomind/RoboMIND`, `h5_franka_1rgb`) was the
  intended source, but the notebook's `RoboMINDDataset` currently yields a **placeholder** sample
  (random image, constant `"perception"` label). ⚠️ **`failure_head.pt` is therefore
  architecturally valid but not yet a meaningful classifier** — replace the dataset with real
  labeled failures and retrain before relying on `method="learned"` classifications.

## How the API uses them

Off by default. Enable with the `[ml]` extra and a flag (see
[`docs/ML_PIPELINE.md`](../../docs/ML_PIPELINE.md)):

```bash
cd apps/api && pip install -e ".[ml]"
export APERTURE_USE_LEARNED_MODELS=true       # Windows: setx APERTURE_USE_LEARNED_MODELS true
```

When enabled and an episode carries frame images, the classify / attribution / clustering paths
use these models; otherwise they fall back to the deterministic heuristic/simulated path. Weights
download and cache from Hugging Face on first use, or set `APERTURE_LOCAL_MODEL_DIR` to load
`policy.pt` / `failure_head.pt` from disk (offline).

No paid inference API is used anywhere.
