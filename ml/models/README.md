# Model checkpoints

Checkpoint **configs, provenance and metrics** live here; weights are gitignored (never commit
`.bin`/`.safetensors`/`.pt`). Training writes `policy.pt`, `failure_head.pt` and `metrics.json`
into this directory, which is also where the API looks for them by default.

## Producing them

```bash
cd ml
pip install -e "../apps/api[ml]" pyarrow av
python -m training.train --download --device auto --epochs 10
```

Run A checkpoints every epoch into `checkpoints/` and resumes from the newest, so an interrupted
run costs at most one epoch. [`../notebooks/train_aperture.ipynb`](../notebooks/train_aperture.ipynb)
is the same thing with a GPU runtime and plots around it.

Training autocasts to **bfloat16** on a GPU (`--precision`), which measured ~1.6x faster than
fp32 on MPS. Parameters stay fp32, so checkpoints are byte-identical in format and load into the
fp32 inference path unchanged. bfloat16 rather than float16 because it carries fp32's exponent
range, so gradients cannot underflow and no gradient scaler is needed.

| File | Class | Size | What it is |
|---|---|---|---|
| `policy.pt` | `AperturePolicy` | ~181 MB | ViT-Small vision + frozen MiniLM language, cross-attention fusion, Gaussian action head (mean + logvar over a 7-dim action) |
| `failure_head.pt` | `FailureHead` | ~0.4 MB | 3-class MLP (perception / grounding / motor) over the pooled visual embedding |
| `metrics.json` | — | small | both runs' loss history and Run B's held-out report; committed |

The class definitions live in
[`apps/api/aperture/ml/model.py`](../../apps/api/aperture/ml/model.py) and training **imports
them from there**. Lockstep between training and inference is structural, not a convention
anyone has to remember.

### Architecture (as trained)

- **Vision:** `timm` `vit_small_patch16_224` (`num_features = 384`), 14×14 = 196 patch tokens (+1 cls).
  ImageNet weights are the initialisation; the tower is fine-tuned in Run A.
- **Language:** `sentence-transformers` `all-MiniLM-L6-v2` (384-dim), **frozen**.
- **Fusion:** the projected instruction embedding is the *query* into an 8-head
  `MultiheadAttention` over the patch tokens. The returned attention weights (language → patches)
  are the interpretability signal, reshaped to the 14×14 grid for the attention heatmap.
- **Action head:** two MLPs → `action_mean` and `action_logvar`; trained with a Gaussian NLL.
- **Preprocessing:** images are only `Resize((224,224))`d — **no ImageNet normalization**.
  Inference must match exactly (see `apps/api/aperture/ml/preprocess.py`); adding the usual
  normalization silently shifts the input distribution and degrades every prediction.

## Training data

Both runs use **`lerobot/xarm_lift_medium`** (LeRobot v3.0 / Open X-Embodiment): 800 episodes,
20,000 frames at 15fps, one task — *"Pick up the cube and lift it."* — with 4-dim actions padded
to the policy's 7. It is ~18 MB, public and ungated, so the whole pipeline is reproducible from
a clean checkout.

Train/val is split **by episode** (720/80). Neighbouring frames in a 15fps episode are
near-duplicates; splitting by frame would leak them across the split and inflate every number
below.

### Run A — policy

Behaviour cloning: `(frame, instruction) → action`, Gaussian NLL. Reported as train and val NLL
per epoch in `metrics.json`.

**The vision tower is frozen** (`--freeze-vision`, on by default); only the language projection,
the cross-attention fusion and the action heads train. This is not a shortcut — it is the single
largest quality decision in the pipeline, and it was measured:

| Failure head trained on | Accuracy | motor recall |
|---|---|---|
| ViT fine-tuned by Run A's behaviour cloning | 0.691 | 0.354 |
| **Frozen ImageNet ViT** | **0.993** | **0.982** |

Identical data, identical head, identical schedule — the tower is the only variable. Fine-tuning
a ViT-Small on one task's behaviour cloning, across 20k frames of a single scene, collapses the
general visual representation: the features get good at predicting this task's actions and lose
the detail that separates a blurred frame from a clean one. The failure head reads the same
tower, so it inherited the damage — motor recall fell to 0.354, barely above chance.

Freezing is the right trade because **Aperture's policy is an interpretability instrument, not a
controller.** What the product needs from it is the language→patch attention map and a
counterfactual probe; nobody deploys `policy.pt` to drive a robot. Keeping a general
representation serves both consumers from one tower, with no second encoder at inference.

Use `--no-freeze-vision` if you are optimising the policy for control rather than attribution.

### Run B — failure head

`FailureHead` has to name the failure *surface* from one frame's pooled visual embedding. No
public robotics dataset carries that label, so the labels are constructed —
[`training/failure_surfaces.py`](../training/failure_surfaces.py) realises each surface as the
visual condition that actually distinguishes it, on real frames:

| Surface | How the sample is built | Signal |
|---|---|---|
| **perception** | real frame, visual channel degraded: blur, sensor noise, under/over-exposure, occlusion, colour loss | the scene is intact; the *image* of it is not |
| **grounding** | real frame, image quality untouched, the cube cloned into 2–3 other plausible table positions (identical or re-hued) | several equally plausible referents for "the cube" |
| **motor** | real frame, untouched, sampled at timesteps where the **logged action trace** is anomalous (high command jerk / saturated commands) | nothing is wrong with the image; actuation is what failed |

The vision tower is frozen for Run B, so features are extracted once and the head then trains in
seconds.

## Measured results

From the committed `metrics.json` (10 epochs, bfloat16, frozen tower, 4000 frames per surface,
episode-level split):

| | |
|---|---|
| Run A best validation NLL | **−2.267** |
| Run B held-out accuracy | **0.997** |
| perception | P 0.996 / R 0.998 |
| grounding | P 0.998 / R 0.998 |
| motor | P 0.996 / R 0.994 |

**Do not quote 0.997 on its own.** It is measured on the same six corruption types the head
trained against, so it cannot separate "understands the visual channel failed" from "memorised
these six corruptions". The leave-one-variant-out check answers that — each perception variant
is held out of training entirely, and the head is scored only on it:

| Held-out corruption | Perception recall |
|---|---|
| overexposure | 1.000 |
| underexposure | 0.996 |
| sensor_noise | 0.871 |
| blur | 0.816 |
| **occlusion** | **0.180** |
| **colour_loss** | **0.129** |
| mean | **0.665** |

That spread is the real finding. The head generalises confidently to unseen corruptions that
shift *global image statistics* — exposure, noise, blur all change the whole frame's
distribution — and largely fails on the two that do not. An occlusion is a local patch in an
otherwise normal frame; colour loss preserves structure and moves only saturation. So what the
head has actually learned is closer to "this frame's global statistics are abnormal" than to a
general concept of a failed visual channel.

Expect roughly **0.997 on corruption types you have trained on, and ~0.67 on ones you have
not.** Retraining with your own failure modes represented is what closes that gap.

### The attention map is instruction-sensitive, less so frame-sensitive

Freezing the tower puts the whole interpretability burden on the cross-attention fusion, so the
map was checked for degeneracy. It is not degenerate: the peak is ~22× uniform, and rephrasing
the instruction moves it substantially (L1 0.94 between the real task and "Do nothing at all").
But across frames *within this one scene* the peak barely moves — the layout is nearly constant,
and the peak only relocates for genuinely different imagery (blur, noise, a black frame). Read
the heatmap as "what the instruction attends to in this kind of scene", not as a precise
per-frame object localiser. One task's worth of data is the limit here, not the method.

## Honest limits

1. **Run B is programmatic (weak) supervision, not labeled fleet failures.** It is a real,
   held-out-measurable task and a working classifier — but of what a *single frame* can show:
   whether the visual channel is at fault, whether the scene is referentially ambiguous, or
   neither. Retrain on your own labeled failures before treating `method="learned"` as ground
   truth for a pilot. Read the generalization table above, not the headline accuracy.
2. **One task, one embodiment, simulated.** `xarm_lift_medium` is a single sim task. The policy
   is a working reference for the interpretability path, not a general manipulation policy.
3. **The motor class is defined by the action trace, not the image.** A motor sample is an
   untouched frame at an anomalous actuation timestep, so the head can only get it right by
   ruling the other two out. It scores 0.994 here because perception and grounding are visually
   obvious; on real footage where they are subtler, expect motor to degrade first.
4. **Preprocessing must never add ImageNet normalization** or the checkpoint degrades.

## How the API uses them

Off by default. Enable with the `[ml]` extra and a flag (see
[`docs/ML_PIPELINE.md`](../../docs/ML_PIPELINE.md)):

```bash
cd apps/api && pip install -e ".[ml]"
export APERTURE_USE_LEARNED_MODELS=true      # Windows: setx APERTURE_USE_LEARNED_MODELS true
```

`APERTURE_LOCAL_MODEL_DIR` defaults to this directory, so a local train is picked up with no
further configuration. When a checkpoint is missing here the runtime falls back to
`APERTURE_HF_MODEL_REPO` on the Hugging Face Hub — there is **no public reference repo**, so
point that at one you own if you want hosted weights.

When enabled and an episode carries frame images, the classify / attribution / clustering paths
use these models; otherwise they fall back to the deterministic heuristic/simulated path. No paid
inference API is used anywhere.
