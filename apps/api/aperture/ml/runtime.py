"""Inference runtime for the learned models — the bridge the API layers call.

Contract with the rest of the app:
  * Importing this module never imports torch/timm (so the base install stays light).
  * `is_available()` returns False (never raises) when the `[ml]` extra is missing or the
    checkpoints cannot be loaded, so every caller can fall back to the simulated/heuristic path.
  * The models are loaded once, lazily, and cached — the first call pays the download/load cost.

Weights are resolved from `APERTURE_LOCAL_MODEL_DIR` if set (offline), otherwise downloaded and
cached from the Hugging Face repo in settings (`KavinandHobbes/aperture-reference-policy`).
""" 

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Class-index order the failure head was trained with (notebook: surface_to_idx).
SURFACES = ["perception", "grounding", "motor"]
GRID = 14  # 14x14 == 196 patch tokens for vit_small_patch16_224 @ 224px


@functools.lru_cache(maxsize=1)
def _deps_importable() -> bool:
    """True iff the optional `[ml]` extra is installed."""
    try:
        import huggingface_hub  # noqa: F401
        import sentence_transformers  # noqa: F401
        import timm  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        from PIL import Image  # noqa: F401

        return True
    except Exception:  # pragma: no cover - import guard
        return False


class _Models:
    def __init__(self, policy, failure_head, device: str) -> None:
        self.policy = policy
        self.failure_head = failure_head
        self.device = device


def _resolve_weight(repo: str, filename: str, local_dir: str | None) -> str:
    if local_dir:
        p = Path(local_dir) / filename
        if p.exists():
            return str(p)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


@functools.lru_cache(maxsize=1)
def _load_models() -> _Models:
    """Load policy + failure head once. Raises on failure; callers go through `is_available()`."""
    import torch

    from aperture.core.config import get_settings
    from aperture.ml.model import AperturePolicy, FailureHead

    s = get_settings()
    device = s.model_device

    policy_path = _resolve_weight(s.hf_model_repo, s.hf_policy_file, s.local_model_dir)
    head_path = _resolve_weight(s.hf_model_repo, s.hf_failure_head_file, s.local_model_dir)

    policy = AperturePolicy().to(device).eval()
    policy.load_state_dict(torch.load(policy_path, map_location=device))

    failure_head = FailureHead(policy.vision.num_features).to(device).eval()
    failure_head.load_state_dict(torch.load(head_path, map_location=device))

    logger.info("Loaded learned models on %s (policy=%s, head=%s)", device, policy_path, head_path)
    return _Models(policy, failure_head, device)


def is_available() -> bool:
    """True iff learned inference can actually run (deps present + checkpoints loadable)."""
    if not _deps_importable():
        return False
    try:
        _load_models()
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Learned models unavailable, falling back to simulated path: %s", e)
        return False


# --- Inference entry points ---------------------------------------------------


def predict_action(image_bytes: bytes, instruction: str | None) -> tuple[list[float], list[float]]:
    """Return (action_mean, action_logvar), each a length-7 list."""
    import torch

    m = _load_models()
    from aperture.ml.preprocess import image_bytes_to_tensor

    x = image_bytes_to_tensor(image_bytes, m.device)
    with torch.no_grad():
        mean, logvar, _ = m.policy(x, [instruction or ""])
    return mean.squeeze(0).cpu().tolist(), logvar.squeeze(0).cpu().tolist()


def classify_surface(image_bytes: bytes, instruction: str | None) -> tuple[str, float, dict]:
    """Learned failure-surface classification: (surface, confidence, details)."""
    import torch

    m = _load_models()
    from aperture.ml.preprocess import image_bytes_to_tensor

    x = image_bytes_to_tensor(image_bytes, m.device)
    with torch.no_grad():
        feats = m.policy.vision_features(x)
        probs = torch.softmax(m.failure_head(feats), dim=-1).squeeze(0)
    idx = int(probs.argmax().item())
    details = {s: round(float(probs[i].item()), 4) for i, s in enumerate(SURFACES)}
    return SURFACES[idx], float(probs[idx].item()), {"probs": details}


def attention_heatmap(frames: list[tuple[int, bytes]], instruction: str | None) -> dict:
    """Real attention heatmap sequence from the policy's language->patch cross-attention.

    `frames` is a list of (timestep, image_bytes). One GRIDxGRID row-normalized grid per frame,
    same JSON shape as the simulated path but flagged `simulated: False`.
    """
    import torch

    m = _load_models()
    from aperture.ml.preprocess import image_bytes_to_tensor

    out_frames: list[dict] = []
    for t, data in frames:
        x = image_bytes_to_tensor(data, m.device)
        with torch.no_grad():
            _, _, attn = m.policy(x, [instruction or ""])  # [1, 1, n_tokens]
        weights = attn.squeeze(0).squeeze(0)  # [n_tokens]
        # Drop prefix (cls) tokens, keep the GRID*GRID patch tokens.
        prefix = max(0, weights.shape[0] - GRID * GRID)
        patch = weights[prefix : prefix + GRID * GRID]
        grid_t = patch.reshape(GRID, GRID)
        total = float(grid_t.sum())
        if total > 0:
            grid_t = grid_t / total
        grid = [[round(float(v), 5) for v in row] for row in grid_t.tolist()]
        fy, fx = divmod(int(patch.argmax().item()), GRID)
        out_frames.append({"t": t, "grid": grid, "focus": [float(fx), float(fy)]})

    return {"simulated": False, "method": "cross_attention", "grid_size": GRID, "frames": out_frames}


def embed_image(image_bytes: bytes, instruction: str | None = None):
    """Return the 384-dim pooled visual embedding (numpy float64) for clustering."""
    import numpy as np
    import torch

    m = _load_models()
    from aperture.ml.preprocess import image_bytes_to_tensor

    x = image_bytes_to_tensor(image_bytes, m.device)
    with torch.no_grad():
        feats = m.policy.vision_features(x).squeeze(0).cpu().numpy()
    return feats.astype(np.float64)


class RealPolicyProbe:
    """`PolicyProbe` backed by the real policy: re-runs the *same image* with a rephrased
    instruction and quantizes the predicted action into a stable target token. Trivial
    rephrasings that leave the action essentially unchanged map to the same token; ones that
    move the action map elsewhere — the grounding-sensitivity signal counterfactual probing
    is looking for.
    """

    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes

    def predicted_target(self, instruction: str, visual_signature: str = "") -> str:
        mean, _ = predict_action(self.image_bytes, instruction)
        # Coarse quantization (0.5 units) so numerically tiny action jitter is not treated as a
        # target change; larger shifts move the bucket and flip the token.
        bucket = tuple(int(round(v * 2)) for v in mean)
        h = hashlib.sha256(str(bucket).encode()).hexdigest()[:6]
        return f"act_{h}"
