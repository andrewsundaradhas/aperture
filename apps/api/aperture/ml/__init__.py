"""Learned VLA models for Aperture.

This package holds the model definitions and the inference runtime for the two checkpoints
produced by `ml/training/train.py` (see `ml/models/README.md` for provenance and held-out
metrics). The checkpoints are not published anywhere — you train them yourself, and the
runtime resolves them from `ml/models/` by default:

  * `AperturePolicy`   -> policy.pt        (ViT-Small vision + frozen MiniLM language,
                                            cross-attention fusion, Gaussian action head)
  * `FailureHead`      -> failure_head.pt  (3-class perception/grounding/motor classifier)

Everything here depends on heavy ML libraries (torch, timm, sentence-transformers) that are
installed only via the optional `[ml]` extra. Importing this subpackage must therefore never
pull those libraries at import time — the model classes live in `model.py` and are imported
lazily by `runtime.py`, which also degrades gracefully (`is_available()` -> False) when the
extra is not installed. That keeps the base API runnable anywhere with zero ML dependencies.
"""

from __future__ import annotations

from aperture.ml.runtime import (
    RealPolicyProbe,
    attention_heatmap,
    classify_surface,
    embed_image,
    is_available,
    predict_action,
)

__all__ = [
    "RealPolicyProbe",
    "attention_heatmap",
    "classify_surface",
    "embed_image",
    "is_available",
    "predict_action",
]
