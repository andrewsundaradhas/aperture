"""Learned VLA models for Aperture.

This package hosts the *real* trained models that were produced in the Colab training
notebook (`ml/notebooks/train_aperture.ipynb`) and published to the Hugging Face repo
`KavinandHobbes/aperture-reference-policy`:

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
