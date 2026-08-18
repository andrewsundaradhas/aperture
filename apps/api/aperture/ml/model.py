"""Model definitions — a faithful, byte-compatible reproduction of the architecture trained in
`ml/notebooks/train_aperture.ipynb`.

These class definitions MUST stay in lockstep with the notebook: the published checkpoints
(`policy.pt`, `failure_head.pt`) are plain `state_dict`s, so any change to layer names, shapes,
or submodule attribute names here will break `load_state_dict`. If you retrain with a different
architecture, retrain end to end and re-publish both files together.

Imported lazily (only when the `[ml]` extra is installed) — never import this module at package
import time. See `aperture.ml.runtime`.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Fixed by the trained checkpoints. Do not change without retraining.
VISION_MODEL = "vit_small_patch16_224"   # timm; num_features == 384
LANG_MODEL = "all-MiniLM-L6-v2"          # sentence-transformers; embedding dim == 384
ACTION_DIM = 7                           # padded/truncated action vector
N_SURFACES = 3                           # perception / grounding / motor
PATCH_GRID = 14                          # 224 / 16 -> 14x14 == 196 patch tokens (+1 cls == 197)


class AperturePolicy(nn.Module):
    """Vision-Language-Action policy: image + instruction -> Gaussian over a 7-dim action.

    The instruction embedding is the *query* into a cross-attention over the ViT patch tokens;
    the returned attention weights (language -> image patches) are Aperture's interpretability
    signal, reshaped to the 14x14 patch grid for the attention heatmap.
    """

    def __init__(self, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        # pretrained=False: the checkpoint carries all vision weights, so we avoid re-downloading
        # ImageNet weights at load time. (Training used pretrained=True; the saved state_dict is
        # identical either way once loaded.)
        self.vision = timm.create_model(VISION_MODEL, pretrained=False, num_classes=0)
        self.lang = SentenceTransformer(LANG_MODEL)

        for p_ in self.lang.parameters():
            p_.requires_grad = False

        d_v, d_l = self.vision.num_features, 384
        self.lang_proj = nn.Linear(d_l, d_v)
        self.cross_attn = nn.MultiheadAttention(d_v, num_heads=8, batch_first=True)
        self.action_mean = nn.Sequential(nn.Linear(d_v, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.action_logvar = nn.Sequential(nn.Linear(d_v, 256), nn.ReLU(), nn.Linear(256, action_dim))

    def forward(self, image: "torch.Tensor", instruction_texts: list[str]):
        patch_tokens = self.vision.forward_features(image)  # [B, 197, 384]
        with torch.no_grad():
            lang_emb = torch.tensor(self.lang.encode(instruction_texts)).to(image.device)
        query = self.lang_proj(lang_emb).unsqueeze(1)  # [B, 1, 384]
        fused, attn_weights = self.cross_attn(query, patch_tokens, patch_tokens)
        fused = fused.squeeze(1)  # [B, 384]
        return self.action_mean(fused), self.action_logvar(fused), attn_weights

    def vision_features(self, image: "torch.Tensor") -> "torch.Tensor":
        """Mean-pooled patch tokens — the frozen visual embedding used by the failure head and
        by clustering. Matches the notebook's `patch_tokens.mean(dim=1)`."""
        return self.vision.forward_features(image).mean(dim=1)  # [B, 384]


class FailureHead(nn.Module):
    """3-class failure-surface classifier over the pooled visual embedding.

    NOTE: the published `failure_head.pt` was trained on a placeholder dataset (constant label),
    so it is architecturally real but not yet a meaningful classifier — retrain on labeled
    failure data before trusting `method="learned"` classifications in production. See the
    provenance note in `ml/models/README.md`.
    """

    def __init__(self, d_v: int, n_classes: int = N_SURFACES) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_v, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, fused_features: "torch.Tensor") -> "torch.Tensor":
        return self.net(fused_features)
