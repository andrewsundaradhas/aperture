"""Model definitions — the single source of truth for Aperture's learned architecture.

`ml/training/train.py` imports these same classes, so training and inference cannot drift: the
checkpoints (`policy.pt`, `failure_head.pt`) are plain `state_dict`s, and any change to layer
names, shapes, or submodule attribute names here invalidates existing weights. Change the
architecture only alongside an end-to-end retrain of both files.

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
LANG_CACHE_MAX = 4096                    # distinct instructions held in the embedding cache


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

        # Frozen language tower => an instruction's embedding never changes, so re-running
        # MiniLM on it every batch is pure waste. Training especially: a single-task dataset
        # encodes the *same* string 32 times a batch, hundreds of batches an epoch.
        #
        # A plain dict, deliberately not a parameter or buffer, so it stays out of
        # `state_dict()` and existing checkpoints keep loading unchanged.
        self._lang_cache: dict[str, "torch.Tensor"] = {}

    def encode_instructions(self, instruction_texts: list[str], device) -> "torch.Tensor":
        """Embeddings for a batch of instructions, computing each distinct one at most once."""
        # dict.fromkeys dedups while preserving order, so a batch of N identical instructions
        # costs exactly one encode.
        missing = [t for t in dict.fromkeys(instruction_texts) if t not in self._lang_cache]
        if missing:
            with torch.no_grad():
                vectors = self.lang.encode(missing, convert_to_numpy=True)
            if len(self._lang_cache) + len(missing) > LANG_CACHE_MAX:
                self._lang_cache.clear()  # unbounded fleets: drop it all rather than grow forever
            for text, vector in zip(missing, vectors):
                self._lang_cache[text] = torch.from_numpy(vector.copy())
        return torch.stack([self._lang_cache[t] for t in instruction_texts]).to(device)

    def forward(self, image: "torch.Tensor", instruction_texts: list[str]):
        patch_tokens = self.vision.forward_features(image)  # [B, 197, 384]
        lang_emb = self.encode_instructions(instruction_texts, image.device)
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

    Trained by `ml/training/train.py` under *programmatic* supervision — real robot frames with
    each surface realised as the visual condition that distinguishes it (degraded sensor,
    ambiguous referent, or neither). That makes it a working classifier of what a single frame
    can actually show, not a substitute for labels from your own fleet. See the provenance and
    held-out numbers in `ml/models/README.md` before relying on `method="learned"`.
    """

    def __init__(self, d_v: int, n_classes: int = N_SURFACES) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_v, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, fused_features: "torch.Tensor") -> "torch.Tensor":
        return self.net(fused_features)
