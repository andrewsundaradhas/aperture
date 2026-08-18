"""Attention rollout + confidence trace.

ATTENTION ROLLOUT METHOD (self-contained description for the patent disclosure — keep this
docstring accurate; it is the precise wording referenced in the invention-disclosure record):

    We compute attention rollout over the vision transformer stack of the VLA policy. For a
    frame, let A_l be the [tokens x tokens] attention matrix of layer l, averaged across
    heads (mean head fusion). We add the residual connection and re-normalize each layer:

        Â_l = 0.5 * A_l + 0.5 * I,   then row-normalize Â_l so each row sums to 1.

    Rollout is the ordered matrix product across all L layers:

        R = Â_L · Â_{L-1} · … · Â_1.

    The attention flowing from the action/CLS query token to each image patch token is the
    corresponding row of R; reshaped to the patch grid (e.g. 14x14 for a ViT-B/16 at 224px)
    and upsampled, it gives the per-frame heatmap over the input image. We emit one heatmap
    per timestep, forming a heatmap *sequence* aligned to the episode's frames.

    This differs from vanilla ViT rollout in that the query is the policy's continuous-action
    read-out token rather than a classification token, and heatmaps are produced per timestep
    over a temporal sequence rather than for a single image — the adaptation to VLA
    (continuous-output, temporally-extended) policies.

Production runs this on a free Colab/Kaggle GPU against an open OpenVLA checkpoint via the
notebook in ml/notebooks/attention_rollout.ipynb, writing heatmaps to R2. Locally — no GPU,
no weights — `simulate_attention_heatmap` produces a deterministic placeholder heatmap
sequence (clearly flagged `simulated: true`) so the full pipeline and dashboard panel work
end to end. The grid is stored as JSON (not a PNG) to avoid an image dependency; the frontend
renders it as a heat grid.
"""

from __future__ import annotations

import hashlib
import json
import math

from aperture.core.models import Episode

GRID = 14  # patch grid side (ViT-B/16 @ 224px)


def confidence_trace(episode: Episode) -> list[dict]:
    """Per-timestep action-confidence trace, reusing the frame signals from ingestion."""
    return [
        {"t": f.t, "action_confidence": f.action_confidence, "contact_force": f.contact_force}
        for f in episode.frames
    ]


def visual_signature(episode: Episode) -> str:
    """Stable per-episode signature standing in for the frozen vision-backbone embedding.

    Used to keep counterfactual probing anchored to 'the same visual input' across rephrasings.
    """
    return hashlib.sha256(episode.id.encode()).hexdigest()[:16]


def simulate_attention_heatmap(episode: Episode, max_frames: int = 8) -> dict:
    """Deterministic placeholder heatmap sequence, concentrated on a plausible region.

    Returns a dict with a `frames` list, each a GRID x GRID row-normalized grid. The focus
    location is derived from the episode id so it is stable but varied across episodes, and it
    drifts across timesteps to look like tracking. Flagged `simulated: true`.
    """
    h = hashlib.sha256(episode.id.encode()).hexdigest()
    cx0 = int(h[0:2], 16) / 255 * (GRID - 1)
    cy0 = int(h[2:4], 16) / 255 * (GRID - 1)
    n = min(max_frames, max(1, len(episode.frames)))

    frames = []
    for k in range(n):
        cx = (cx0 + k * 0.4) % GRID
        cy = (cy0 + k * 0.25) % GRID
        grid = [[0.0] * GRID for _ in range(GRID)]
        total = 0.0
        for y in range(GRID):
            for x in range(GRID):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                v = math.exp(-d2 / (2 * 2.2**2))
                grid[y][x] = v
                total += v
        if total > 0:
            grid = [[round(v / total, 5) for v in row] for row in grid]
        frames.append({"t": k, "grid": grid, "focus": [round(cx, 2), round(cy, 2)]})

    return {"simulated": True, "grid_size": GRID, "frames": frames}


def heatmap_to_bytes(heatmap: dict) -> bytes:
    return json.dumps(heatmap).encode("utf-8")
