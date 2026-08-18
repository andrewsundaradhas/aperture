"""Failure-episode embeddings for clustering.

Production embeds each failed episode with a frozen open vision-language encoder (CLIP
ViT-B/32 is fine for MVP) over its key frames and stores the vector in pgvector. That path
needs model weights + a GPU and is wired in ml/notebooks. For a zero-dependency, runs-anywhere
build we compute a deterministic *failure-signature* embedding from the normalized per-frame
signals and the heuristic verdict. It captures the same thing clustering cares about — the
shape of the failure — so episodes with similar failure signatures land together and
dissimilar ones don't, which is exactly the property the verify step checks.

The vector is fixed-length and comparable across episodes; swapping in CLIP means replacing
this one function and widening the pgvector column.
"""

from __future__ import annotations

import numpy as np

from aperture.core.models import Episode

_SURFACES = ["perception", "grounding", "motor"]
EMBED_DIM = 10


def embed_episode(episode: Episode) -> np.ndarray:
    confs = [f.action_confidence for f in episode.frames if f.action_confidence is not None]
    forces = [f.contact_force for f in episode.frames if f.contact_force is not None]
    subgoals = [f.subgoal for f in episode.frames if f.subgoal]

    conf_mean = float(np.mean(confs)) if confs else 0.5
    conf_min = float(np.min(confs)) if confs else 0.5
    conf_std = float(np.std(confs)) if confs else 0.0
    force_max = float(np.max(np.abs(forces))) if forces else 0.0
    force_std = float(np.std(forces)) if forces else 0.0

    distinct_subgoals = len(set(subgoals))
    reissue_ratio = 0.0
    if subgoals:
        blocks = [subgoals[0]]
        for s in subgoals[1:]:
            if s != blocks[-1]:
                blocks.append(s)
        seen: set[str] = set()
        reissues = sum(1 for b in blocks if (b in seen) or seen.add(b))  # count re-entries
        reissue_ratio = reissues / max(1, len(blocks) - 1)

    surface_onehot = [0.0, 0.0, 0.0]
    if episode.classification is not None:
        surface = episode.classification.surface
        if surface in _SURFACES:
            surface_onehot[_SURFACES.index(surface)] = 1.0

    vec = np.array(
        [
            conf_mean,
            conf_min,
            conf_std,
            np.tanh(force_max / 5.0),
            np.tanh(force_std / 5.0),
            np.tanh(distinct_subgoals / 5.0),
            reissue_ratio,
            *surface_onehot,
        ],
        dtype=np.float64,
    )
    assert vec.shape[0] == EMBED_DIM
    return vec
