"""Embedding-space clustering of failed episodes for an org.

Runs HDBSCAN over failure-signature embeddings. HDBSCAN needs no preset cluster count and
labels sparse points as noise (-1), which suits fleet failure data where most episodes are
one-offs and a few recur. Falls back to a small deterministic agglomerative grouping if the
installed scikit-learn predates `sklearn.cluster.HDBSCAN`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.clustering.embed import embed_episode
from aperture.core.models import (
    ClusterEpisode,
    Episode,
    FailureCluster,
    FailureClassification,
    Organization,
)
from aperture.ml import gateway, runtime


def _absorb_noise(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give a noise point to the nearest cluster when it lies within that cluster's own extent.

    HDBSCAN marks the fringe of a single dense group as noise even when those points sit inside
    the group's radius; on a fleet where most failures look alike that left most of them
    unclustered. A point farther out than every member of its nearest cluster stays noise.
    """
    ids = sorted(set(labels.tolist()) - {-1})
    if not ids or (labels != -1).all():
        return labels
    centroids = {k: matrix[labels == k].mean(axis=0) for k in ids}
    radius = {
        k: float(np.linalg.norm(matrix[labels == k] - centroids[k], axis=1).max()) for k in ids
    }
    out = labels.copy()
    for i in np.where(labels == -1)[0]:
        dist = {k: float(np.linalg.norm(matrix[i] - centroids[k])) for k in ids}
        k = min(dist, key=dist.get)
        if dist[k] <= radius[k] + 1e-9:
            out[i] = k
    return out


def _hdbscan_labels(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.full(len(matrix), -1, dtype=int)
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3

        min_size = max(2, int(0.15 * len(matrix)) or 2)
        # allow_single_cluster: without it HDBSCAN refuses to report "these are all one group"
        # and labels the whole set noise — exactly the case of a fleet failing the same way.
        labels = HDBSCAN(
            min_cluster_size=min_size, min_samples=2, allow_single_cluster=True
        ).fit_predict(matrix)
        return _absorb_noise(matrix, labels)
    except Exception:
        from sklearn.cluster import AgglomerativeClustering

        n = max(1, min(len(matrix) - 1, max(2, len(matrix) // 3)))
        return AgglomerativeClustering(n_clusters=n).fit_predict(matrix)


def _embedding_groups(episodes: list[Episode]) -> list[tuple[list[Episode], np.ndarray]]:
    """Split the batch by embedding space; each group is clustered on its own.

    Vectors from different spaces are not comparable (384-dim visual vs 10-dim signature), so
    they cannot share a matrix — but one image-less episode used to force the *whole org* onto
    signatures. Now episodes with a frame image use the learned visual embedding, the rest use
    the failure signature, and a cluster never mixes the two.
    """
    visual: list[tuple[Episode, np.ndarray]] = []
    signature: list[Episode] = []
    use_visual = gateway.learned_enabled()
    for ep in episodes:
        img = gateway.first_frame_image(ep) if use_visual and gateway.episode_has_images(ep) else None
        if img is not None:
            visual.append((ep, np.asarray(runtime.embed_image(img, ep.instruction), dtype=np.float64)))
        else:
            signature.append(ep)
    groups = []
    if visual:
        groups.append(([e for e, _ in visual], np.vstack([v for _, v in visual])))
    if signature:
        groups.append((signature, np.vstack([embed_episode(e) for e in signature])))
    return groups


def recompute_clusters(db: Session, org: Organization) -> list[FailureCluster]:
    """Recluster all of the org's failed, classified episodes. Idempotent: replaces prior clusters."""
    episodes = db.execute(
        select(Episode)
        .join(FailureClassification, FailureClassification.episode_id == Episode.id)
        .where(Episode.org_id == org.id, Episode.outcome == "fail")
    ).scalars().all()

    # Clear existing clusters for the org (recompute is a full refresh).
    for old in db.execute(select(FailureCluster).where(FailureCluster.org_id == org.id)).scalars():
        db.delete(old)
    db.flush()

    if not episodes:
        db.commit()
        return []

    created: list[FailureCluster] = []
    next_label = 0
    for group, matrix in _embedding_groups(episodes):
        labels = _hdbscan_labels(matrix)
        clusters: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            if lab == -1:  # noise / singleton failures are not a fleet pattern
                continue
            clusters.setdefault(int(lab), []).append(idx)

        for _lab, idxs in sorted(clusters.items()):
            lab = next_label
            next_label += 1
            members = [group[i] for i in idxs]
            centroid = matrix[idxs].mean(axis=0)
            surfaces = [m.classification.surface for m in members if m.classification]
            dominant = Counter(surfaces).most_common(1)[0][0] if surfaces else None
            fc = FailureCluster(
                org_id=org.id,
                label=f"{dominant or 'mixed'}-cluster-{lab}",
                dominant_surface=dominant,
                embedding=[round(float(x), 6) for x in centroid.tolist()],
                episode_count=len(members),
            )
            db.add(fc)
            db.flush()
            for m in members:
                db.add(ClusterEpisode(cluster_id=fc.id, episode_id=m.id))
            created.append(fc)

    db.commit()
    created.sort(key=lambda c: c.episode_count, reverse=True)
    return created
