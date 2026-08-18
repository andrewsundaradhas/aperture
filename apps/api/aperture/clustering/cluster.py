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


def _hdbscan_labels(matrix: np.ndarray) -> np.ndarray:
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3

        min_size = max(2, int(0.15 * len(matrix)) or 2)
        return HDBSCAN(min_cluster_size=min_size).fit_predict(matrix)
    except Exception:
        from sklearn.cluster import AgglomerativeClustering

        n = max(1, min(len(matrix) - 1, max(2, len(matrix) // 3)))
        if len(matrix) < 2:
            return np.zeros(len(matrix), dtype=int)
        return AgglomerativeClustering(n_clusters=n).fit_predict(matrix)


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

    matrix = np.vstack([embed_episode(ep) for ep in episodes])
    labels = _hdbscan_labels(matrix)

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        if lab == -1:  # noise / singleton failures are not a fleet pattern
            continue
        clusters.setdefault(int(lab), []).append(idx)

    created: list[FailureCluster] = []
    for lab, idxs in sorted(clusters.items()):
        members = [episodes[i] for i in idxs]
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
