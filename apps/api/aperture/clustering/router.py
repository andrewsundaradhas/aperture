"""Clustering Layer API — GET /v1/clusters, POST /v1/clusters/recompute, GET /v1/clusters/{id}."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.clustering.cluster import recompute_clusters
from aperture.core.auth import resolve_org
from aperture.core.db import get_db
from aperture.core.models import ClusterEpisode, Episode, FailureCluster, Organization

router = APIRouter(prefix="/v1/clusters", tags=["clustering"])


class ClusterOut(BaseModel):
    id: str
    label: str
    dominant_surface: str | None
    episode_count: int
    representative_episode_id: str | None = None


class ClusterEpisodeOut(BaseModel):
    id: str
    robot_id: str
    outcome: str
    surface: str | None
    instruction: str | None


class ClusterDetailOut(ClusterOut):
    episodes: list[ClusterEpisodeOut]


def _load_cluster(db: Session, org: Organization, cluster_id: str) -> FailureCluster:
    fc = db.get(FailureCluster, cluster_id)
    if fc is None or fc.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cluster not found.")
    return fc


def _representative(db: Session, cluster_id: str) -> str | None:
    """First member episode — stands in for the 'representative episode thumbnail' (Phase 4).

    Real thumbnails need decoded video frames, which arrive with a design partner's actual
    fleet data; the episode id links to that episode's attention heatmap in the meantime.
    """
    row = db.execute(
        select(ClusterEpisode.episode_id).where(ClusterEpisode.cluster_id == cluster_id).limit(1)
    ).scalar_one_or_none()
    return row


def _to_out(db: Session, c: FailureCluster) -> "ClusterOut":
    return ClusterOut(
        id=c.id,
        label=c.label,
        dominant_surface=c.dominant_surface,
        episode_count=c.episode_count,
        representative_episode_id=_representative(db, c.id),
    )


@router.post("/recompute", response_model=list[ClusterOut])
def recompute(
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[ClusterOut]:
    clusters = recompute_clusters(db, org)
    return [_to_out(db, c) for c in clusters]


@router.get("", response_model=list[ClusterOut])
def list_clusters(
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[ClusterOut]:
    clusters = db.execute(
        select(FailureCluster)
        .where(FailureCluster.org_id == org.id)
        .order_by(FailureCluster.episode_count.desc())
    ).scalars().all()
    return [_to_out(db, c) for c in clusters]


@router.get("/{cluster_id}", response_model=ClusterDetailOut)
def get_cluster(
    cluster_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> ClusterDetailOut:
    fc = _load_cluster(db, org, cluster_id)
    member_ids = [
        m.episode_id
        for m in db.execute(
            select(ClusterEpisode).where(ClusterEpisode.cluster_id == fc.id)
        ).scalars()
    ]
    episodes = db.execute(select(Episode).where(Episode.id.in_(member_ids))).scalars().all()
    return ClusterDetailOut(
        id=fc.id,
        label=fc.label,
        dominant_surface=fc.dominant_surface,
        episode_count=fc.episode_count,
        episodes=[
            ClusterEpisodeOut(
                id=ep.id,
                robot_id=ep.robot_id,
                outcome=ep.outcome,
                surface=ep.classification.surface if ep.classification else None,
                instruction=ep.instruction,
            )
            for ep in episodes
        ],
    )
