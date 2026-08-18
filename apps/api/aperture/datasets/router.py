"""Dataset export API — POST /v1/clusters/{id}/dataset-export."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.db import get_db
from aperture.core.models import (
    ClusterEpisode,
    DatasetExport,
    Episode,
    FailureCluster,
    Organization,
)
from aperture.core.storage import get_storage
from aperture.datasets.export import build_export

router = APIRouter(prefix="/v1/clusters", tags=["datasets"])


class ExportRequest(BaseModel):
    format: str = "lerobot"  # rlds|lerobot


class ExportOut(BaseModel):
    export_id: str
    cluster_id: str
    format: str
    episode_count: int
    download_url: str


@router.post("/{cluster_id}/dataset-export", response_model=ExportOut)
def dataset_export(
    cluster_id: str,
    body: ExportRequest,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> ExportOut:
    fc = db.get(FailureCluster, cluster_id)
    if fc is None or fc.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cluster not found.")
    if body.format not in ("rlds", "lerobot"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "format must be 'rlds' or 'lerobot'.")

    member_ids = [
        m.episode_id
        for m in db.execute(select(ClusterEpisode).where(ClusterEpisode.cluster_id == fc.id)).scalars()
    ]
    episodes = db.execute(select(Episode).where(Episode.id.in_(member_ids))).scalars().all()

    payload = build_export(episodes, body.format)
    key = f"{org.slug}/exports/{fc.id}/{body.format}-{fc.episode_count}ep.json"
    storage = get_storage()
    storage.put_bytes(key, payload)
    url = storage.signed_url(key)

    row = DatasetExport(
        cluster_id=fc.id,
        format=body.format,
        export_uri=key,
        episode_count=len(episodes),
    )
    db.add(row)
    db.commit()

    return ExportOut(
        export_id=row.id,
        cluster_id=fc.id,
        format=body.format,
        episode_count=len(episodes),
        download_url=url,
    )
