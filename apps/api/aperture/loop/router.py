"""Loop-closure API — POST /v1/clusters/{id}/verify.

Accepts a batch of post-retrain episode files (same formats as ingestion), ingests them
through the shared persistence path, classifies each, then computes the before/after delta
scoped to the cluster's failure signature.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import (
    ClusterEpisode,
    Episode,
    FailureCluster,
    Organization,
    VerificationRun,
)
from aperture.evaluation.router import run_classification
from aperture.ingestion.normalize import IngestionError, normalize
from aperture.ingestion.service import persist_episode
from aperture.loop.verify import compute_verification

router = APIRouter(prefix="/v1/clusters", tags=["loop-closure"])


class VerifyOut(BaseModel):
    cluster_id: str
    verification_run_id: str
    pre_success_rate: float
    post_success_rate: float
    delta: float
    pre_n: int
    post_n: int


@router.post("/{cluster_id}/verify", response_model=VerifyOut)
async def verify_cluster(
    cluster_id: str,
    files: list[UploadFile] = File(...),
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> VerifyOut:
    fc = db.get(FailureCluster, cluster_id)
    if fc is None or fc.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cluster not found.")

    settings = get_settings()
    if len(files) > settings.max_batch_files:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Batch too large.")

    # Ingest + classify the post-retrain batch through the same path as Phase 1/2.
    new_batch: list[Episode] = []
    for f in files:
        raw = await f.read()
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"'{f.filename}' too large.")
        try:
            ne = normalize(raw, f.filename or "episode.bin")
        except IngestionError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{f.filename}: {e}") from e
        ep = persist_episode(db, org, ne)
        run_classification(db, ep)
        new_batch.append(ep)

    member_ids = [
        m.episode_id
        for m in db.execute(select(ClusterEpisode).where(ClusterEpisode.cluster_id == fc.id)).scalars()
    ]
    cluster_episodes = db.execute(select(Episode).where(Episode.id.in_(member_ids))).scalars().all()

    result = compute_verification(fc, cluster_episodes, new_batch)

    run = VerificationRun(
        cluster_id=fc.id,
        pre_success_rate=result.pre_success_rate,
        post_success_rate=result.post_success_rate,
        pre_n=result.pre_n,
        post_n=result.post_n,
    )
    db.add(run)
    db.commit()

    return VerifyOut(
        cluster_id=fc.id,
        verification_run_id=run.id,
        pre_success_rate=result.pre_success_rate,
        post_success_rate=result.post_success_rate,
        delta=result.delta,
        pre_n=result.pre_n,
        post_n=result.post_n,
    )
