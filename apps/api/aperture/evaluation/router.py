"""Evaluation Layer API — POST /v1/episodes/{id}/classify."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.ratelimit import recompute_limit
from aperture.core.db import get_db
from aperture.core.models import Episode, FailureClassification, Organization
from aperture.evaluation.classifier import classify_frames
from aperture.ingestion.schemas import NormalizedFrame
from aperture.ml import gateway, runtime

router = APIRouter(prefix="/v1/episodes", tags=["evaluation"])


class ClassificationOut(BaseModel):
    episode_id: str
    surface: str
    confidence: float
    method: str
    details: dict


def _learned_classification(episode: Episode) -> tuple[str, float, dict] | None:
    """Try the trained FailureHead over the episode's first frame image. None if unavailable."""
    if not gateway.learned_enabled():
        return None
    img = gateway.first_frame_image(episode)
    if img is None:
        return None
    return runtime.classify_surface(img, episode.instruction)


def run_classification(db: Session, episode: Episode) -> FailureClassification:
    """Classify an episode and upsert its failure_classification row.

    Uses the trained FailureHead when learned models are enabled and the episode has a frame
    image; otherwise the deterministic heuristic classifier over the per-frame signals.
    """
    learned = _learned_classification(episode)
    # The FailureHead's "motor" class is trained on *untouched* frames: it means "nothing is
    # visibly wrong with this image", not "actuation failed". On real fleet footage that is most
    # frames, so trusting it labelled every real failure motor (40/40 on xarm_lift_medium). Only
    # its visual verdicts are evidence; for "motor" we ask the signal channels instead and keep
    # the head's probabilities alongside for the record.
    if learned is not None and learned[0] != "motor":
        surface, confidence, details = learned
        method = "learned"
    else:
        frames = [
            NormalizedFrame(
                t=f.t,
                action_confidence=f.action_confidence,
                contact_force=f.contact_force,
                subgoal=f.subgoal,
                action=f.action,
            )
            for f in episode.frames
        ]
        result = classify_frames(frames)
        surface, confidence, details, method = (
            result.surface,
            result.confidence,
            result.details,
            result.method,
        )
        if learned is not None:
            details = {**details, "_learned": {"surface": learned[0], **learned[2]}}

    # An atomic upsert, not "read, then insert if missing". Uploads queue a background classify
    # job, so a person pressing Classify right after uploading races the worker for the same
    # episode; with read-then-insert both saw no row and the loser 500'd on the unique index.
    # (Invisible locally, where the worker always won; reproduced on Supabase.)
    import uuid

    from sqlalchemy import select

    values = {"surface": surface, "confidence": confidence, "method": method, "details": details}
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    db.execute(
        insert(FailureClassification)
        .values(id=str(uuid.uuid4()), episode_id=episode.id, **values)
        .on_conflict_do_update(index_elements=["episode_id"], set_=values)
    )
    db.expire(episode, ["classification"])
    return db.execute(
        select(FailureClassification).where(FailureClassification.episode_id == episode.id)
    ).scalar_one()


@router.post("/{episode_id}/classify", response_model=ClassificationOut)
def classify_episode(
    episode_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> ClassificationOut:
    ep = db.get(Episode, episode_id)
    if ep is None or ep.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Episode not found.")
    row = run_classification(db, ep)
    db.commit()
    return ClassificationOut(
        episode_id=ep.id,
        surface=row.surface,
        confidence=row.confidence,
        method=row.method,
        details=row.details,
    )


class ClassifyPendingOut(BaseModel):
    job_id: str
    status: str
    poll: str


@router.post(
    "/classify-pending",
    response_model=ClassifyPendingOut,
    status_code=status.HTTP_202_ACCEPTED,
)
def classify_pending(
    reclassify: bool = False,
    org: Organization = Depends(recompute_limit),
    db: Session = Depends(get_db),
) -> ClassifyPendingOut:
    """Classify every failed episode that has no verdict yet, in the background.

    `reclassify=true` redoes all of them — after a classifier upgrade, or after enabling the
    learned models. Poll the returned job; run `POST /v1/clusters/recompute` once it is done.
    """
    from aperture.jobs.queue import enqueue

    job = enqueue(db, org, "classify_pending", {"reclassify": reclassify})
    db.commit()
    return ClassifyPendingOut(job_id=job.id, status=job.status, poll=f"/v1/jobs/{job.id}")
