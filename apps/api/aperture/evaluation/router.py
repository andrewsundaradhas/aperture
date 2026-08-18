"""Evaluation Layer API — POST /v1/episodes/{id}/classify."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
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
    if learned is not None:
        surface, confidence, details = learned
        method = "learned"
    else:
        frames = [
            NormalizedFrame(
                t=f.t,
                action_confidence=f.action_confidence,
                contact_force=f.contact_force,
                subgoal=f.subgoal,
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

    row = episode.classification
    if row is None:
        row = FailureClassification(episode_id=episode.id)
        db.add(row)
    row.surface = surface
    row.confidence = confidence
    row.method = method
    row.details = details
    db.flush()
    return row


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
