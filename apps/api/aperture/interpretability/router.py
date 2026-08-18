"""Interpretability Layer API.

  POST /v1/episodes/{id}/attribution   -> enqueue attention-rollout job, run confidence trace
                                          and (for grounding failures) counterfactual probing
  GET  /v1/episodes/{id}/attribution   -> current attribution: job status, heatmap, trace, CF
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Attribution, AttributionJob, Episode, Organization
from aperture.interpretability import rollout
from aperture.interpretability.counterfactual import MockPolicyProbe, probe_episode
from aperture.core.storage import get_storage
from aperture.ml import gateway, runtime

router = APIRouter(prefix="/v1/episodes", tags=["interpretability"])


class AttributionOut(BaseModel):
    episode_id: str
    job_status: str
    attention_map_uri: str | None
    confidence_trace: list[dict]
    counterfactual_result: dict | None


def _load(db: Session, org: Organization, episode_id: str) -> Episode:
    ep = db.get(Episode, episode_id)
    if ep is None or ep.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Episode not found.")
    return ep


def _run_counterfactual(db: Session, ep: Episode) -> dict | None:
    """Run counterfactual probing when the episode is a grounding failure.

    Prefers the real policy as the probe (re-running the same frame image under rephrasings)
    when learned models are enabled and an image is present; otherwise the deterministic
    MockPolicyProbe driven by the episode's grounding evidence.
    """
    cls = ep.classification
    if cls is None or cls.surface != "grounding":
        return None

    if gateway.learned_enabled():
        img = gateway.first_frame_image(ep)
        if img is not None:
            return probe_episode(ep.instruction, rollout.visual_signature(ep), runtime.RealPolicyProbe(img))

    grounding_conf = cls.details.get("grounding", {}).get("confidence", 0.0) if cls.details else 0.0
    probe = MockPolicyProbe(ambiguity=grounding_conf)
    return probe_episode(ep.instruction, rollout.visual_signature(ep), probe)


@router.post("/{episode_id}/attribution", response_model=AttributionOut)
def create_attribution(
    episode_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> AttributionOut:
    ep = _load(db, org, episode_id)
    settings = get_settings()

    # Enqueue a polling job for the GPU worker (Colab notebook consumes it).
    job = AttributionJob(episode_id=ep.id, org_id=org.id, status="queued")
    db.add(job)
    db.flush()

    attribution = ep.attribution or Attribution(episode_id=ep.id)
    if ep.attribution is None:
        db.add(attribution)

    # Attention heatmap. Preference order:
    #   1. Learned models enabled + the episode has frame images -> run real cross-attention
    #      rollout in-process now.
    #   2. No external GPU worker (R2) configured -> synthesize a deterministic heatmap so the
    #      pipeline and dashboard panel work end to end.
    #   3. Otherwise leave the job queued for the external GPU worker (Colab notebook) to fill.
    heatmap: dict | None = None
    if gateway.learned_enabled():
        frame_images = gateway.episode_frame_images(ep)
        if frame_images:
            heatmap = runtime.attention_heatmap(frame_images, ep.instruction)
    if heatmap is None and not (settings.r2_endpoint_url and settings.r2_access_key_id):
        heatmap = rollout.simulate_attention_heatmap(ep)

    if heatmap is not None:
        key = f"{org.slug}/attributions/{ep.id}/attention.json"
        uri = get_storage().put_bytes(key, rollout.heatmap_to_bytes(heatmap))
        job.status = "done"
        job.attention_map_uri = uri
        attribution.attention_map_uri = uri

    attribution.counterfactual_result = _run_counterfactual(db, ep)
    db.commit()

    return AttributionOut(
        episode_id=ep.id,
        job_status=job.status,
        attention_map_uri=attribution.attention_map_uri,
        confidence_trace=rollout.confidence_trace(ep),
        counterfactual_result=attribution.counterfactual_result,
    )


@router.get("/{episode_id}/attribution", response_model=AttributionOut)
def get_attribution(
    episode_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> AttributionOut:
    ep = _load(db, org, episode_id)
    job = db.execute(
        select(AttributionJob)
        .where(AttributionJob.episode_id == ep.id)
        .order_by(AttributionJob.created_at.desc())
    ).scalars().first()
    attribution = ep.attribution
    return AttributionOut(
        episode_id=ep.id,
        job_status=job.status if job else "none",
        attention_map_uri=attribution.attention_map_uri if attribution else None,
        confidence_trace=rollout.confidence_trace(ep),
        counterfactual_result=attribution.counterfactual_result if attribution else None,
    )
