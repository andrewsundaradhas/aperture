"""Attribution-job queue API — the contract between Aperture and an external GPU worker.

`POST /v1/episodes/{id}/attribution` enqueues an `AttributionJob`. When the API host can run
the learned models in-process it fills the heatmap immediately and the job is already `done`.
When it can't — the interesting case, a 7B VLA like OpenVLA that will not fit on a free API
dyno — the job stays `queued` and a worker on borrowed GPU (the Colab/Kaggle notebooks in
`ml/notebooks/`) picks it up through these endpoints:

    GET  /v1/attribution_jobs?status=queued   what is waiting for me
    POST /v1/attribution_jobs/{id}/claim      queued -> running, so two workers don't duplicate
    POST /v1/attribution_jobs/{id}/done       -> done, with the heatmap's storage uri
         /v1/attribution_jobs/{id}/failed     -> failed, with the reason

`done` and `failed` accept a job that is `queued` or `running` — a lone worker that skips the
claim step still works — and reject one that has already reached a terminal state.

Every query is scoped to the caller's organization, exactly like every other layer: a worker
holding one org's API key can neither see nor complete another org's jobs.

Claiming is a read-then-write under one transaction and one row lock, which is sufficient for
the intended deployment (a handful of notebook workers against Postgres or SQLite). It is not
a general work queue — at real worker counts this belongs in `SELECT ... FOR UPDATE SKIP
LOCKED` or a purpose-built broker.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.db import get_db
from aperture.core.models import Attribution, AttributionJob, Episode, Organization
from aperture.core.pagination import Page, page_params

router = APIRouter(prefix="/v1/attribution_jobs", tags=["interpretability"])

_TERMINAL = {"done", "failed"}


class JobOut(BaseModel):
    id: str
    episode_id: str
    status: str
    attention_map_uri: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    # Denormalized so a worker can start work from the listing alone, without a second
    # round-trip per job just to learn what the robot was told to do.
    instruction: str | None
    rlds_uri: str | None


class DoneIn(BaseModel):
    attention_map_uri: str = Field(
        ...,
        min_length=1,
        description="Storage uri of the heatmap the worker wrote (e.g. 'r2://bucket/key').",
    )


class FailedIn(BaseModel):
    error: str = Field(..., min_length=1, description="Why the rollout could not be produced.")


def _to_out(job: AttributionJob, episode: Episode | None) -> JobOut:
    return JobOut(
        id=job.id,
        episode_id=job.episode_id,
        status=job.status,
        attention_map_uri=job.attention_map_uri,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
        instruction=episode.instruction if episode else None,
        rlds_uri=episode.rlds_uri if episode else None,
    )


def _load_job(db: Session, org: Organization, job_id: str) -> AttributionJob:
    job = db.get(AttributionJob, job_id)
    if job is None or job.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Attribution job not found.")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(
    status_filter: str | None = Query(
        default="queued",
        alias="status",
        description="Filter by status; pass an empty value for all of this org's jobs.",
    ),
    page: Page = Depends(page_params),
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    """Jobs for the caller's org, oldest first so the queue drains in FIFO order.

    Paginated with the same `limit`/`offset` as every other list endpoint.
    """
    stmt = select(AttributionJob).where(AttributionJob.org_id == org.id)
    if status_filter:
        stmt = stmt.where(AttributionJob.status == status_filter)
    jobs = db.execute(
        stmt.order_by(AttributionJob.created_at, AttributionJob.id)
        .limit(page.limit)
        .offset(page.offset)
    ).scalars().all()

    episodes = {
        e.id: e
        for e in db.execute(
            select(Episode).where(Episode.id.in_([j.episode_id for j in jobs]))
        ).scalars()
    }
    return [_to_out(j, episodes.get(j.episode_id)) for j in jobs]


@router.post("/{job_id}/claim", response_model=JobOut)
def claim_job(
    job_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> JobOut:
    """Take ownership of a queued job. A job already claimed by another worker returns 409, which
    is the signal to move on to the next one rather than duplicate its GPU time."""
    job = _load_job(db, org, job_id)
    if job.status != "queued":
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Job is '{job.status}', not 'queued' — already taken."
        )
    job.status = "running"
    db.commit()
    return _to_out(job, db.get(Episode, job.episode_id))


@router.post("/{job_id}/done", response_model=JobOut)
def complete_job(
    job_id: str,
    payload: DoneIn,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> JobOut:
    """Record a finished rollout and publish it onto the episode's attribution.

    Writing `Attribution.attention_map_uri` here is what makes the result visible to
    `GET /v1/episodes/{id}/attribution`, and therefore to the dashboard — without it the work
    would land in the jobs table and never reach a user.
    """
    job = _load_job(db, org, job_id)
    if job.status in _TERMINAL:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job is already '{job.status}'.")

    job.status = "done"
    job.attention_map_uri = payload.attention_map_uri
    job.error = None

    episode = db.get(Episode, job.episode_id)
    if episode is not None:
        attribution = episode.attribution
        if attribution is None:
            attribution = Attribution(episode_id=episode.id)
            db.add(attribution)
        attribution.attention_map_uri = payload.attention_map_uri

    db.commit()
    return _to_out(job, episode)


@router.post("/{job_id}/failed", response_model=JobOut)
def fail_job(
    job_id: str,
    payload: FailedIn,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> JobOut:
    """Mark a job failed with a reason, so a dead worker is distinguishable from a slow one."""
    job = _load_job(db, org, job_id)
    if job.status in _TERMINAL:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job is already '{job.status}'.")
    job.status = "failed"
    job.error = payload.error
    db.commit()
    return _to_out(job, db.get(Episode, job.episode_id))
