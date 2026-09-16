"""Job status API — how a client follows work it started.

`POST /v1/clusters/recompute` and the presigned ingest flow both return a job here instead of
blocking. The dashboard polls `GET /v1/jobs/{id}` until the status is terminal.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.db import get_db
from aperture.core.models import Job, Organization
from aperture.core.pagination import Page, page_params

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class JobOut(BaseModel):
    id: str
    kind: str
    status: str            # queued|running|done|failed
    result: dict | None
    error: str | None
    attempts: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[JobOut])
def list_jobs(
    kind: str | None = Query(default=None, description="Filter by job kind."),
    status_filter: str | None = Query(default=None, alias="status"),
    page: Page = Depends(page_params),
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    """This org's jobs, newest first."""
    stmt = select(Job).where(Job.org_id == org.id)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    if status_filter:
        stmt = stmt.where(Job.status == status_filter)
    jobs = db.execute(
        stmt.order_by(Job.created_at.desc(), Job.id).limit(page.limit).offset(page.offset)
    ).scalars().all()
    return [JobOut.model_validate(j) for j in jobs]


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> JobOut:
    job = db.get(Job, job_id)
    if job is None or job.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return JobOut.model_validate(job)
