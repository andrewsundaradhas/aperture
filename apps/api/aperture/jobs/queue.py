"""A small durable job queue backed by the `jobs` table.

Why not Celery/RQ/Arq: all three want a broker (Redis or RabbitMQ) that this deployment does not
otherwise need, and Aperture already has the one piece of infrastructure a queue requires — a
transactional database. At the volume one robotics pilot generates, a Postgres-backed queue is
both sufficient and one less service to run on-prem.

**Claiming is the only subtle part.** Two workers must never run the same job. `claim_next`
issues a conditional `UPDATE ... WHERE status = 'queued'` and treats "rows updated == 1" as
winning the race — the database's own row lock does the arbitration, so it is correct without a
`SELECT ... FOR UPDATE` and works identically on SQLite and Postgres. At high worker counts
`FOR UPDATE SKIP LOCKED` is the better instrument; this is not that scale.

Jobs are retried up to `MAX_ATTEMPTS`, then marked `failed` with the last error. A job that
fails deterministically must stop consuming the worker rather than spin at the head of the
queue forever.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from aperture.core.models import Job, Organization

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
TERMINAL = {DONE, FAILED}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(db: Session, org: Organization, kind: str, payload: dict | None = None) -> Job:
    """Queue a unit of work for `org`. The caller commits."""
    from aperture.jobs.handlers import HANDLERS

    if kind not in HANDLERS:
        raise ValueError(f"unknown job kind {kind!r}; known: {sorted(HANDLERS)}")
    job = Job(org_id=org.id, kind=kind, payload=payload or {})
    db.add(job)
    db.flush()
    return job


def claim_next(db: Session, kinds: list[str] | None = None) -> Job | None:
    """Atomically take the oldest queued job, or None.

    The conditional UPDATE is the lock: if another worker claimed this row first, the WHERE no
    longer matches and `rowcount` is 0, so we look at the next candidate instead of running a
    job someone else owns.
    """
    stmt = select(Job).where(Job.status == QUEUED)
    if kinds:
        stmt = stmt.where(Job.kind.in_(kinds))
    candidates = db.execute(stmt.order_by(Job.created_at, Job.id).limit(10)).scalars().all()

    for candidate in candidates:
        won = db.execute(
            update(Job)
            .where(Job.id == candidate.id, Job.status == QUEUED)
            .values(status=RUNNING, started_at=_now(), attempts=Job.attempts + 1)
        ).rowcount
        db.commit()
        if won:
            db.refresh(candidate)
            return candidate
    return None


def run_once(db: Session, kinds: list[str] | None = None) -> Job | None:
    """Claim and execute one job. Returns it, or None when the queue is empty."""
    from aperture.jobs.handlers import HANDLERS

    job = claim_next(db, kinds)
    if job is None:
        return None

    handler = HANDLERS.get(job.kind)
    if handler is None:
        # The kind was valid when enqueued and is not now — a deploy removed it. Fail loudly
        # rather than retry something nothing can run.
        job.status = FAILED
        job.error = f"no handler registered for kind {job.kind!r}"
        job.finished_at = _now()
        db.commit()
        return job

    try:
        job.result = handler(db, job) or {}
        job.status = DONE
        job.error = None
    except Exception as exc:  # noqa: BLE001 - a handler failing must not kill the worker
        from aperture.jobs.handlers import PermanentJobError

        db.rollback()
        # Re-read: the rollback detached whatever the handler had staged.
        job = db.get(Job, job.id)
        job.error = f"{type(exc).__name__}: {exc}"[:2000]
        # Bad input will fail identically every time; only retry what might succeed later.
        permanent = isinstance(exc, PermanentJobError)
        job.status = FAILED if (permanent or job.attempts >= MAX_ATTEMPTS) else QUEUED
        logger.warning(
            "job %s (%s) failed on attempt %d/%d: %s\n%s",
            job.id, job.kind, job.attempts, MAX_ATTEMPTS, job.error, traceback.format_exc(),
        )
    finally:
        if job.status in TERMINAL:
            job.finished_at = _now()
        db.commit()

    return job
