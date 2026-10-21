"""What each job kind actually does.

A handler receives an open session and its `Job`, and returns a JSON-serialisable result that
lands on `job.result`. It must not commit — `run_once` owns the transaction so a handler that
raises leaves nothing half-written.

Every handler is also callable synchronously; the API still uses that path for small workloads
and the tests exercise the same code either way.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from aperture.core.models import Job, Organization
from aperture.core.storage import read_uri

logger = logging.getLogger(__name__)


def cluster_recompute(db: Session, job: Job) -> dict:
    """Recluster an org's failed episodes.

    O(all failed episodes) with an HDBSCAN fit in the middle — the operation that made
    `POST /v1/clusters/recompute` a timeout waiting to happen once a fleet had real volume.
    """
    from aperture.clustering.cluster import recompute_clusters

    org = db.get(Organization, job.org_id)
    clusters = recompute_clusters(db, org)
    return {
        "cluster_count": len(clusters),
        "cluster_ids": [c.id for c in clusters],
        "episode_count": sum(c.episode_count for c in clusters),
    }


def episode_ingest(db: Session, job: Job) -> dict:
    """Normalize and persist a dataset that was uploaded straight to object storage.

    The upload endpoint holds the whole file in memory to parse it; a LeRobot v3 archive of real
    fleet video does not fit that shape. The presigned flow puts the bytes in the bucket first,
    then queues this — so the request that starts an ingest is tiny no matter how big the dataset.
    """
    from aperture.core.config import get_settings
    from aperture.ingestion.normalize import IngestionError, normalize_many
    from aperture.ingestion.service import persist_episode

    org = db.get(Organization, job.org_id)
    uri = job.payload["uri"]
    filename = job.payload.get("filename") or "upload.bin"

    raw = read_uri(uri)
    try:
        episodes = normalize_many(
            raw, filename,
            outcome=job.payload.get("outcome") or "auto",
            max_frames_per_episode=get_settings().max_frames_per_episode or None,
            images_per_episode=get_settings().max_images_per_episode,
        )
    except IngestionError as exc:
        # A malformed upload is the user's problem, not a transient fault — do not burn retries.
        raise PermanentJobError(str(exc)) from exc

    episode_ids = [persist_episode(db, org, ne).id for ne in episodes]
    classify_job = enqueue_classification(db, org, episode_ids)
    return {
        "episode_ids": episode_ids, "count": len(episode_ids), "filename": filename,
        "classify_job_id": classify_job,
    }


def classify_pending(db: Session, job: Job) -> dict:
    """Classify failed episodes that have no verdict yet (or all of them, with `reclassify`).

    Queued automatically after every upload and ingest, so clustering — which only considers
    classified failures — has something to work with without one API call per episode.
    Commits in batches so a large backlog makes visible progress and a crash loses one batch.
    """
    from sqlalchemy import select

    from aperture.core.models import Episode, FailureClassification
    from aperture.evaluation.router import run_classification

    payload = job.payload or {}
    stmt = select(Episode).where(Episode.org_id == job.org_id, Episode.outcome == "fail")
    if payload.get("episode_ids"):
        stmt = stmt.where(Episode.id.in_(payload["episode_ids"]))
    if not payload.get("reclassify"):
        stmt = stmt.outerjoin(
            FailureClassification, FailureClassification.episode_id == Episode.id
        ).where(FailureClassification.id.is_(None))
    ids = [e.id for e in db.execute(stmt.order_by(Episode.id)).scalars()]

    surfaces: dict[str, int] = {}
    for i, eid in enumerate(ids, 1):
        row = run_classification(db, db.get(Episode, eid))
        surfaces[row.surface] = surfaces.get(row.surface, 0) + 1
        if i % CLASSIFY_BATCH == 0:
            db.commit()
    db.flush()
    return {"classified": len(ids), "surfaces": surfaces}


CLASSIFY_BATCH = 100


def enqueue_classification(db: Session, org: Organization, episode_ids: list[str]) -> str | None:
    """Queue `classify_pending` for freshly ingested episodes when auto-classify is on."""
    from aperture.core.config import get_settings
    from aperture.jobs.queue import enqueue

    if not episode_ids or not get_settings().auto_classify:
        return None
    return enqueue(db, org, "classify_pending", {"episode_ids": episode_ids}).id


class PermanentJobError(Exception):
    """A failure retrying cannot fix — bad input rather than a flaky dependency."""


HANDLERS = {
    "cluster_recompute": cluster_recompute,
    "episode_ingest": episode_ingest,
    "classify_pending": classify_pending,
}
