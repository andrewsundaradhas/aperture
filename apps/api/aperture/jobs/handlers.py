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
            max_frames_per_episode=get_settings().max_frames_per_episode or None,
        )
    except IngestionError as exc:
        # A malformed upload is the user's problem, not a transient fault — do not burn retries.
        raise PermanentJobError(str(exc)) from exc

    episode_ids = [persist_episode(db, org, ne).id for ne in episodes]
    return {"episode_ids": episode_ids, "count": len(episode_ids), "filename": filename}


class PermanentJobError(Exception):
    """A failure retrying cannot fix — bad input rather than a flaky dependency."""


HANDLERS = {
    "cluster_recompute": cluster_recompute,
    "episode_ingest": episode_ingest,
}
