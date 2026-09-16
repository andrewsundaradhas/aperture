"""Large-dataset ingestion: upload straight to object storage, then queue the parse.

`POST /v1/episodes/upload` reads the whole file into the API process to parse it. That is fine
for a JSON episode and impossible for a LeRobot v3 archive of real fleet video, which is
gigabytes: the bytes would cross the network twice and sit in the API's memory in between.

The flow here keeps the API out of the data path entirely:

    1. POST /v1/uploads/presign   -> {upload_url, key, uri}
    2. PUT the bytes to upload_url (straight to R2/S3/MinIO; no API involvement)
    3. POST /v1/uploads/{key}/ingest -> 202 + job id, worker parses and persists
    4. GET /v1/jobs/{id}          -> episode_ids when done

Step 2 is a real presigned URL against R2/S3. On the local filesystem backend there is nothing
to presign, so `PUT /v1/blobs/{key}` stands in — the client code is identical either way, which
is the point of routing it through `StorageBackend`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Organization
from aperture.core.ratelimit import ingest_limit
from aperture.core.storage import get_storage
from aperture.jobs.queue import enqueue

router = APIRouter(prefix="/v1/uploads", tags=["ingestion"])

UPLOAD_URL_TTL_SECONDS = 3600


class PresignRequest(BaseModel):
    filename: str = Field(
        ...,
        min_length=1,
        description="Original filename. Its extension selects the reader, so keep the real one "
                    "('fleet.zip', 'shard.tfrecord', 'episode.rlds.json').",
    )


class PresignResponse(BaseModel):
    key: str
    uri: str
    upload_url: str
    method: str = "PUT"
    expires_in: int = UPLOAD_URL_TTL_SECONDS
    ingest_url: str


def _safe_name(filename: str) -> str:
    """Strip any path from a client-supplied filename.

    The key is built from this and becomes a path in the bucket, so `../../` in a filename must
    not be able to walk out of the org's prefix and write over another tenant's objects.
    """
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or name in (".", ".."):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid filename.")
    return name


def _key_for(org: Organization, filename: str) -> str:
    import uuid

    # The uuid keeps two uploads of the same filename from colliding, and keeps one tenant's
    # key from ever being guessable by another.
    return f"{org.slug}/uploads/{uuid.uuid4()}/{_safe_name(filename)}"


@router.post("/presign", response_model=PresignResponse)
def presign_upload(
    body: PresignRequest,
    org: Organization = Depends(ingest_limit),
) -> PresignResponse:
    """A URL to PUT a dataset to, and the key to hand back when it is there."""
    storage = get_storage()
    key = _key_for(org, body.filename)
    return PresignResponse(
        key=key,
        uri=f"local://{key}" if not get_settings().r2_endpoint_url else f"r2://{get_settings().r2_bucket}/{key}",
        upload_url=storage.signed_upload_url(key, expires_in=UPLOAD_URL_TTL_SECONDS),
        ingest_url=f"/v1/uploads/ingest",
    )


class IngestRequest(BaseModel):
    key: str = Field(..., min_length=1, description="The key returned by /presign.")
    filename: str | None = Field(
        default=None,
        description="Overrides the reader selection; defaults to the key's own basename.",
    )


class IngestAccepted(BaseModel):
    job_id: str
    status: str
    poll: str


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=IngestAccepted)
def ingest_uploaded(
    body: IngestRequest,
    org: Organization = Depends(ingest_limit),
    db: Session = Depends(get_db),
) -> IngestAccepted:
    """Queue parsing of an object already in storage.

    The key must sit under the caller's own prefix — otherwise one tenant could name another
    tenant's object and have the worker ingest it into their own organization.
    """
    if not body.key.startswith(f"{org.slug}/"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Key does not belong to this organization.")

    storage = get_storage()
    if not storage.exists(body.key):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No object at that key. PUT the file to the presigned URL before calling ingest.",
        )

    settings = get_settings()
    uri = f"r2://{settings.r2_bucket}/{body.key}" if settings.r2_endpoint_url else f"local://{body.key}"
    job = enqueue(
        db, org, "episode_ingest",
        {"uri": uri, "filename": body.filename or body.key.split("/")[-1]},
    )
    db.commit()
    return IngestAccepted(job_id=job.id, status=job.status, poll=f"/v1/jobs/{job.id}")
