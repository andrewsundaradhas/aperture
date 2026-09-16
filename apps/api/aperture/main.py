"""Aperture FastAPI application entrypoint.

Wires the four layers (ingestion -> evaluation -> interpretability -> clustering/loop-closure),
a health check, blob serving for the local storage backend, and pilot onboarding. Sentry is
initialized when APERTURE_SENTRY_DSN is set (free tier, Phase 6).
"""

from __future__ import annotations

import os

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import init_db
from aperture.core.models import Organization
from aperture.core.storage import get_storage

# --- Optional Sentry (Phase 6) ------------------------------------------------
_sentry_dsn = os.getenv("APERTURE_SENTRY_DSN")
if _sentry_dsn:
    try:
        import sentry_sdk

        sentry_sdk.init(dsn=_sentry_dsn, traces_sample_rate=0.1)
    except ImportError:
        pass

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Local convenience: ensure tables exist. Production uses Alembic migrations.
    init_db()

    # Without a worker, queued jobs sit forever and `uvicorn aperture.main:app` on its own would
    # silently do nothing. On by default so the quick-start works; docker-compose turns it off
    # and runs `python -m aperture.jobs.worker` as its own service.
    if settings.inline_worker:
        from aperture.jobs.worker import start_inline_worker, stop_inline_worker

        start_inline_worker()
        try:
            yield
        finally:
            stop_inline_worker()
        return

    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# Any localhost/127.0.0.1 port is allowed: `next dev` auto-increments past 3000 when that
# port is busy, and a hardcoded origin would silently break every dashboard fetch.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}


@app.get("/v1/blobs/{key:path}", tags=["meta"])
def get_blob(key: str, org: Organization = Depends(resolve_org)) -> Response:
    """Serve a blob by key, scoped to the calling organization.

    This route is registered unconditionally and reads through `get_storage()`, so it serves
    whichever backend is configured — including R2 in production, not just the local filesystem.
    It therefore authenticates and confines each org to its own key prefix, exactly as the PUT
    below does. Without that it was an unauthenticated read of any blob whose key was known, and
    exported datasets and attention maps are episode data.

    A real R2 presigned GET carries its own scoped authorisation and bypasses this route
    entirely; this is the path used when the client is handed a `/v1/blobs/...` URL instead.
    """
    if not key.startswith(f"{org.slug}/"):
        raise HTTPException(403, "Key does not belong to this organization.")
    try:
        data = get_storage().get_bytes(key)
    except FileNotFoundError as e:
        raise HTTPException(404, "Blob not found.") from e
    media = "application/json" if key.endswith(".json") else "application/octet-stream"
    return Response(content=data, media_type=media)


@app.put("/v1/blobs/{key:path}", tags=["meta"], status_code=201)
async def put_blob(key: str, request: Request, org: Organization = Depends(resolve_org)) -> dict:
    """Local stand-in for a presigned upload URL.

    R2/S3 hand out a real presigned PUT and the bytes never touch this process. The filesystem
    backend has nothing to presign, so `LocalStorage.signed_upload_url` points here and the
    client flow stays identical across backends.

    Unlike a presigned URL — which carries its own scoped authorisation — this route is
    reachable by anyone, so it authenticates and confines each org to its own key prefix.
    """
    if not key.startswith(f"{org.slug}/"):
        raise HTTPException(403, "Key does not belong to this organization.")

    settings = get_settings()
    body = await request.body()
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(413, f"Upload is {len(body)} bytes, over the configured limit.")

    get_storage().put_bytes(key, body)
    return {"key": key, "bytes": len(body)}


# --- Layer routers ------------------------------------------------------------
from aperture.clustering.router import router as clustering_router  # noqa: E402
from aperture.datasets.router import router as datasets_router  # noqa: E402
from aperture.evaluation.router import router as evaluation_router  # noqa: E402
from aperture.ingestion.router import router as ingestion_router  # noqa: E402
from aperture.interpretability.jobs import router as attribution_jobs_router  # noqa: E402
from aperture.interpretability.router import router as interpretability_router  # noqa: E402
from aperture.ingestion.uploads import router as uploads_router  # noqa: E402
from aperture.jobs.router import router as jobs_router  # noqa: E402
from aperture.loop.router import router as loop_router  # noqa: E402
from aperture.onboarding import router as onboarding_router  # noqa: E402

app.include_router(ingestion_router)
app.include_router(evaluation_router)
app.include_router(interpretability_router)
app.include_router(attribution_jobs_router)
app.include_router(clustering_router)
app.include_router(datasets_router)
app.include_router(loop_router)
app.include_router(jobs_router)
app.include_router(uploads_router)
app.include_router(onboarding_router)
