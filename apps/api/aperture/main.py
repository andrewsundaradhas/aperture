"""Aperture FastAPI application entrypoint.

Wires the four layers (ingestion -> evaluation -> interpretability -> clustering/loop-closure),
a health check, blob serving for the local storage backend, and pilot onboarding. Sentry is
initialized when APERTURE_SENTRY_DSN is set (free tier, Phase 6).
"""

from __future__ import annotations

import os

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from aperture.core.config import get_settings
from aperture.core.db import init_db
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
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}


@app.get("/v1/blobs/{key:path}", tags=["meta"])
def get_blob(key: str) -> Response:
    """Serve a local-storage blob (dev only). In production R2 presigned URLs serve blobs."""
    try:
        data = get_storage().get_bytes(key)
    except FileNotFoundError as e:
        raise HTTPException(404, "Blob not found.") from e
    media = "application/json" if key.endswith(".json") else "application/octet-stream"
    return Response(content=data, media_type=media)


# --- Layer routers ------------------------------------------------------------
from aperture.clustering.router import router as clustering_router  # noqa: E402
from aperture.datasets.router import router as datasets_router  # noqa: E402
from aperture.evaluation.router import router as evaluation_router  # noqa: E402
from aperture.ingestion.router import router as ingestion_router  # noqa: E402
from aperture.interpretability.router import router as interpretability_router  # noqa: E402
from aperture.loop.router import router as loop_router  # noqa: E402
from aperture.onboarding import router as onboarding_router  # noqa: E402

app.include_router(ingestion_router)
app.include_router(evaluation_router)
app.include_router(interpretability_router)
app.include_router(clustering_router)
app.include_router(datasets_router)
app.include_router(loop_router)
app.include_router(onboarding_router)
