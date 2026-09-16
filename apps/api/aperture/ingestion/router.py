"""Ingestion Layer API — POST /v1/episodes/upload, GET /v1/episodes/{id}, GET /v1/episodes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Episode, Organization
from aperture.core.pagination import Page, page_params
from aperture.core.ratelimit import ingest_limit
from aperture.ingestion.normalize import IngestionError, normalize_many
from aperture.ingestion.schemas import (
    EpisodeClassificationOut,
    EpisodeOut,
    EpisodeSummary,
    FrameOut,
    UploadResult,
)
from aperture.ingestion.service import persist_episode

router = APIRouter(prefix="/v1/episodes", tags=["ingestion"])


@router.post("/upload", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
async def upload_episodes(
    files: list[UploadFile] = File(...),
    org: Organization = Depends(ingest_limit),
    db: Session = Depends(get_db),
) -> UploadResult:
    """Accept a batch of episode files; normalize and store every episode in them.

    Each file may be a real LeRobot v3 dataset archive, an RLDS `.tfrecord` stream, or a legacy
    single-episode JSON document — and a dataset carries many episodes, so the number of
    episodes created is not the number of files uploaded.

    Malformed files fail with a specific 4xx (Phase 6 hardening), never a 500.
    """
    settings = get_settings()
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files supplied.")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Batch of {len(files)} exceeds the limit of {settings.max_batch_files} files.",
        )

    episode_ids: list[str] = []
    for f in files:
        raw = await f.read()
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"'{f.filename}' is {len(raw)} bytes, over the {settings.max_upload_bytes}-byte limit.",
            )
        try:
            normalized = normalize_many(
                raw,
                f.filename or "episode.bin",
                max_frames_per_episode=settings.max_frames_per_episode or None,
            )
        except IngestionError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{f.filename}: {e}") from e
        for ne in normalized:
            episode_ids.append(persist_episode(db, org, ne).id)

    db.commit()
    return UploadResult(episode_ids=episode_ids, count=len(episode_ids))


@router.get("", response_model=list[EpisodeSummary])
def list_episodes(
    robot_id: str | None = None,
    surface: str | None = None,
    outcome: str | None = None,
    page: Page = Depends(page_params),
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[EpisodeSummary]:
    """List episodes for the caller's org, filterable by robot / failure surface / outcome.

    Paginated with `limit`/`offset` (default 50). `surface` filters after the query because the
    verdict lives on a related row; `limit` is therefore applied to the pre-filter page, which
    is why a surface-filtered page can return fewer than `limit` rows.
    """
    stmt = select(Episode).where(Episode.org_id == org.id).order_by(
        Episode.started_at.desc(), Episode.id
    )
    if robot_id:
        stmt = stmt.where(Episode.robot_id == robot_id)
    if outcome:
        stmt = stmt.where(Episode.outcome == outcome)

    episodes = db.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all()
    out: list[EpisodeSummary] = []
    for ep in episodes:
        cls = ep.classification
        if surface and (cls is None or cls.surface != surface):
            continue
        out.append(
            EpisodeSummary(
                id=ep.id,
                robot_id=ep.robot_id,
                started_at=ep.started_at,
                outcome=ep.outcome,
                source_format=ep.source_format,
                instruction=ep.instruction,
                surface=cls.surface if cls else None,
                classification_confidence=cls.confidence if cls else None,
            )
        )
    return out


@router.get("/{episode_id}", response_model=EpisodeOut)
def get_episode(
    episode_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> EpisodeOut:
    """Return normalized metadata + per-frame signals. Scoped to the caller's org."""
    ep = _load_episode(db, org, episode_id)
    cls = ep.classification
    return EpisodeOut(
        id=ep.id,
        robot_id=ep.robot_id,
        org_id=ep.org_id,
        started_at=ep.started_at,
        outcome=ep.outcome,
        source_format=ep.source_format,
        instruction=ep.instruction,
        rlds_uri=ep.rlds_uri,
        frames=[
            FrameOut(
                t=fr.t,
                action_confidence=fr.action_confidence,
                contact_force=fr.contact_force,
                subgoal=fr.subgoal,
            )
            for fr in ep.frames
        ],
        classification=(
            EpisodeClassificationOut(
                surface=cls.surface,
                confidence=cls.confidence,
                method=cls.method,
                details=cls.details or {},
            )
            if cls
            else None
        ),
    )


def _load_episode(db: Session, org: Organization, episode_id: str) -> Episode:
    ep = db.get(Episode, episode_id)
    # Tenant isolation: an episode from another org is indistinguishable from a missing one.
    if ep is None or ep.org_id != org.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Episode not found.")
    return ep
