"""Persistence for normalized episodes — shared by the upload endpoint and the loop-closure
verify flow (which ingests a post-retrain batch through the same path).
"""

from __future__ import annotations

import base64
import binascii

from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.models import Episode, EpisodeFrame, Organization, Robot
from aperture.core.storage import get_storage
from aperture.ingestion.schemas import NormalizedEpisode, NormalizedFrame


def _decode_image(frame: NormalizedFrame) -> bytes | None:
    """Decode a frame's base64 image payload, tolerating a `data:` URI prefix. Returns None on
    absent or malformed data (ingestion never fails just because an image is unreadable)."""
    if not frame.image_b64:
        return None
    payload = frame.image_b64
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None


def _get_or_create_robot(db: Session, org: Organization, ne: NormalizedEpisode) -> Robot:
    robot = db.execute(
        select(Robot).where(
            Robot.org_id == org.id,
            Robot.embodiment_type == ne.embodiment_type,
            Robot.policy_name == ne.policy_name,
        )
    ).scalar_one_or_none()
    if robot is None:
        robot = Robot(org_id=org.id, embodiment_type=ne.embodiment_type, policy_name=ne.policy_name)
        db.add(robot)
        db.flush()
    return robot


def persist_episode(db: Session, org: Organization, ne: NormalizedEpisode) -> Episode:
    """Store raw blob in object storage; write metadata + per-frame signals to the DB."""
    robot = _get_or_create_robot(db, org, ne)

    episode = Episode(
        robot_id=robot.id,
        org_id=org.id,
        outcome=ne.outcome,
        source_format=ne.source_format,
        instruction=ne.instruction,
    )
    if ne.started_at is not None:
        episode.started_at = ne.started_at
    db.add(episode)
    db.flush()  # assigns episode.id

    if ne.raw_blob is not None:
        key = f"{org.slug}/episodes/{episode.id}/{ne.raw_blob_name or 'raw.bin'}"
        episode.rlds_uri = get_storage().put_bytes(key, ne.raw_blob)

    storage = get_storage()
    for f in ne.frames:
        image_uri: str | None = None
        img = _decode_image(f)
        if img is not None:
            key = f"{org.slug}/episodes/{episode.id}/frames/{f.t}.img"
            image_uri = storage.put_bytes(key, img)
        db.add(
            EpisodeFrame(
                episode_id=episode.id,
                t=f.t,
                action_confidence=f.action_confidence,
                contact_force=f.contact_force,
                subgoal=f.subgoal,
                image_uri=image_uri,
                action=f.action,
                state=f.state,
            )
        )
    db.flush()
    return episode
