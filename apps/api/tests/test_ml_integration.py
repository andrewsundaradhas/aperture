"""Learned-model integration: image threading through ingestion, and graceful fallback when the
optional `[ml]` extra / weights are absent (the default in CI and the base install)."""

from __future__ import annotations

import base64
import json

from sqlalchemy import select

from aperture.core.models import Episode, Organization
from aperture.core.storage import read_uri
from aperture.ingestion.normalize import normalize
from aperture.ingestion.service import persist_episode

# A minimal valid 1x1 PNG — enough to exercise decode + blob storage without needing torch/PIL.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _rlds_with_image() -> bytes:
    img_b64 = base64.b64encode(_PNG_1x1).decode()
    return json.dumps(
        {
            "embodiment_type": "franka",
            "policy_name": "openvla-7b",
            "outcome": "fail",
            "steps": [
                {
                    "observation": {"action_confidence": 0.2, "contact_force": 1.0, "image": img_b64},
                    "language_instruction": "pick up the red block",
                    "subgoal": "reach",
                }
            ],
        }
    ).encode("utf-8")


def test_frame_image_is_parsed_and_stored(db_session):
    ne = normalize(_rlds_with_image(), "e.rlds.json")
    assert ne.frames[0].image_b64 is not None

    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    ep = persist_episode(db_session, org, ne)
    db_session.commit()

    ep = db_session.get(Episode, ep.id)
    frame = ep.frames[0]
    assert frame.image_uri is not None
    # The stored blob round-trips back to the original image bytes.
    assert read_uri(frame.image_uri) == _PNG_1x1


def test_episode_without_image_still_ingests(db_session):
    from aperture import fixtures

    ne = normalize(fixtures.perception_failure_rlds(), "e.rlds.json")
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    ep = persist_episode(db_session, org, ne)
    db_session.commit()
    assert all(f.image_uri is None for f in db_session.get(Episode, ep.id).frames)


def test_learned_path_disabled_by_default():
    """With default settings the learned path is inert, so every layer uses the base path."""
    from aperture.ml import gateway

    assert gateway.learned_enabled() is False
