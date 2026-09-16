"""A dataset export has to be trainable, not just descriptive.

A policy learns from (observation, action) pairs. Before this suite existed the export carried
only diagnostic scalars — action confidence, contact force, subgoal — which no fine-tune can
consume. These tests exist so that regression is loud instead of silent.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from aperture import fixtures
from aperture.core.models import Episode, Organization
from aperture.datasets.export import (
    EXPORT_SCHEMA_VERSION,
    build_export,
    export_readiness,
    serialize_lerobot,
    serialize_rlds,
)
from aperture.ingestion.normalize import normalize
from aperture.ingestion.service import persist_episode


@pytest.fixture
def stored_episode(db_session):
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    ne = normalize(fixtures.perception_failure_rlds(), "e.rlds.json")
    ep = persist_episode(db_session, org, ne)
    db_session.commit()
    return db_session.get(Episode, ep.id)


# --- the regression guard -------------------------------------------------------------------


def test_rlds_export_carries_actions(stored_episode):
    """THE regression test: every step must carry a non-empty action vector."""
    doc = serialize_rlds(stored_episode)
    assert doc["steps"], "export produced no steps"
    for i, step in enumerate(doc["steps"]):
        assert step["action"], f"step {i} has no action — this export cannot be fine-tuned on"
        assert len(step["action"]) == fixtures.ARM_DOF


def test_lerobot_export_carries_actions(stored_episode):
    doc = serialize_lerobot(stored_episode)
    assert doc["frames"], "export produced no frames"
    for i, frame in enumerate(doc["frames"]):
        assert frame["action"], f"frame {i} has no action — this export cannot be fine-tuned on"
        assert len(frame["action"]) == fixtures.ARM_DOF


def test_built_export_document_contains_actions(stored_episode):
    """The same guarantee through the public entry point, on serialized bytes."""
    for fmt in ("rlds", "lerobot"):
        doc = json.loads(build_export([stored_episode], fmt))
        assert doc["schema_version"] == EXPORT_SCHEMA_VERSION
        assert doc["readiness"]["trainable"] is True
        blob = json.dumps(doc)
        assert '"action"' in blob and "null" not in blob.split('"action":')[1][:20]


# --- observations ---------------------------------------------------------------------------


def test_export_carries_state_vectors(stored_episode):
    rlds = serialize_rlds(stored_episode)
    lerobot = serialize_lerobot(stored_episode)
    assert all(len(s["observation"]["state"]) == fixtures.ARM_DOF for s in rlds["steps"])
    assert all(len(f["observation.state"]) == fixtures.ARM_DOF for f in lerobot["frames"])


def test_export_covers_every_stored_frame(stored_episode):
    """Not a sample — the whole episode, or a fine-tune sees a truncated trajectory."""
    n = len(stored_episode.frames)
    assert len(serialize_rlds(stored_episode)["steps"]) == n
    assert len(serialize_lerobot(stored_episode)["frames"]) == n


def test_export_marks_the_terminal_step(stored_episode):
    steps = serialize_rlds(stored_episode)["steps"]
    assert steps[-1]["is_terminal"] is True
    assert all(s["is_terminal"] is False for s in steps[:-1])


def test_images_are_referenced_not_inlined(db_session):
    """A 50-episode cluster is thousands of frames; inlining base64 would make the document
    unusable. Frames carry a fetchable URL instead."""
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    doc = {
        "embodiment_type": "franka",
        "policy_name": "openvla-7b",
        "outcome": "fail",
        "steps": [
            {
                "observation": {"action_confidence": 0.3, "contact_force": 1.0, "image": base64.b64encode(png).decode()},
                "action": [0.1] * 7,
                "language_instruction": "pick up the red block",
            }
        ],
    }
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    ep = persist_episode(db_session, org, normalize(json.dumps(doc).encode(), "i.rlds.json"))
    db_session.commit()
    ep = db_session.get(Episode, ep.id)

    step = serialize_rlds(ep)["steps"][0]
    assert step["observation"]["image_url"], "a stored frame image must be referenced"
    # The raw bytes must not appear in the document.
    assert base64.b64encode(png).decode() not in json.dumps(step)


# --- readiness reporting --------------------------------------------------------------------


def test_readiness_flags_an_untrainable_export(db_session):
    """Episodes ingested before actions were captured must be reported as untrainable rather
    than handed over as if they were a usable dataset."""
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    doc = {
        "embodiment_type": "franka",
        "policy_name": "openvla-7b",
        "outcome": "fail",
        "steps": [
            {"observation": {"action_confidence": 0.3, "contact_force": 1.0}, "language_instruction": "x"}
        ],
    }
    ep = persist_episode(db_session, org, normalize(json.dumps(doc).encode(), "n.rlds.json"))
    db_session.commit()
    ep = db_session.get(Episode, ep.id)

    readiness = export_readiness([ep])
    assert readiness["trainable"] is False
    assert readiness["frames_with_action"] == 0
    assert "cannot be fine-tuned" in readiness["note"]


def test_readiness_counts_frames(stored_episode):
    readiness = export_readiness([stored_episode])
    assert readiness["total_frames"] == len(stored_episode.frames)
    assert readiness["frames_with_action"] == len(stored_episode.frames)
    assert readiness["trainable"] is True


def test_unknown_format_is_rejected(stored_episode):
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_export([stored_episode], "tfrecord")
