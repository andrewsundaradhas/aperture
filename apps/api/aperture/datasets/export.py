"""Scoped fine-tune dataset export.

Assembles ONLY the episodes in a given failure cluster — not the whole fleet — into the
customer's chosen format (RLDS or LeRobot) and writes a single artifact to object storage,
returning a signed download URL. Scoping the dataset to the failure cluster (rather than the
whole fleet) is a core claim: the fine-tune targets exactly the failure mode, so less data
and less compute are needed to fix it.

**What makes the export trainable.** A policy learns from (observation, action) pairs. Each
exported frame therefore carries the commanded `action`, the observed `state`, and a
reference to that timestep's observation image; the diagnostic scalars Aperture classifies on
(action confidence, contact force, subgoal) ride along but are not the learning target. An
export whose frames have no actions is a report, not a dataset — `export_readiness` says so
explicitly rather than letting a customer discover it at fine-tune time.

Images are exported **by reference**, not inlined. A cluster of 50 episodes at 200 frames is
10,000 images; base64-inlining them would produce a multi-gigabyte JSON document that no
training pipeline wants to parse. Each frame carries `image_url`, a URL the customer's loader
can fetch with its API key.

The serializers round-trip the normalized frames back into each format's encoding, so the
export is re-ingestable by the customer's training pipeline (and by this system's own
loop-closure verify step).
"""

from __future__ import annotations

import json

from aperture.core.models import Episode
from aperture.core.storage import get_storage, key_from_uri

# Bumped when the emitted document's shape changes, so a customer's loader can refuse a
# version it does not understand instead of silently mis-reading fields.
EXPORT_SCHEMA_VERSION = 2


def _image_url(image_uri: str | None) -> str | None:
    """A fetchable URL for a frame's observation image, or None when the frame has none.

    Local storage yields an API-relative path; R2 yields a presigned URL. Both are resolved
    at export time so the document is self-contained.
    """
    if not image_uri:
        return None
    try:
        return get_storage().signed_url(key_from_uri(image_uri))
    except Exception:
        # A blob that cannot be signed should not sink the whole export — the frame's
        # action and state are still trainable without the image.
        return None


def serialize_rlds(ep: Episode) -> dict:
    """RLDS-shaped episode: a `steps` list, each step an observation + the action taken."""
    return {
        "embodiment_type": ep.robot.embodiment_type if ep.robot else "unknown",
        "policy_name": ep.robot.policy_name if ep.robot else "unknown",
        "outcome": ep.outcome,
        "steps": [
            {
                "observation": {
                    "state": f.state,
                    "image_url": _image_url(f.image_uri),
                    "action_confidence": f.action_confidence,
                    "contact_force": f.contact_force,
                },
                "action": f.action,
                "language_instruction": ep.instruction,
                "subgoal": f.subgoal,
                "is_terminal": f.t == ep.frames[-1].t if ep.frames else True,
            }
            for f in ep.frames
        ],
    }


def serialize_lerobot(ep: Episode) -> dict:
    """LeRobot-shaped episode: flat per-frame rows with dotted observation keys."""
    return {
        "meta": {
            "robot_type": ep.robot.embodiment_type if ep.robot else "unknown",
            "policy": ep.robot.policy_name if ep.robot else "unknown",
            "tasks": [ep.instruction] if ep.instruction else [],
        },
        "success": ep.outcome == "success",
        "frames": [
            {
                "frame_index": f.t,
                "action": f.action,
                "observation.state": f.state,
                "observation.image_url": _image_url(f.image_uri),
                "observation.confidence": f.action_confidence,
                "observation.force": f.contact_force,
                "subtask": f.subgoal,
            }
            for f in ep.frames
        ],
    }


def export_readiness(episodes: list[Episode]) -> dict:
    """Whether this export can actually be fine-tuned on, and what is missing if not.

    Surfaced in the document itself so the gap is visible before a customer builds a training
    run around it, rather than after.
    """
    total = sum(len(ep.frames) for ep in episodes)
    with_action = sum(1 for ep in episodes for f in ep.frames if f.action)
    with_image = sum(1 for ep in episodes for f in ep.frames if f.image_uri)
    return {
        "total_frames": total,
        "frames_with_action": with_action,
        "frames_with_image": with_image,
        "trainable": with_action > 0,
        "note": (
            "Frames carry actions; this export can be fine-tuned on."
            if with_action == total and total
            else "No frame carries an action — this export is diagnostic only and cannot be "
            "fine-tuned on. Re-ingest these episodes from a source that includes actions."
            if with_action == 0
            else f"{with_action}/{total} frames carry actions; the remainder are diagnostic only."
        ),
    }


def build_export(episodes: list[Episode], fmt: str) -> bytes:
    if fmt not in ("rlds", "lerobot"):
        raise ValueError(f"Unsupported export format '{fmt}'. Use 'rlds' or 'lerobot'.")
    serializer = serialize_rlds if fmt == "rlds" else serialize_lerobot
    doc = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": fmt,
        "episode_count": len(episodes),
        "readiness": export_readiness(episodes),
        "episodes": [{"episode_id": ep.id, "data": serializer(ep)} for ep in episodes],
    }
    return json.dumps(doc, indent=2).encode("utf-8")
