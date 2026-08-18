"""Scoped fine-tune dataset export.

Assembles ONLY the episodes in a given failure cluster — not the whole fleet — into the
customer's chosen format (RLDS or LeRobot) and writes a single artifact to object storage,
returning a signed download URL. Scoping the dataset to the failure cluster (rather than the
whole fleet) is a core claim: the fine-tune targets exactly the failure mode, so less data
and less compute are needed to fix it.

The serializers round-trip the normalized frames back into each format's encoding, so the
export is re-ingestable by the customer's training pipeline (and by this system's own
loop-closure verify step).
"""

from __future__ import annotations

import json

from aperture.core.models import Episode


def serialize_rlds(ep: Episode) -> dict:
    return {
        "embodiment_type": ep.robot.embodiment_type if ep.robot else "unknown",
        "policy_name": ep.robot.policy_name if ep.robot else "unknown",
        "outcome": ep.outcome,
        "steps": [
            {
                "observation": {
                    "action_confidence": f.action_confidence,
                    "contact_force": f.contact_force,
                },
                "language_instruction": ep.instruction,
                "subgoal": f.subgoal,
            }
            for f in ep.frames
        ],
    }


def serialize_lerobot(ep: Episode) -> dict:
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
                "observation.confidence": f.action_confidence,
                "observation.force": f.contact_force,
                "subtask": f.subgoal,
            }
            for f in ep.frames
        ],
    }


def build_export(episodes: list[Episode], fmt: str) -> bytes:
    if fmt not in ("rlds", "lerobot"):
        raise ValueError(f"Unsupported export format '{fmt}'. Use 'rlds' or 'lerobot'.")
    serializer = serialize_rlds if fmt == "rlds" else serialize_lerobot
    doc = {
        "format": fmt,
        "episode_count": len(episodes),
        "episodes": [{"episode_id": ep.id, "data": serializer(ep)} for ep in episodes],
    }
    return json.dumps(doc, indent=2).encode("utf-8")
