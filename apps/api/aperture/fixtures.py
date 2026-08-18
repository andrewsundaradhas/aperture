"""Synthetic RLDS / LeRobot episode generators.

Used as test fixtures and by the seed script (no real fleet data needed for the MVP). Each
generator produces bytes in the portable JSON encoding the ingestion normalizer accepts, and
is engineered to trigger exactly one failure surface (or to be deliberately ambiguous / a
success). Deterministic given their arguments so tests are stable.
"""

from __future__ import annotations

import json


def _rlds(embodiment: str, policy: str, instruction: str | None, steps: list[dict], outcome: str) -> bytes:
    return json.dumps(
        {
            "embodiment_type": embodiment,
            "policy_name": policy,
            "outcome": outcome,
            "steps": [{**s, "language_instruction": instruction} for s in steps],
        }
    ).encode("utf-8")


def _lerobot(robot_type: str, policy: str, task: str | None, frames: list[dict], success: bool) -> bytes:
    return json.dumps(
        {
            "meta": {"robot_type": robot_type, "policy": policy, "tasks": [task] if task else []},
            "success": success,
            "frames": frames,
        }
    ).encode("utf-8")


# --- Perception failure: action-confidence collapse (RLDS) --------------------
def perception_failure_rlds(instruction: str = "pick up the red block") -> bytes:
    steps = []
    for t in range(12):
        conf = 0.92 if t < 5 else 0.2  # sharp collapse after t=5
        steps.append({"observation": {"action_confidence": conf, "contact_force": 1.0}, "subgoal": "reach"})
    return _rlds("franka", "openvla-7b", instruction, steps, "fail")


# --- Motor failure: contact-force anomaly / spike (LeRobot) --------------------
def motor_failure_lerobot(task: str = "insert the peg") -> bytes:
    frames = []
    for t in range(12):
        force = 1.0 if t != 8 else 40.0  # large spike at t=8
        frames.append(
            {"frame_index": t, "observation.confidence": 0.85, "observation.force": force, "subtask": "insert"}
        )
    return _lerobot("so100", "act", task, frames, success=False)


# --- Grounding failure: replanning / repeated sub-goal re-issue (RLDS) ---------
def grounding_failure_rlds(instruction: str = "move object to the shelf") -> bytes:
    # sub-goal thrashes: reach -> grasp -> reach -> grasp -> reach (re-issues 'reach' repeatedly)
    subgoals = ["reach", "grasp", "reach", "grasp", "reach", "grasp", "reach"]
    steps = [
        {"observation": {"action_confidence": 0.8, "contact_force": 1.0}, "subgoal": sg} for sg in subgoals
    ]
    return _rlds("franka", "openvla-7b", instruction, steps, "fail")


# --- Ambiguous: weak signals across the board (LeRobot) ------------------------
def ambiguous_lerobot(task: str = "tidy the table") -> bytes:
    frames = []
    for t in range(10):
        frames.append(
            {
                "frame_index": t,
                "observation.confidence": 0.7 - t * 0.02,  # gentle, not a collapse
                "observation.force": 1.0 + (0.2 if t % 2 else -0.2),  # small jitter, no spike
                "subtask": "reach" if t < 5 else "grasp",  # one clean transition, no re-issues
            }
        )
    return _lerobot("so100", "act", task, frames, success=False)


# --- Success episode ----------------------------------------------------------
def success_rlds(instruction: str = "pick up the red block") -> bytes:
    steps = [
        {"observation": {"action_confidence": 0.9, "contact_force": 1.0}, "subgoal": sg}
        for sg in ["reach", "grasp", "lift"]
    ]
    return _rlds("franka", "openvla-7b", instruction, steps, "success")


def success_lerobot(task: str, subtasks: list[str] | None = None) -> bytes:
    subtasks = subtasks or ["reach", "grasp", "lift"]
    frames = [
        {"frame_index": t, "observation.confidence": 0.9, "observation.force": 1.0, "subtask": sg}
        for t, sg in enumerate(subtasks)
    ]
    return _lerobot("so100", "act", task, frames, success=True)


# --- A cluster of similar grounding failures (same task, same signature) -------
def grounding_cluster(instruction: str, n: int = 4) -> list[bytes]:
    out = []
    for i in range(n):
        subgoals = ["reach", "grasp", "reach", "grasp", "reach"] + (["grasp"] if i % 2 else [])
        steps = [
            {"observation": {"action_confidence": 0.82, "contact_force": 1.0}, "subgoal": sg}
            for sg in subgoals
        ]
        out.append(_rlds("franka", "openvla-7b", instruction, steps, "fail"))
    return out
