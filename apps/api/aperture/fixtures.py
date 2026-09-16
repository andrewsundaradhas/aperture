"""Synthetic RLDS / LeRobot episode generators.

Used as test fixtures and by the seed script (no real fleet data needed for the MVP). Each
generator produces bytes in the portable JSON encoding the ingestion normalizer accepts, and
is engineered to trigger exactly one failure surface (or to be deliberately ambiguous / a
success). Deterministic given their arguments so tests are stable.
"""

from __future__ import annotations

import json
import math

# 6-DoF end-effector command + 1 gripper channel. Matches the width the reference policy's
# action head was trained against, so a fixture export drops straight into it.
ARM_DOF = 7


def _state(t: int, phase: float = 0.0) -> list[float]:
    """Deterministic proprioceptive state for timestep `t`.

    Not a physical simulation — a smooth, reproducible trajectory, so that exports carry
    real-shaped vectors and tests can assert on exact values rather than "something non-empty".
    """
    a = t * 0.35 + phase
    return [
        round(v, 4)
        for v in (
            0.40 + 0.05 * math.sin(a),
            0.00 + 0.05 * math.cos(a),
            0.25 + 0.01 * t,
            0.0,
            0.0,
            0.0,
            1.0 if t % 4 < 2 else 0.0,  # gripper: open, then closed
        )
    ]


def _action(t: int, phase: float = 0.0) -> list[float]:
    """The commanded delta that carries `_state(t)` to `_state(t + 1)`."""
    return [round(b - a, 4) for a, b in zip(_state(t, phase), _state(t + 1, phase))]


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
        steps.append(
            {
                "observation": {"action_confidence": conf, "contact_force": 1.0, "state": _state(t)},
                "action": _action(t),
                "subgoal": "reach",
            }
        )
    return _rlds("franka", "openvla-7b", instruction, steps, "fail")


# --- Motor failure: contact-force anomaly / spike (LeRobot) --------------------
def motor_failure_lerobot(task: str = "insert the peg") -> bytes:
    frames = []
    for t in range(12):
        force = 1.0 if t != 8 else 40.0  # large spike at t=8
        frames.append(
            {
                "frame_index": t,
                "observation.confidence": 0.85,
                "observation.force": force,
                "observation.state": _state(t),
                "action": _action(t),
                "subtask": "insert",
            }
        )
    return _lerobot("so100", "act", task, frames, success=False)


# --- Grounding failure: replanning / repeated sub-goal re-issue (RLDS) ---------
def grounding_failure_rlds(instruction: str = "move object to the shelf") -> bytes:
    # sub-goal thrashes: reach -> grasp -> reach -> grasp -> reach (re-issues 'reach' repeatedly)
    subgoals = ["reach", "grasp", "reach", "grasp", "reach", "grasp", "reach"]
    steps = [
        {
            "observation": {"action_confidence": 0.8, "contact_force": 1.0, "state": _state(t)},
            "action": _action(t),
            "subgoal": sg,
        }
        for t, sg in enumerate(subgoals)
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
                "observation.state": _state(t),
                "action": _action(t),
                "subtask": "reach" if t < 5 else "grasp",  # one clean transition, no re-issues
            }
        )
    return _lerobot("so100", "act", task, frames, success=False)


# --- Success episode ----------------------------------------------------------
def success_rlds(instruction: str = "pick up the red block") -> bytes:
    steps = [
        {
            "observation": {"action_confidence": 0.9, "contact_force": 1.0, "state": _state(t)},
            "action": _action(t),
            "subgoal": sg,
        }
        for t, sg in enumerate(["reach", "grasp", "lift"])
    ]
    return _rlds("franka", "openvla-7b", instruction, steps, "success")


def success_lerobot(task: str, subtasks: list[str] | None = None) -> bytes:
    subtasks = subtasks or ["reach", "grasp", "lift"]
    frames = [
        {
            "frame_index": t,
            "observation.confidence": 0.9,
            "observation.force": 1.0,
            "observation.state": _state(t),
            "action": _action(t),
            "subtask": sg,
        }
        for t, sg in enumerate(subtasks)
    ]
    return _lerobot("so100", "act", task, frames, success=True)


# --- A cluster of similar grounding failures (same task, same signature) -------
def grounding_cluster(instruction: str, n: int = 4) -> list[bytes]:
    out = []
    for i in range(n):
        subgoals = ["reach", "grasp", "reach", "grasp", "reach"] + (["grasp"] if i % 2 else [])
        steps = [
            {
                "observation": {"action_confidence": 0.82, "contact_force": 1.0, "state": _state(t, phase=i)},
                "action": _action(t, phase=i),
                "subgoal": sg,
            }
            for t, sg in enumerate(subgoals)
        ]
        out.append(_rlds("franka", "openvla-7b", instruction, steps, "fail"))
    return out
