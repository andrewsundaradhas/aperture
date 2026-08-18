"""RLDS and LeRobot parsers -> one internal representation.

Real fleet data comes in two formats the PRD names as first-class citizens:

  * RLDS (Reinforcement Learning Datasets, Open X-Embodiment): TFRecord-based, organized as
    a list of `steps`, each with `observation`, `action`, `language_instruction`, and often
    `is_terminal` / `reward`.
  * LeRobot (HuggingFace `datasets`): a columnar/tabular layout — per-frame rows sharing an
    `episode_index`, plus an episode-level `tasks` string.

Production plugs a real TFRecord reader (tensorflow-datasets) and a parquet reader (pyarrow)
into `parse_rlds` / `parse_lerobot`. For a zero-dependency, runs-anywhere MVP we accept a
portable JSON encoding of each format — the two encodings are genuinely *different shapes*,
and the point of this module is that both collapse to an identical `NormalizedEpisode`.

Both parsers derive the three signals the heuristic classifier needs:
  action_confidence, contact_force, subgoal (for replanning detection).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from aperture.ingestion.schemas import NormalizedEpisode, NormalizedFrame


class IngestionError(ValueError):
    """Raised on malformed input so the API can return a clean 4xx, not a 500."""


def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_str(v) -> str | None:
    """Coerce an optional value to a non-empty string, else None (used for base64 image data)."""
    if v is None:
        return None
    s = str(v)
    return s or None


def parse_rlds(raw: bytes, filename: str = "episode.rlds.json") -> NormalizedEpisode:
    """Parse an RLDS / Open X-Embodiment episode.

    Expected JSON shape (portable stand-in for the TFRecord schema)::

        {
          "embodiment_type": "franka",
          "policy_name": "openvla-7b",
          "steps": [
            {
              "observation": {"action_confidence": 0.9, "contact_force": 1.2},
              "action": [...],
              "language_instruction": "pick up the red block",
              "subgoal": "reach",
              "is_terminal": false,
              "reward": 0.0
            },
            ...
          ]
        }
    """
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IngestionError(f"RLDS file is not valid JSON: {e}") from e

    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise IngestionError("RLDS file has no 'steps' array.")

    instruction: str | None = None
    frames: list[NormalizedFrame] = []
    final_reward = 0.0
    for t, step in enumerate(steps):
        if not isinstance(step, dict):
            raise IngestionError(f"RLDS step {t} is not an object.")
        obs = step.get("observation", {}) or {}
        instr = step.get("language_instruction")
        if instr and instruction is None:
            instruction = str(instr)
        if "reward" in step:
            final_reward = _as_float(step.get("reward")) or final_reward
        frames.append(
            NormalizedFrame(
                t=t,
                action_confidence=_as_float(obs.get("action_confidence")),
                contact_force=_as_float(obs.get("contact_force")),
                subgoal=(str(step["subgoal"]) if step.get("subgoal") is not None else None),
                image_b64=_as_str(obs.get("image")),
            )
        )

    outcome = _resolve_outcome(doc.get("outcome"), final_reward)
    return NormalizedEpisode(
        source_format="rlds",
        embodiment_type=str(doc.get("embodiment_type", "unknown")),
        policy_name=str(doc.get("policy_name", "unknown")),
        instruction=instruction,
        outcome=outcome,
        started_at=_parse_ts(doc.get("started_at")),
        frames=frames,
        raw_blob=raw,
        raw_blob_name=filename,
    )


def parse_lerobot(raw: bytes, filename: str = "episode.lerobot.json") -> NormalizedEpisode:
    """Parse a LeRobot (HF datasets) episode.

    Expected JSON shape (portable stand-in for the parquet/HF-datasets schema)::

        {
          "meta": {"robot_type": "so100", "policy": "act", "tasks": ["pick up the red block"]},
          "frames": [
            {"frame_index": 0, "action_is_pad": false,
             "observation.confidence": 0.9, "observation.force": 1.2, "task_index": 0,
             "subtask": "reach"},
            ...
          ],
          "success": false
        }

    Note the deliberately different key names/nesting from RLDS — the normalizer maps them.
    """
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IngestionError(f"LeRobot file is not valid JSON: {e}") from e

    rows = doc.get("frames")
    if not isinstance(rows, list) or not rows:
        raise IngestionError("LeRobot file has no 'frames' array.")

    meta = doc.get("meta", {}) or {}
    tasks = meta.get("tasks") or []
    instruction = str(tasks[0]) if tasks else None

    frames: list[NormalizedFrame] = []
    for i, row in enumerate(sorted(rows, key=lambda r: r.get("frame_index", 0))):
        if not isinstance(row, dict):
            raise IngestionError(f"LeRobot frame {i} is not an object.")
        frames.append(
            NormalizedFrame(
                t=int(row.get("frame_index", i)),
                action_confidence=_as_float(row.get("observation.confidence")),
                contact_force=_as_float(row.get("observation.force")),
                subgoal=(str(row["subtask"]) if row.get("subtask") is not None else None),
                image_b64=_as_str(row.get("observation.image")),
            )
        )

    outcome = _resolve_outcome(
        "success" if doc.get("success") is True else ("fail" if doc.get("success") is False else None),
        None,
    )
    return NormalizedEpisode(
        source_format="lerobot",
        embodiment_type=str(meta.get("robot_type", "unknown")),
        policy_name=str(meta.get("policy", "unknown")),
        instruction=instruction,
        outcome=outcome,
        started_at=_parse_ts(doc.get("started_at")),
        frames=frames,
        raw_blob=raw,
        raw_blob_name=filename,
    )


def normalize(raw: bytes, filename: str) -> NormalizedEpisode:
    """Dispatch to the right parser based on the declared format in the filename.

    Convention: `*.rlds.json` -> RLDS, `*.lerobot.json` -> LeRobot. In production the format
    is also carried by the upload's declared content-type / multipart field.
    """
    lower = filename.lower()
    if ".rlds" in lower:
        return parse_rlds(raw, filename)
    if ".lerobot" in lower:
        return parse_lerobot(raw, filename)
    raise IngestionError(
        f"Cannot determine format for '{filename}'. Expected a '.rlds.json' or '.lerobot.json' file."
    )


def _resolve_outcome(explicit: str | None, final_reward: float | None) -> str:
    if explicit in ("success", "fail"):
        return explicit
    if final_reward is not None:
        return "success" if final_reward >= 1.0 else "fail"
    return "fail"


def _parse_ts(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None
