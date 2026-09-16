"""RLDS and LeRobot parsers -> one internal representation.

Real fleet data comes in two formats the PRD names as first-class citizens:

  * RLDS (Reinforcement Learning Datasets, Open X-Embodiment): TFRecord-based, organized as
    a list of `steps`, each with `observation`, `action`, `language_instruction`, and often
    `is_terminal` / `reward`.
  * LeRobot (HuggingFace `datasets`): a columnar/tabular layout — per-frame rows sharing an
    `episode_index`, plus an episode-level `tasks` string.

Two ways in, both landing on `NormalizedEpisode`:

* **Real fleet data** — a LeRobot v3 dataset archive (parquet + mp4) or an RLDS `.tfrecord`
  stream. One upload carries many episodes, so these go through `normalize_many`. Readers live
  in `lerobot_v3.py` and `rlds_tfrecord.py`; neither needs `lerobot` or `tensorflow`.
* **Portable JSON** (`*.rlds.json` / `*.lerobot.json`) — **legacy**. It predates real-format
  support and is kept so existing fixtures, the seed script, and any partner already posting it
  keep working. New integrations should send the real thing.

The parsers derive the three signals the heuristic classifier needs (action_confidence,
contact_force, subgoal) plus the `action`/`state` vectors that make a dataset export trainable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from aperture.ingestion.archive import is_archive
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


def _as_vector(v) -> list[float] | None:
    """Coerce an action/state value to a flat list of floats, else None.

    Tolerates the shapes these fields actually arrive in: a list, a bare scalar (a 1-DoF
    gripper command), or a nested list (some RLDS producers wrap the action in a batch
    dimension of 1). Anything with a non-numeric element is rejected whole rather than
    silently zero-filled — a partially-parsed action is worse than no action, because it
    would be trained on.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)]
    if not isinstance(v, (list, tuple)):
        return None
    # Unwrap a single nested row, e.g. [[0.1, 0.2, 0.3]].
    if len(v) == 1 and isinstance(v[0], (list, tuple)):
        v = v[0]
    out: list[float] = []
    for item in v:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return out or None


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
                action=_as_vector(step.get("action")),
                # RLDS producers put the proprioceptive state under the observation, and are
                # inconsistent about whether it is nested or a flat dotted key.
                state=_as_vector(obs.get("state", obs.get("observation.state"))),
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
                action=_as_vector(row.get("action")),
                state=_as_vector(row.get("observation.state")),
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
    """One episode from a legacy portable-JSON upload.

    Convention: `*.rlds.json` -> RLDS, `*.lerobot.json` -> LeRobot. Real multi-episode datasets
    go through `normalize_many`, which falls back here for these single-episode files.
    """
    lower = filename.lower()
    if ".rlds" in lower:
        return parse_rlds(raw, filename)
    if ".lerobot" in lower:
        return parse_lerobot(raw, filename)
    raise IngestionError(
        f"Cannot determine format for '{filename}'. Expected a '.rlds.json' or '.lerobot.json' "
        f"file, a '.tfrecord' stream, or a dataset archive."
    )


def normalize_many(raw: bytes, filename: str, **kwargs) -> list[NormalizedEpisode]:
    """Every episode in an upload.

    A legacy JSON file yields a one-element list; a LeRobot v3 archive or an RLDS `.tfrecord`
    yields one entry per episode in it. `kwargs` (embodiment_type, policy_name,
    max_frames_per_episode, ...) are forwarded to the real-format readers and ignored by the
    JSON path, which carries that metadata in the document itself.
    """
    lower = filename.lower()

    if lower.endswith(".tfrecord") or lower.endswith(".tfrecords"):
        from aperture.ingestion.rlds_tfrecord import TFRecordError, parse_tfrecord_episodes

        try:
            return parse_tfrecord_episodes(raw, **_reader_kwargs(kwargs, frames_key="max_frames"))
        except TFRecordError as e:
            raise IngestionError(f"{filename}: {e}") from e

    if is_archive(filename):
        return _normalize_archive(raw, filename, kwargs)

    return [normalize(raw, filename)]


def _reader_kwargs(kwargs: dict, frames_key: str) -> dict:
    """Reader kwargs, renaming the frame cap to whatever that reader calls it."""
    out = {k: v for k, v in kwargs.items() if k in ("embodiment_type", "policy_name")}
    cap = kwargs.get("max_frames_per_episode")
    if cap is not None:
        out[frames_key] = cap
    return out


def _normalize_archive(raw: bytes, filename: str, kwargs: dict) -> list[NormalizedEpisode]:
    """Unpack once, then dispatch on what the archive actually contains.

    A LeRobot v3 dataset is identified by `meta/info.json`; an RLDS export by its `.tfrecord`
    shards. Sniffing the contents beats trusting the archive's name, which says nothing about
    which format is inside.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from aperture.ingestion.archive import ArchiveError, unpack

    workdir = Path(tempfile.mkdtemp(prefix="aperture-ingest-"))
    try:
        try:
            root = unpack(raw, filename, workdir)
        except ArchiveError as e:
            raise IngestionError(f"{filename}: {e}") from e

        if next(root.rglob("meta/info.json"), None) is not None:
            from aperture.ingestion.lerobot_v3 import find_dataset_root, to_normalized_episodes

            try:
                return to_normalized_episodes(find_dataset_root(root), **kwargs)
            except ImportError as e:
                raise IngestionError(
                    f"{filename}: reading a LeRobot v3 dataset needs the 'datasets' extra "
                    f"(pip install -e '.[datasets]') — {e}"
                ) from e
            except Exception as e:
                raise IngestionError(f"{filename}: unreadable LeRobot v3 dataset: {e}") from e

        shards = sorted(p for p in root.rglob("*") if p.suffix in (".tfrecord", ".tfrecords"))
        if shards:
            from aperture.ingestion.rlds_tfrecord import TFRecordError, parse_tfrecord_episodes

            reader_kwargs = _reader_kwargs(kwargs, frames_key="max_frames")
            episodes: list[NormalizedEpisode] = []
            try:
                for shard in shards:
                    episodes.extend(parse_tfrecord_episodes(shard.read_bytes(), **reader_kwargs))
            except TFRecordError as e:
                raise IngestionError(f"{filename}: {e}") from e
            return episodes

        raise IngestionError(
            f"{filename}: archive contains neither a LeRobot v3 dataset (meta/info.json) "
            f"nor RLDS .tfrecord shards."
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
