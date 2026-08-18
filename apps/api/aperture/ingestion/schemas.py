"""Pydantic schemas: the single internal episode representation, plus API request/response
shapes. RLDS and LeRobot inputs are both normalized into `NormalizedEpisode` so every
downstream layer is format-agnostic (per PRD: Aperture must work across embodiments/policies).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NormalizedFrame(BaseModel):
    t: int
    action_confidence: float | None = None
    contact_force: float | None = None
    subgoal: str | None = None  # policy's current sub-goal token, used for replanning signal


class NormalizedEpisode(BaseModel):
    """Source-format-agnostic episode. Both RLDS and LeRobot parse into exactly this."""

    source_format: str  # rlds|lerobot
    embodiment_type: str
    policy_name: str
    instruction: str | None = None
    outcome: str  # success|fail
    started_at: datetime | None = None
    frames: list[NormalizedFrame] = Field(default_factory=list)
    # Raw bytes get stored to object storage; this holds the original payload for the blob.
    raw_blob: bytes | None = None
    raw_blob_name: str | None = None


# ---- API I/O -----------------------------------------------------------------


class FrameOut(BaseModel):
    t: int
    action_confidence: float | None
    contact_force: float | None
    subgoal: str | None


class EpisodeOut(BaseModel):
    id: str
    robot_id: str
    org_id: str
    started_at: datetime
    outcome: str
    source_format: str
    instruction: str | None
    rlds_uri: str | None
    frames: list[FrameOut]

    model_config = {"from_attributes": True}


class EpisodeSummary(BaseModel):
    id: str
    robot_id: str
    started_at: datetime
    outcome: str
    source_format: str
    instruction: str | None
    surface: str | None = None
    classification_confidence: float | None = None

    model_config = {"from_attributes": True}


class UploadResult(BaseModel):
    episode_ids: list[str]
    count: int
