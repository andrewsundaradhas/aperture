"""Core data model — the embodiment-agnostic schema.

Deliberately no robot-specific columns: Aperture must work across OpenVLA, GR00T-based,
and proprietary policies. IDs are string UUIDs so the same models run on SQLite locally
and Supabase Postgres in production. `failure_clusters.embedding` is stored as JSON here;
in production it becomes a pgvector column (see infra/supabase/migrations).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aperture.core.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    tier: Mapped[str] = mapped_column(String, default="pilot")  # pilot|growth|enterprise

    robots: Mapped[list["Robot"]] = relationship(back_populates="org", cascade="all, delete-orphan")


class Robot(Base):
    __tablename__ = "robots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    embodiment_type: Mapped[str] = mapped_column(String)  # e.g. "franka", "so100", "sim"
    policy_name: Mapped[str] = mapped_column(String)

    org: Mapped[Organization] = relationship(back_populates="robots")
    episodes: Mapped[list["Episode"]] = relationship(back_populates="robot")


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    robot_id: Mapped[str] = mapped_column(ForeignKey("robots.id"), index=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    outcome: Mapped[str] = mapped_column(String)  # success|fail
    source_format: Mapped[str] = mapped_column(String)  # rlds|lerobot
    instruction: Mapped[str | None] = mapped_column(String, nullable=True)
    rlds_uri: Mapped[str | None] = mapped_column(String, nullable=True)  # raw blob in R2/local

    robot: Mapped[Robot] = relationship(back_populates="episodes")
    frames: Mapped[list["EpisodeFrame"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", order_by="EpisodeFrame.t"
    )
    classification: Mapped["FailureClassification | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )
    attribution: Mapped["Attribution | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )


class EpisodeFrame(Base):
    __tablename__ = "episode_frames"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), index=True)
    t: Mapped[int] = mapped_column(Integer)  # timestep index
    action_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    contact_force: Mapped[float | None] = mapped_column(Float, nullable=True)
    subgoal: Mapped[str | None] = mapped_column(String, nullable=True)  # for replanning signal

    episode: Mapped[Episode] = relationship(back_populates="frames")


class FailureClassification(Base):
    __tablename__ = "failure_classifications"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), index=True, unique=True)
    surface: Mapped[str] = mapped_column(String)  # perception|grounding|motor
    method: Mapped[str] = mapped_column(String, default="heuristic")  # heuristic|learned
    confidence: Mapped[float] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSON, default=dict)  # per-heuristic scores

    episode: Mapped[Episode] = relationship(back_populates="classification")


class Attribution(Base):
    __tablename__ = "attributions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), index=True, unique=True)
    attention_map_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    counterfactual_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    episode: Mapped[Episode] = relationship(back_populates="attribution")


class AttributionJob(Base):
    """A minimal polling table (not a full queue) for the GPU attention-rollout job.

    The API enqueues a row (status=queued); the Colab notebook in ml/notebooks polls for
    queued rows, runs attention rollout on a free GPU, writes the heatmap to R2, and marks
    the row done with the resulting attention_map_uri. See ml/notebooks/attention_rollout.ipynb.
    """

    __tablename__ = "attribution_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), index=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|done|failed
    attention_map_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class FailureCluster(Base):
    __tablename__ = "failure_clusters"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    label: Mapped[str] = mapped_column(String)
    dominant_surface: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding: Mapped[list] = mapped_column(JSON, default=list)  # centroid; pgvector in prod
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    members: Mapped[list["ClusterEpisode"]] = relationship(
        back_populates="cluster", cascade="all, delete-orphan"
    )


class ClusterEpisode(Base):
    __tablename__ = "cluster_episodes"
    __table_args__ = (UniqueConstraint("cluster_id", "episode_id", name="uq_cluster_episode"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("failure_clusters.id"), index=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), index=True)

    cluster: Mapped[FailureCluster] = relationship(back_populates="members")


class DatasetExport(Base):
    __tablename__ = "dataset_exports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("failure_clusters.id"), index=True)
    format: Mapped[str] = mapped_column(String)  # rlds|lerobot
    export_uri: Mapped[str] = mapped_column(String)
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class VerificationRun(Base):
    __tablename__ = "verification_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("failure_clusters.id"), index=True)
    pre_success_rate: Mapped[float] = mapped_column(Float)
    post_success_rate: Mapped[float] = mapped_column(Float)
    pre_n: Mapped[int] = mapped_column(Integer, default=0)
    post_n: Mapped[int] = mapped_column(Integer, default=0)
    verified_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
