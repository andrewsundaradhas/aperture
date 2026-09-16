"""Real LeRobot v3 ingestion: parquet tables + an mp4, the layout the Hub actually serves.

Skipped when `pyarrow`/`av` are absent (the base install), since building a genuine dataset to
read requires writing one. The reader itself is shared with `ml/training` — `ml/tests` cover it
against the real `lerobot/xarm_lift_medium` download; these cover the *ingestion* half.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import select

from aperture.core.models import Episode, Organization
from aperture.datasets.export import build_export
from aperture.ingestion.normalize import normalize_many
from aperture.ingestion.service import persist_episode

pa = pytest.importorskip("pyarrow", reason="pyarrow is only in the training/dataset extra")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av", reason="av is only in the training/dataset extra")
np = pytest.importorskip("numpy")

N_EPISODES = 3
PER_EPISODE = 4
ACTION_DIM = 4
SIZE = 64
TASK = "Pick up the cube and lift it."


def _write_video(path: Path, n_frames: int) -> None:
    """A real mp4 whose frames decode back in row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=15)
    stream.width, stream.height, stream.pix_fmt = SIZE, SIZE, "yuv420p"
    for i in range(n_frames):
        img = np.full((SIZE, SIZE, 3), (i * 7) % 256, dtype=np.uint8)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
    container.mux(stream.encode())
    container.close()


@pytest.fixture(scope="module")
def lerobot_dataset(tmp_path_factory) -> Path:
    """A minimal but genuine LeRobot v3.0 dataset directory."""
    root = tmp_path_factory.mktemp("lerobot_v3")
    total = N_EPISODES * PER_EPISODE

    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 15, "total_frames": total}))

    episode_index, frame_index, actions, states, rewards = [], [], [], [], []
    for ep in range(N_EPISODES):
        for t in range(PER_EPISODE):
            episode_index.append(ep)
            frame_index.append(t)
            actions.append([float(ep * 100 + t * 10 + j) for j in range(ACTION_DIM)])
            states.append([float(-(ep * 100 + t * 10 + j)) for j in range(ACTION_DIM)])
            # Only episode 0 ends successfully.
            rewards.append(1.0 if (ep == 0 and t == PER_EPISODE - 1) else 0.0)

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "action": actions,
                "observation.state": states,
                "next.reward": rewards,
            }
        ),
        data_dir / "file-000.parquet",
    )

    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"episode_index": list(range(N_EPISODES)), "tasks": [[TASK]] * N_EPISODES}),
        ep_dir / "file-000.parquet",
    )

    _write_video(root / "videos" / "observation.image" / "chunk-000" / "file-000.mp4", total)
    return root


@pytest.fixture(scope="module")
def dataset_zip(lerobot_dataset) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path in sorted(lerobot_dataset.rglob("*")):
            if path.is_file():
                zf.write(path, Path("dataset") / path.relative_to(lerobot_dataset))
    return buf.getvalue()


def test_archive_yields_every_episode(dataset_zip):
    """One upload, many episodes — the number of files says nothing about the episode count."""
    episodes = normalize_many(dataset_zip, "xarm.zip")
    assert len(episodes) == N_EPISODES
    assert all(len(e.frames) == PER_EPISODE for e in episodes)


def test_actions_and_states_survive_the_real_reader(dataset_zip):
    episodes = normalize_many(dataset_zip, "xarm.zip")
    for ep, episode in enumerate(episodes):
        for t, frame in enumerate(episode.frames):
            assert frame.action == [float(ep * 100 + t * 10 + j) for j in range(ACTION_DIM)]
            assert frame.state == [float(-(ep * 100 + t * 10 + j)) for j in range(ACTION_DIM)]


def test_instruction_comes_from_the_episode_metadata(dataset_zip):
    assert all(e.instruction == TASK for e in normalize_many(dataset_zip, "xarm.zip"))


def test_outcome_is_derived_from_the_final_reward(dataset_zip):
    outcomes = [e.outcome for e in normalize_many(dataset_zip, "xarm.zip")]
    assert outcomes == ["success", "fail", "fail"]


def test_frames_carry_decoded_images(dataset_zip):
    """The mp4 is decoded per frame, so the learned path has pixels to work with."""
    import base64

    episode = normalize_many(dataset_zip, "xarm.zip")[0]
    assert all(f.image_b64 for f in episode.frames)
    assert base64.b64decode(episode.frames[0].image_b64).startswith(b"\x89PNG")


def test_frame_cap_truncates_and_is_configurable(dataset_zip):
    episodes = normalize_many(dataset_zip, "xarm.zip", max_frames_per_episode=2)
    assert all(len(e.frames) == 2 for e in episodes)


def test_round_trip_to_a_trainable_export(dataset_zip, db_session):
    """Real LeRobot v3 in, trainable dataset out — the whole point of the format work."""
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()

    stored = []
    for ne in normalize_many(dataset_zip, "xarm.zip"):
        stored.append(persist_episode(db_session, org, ne))
    db_session.commit()
    stored = [db_session.get(Episode, e.id) for e in stored]

    doc = json.loads(build_export(stored, "lerobot"))
    assert doc["episode_count"] == N_EPISODES
    assert doc["readiness"]["trainable"] is True
    assert doc["readiness"]["frames_with_action"] == N_EPISODES * PER_EPISODE
    assert doc["readiness"]["frames_with_image"] == N_EPISODES * PER_EPISODE

    frames = doc["episodes"][0]["data"]["frames"]
    assert [f["action"] for f in frames] == [
        [float(t * 10 + j) for j in range(ACTION_DIM)] for t in range(PER_EPISODE)
    ]
    assert all(f["observation.image_url"] for f in frames)
