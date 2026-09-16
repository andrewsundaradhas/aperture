"""Reader for a LeRobot v3.0 dataset directory (parquet tables + one concatenated mp4).

This is the single LeRobot v3 reader in the repo: the ingestion API and `ml/training` both use
it (`ml/training/lerobot_data.py` re-exports from here), so a dataset read for training and the
same dataset read for ingestion cannot drift apart.

Deliberately does not depend on the `lerobot` package. The v3.0 layout is small and stable, and
pinning a heavyweight dataset library (plus its torchcodec/ffmpeg matrix) is the single biggest
reason the original Colab notebook could not be reproduced off Colab.

Layout consumed here:

    meta/info.json                     fps, feature shapes, totals
    meta/tasks.parquet                 task_index -> instruction text
    meta/episodes/**.parquet           per-episode row ranges + task
    data/**.parquet                    per-frame state/action/reward, one row per frame
    videos/<key>/**.mp4                every episode's frames, concatenated in row order

With a single data file and a single video file (the case for `lerobot/xarm_lift_medium`),
video decode order is exactly the parquet row order, which is what `frames[i]` relies on.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HF_DATASET_REPO = "lerobot/xarm_lift_medium"


def download_dataset(root: Path, repo_id: str = HF_DATASET_REPO) -> Path:
    """Fetch the dataset from the Hugging Face Hub into `root` (no-op if already present)."""
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(root))
    return root


def _decode_video(path: Path) -> np.ndarray:
    import av

    container = av.open(str(path))
    out = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return np.asarray(out, dtype=np.uint8)


@dataclass
class LeRobotV3Dataset:
    """One in-memory LeRobot v3.0 dataset. Frames are uint8 `[N, H, W, 3]` in row order."""

    frames: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    episode_task: dict[int, str]
    fps: float

    @classmethod
    def load(cls, root: str | Path, use_cache: bool = True) -> "LeRobotV3Dataset":
        import pyarrow as pa
        import pyarrow.parquet as pq

        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())

        data_files = sorted((root / "data").rglob("*.parquet"))
        table = pa.concat_tables([pq.read_table(f) for f in data_files]).to_pydict()
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        rewards = np.asarray(table["next.reward"], dtype=np.float32)
        episode_index = np.asarray(table["episode_index"], dtype=np.int64)
        frame_index = np.asarray(table["frame_index"], dtype=np.int64)

        ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pydict()
        episode_task = {
            int(ep): (tasks[0] if tasks else "")
            for ep, tasks in zip(ep_table["episode_index"], ep_table["tasks"])
        }

        video_files = sorted((root / "videos").rglob("*.mp4"))
        cache = root / ".cache" / "frames.npy"
        if use_cache and cache.exists():
            frames = np.load(cache, mmap_mode="r")
        else:
            frames = np.concatenate([_decode_video(v) for v in video_files])
            if use_cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, frames)

        if len(frames) != len(actions):
            raise ValueError(
                f"{len(frames)} video frames but {len(actions)} parquet rows in {root} — "
                "the frame/row alignment this reader assumes does not hold."
            )

        return cls(
            frames=frames,
            states=states,
            actions=actions,
            rewards=rewards,
            episode_index=episode_index,
            frame_index=frame_index,
            episode_task=episode_task,
            fps=float(info["fps"]),
        )

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def episode_ids(self) -> np.ndarray:
        return np.unique(self.episode_index)

    def rows_for(self, episodes: np.ndarray) -> np.ndarray:
        """Row indices belonging to any of `episodes`, in dataset order."""
        return np.flatnonzero(np.isin(self.episode_index, episodes))

    def split_episodes(self, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Episode-level train/val split. Splitting by episode (not by frame) is what keeps
        near-duplicate neighbouring frames from straddling the split and inflating val scores."""
        ids = self.episode_ids()
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(ids)
        n_val = max(1, int(round(len(ids) * val_fraction)))
        return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def pad_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    """Pad with zeros / truncate to `action_dim`, matching the trained policy's action head."""
    have = actions.shape[1]
    if have == action_dim:
        return actions
    if have > action_dim:
        return actions[:, :action_dim]
    pad = np.zeros((actions.shape[0], action_dim - have), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=1)


# --- Ingestion bridge -----------------------------------------------------------------------
#
# Everything below turns a real LeRobot v3 dataset into Aperture's `NormalizedEpisode`s. The
# reader above is shared with `ml/training`; this half is what the upload endpoint calls.


# A LeRobot reward is usually shaped and continuous rather than a success flag, so "did this
# episode succeed" has to be inferred. The same threshold the RLDS parser uses, so both formats
# agree; override per-upload when a dataset encodes success differently.
SUCCESS_REWARD_THRESHOLD = 1.0

def extract_archive(raw: bytes, filename: str, dest: "Path") -> "Path":
    """Unpack a dataset archive into `dest` and return the LeRobot v3 root inside it."""
    from aperture.ingestion.archive import unpack

    return find_dataset_root(unpack(raw, filename, dest))


def find_dataset_root(base: "Path") -> "Path":
    """The directory holding `meta/info.json`, which archives commonly nest one level deep."""
    info = sorted(base.rglob("meta/info.json"), key=lambda p: len(p.parts))
    if not info:
        raise ValueError("archive contains no meta/info.json — not a LeRobot v3 dataset.")
    return info[0].parent.parent


@functools.lru_cache(maxsize=1)
def _pillow_or_warn():
    """Pillow, or None with a single warning — frames then ingest without images rather than
    failing the whole upload, but the degradation is never silent."""
    try:
        from PIL import Image

        return Image
    except ImportError:
        logging.getLogger(__name__).warning(
            "Pillow is not installed; frames will be ingested without images and the learned "
            "path will fall back to heuristics. Install the 'datasets' extra to keep them."
        )
        return None


def _png_b64(frame) -> str | None:
    """Encode one uint8 HWC frame as base64 PNG, or None if Pillow is unavailable."""
    import base64
    import io

    Image = _pillow_or_warn()
    if Image is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_normalized_episodes(
    root,
    *,
    embodiment_type: str = "unknown",
    policy_name: str = "unknown",
    max_frames_per_episode: int | None = None,
    success_reward_threshold: float = SUCCESS_REWARD_THRESHOLD,
    include_images: bool = True,
) -> list:
    """Every episode in a LeRobot v3 dataset, as `NormalizedEpisode`s.

    `max_frames_per_episode` truncates long episodes and logs each truncation — an episode
    silently cut short would teach a fine-tune that the task ends early. None keeps every frame.
    """
    import logging

    import numpy as np

    from aperture.ingestion.schemas import NormalizedEpisode, NormalizedFrame

    logger = logging.getLogger(__name__)
    dataset = LeRobotV3Dataset.load(root, use_cache=False)

    episodes = []
    for episode_id in dataset.episode_ids():
        rows = np.flatnonzero(dataset.episode_index == episode_id)
        if rows.size == 0:
            continue

        kept = rows
        if max_frames_per_episode is not None and rows.size > max_frames_per_episode:
            kept = rows[:max_frames_per_episode]
            logger.warning(
                "episode %s truncated from %d to %d frames by max_frames_per_episode",
                episode_id, rows.size, max_frames_per_episode,
            )

        frames = [
            NormalizedFrame(
                t=int(dataset.frame_index[row]),
                action=[float(v) for v in dataset.actions[row]],
                state=[float(v) for v in dataset.states[row]],
                image_b64=_png_b64(np.asarray(dataset.frames[row])) if include_images else None,
            )
            for row in kept
        ]

        final_reward = float(dataset.rewards[rows[-1]])
        episodes.append(
            NormalizedEpisode(
                source_format="lerobot",
                embodiment_type=embodiment_type,
                policy_name=policy_name,
                instruction=dataset.episode_task.get(int(episode_id)) or None,
                outcome="success" if final_reward >= success_reward_threshold else "fail",
                frames=frames,
                # Deliberately no raw blob: the source is one dataset archive shared by every
                # episode in it, and storing a copy per episode would multiply it N times.
                raw_blob=None,
                raw_blob_name=None,
            )
        )
    return episodes


def from_archive(raw: bytes, filename: str, **kwargs) -> list:
    """`NormalizedEpisode`s from an uploaded `.zip` / `.tar.gz` LeRobot v3 dataset."""
    import shutil
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="aperture-lerobot-"))
    try:
        return to_normalized_episodes(extract_archive(raw, filename, workdir), **kwargs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
