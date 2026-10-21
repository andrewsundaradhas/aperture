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


# Images kept per episode, evenly spaced (first and last included). Every frame's state and
# action are always kept; images are what make an upload heavy, and the product reads only a
# few per episode (first frame for classification, up to 8 for attention). 0 = keep every one.
DEFAULT_IMAGES_PER_EPISODE = 32
# Stored images are downscaled so the longest side is at most this. The learned models read
# 224x224, and full-resolution frames from real robot cameras are 10-40x larger for no gain.
IMAGE_MAX_SIDE = 256


def _camera_key(info: dict, root: "Path | None" = None) -> str | None:
    """The camera to read. Multi-camera datasets (most real robots) get their first camera —
    decoding every camera into one stream breaks the one-frame-per-row alignment. Falls back to
    the folders under videos/ when meta/info.json does not list its features."""
    feats = info.get("features", {})
    visual = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") in ("video", "image")]
    if not visual and root is not None and (root / "videos").is_dir():
        visual = sorted(p.name for p in (root / "videos").iterdir() if p.is_dir())
    if not visual:
        return None
    return "observation.image" if "observation.image" in visual else visual[0]


def _sample_rows(rows, k: int | None) -> set[int]:
    import numpy as np

    if not k or k <= 0 or rows.size <= k:
        return set(int(r) for r in rows)
    idx = np.unique(np.linspace(0, rows.size - 1, k).round().astype(int))
    return set(int(r) for r in rows[idx])


def _encode_image(image) -> str | None:
    """A PIL image (or HWC uint8 array) -> base64 PNG, downscaled to IMAGE_MAX_SIDE."""
    import base64
    import io

    Image = _pillow_or_warn()
    if Image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def _stream_video_images(files: list, n_rows: int, wanted: set[int]) -> dict[int, str]:
    """Decode the camera's videos frame by frame, keeping (downscaled) only `wanted` rows.

    Streaming is what keeps memory flat: the previous reader decoded every frame of every video
    at full resolution into one array, which for a real robot dataset is many gigabytes.
    """
    import av

    out: dict[int, str] = {}
    row = 0
    for path in files:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if row in wanted:
                    out[row] = _encode_image(frame.to_image())
                row += 1
    if row != n_rows:
        raise ValueError(
            f"{row} video frames but {n_rows} parquet rows — the one-frame-per-row alignment "
            "this reader relies on does not hold for this dataset."
        )
    return out


def to_normalized_episodes(
    root,
    *,
    embodiment_type: str = "unknown",
    policy_name: str = "unknown",
    max_frames_per_episode: int | None = None,
    success_reward_threshold: float = SUCCESS_REWARD_THRESHOLD,
    include_images: bool = True,
    images_per_episode: int | None = DEFAULT_IMAGES_PER_EPISODE,
    default_outcome: str = "success",
) -> list:
    """Every episode in a LeRobot v3 dataset, as `NormalizedEpisode`s.

    Outcome, most trustworthy signal first: the dataset's own `next.success` flag (an episode
    succeeded if it ever reached success); else the final `next.reward` against the threshold
    (LeRobot rewards are usually shaped and continuous); else `default_outcome` — a dataset with
    neither is typically teleoperated demonstrations. Uploaders can override per batch.

    `max_frames_per_episode` truncates long episodes and logs each truncation — an episode
    silently cut short would teach a fine-tune that the task ends early. None keeps every frame.
    """
    import logging

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aperture.ingestion.schemas import NormalizedEpisode, NormalizedFrame

    logger = logging.getLogger(__name__)
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    if embodiment_type == "unknown":
        # LeRobot v3 records the robot in meta/info.json; use it unless the caller named one.
        robot_type = info.get("robot_type")
        if robot_type and robot_type != "unknown":
            embodiment_type = str(robot_type)

    camera = _camera_key(info, root)
    camera_is_image_column = bool(camera) and info.get("features", {}).get(camera, {}).get("dtype") == "image"
    wanted_cols = ["episode_index", "frame_index", "action", "observation.state", "next.reward", "next.success"]
    if camera_is_image_column:
        wanted_cols.append(camera)
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError("no data/*.parquet files in the dataset.")
    present = set(pq.read_schema(data_files[0]).names)
    if "episode_index" not in present:
        raise ValueError("data parquet has no episode_index column.")
    table = pa.concat_tables(
        [pq.read_table(f, columns=[c for c in wanted_cols if c in present]) for f in data_files]
    ).to_pydict()
    n_rows = len(table["episode_index"])
    episode_index = np.asarray(table["episode_index"], dtype=np.int64)
    frame_index = np.asarray(table.get("frame_index") or range(n_rows), dtype=np.int64)
    actions = table.get("action")
    states = table.get("observation.state")
    rewards = np.asarray(table["next.reward"], dtype=np.float32) if "next.reward" in table else None
    successes = np.asarray(table["next.success"], dtype=bool) if "next.success" in table else None

    episode_task: dict[int, str] = {}
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if ep_files:
        ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pydict()
        for ep, tasks in zip(ep_table.get("episode_index", []), ep_table.get("tasks", [])):
            episode_task[int(ep)] = (tasks[0] if tasks else "") or ""

    ids = np.unique(episode_index)
    rows_by_episode = {int(e): np.flatnonzero(episode_index == e) for e in ids}
    kept_by_episode = {}
    for e, rows in rows_by_episode.items():
        kept = rows
        if max_frames_per_episode is not None and rows.size > max_frames_per_episode:
            kept = rows[:max_frames_per_episode]
            logger.warning(
                "episode %s truncated from %d to %d frames by max_frames_per_episode",
                e, rows.size, max_frames_per_episode,
            )
        kept_by_episode[e] = kept

    images: dict[int, str] = {}
    if include_images and camera:
        wanted: set[int] = set()
        for kept in kept_by_episode.values():
            wanted |= _sample_rows(kept, images_per_episode)
        if camera_is_image_column:
            import io

            Image = _pillow_or_warn()
            for r in wanted:
                cell = table[camera][r]
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                if raw and Image is not None:
                    images[r] = _encode_image(Image.open(io.BytesIO(raw)))
        else:
            video_files = sorted((root / "videos" / camera).rglob("*.mp4"))
            if video_files:
                images = _stream_video_images(video_files, n_rows, wanted)

    episodes = []
    for e, rows in rows_by_episode.items():
        kept = kept_by_episode[e]
        frames = [
            NormalizedFrame(
                t=int(frame_index[r]),
                action=[float(v) for v in actions[r]] if actions is not None else None,
                state=[float(v) for v in states[r]] if states is not None else None,
                image_b64=images.get(int(r)),
            )
            for r in kept
        ]
        if successes is not None:
            outcome, source = ("success" if successes[rows].any() else "fail"), "flag"
        elif rewards is not None:
            ok = float(rewards[rows[-1]]) >= success_reward_threshold
            outcome, source = ("success" if ok else "fail"), "reward"
        else:
            outcome, source = default_outcome, "none"
        episodes.append(
            NormalizedEpisode(
                source_format="lerobot",
                embodiment_type=embodiment_type,
                policy_name=policy_name,
                instruction=episode_task.get(e) or None,
                outcome=outcome,
                outcome_source=source,
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
