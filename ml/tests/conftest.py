"""Put `ml/` on the path so `training.*` imports resolve the same way `python -m training.train`
does, and build the synthetic dataset the surface tests run against."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from training.lerobot_data import LeRobotV3Dataset  # noqa: E402

TABLE_GREY = 150  # bright + unsaturated -> matches table_mask
CUBE_RGB = (40, 120, 110)  # green and blue clearly above red -> matches cube_mask


def _frame(size: int, cube_yx: tuple[int, int], cube_px: int) -> np.ndarray:
    """One synthetic xarm-like frame: grey table, black border, a single teal cube."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[4:-4, 4:-4] = TABLE_GREY  # table surface, inset so there is non-table border
    y, x = cube_yx
    img[y : y + cube_px, x : x + cube_px] = CUBE_RGB
    return img


@pytest.fixture(scope="session")
def fake_dataset() -> LeRobotV3Dataset:
    """A LeRobotV3Dataset built in memory — no Hub download, no video decode.

    12 episodes x 10 frames. The cube walks across the table so its position varies, and the
    action trace carries a deliberate jerk spike in the back half of every episode.
    """
    rng = np.random.default_rng(0)
    n_eps, per_ep, size = 12, 10, 84
    frames, actions, episode_index, frame_index = [], [], [], []

    for ep in range(n_eps):
        for t in range(per_ep):
            frames.append(_frame(size, (30 + (ep % 5) * 4, 20 + t * 4), 8))
            a = rng.normal(0, 0.05, size=4)
            if t >= per_ep // 2:  # a real jerk spike the anomaly score should find
                a = a + 0.9
            actions.append(a)
            episode_index.append(ep)
            frame_index.append(t)

    n = n_eps * per_ep
    return LeRobotV3Dataset(
        frames=np.stack(frames),
        states=np.zeros((n, 4), dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.zeros(n, dtype=np.float32),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        episode_task={ep: "Pick up the cube and lift it." for ep in range(n_eps)},
        fps=15.0,
    )
