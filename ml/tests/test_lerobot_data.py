"""The dataset reader's two load-bearing guarantees: action padding matches the policy's action
head, and the train/val split is episode-level (the thing that keeps val scores honest)."""

from __future__ import annotations

import numpy as np

from training.lerobot_data import pad_actions


def test_pad_actions_pads_short_actions_with_zeros():
    actions = np.arange(8, dtype=np.float32).reshape(2, 4)
    padded = pad_actions(actions, 7)
    assert padded.shape == (2, 7)
    np.testing.assert_array_equal(padded[:, :4], actions)
    assert not padded[:, 4:].any()


def test_pad_actions_truncates_long_actions():
    actions = np.arange(20, dtype=np.float32).reshape(2, 10)
    truncated = pad_actions(actions, 7)
    assert truncated.shape == (2, 7)
    np.testing.assert_array_equal(truncated, actions[:, :7])


def test_pad_actions_is_a_noop_at_the_right_width():
    actions = np.zeros((3, 7), dtype=np.float32)
    assert pad_actions(actions, 7) is actions


def test_split_is_by_episode_and_covers_everything(fake_dataset):
    train, val = fake_dataset.split_episodes(0.25, seed=0)
    assert not set(train) & set(val), "an episode may not be on both sides of the split"
    assert set(train) | set(val) == set(fake_dataset.episode_ids().tolist())
    assert len(val) == 3  # 25% of 12


def test_no_frame_is_shared_across_the_split(fake_dataset):
    """The property the episode-level split exists for: neighbouring frames in an episode are
    near-duplicates, so a frame-level split would leak them and inflate every val number."""
    train, val = fake_dataset.split_episodes(0.25, seed=0)
    train_rows = set(fake_dataset.rows_for(train).tolist())
    val_rows = set(fake_dataset.rows_for(val).tolist())
    assert not train_rows & val_rows
    assert len(train_rows) + len(val_rows) == len(fake_dataset.frames)


def test_split_is_deterministic_for_a_seed(fake_dataset):
    a = fake_dataset.split_episodes(0.25, seed=3)
    b = fake_dataset.split_episodes(0.25, seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_val_split_is_never_empty(fake_dataset):
    """A tiny fraction must still yield a val episode rather than an empty report."""
    _, val = fake_dataset.split_episodes(0.001, seed=0)
    assert len(val) >= 1
