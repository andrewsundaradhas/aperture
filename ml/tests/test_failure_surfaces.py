"""The failure-surface labels are constructed, so what makes them trustworthy is that each
construction really produces the condition it claims. These tests pin that down, plus the
split-scoping and class balance the held-out report depends on."""

from __future__ import annotations

import numpy as np
import pytest

from training import failure_surfaces as fs


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def frame(fake_dataset) -> np.ndarray:
    return np.asarray(fake_dataset.frames[0])


# --- masks ----------------------------------------------------------------------------------


def test_cube_mask_finds_the_cube_and_only_the_cube(frame):
    mask = fs.cube_mask(frame)
    assert mask.sum() == 64, "the 8x8 synthetic cube, exactly"
    assert not fs.table_mask(frame)[mask].any(), "cube pixels must not read as table"


def test_table_mask_excludes_the_cube_and_the_black_border(frame):
    table = fs.table_mask(frame)
    assert table.sum() > 0
    assert not table[fs.cube_mask(frame)].any()
    assert not table[0, 0], "the black border is not table"


def test_bbox_is_none_when_nothing_is_masked():
    assert fs._bbox(np.zeros((10, 10), dtype=bool)) is None


# --- perception: the visual channel failed ---------------------------------------------------


@pytest.mark.parametrize("variant", fs.PERCEPTION_VARIANTS)
def test_every_perception_variant_actually_degrades_the_image(frame, rng, variant):
    out, reported = fs.degrade_perception(frame, rng, variant)
    assert reported == variant
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not np.array_equal(out, frame), f"{variant} left the frame untouched"


def test_perception_variants_stay_in_range(frame, rng):
    """Exposure variants scale pixels well past the byte range; clipping must hold."""
    for variant in fs.PERCEPTION_VARIANTS:
        out, _ = fs.degrade_perception(frame, rng, variant)
        assert out.min() >= 0 and out.max() <= 255


def test_unknown_perception_variant_is_rejected(frame, rng):
    with pytest.raises(ValueError, match="unknown perception variant"):
        fs.degrade_perception(frame, rng, "not_a_variant")


def test_perception_variant_is_sampled_when_unspecified(frame, rng):
    _, variant = fs.degrade_perception(frame, rng)
    assert variant in fs.PERCEPTION_VARIANTS


# --- grounding: the referent is ambiguous ----------------------------------------------------


@pytest.mark.parametrize("variant", fs.GROUNDING_VARIANTS)
def test_grounding_adds_extra_candidate_referents(frame, rng, variant):
    result = fs.ambiguate_grounding(frame, rng, variant)
    assert result is not None
    out, reported = result
    assert reported == variant
    # The whole point: more cube-coloured pixels than the single real referent.
    assert fs.cube_mask(out).sum() > fs.cube_mask(frame).sum()


def test_grounding_leaves_the_original_cube_in_place(frame, rng):
    out, _ = fs.ambiguate_grounding(frame, rng, "identical_distractors")
    original = fs.cube_mask(frame)
    assert fs.cube_mask(out)[original].all(), "the real referent must survive"


def test_grounding_does_not_degrade_image_quality(frame, rng):
    """Grounding samples must be visually clean — otherwise they carry the perception signal
    too and the head learns the wrong distinction."""
    out, _ = fs.ambiguate_grounding(frame, rng, "identical_distractors")
    untouched = ~(fs.cube_mask(out) | fs.cube_mask(frame))
    np.testing.assert_array_equal(out[untouched], frame[untouched])


def test_grounding_returns_none_when_no_cube_is_visible(rng):
    """A frame whose referent is occluded cannot illustrate referential ambiguity; the caller
    relies on None to go pick a different frame."""
    table_only = np.full((84, 84, 3), 150, dtype=np.uint8)  # table grey, no cube anywhere
    assert fs.cube_mask(table_only).sum() == 0
    assert fs.ambiguate_grounding(table_only, rng) is None


# --- motor: actuation failed, invisibly ------------------------------------------------------


def test_anomaly_score_finds_the_jerk_spike(fake_dataset):
    scores = fs.actuation_anomaly_score(fake_dataset.actions, fake_dataset.episode_index)
    assert scores.shape == (len(fake_dataset.actions),)
    # The conftest steps every action by +0.9 from the midpoint of each episode, so the single
    # frame where that step lands is the one with real command jerk.
    per_ep = int(fake_dataset.frame_index.max()) + 1
    spike = fake_dataset.frame_index == per_ep // 2
    assert scores[spike].mean() > scores[~spike].mean()


def test_anomaly_score_does_not_leak_jerk_across_episode_boundaries(fake_dataset):
    """The first frame of an episode has no previous command, so its jerk term must be zero —
    otherwise every episode start looks like a motor failure."""
    scores_first = fs.actuation_anomaly_score(
        fake_dataset.actions, fake_dataset.episode_index
    )
    starts = np.flatnonzero(fake_dataset.frame_index == 0)
    # Rebuild the jerk term alone to assert it is zero at every episode start.
    jerk = np.zeros(len(fake_dataset.actions))
    diff = np.linalg.norm(np.diff(fake_dataset.actions, axis=0), axis=1)
    same = fake_dataset.episode_index[1:] == fake_dataset.episode_index[:-1]
    jerk[1:] = np.where(same, diff, 0.0)
    assert not jerk[starts].any()
    assert scores_first.shape == jerk.shape


# --- dataset assembly ------------------------------------------------------------------------


def test_build_failure_dataset_is_balanced_and_well_formed(fake_dataset):
    train, _ = fake_dataset.split_episodes(0.25, seed=0)
    images, labels, variants = fs.build_failure_dataset(fake_dataset, train, n_per_class=20, seed=0)

    assert images.shape == (60, 84, 84, 3) and images.dtype == np.uint8
    assert labels.shape == (60,) and labels.dtype == np.int64
    assert len(variants) == 60
    counts = np.bincount(labels, minlength=len(fs.SURFACES))
    assert (counts == 20).all(), f"classes must be balanced, got {counts}"


def test_build_failure_dataset_respects_the_episode_split(fake_dataset):
    """Samples must be drawn only from the requested side of the split — this is what makes the
    held-out report in metrics.json mean anything."""
    train, val = fake_dataset.split_episodes(0.25, seed=0)
    motor_idx = fs.SURFACE_TO_IDX["motor"]

    # Motor samples are untouched real frames, so they can be matched back to source rows.
    images, labels, _ = fs.build_failure_dataset(fake_dataset, val, n_per_class=10, seed=1)
    val_frames = {fake_dataset.frames[r].tobytes() for r in fake_dataset.rows_for(val)}
    for img, label in zip(images, labels):
        if label == motor_idx:
            assert img.tobytes() in val_frames, "a motor sample came from outside the split"


def test_build_failure_dataset_rejects_an_empty_split(fake_dataset):
    with pytest.raises(ValueError, match="no rows"):
        fs.build_failure_dataset(fake_dataset, np.array([], dtype=np.int64), n_per_class=4)


def test_variants_are_recorded_for_every_sample(fake_dataset):
    """`metrics.json` breaks accuracy down per variant; every sample needs a usable tag."""
    train, _ = fake_dataset.split_episodes(0.25, seed=0)
    _, labels, variants = fs.build_failure_dataset(fake_dataset, train, n_per_class=12, seed=0)
    known = set(fs.PERCEPTION_VARIANTS) | set(fs.GROUNDING_VARIANTS) | {"actuation_anomaly"}
    assert set(variants) <= known
    for label, variant in zip(labels, variants):
        if label == fs.SURFACE_TO_IDX["motor"]:
            assert variant == "actuation_anomaly"


def test_surface_indices_match_the_api_runtime():
    """`SURFACES` here is the class-index order the checkpoint is trained with; the API decodes
    predictions with its own copy. If these ever diverge, every learned label is silently wrong."""
    from aperture.ml.runtime import SURFACES as RUNTIME_SURFACES

    assert fs.SURFACES == RUNTIME_SURFACES


def test_perception_variants_can_be_restricted(fake_dataset):
    """Holding a corruption out of training is what makes the generalization check possible."""
    train, _ = fake_dataset.split_episodes(0.25, seed=0)
    _, labels, variants = fs.build_failure_dataset(
        fake_dataset, train, n_per_class=12, seed=0, perception_variants=("blur",)
    )
    perception = [v for v, y in zip(variants, labels) if y == fs.SURFACE_TO_IDX["perception"]]
    assert perception and set(perception) == {"blur"}


def test_restricting_variants_leaves_other_classes_alone(fake_dataset):
    train, _ = fake_dataset.split_episodes(0.25, seed=0)
    _, labels, _ = fs.build_failure_dataset(
        fake_dataset, train, n_per_class=12, seed=0, perception_variants=("occlusion",)
    )
    assert np.bincount(labels, minlength=3).tolist() == [12, 12, 12]
