"""Build a labeled perception / grounding / motor dataset from real robot frames.

Why this module exists
----------------------
`FailureHead` predicts Aperture's failure *surface* from a single frame's pooled visual
embedding. No public robotics dataset carries that label, which is why the original training
notebook fell back to a placeholder sample (random image, constant `"perception"`) and produced
a checkpoint that was architecturally valid but predictively meaningless.

This module replaces that placeholder with **programmatic supervision over real frames**. Each
class is realised as the visual condition that actually distinguishes it, on real robot imagery:

    perception  the visual channel itself failed — blur, sensor noise, bad exposure, occlusion,
                colour loss. The scene is intact; the image of it is not.
    grounding   the image is clean but the referent is ambiguous — several equally plausible
                candidate objects for "the cube", so the instruction cannot be tied to one.
    motor       the image is clean and the referent unambiguous; what went wrong is actuation,
                which is invisible to the camera. Sampled from timesteps where the *real* action
                trace is anomalous (large jerk / saturated commands).

This is weak supervision, not human-labeled fleet failures — see `ml/models/README.md`. It is
nonetheless a real, learnable, held-out-measurable task, and it is the honest ceiling on what a
single frame can tell you: whether the visual channel is at fault, whether the scene is
referentially ambiguous, or neither.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

SURFACES = ["perception", "grounding", "motor"]
SURFACE_TO_IDX = {s: i for i, s in enumerate(SURFACES)}

PERCEPTION_VARIANTS = ("blur", "sensor_noise", "underexposure", "overexposure", "occlusion", "colour_loss")
GROUNDING_VARIANTS = ("identical_distractors", "recoloured_distractors")


# --- Scene masks --------------------------------------------------------------------------
# The xarm_lift scene is a white arm on a grey table against black, with one teal cube as the
# only strongly-coloured object. That makes the referent cheap to isolate without a detector.


def cube_mask(img: np.ndarray) -> np.ndarray:
    """Pixels belonging to the teal cube: green and blue both clearly above red."""
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > r + 12) & (b > r + 5)


def table_mask(img: np.ndarray) -> np.ndarray:
    """Table surface: bright but not blown out, and unsaturated (so it excludes the cube)."""
    a = img.astype(np.int16)
    sat = a.max(-1) - a.min(-1)
    bright = a.mean(-1)
    return (sat < 18) & (bright > 120) & (bright < 235)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# --- Perception: the visual channel failed --------------------------------------------------


def degrade_perception(img: np.ndarray, rng: np.random.Generator, variant: str | None = None) -> tuple[np.ndarray, str]:
    variant = variant or str(rng.choice(PERCEPTION_VARIANTS))
    out = img.astype(np.float32)

    if variant == "blur":
        radius = float(rng.uniform(1.5, 4.0))
        out = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    elif variant == "sensor_noise":
        out = out + rng.normal(0.0, float(rng.uniform(22.0, 60.0)), size=out.shape)
    elif variant == "underexposure":
        out = out * float(rng.uniform(0.12, 0.34))
    elif variant == "overexposure":
        out = out * float(rng.uniform(1.9, 3.2))
    elif variant == "occlusion":
        h, w = img.shape[:2]
        ph, pw = (int(h * rng.uniform(0.35, 0.6)), int(w * rng.uniform(0.35, 0.6)))
        y0, x0 = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
        out[y0 : y0 + ph, x0 : x0 + pw] = float(rng.uniform(0, 60))
    elif variant == "colour_loss":
        grey = out.mean(-1, keepdims=True)
        # Collapse toward grey, then apply a channel cast — a stuck white balance / dead channel.
        out = grey * (1.0 - 0.25) + out * 0.25
        out = out * rng.uniform(0.6, 1.25, size=(1, 1, 3))
    else:
        raise ValueError(f"unknown perception variant {variant!r}")

    return np.clip(out, 0, 255).astype(np.uint8), variant


# --- Grounding: the referent is ambiguous ---------------------------------------------------


def _paste_positions(img: np.ndarray, n: int, patch_h: int, patch_w: int,
                     avoid: tuple[int, int, int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Top-left corners on the table where a `patch_h x patch_w` copy fits and misses `avoid`."""
    h, w = img.shape[:2]
    ys, xs = np.nonzero(table_mask(img))
    if ys.size == 0:
        return []
    ay0, ay1, ax0, ax1 = avoid
    out: list[tuple[int, int]] = []
    for idx in rng.permutation(ys.size):
        y0 = int(ys[idx]) - patch_h // 2
        x0 = int(xs[idx]) - patch_w // 2
        if y0 < 0 or x0 < 0 or y0 + patch_h > h or x0 + patch_w > w:
            continue
        # Reject anything overlapping the real cube (with a margin) or an earlier copy.
        if not (y0 + patch_h < ay0 - 2 or y0 > ay1 + 2 or x0 + patch_w < ax0 - 2 or x0 > ax1 + 2):
            continue
        if any(abs(y0 - py) < patch_h and abs(x0 - px) < patch_w for py, px in out):
            continue
        out.append((y0, x0))
        if len(out) == n:
            break
    return out


def ambiguate_grounding(img: np.ndarray, rng: np.random.Generator,
                        variant: str | None = None) -> tuple[np.ndarray, str] | None:
    """Clone the cube into other plausible table positions so "the cube" has several referents.

    Returns None when the cube is not visible in this frame (occluded by the gripper), which is
    the caller's signal to pick a different frame — a frame with no visible referent cannot
    illustrate referential ambiguity.
    """
    mask = cube_mask(img)
    box = _bbox(mask)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    ph, pw = y1 - y0 + 1, x1 - x0 + 1
    if ph < 3 or pw < 3:
        return None

    variant = variant or str(rng.choice(GROUNDING_VARIANTS))
    patch = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    patch_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    n_copies = int(rng.integers(2, 4))
    positions = _paste_positions(img, n_copies, ph, pw, box, rng)
    if not positions:
        return None

    out = img.copy()
    for i, (py, px) in enumerate(positions):
        copy = patch.copy()
        if variant == "recoloured_distractors":
            # Permuting channels re-hues the cube while preserving its shading, so each
            # distractor reads as a differently-coloured cube rather than a pasted blob.
            perm = [(0, 2, 1), (1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)][i % 5]
            copy[patch_mask] = copy[patch_mask][:, perm]
        region = out[py : py + ph, px : px + pw]
        region[patch_mask] = copy[patch_mask]

    return out, variant


# --- Motor: actuation went wrong, and the camera cannot see it ------------------------------


def actuation_anomaly_score(actions: np.ndarray, episode_index: np.ndarray) -> np.ndarray:
    """Per-row actuation anomaly: command jerk plus how saturated the command is.

    Jerk is computed within an episode (the first frame of each episode scores 0, since there is
    no previous command to compare against). Both terms are real signals from the logged action
    trace — this is what picks *which* clean frames carry the motor label.
    """
    jerk = np.zeros(len(actions), dtype=np.float32)
    diff = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    same_episode = episode_index[1:] == episode_index[:-1]
    jerk[1:] = np.where(same_episode, diff, 0.0)

    saturation = (np.abs(actions) > 0.98).mean(axis=1).astype(np.float32)

    def _z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() or 1.0)

    return _z(jerk) + _z(saturation)


# --- Dataset assembly -----------------------------------------------------------------------


def build_failure_dataset(
    dataset,
    episodes: np.ndarray,
    n_per_class: int,
    seed: int = 0,
    motor_quantile: float = 0.6,
    perception_variants: tuple[str, ...] = PERCEPTION_VARIANTS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return `(images uint8 [N,H,W,3], labels int64 [N], variants)` balanced over the surfaces.

    `episodes` scopes every sample to one side of an episode-level split, so train and val never
    share a source frame.
    """
    rng = np.random.default_rng(seed)
    rows = dataset.rows_for(episodes)
    if rows.size == 0:
        raise ValueError("no rows for the requested episodes")

    scores = actuation_anomaly_score(dataset.actions, dataset.episode_index)[rows]
    anomalous = rows[scores >= np.quantile(scores, motor_quantile)]

    images: list[np.ndarray] = []
    labels: list[int] = []
    variants: list[str] = []

    # motor — real frames, untouched, at genuinely anomalous actuation timesteps.
    picks = rng.choice(anomalous, size=n_per_class, replace=anomalous.size < n_per_class)
    for row in picks:
        images.append(np.asarray(dataset.frames[row]))
        labels.append(SURFACE_TO_IDX["motor"])
        variants.append("actuation_anomaly")

    # perception — real frames with the visual channel degraded. `perception_variants` narrows
    # the set, which is what lets a caller hold one corruption out of training and test on it.
    picks = rng.choice(rows, size=n_per_class, replace=rows.size < n_per_class)
    for i, row in enumerate(picks):
        variant = perception_variants[i % len(perception_variants)]
        img, variant = degrade_perception(np.asarray(dataset.frames[row]), rng, variant)
        images.append(img)
        labels.append(SURFACE_TO_IDX["perception"])
        variants.append(variant)

    # grounding — clean frames made referentially ambiguous. Frames whose cube is hidden are
    # skipped, so draw from a shuffled pool until the quota is met.
    pool = rng.permutation(rows)
    made = 0
    cursor = 0
    while made < n_per_class:
        if cursor >= pool.size:  # exhausted the split; reshuffle for another pass
            pool = rng.permutation(rows)
            cursor = 0
        row = pool[cursor]
        cursor += 1
        variant = GROUNDING_VARIANTS[made % len(GROUNDING_VARIANTS)]
        result = ambiguate_grounding(np.asarray(dataset.frames[row]), rng, variant)
        if result is None:
            continue
        img, variant = result
        images.append(img)
        labels.append(SURFACE_TO_IDX["grounding"])
        variants.append(variant)
        made += 1

    order = rng.permutation(len(images))
    return (
        np.stack([images[i] for i in order]),
        np.asarray([labels[i] for i in order], dtype=np.int64),
        [variants[i] for i in order],
    )
