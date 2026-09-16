"""A labeled evaluation set for the failure classifier.

**Why not reuse `aperture.fixtures`.** Those episodes are engineered to trip exactly one
heuristic — a perception fixture *is* a confidence collapse. Scoring the heuristics on them
measures whether the code runs, not whether it discriminates, and would report ~100% no matter
how brittle the thresholds are.

So these episodes are generated from a **causal model of the failure** instead, and the
heuristics are never consulted while generating them:

    perception  the visual channel degrades, so the policy's confidence in what it sees falls.
                Contact force is normal — nothing is wrong with the arm.
    grounding   the instruction has several plausible referents, so the policy oscillates
                between sub-goals. It stays fairly *confident* about each candidate, and the
                arm behaves normally.
    motor       actuation fails: the force trace spikes on a collision, or flatlines when
                contact is lost. Vision and language are fine.

Three properties make it a real test rather than a restatement of the thresholds:

1. **Severity varies.** Every surface is generated across a severity range, so the set contains
   marginal failures as well as blatant ones. A classifier that only catches the blatant ones
   is supposed to score badly here.
2. **Channels are confounded.** Real failures leak across signals — a failed grasp (motor) makes
   the policy retry a sub-goal (looks like grounding) and dents its confidence (looks like
   perception). Severity controls how much leaks.
3. **Signals go missing.** Real fleet logs drop sensors. A fraction of episodes carry no force
   trace or no sub-goal annotations at all.

Deterministic given a seed, so the reported numbers are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SURFACES = ["perception", "grounding", "motor"]
SUBGOALS = ["reach", "align", "grasp", "lift", "place"]


@dataclass
class LabeledEpisode:
    """One generated episode plus the ground truth used to build it."""

    surface: str                 # the true failure surface
    severity: float              # 0 = marginal, 1 = blatant
    confidences: list[float | None]
    forces: list[float | None]
    subgoals: list[str | None]
    dropped: tuple[str, ...] = ()   # signals deliberately withheld


def _clip01(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.clip(values, 0.01, 1.0)]


def _walk(rng: np.random.Generator, n: int, start: float, drift: float, noise: float) -> np.ndarray:
    """A gently drifting signal — the baseline behaviour of a healthy channel."""
    return start + np.cumsum(rng.normal(drift, noise, size=n))


def _healthy_force(rng: np.random.Generator, n: int) -> np.ndarray:
    """Contact force for an arm that is working: a non-zero hold with sensor noise."""
    return np.abs(_walk(rng, n, start=rng.uniform(0.8, 2.5), drift=0.0, noise=0.05))


def _stable_subgoals(rng: np.random.Generator, n: int) -> list[str]:
    """A monotone plan: each sub-goal entered once, in order, no re-issues."""
    n_blocks = int(rng.integers(2, 5))
    plan = SUBGOALS[:n_blocks]
    # Distribute n frames over the blocks.
    cuts = sorted(rng.choice(np.arange(1, n), size=n_blocks - 1, replace=False)) if n_blocks > 1 else []
    out, previous = [], 0
    for block, cut in enumerate(list(cuts) + [n]):
        out.extend([plan[block]] * (cut - previous))
        previous = cut
    return out[:n]


def _thrashing_subgoals(rng: np.random.Generator, n: int, severity: float) -> list[str]:
    """A plan that keeps re-entering sub-goals it already tried — the grounding signature."""
    reissues = 1 + int(round(severity * 4))
    pattern = ["reach", "align"]
    for _ in range(reissues):
        pattern += [str(rng.choice(["reach", "align", "grasp"])), "grasp"]

    out, index = [], 0
    while len(out) < n:
        hold = int(rng.integers(1, 4))
        out.extend([pattern[index % len(pattern)]] * hold)
        index += 1
    return out[:n]


def _generate(rng: np.random.Generator, surface: str, severity: float) -> LabeledEpisode:
    n = int(rng.integers(12, 40))

    confidences = _clip01(_walk(rng, n, start=rng.uniform(0.75, 0.95), drift=-0.002, noise=0.03))
    forces = _healthy_force(rng, n)
    subgoals: list[str] = _stable_subgoals(rng, n)

    if surface == "perception":
        # Vision degrades from some point on; severity sets the depth and the abruptness.
        onset = int(rng.integers(n // 4, max(n // 4 + 1, 3 * n // 4)))
        depth = 0.25 + 0.65 * severity
        ramp = int(round((1.0 - severity) * 6))  # blatant failures collapse fast
        values = np.array(confidences)
        for i in range(onset, n):
            factor = 1.0 - depth * min(1.0, (i - onset + 1) / max(1, ramp))
            values[i] *= factor
        confidences = _clip01(values + rng.normal(0, 0.02, size=n))
        # Confound: a blind policy sometimes retries its plan.
        if rng.random() < 0.35 * severity:
            subgoals = _thrashing_subgoals(rng, n, severity * 0.4)

    elif surface == "grounding":
        # The referent is ambiguous: the plan oscillates, confidence stays respectable.
        subgoals = _thrashing_subgoals(rng, n, severity)
        values = np.array(confidences) * rng.uniform(0.9, 1.0)
        confidences = _clip01(values)
        # Confound: churning between candidates dents confidence a little.
        if rng.random() < 0.3 * severity:
            confidences = _clip01(np.array(confidences) * rng.uniform(0.75, 0.9))

    elif surface == "motor":
        if rng.random() < 0.5:
            # Collision: a force spike whose height scales with severity.
            at = int(rng.integers(2, n - 1))
            forces[at] += (4.0 + 40.0 * severity) * float(np.median(forces))
        else:
            # Lost contact: the trace goes dead flat near zero.
            at = int(rng.integers(2, n - 1))
            forces[at:] = rng.uniform(0.0, 0.02)
        # Confounds: a failed grasp is retried, and the policy's confidence dips afterwards.
        if rng.random() < 0.5 * severity:
            subgoals = _thrashing_subgoals(rng, n, severity * 0.5)
        if rng.random() < 0.4 * severity:
            values = np.array(confidences)
            values[at:] *= rng.uniform(0.55, 0.85)
            confidences = _clip01(values)
    else:
        raise ValueError(f"unknown surface {surface!r}")

    # Real logs lose sensors. Drop a channel outright on a minority of episodes.
    dropped: list[str] = []
    force_values: list[float | None] = [float(f) for f in forces]
    subgoal_values: list[str | None] = list(subgoals)
    confidence_values: list[float | None] = list(confidences)
    if rng.random() < 0.10:
        force_values = [None] * n
        dropped.append("contact_force")
    if rng.random() < 0.10:
        subgoal_values = [None] * n
        dropped.append("subgoal")
    if rng.random() < 0.05:
        confidence_values = [None] * n
        dropped.append("action_confidence")

    return LabeledEpisode(
        surface=surface,
        severity=severity,
        confidences=confidence_values,
        forces=force_values,
        subgoals=subgoal_values,
        dropped=tuple(dropped),
    )


def build_eval_set(n_per_surface: int = 400, seed: int = 0) -> list[LabeledEpisode]:
    """A balanced, severity-stratified labeled set. Balanced so the majority-class baseline is
    exactly 1/3 and every per-class number is read on equal support."""
    rng = np.random.default_rng(seed)
    episodes: list[LabeledEpisode] = []
    for surface in SURFACES:
        for i in range(n_per_surface):
            # Sweep severity uniformly rather than sampling, so each surface gets the same
            # difficulty distribution and per-band accuracy is comparable across classes.
            severity = (i + 0.5) / n_per_surface
            episodes.append(_generate(rng, surface, severity))
    rng.shuffle(episodes)
    return episodes
