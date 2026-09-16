"""The three MVP failure heuristics, as pure functions over frame signals.

Each heuristic inspects one signal and returns a `HeuristicVerdict`: the failure *surface*
it argues for and a confidence in [0, 1]. The classifier combines the three. No training
data is required, which sidesteps the "limited training data per customer" cold-start risk.

Signal -> surface mapping:

    action-confidence collapse   -> perception   (the policy stopped trusting what it sees)
    replanning frequency         -> grounding    (it keeps re-issuing the same sub-goal)
    contact-force anomaly        -> motor        (physical interaction went wrong)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Tunable thresholds (kept in one place for auditability / patent disclosure) ---------
#
# `CONF_COLLAPSE_RATIO` and `FORCE_SPIKE_Z` were fitted by sweep against the labeled evaluation
# set (`ml/eval/`), tuned on seed 0 and checked on a held-out seed 99: held-out macro F1 rose
# from 0.855 to 0.887, with a tune/held gap of ~0.015, so the values generalise rather than
# memorise that one draw. Re-derive with `make eval` after changing the generator.
CONF_BASELINE_WINDOW = 3          # frames smoothed into each confidence baseline sample
CONF_COLLAPSE_RATIO = 0.75        # a frame is "collapsed" below this fraction of the peak baseline
FORCE_SPIKE_Z = 4.0               # contact-force spike: |value - median| > Z * MAD-ish scale
FORCE_FLATLINE_EPS = 1e-3         # a force trace with less spread than this is "flat"
FORCE_FLATLINE_LEVEL = 0.1        # ...and only anomalous if it's also stuck near zero (lost contact)
REPLAN_MIN_BLOCKS = 2             # need at least this many sub-goal blocks to judge replanning


@dataclass
class HeuristicVerdict:
    surface: str            # perception|grounding|motor
    confidence: float       # 0..1
    signal_present: bool    # False when the input lacked the signal this heuristic reads
    evidence: dict = field(default_factory=dict)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def action_confidence_collapse(confidences: list[float | None]) -> HeuristicVerdict:
    """Perception. Flag a sustained drop of per-timestep confidence below its established peak.

    The baseline is the **best sustained confidence seen so far in the episode**, not a trailing
    window. A trailing window drifts down with a slow decline, so it can only ever see abrupt
    drops: a 0.9 -> 0.3 decay spread over 30 frames scored 0.000 against a rolling baseline,
    while the identical total loss applied in one step scored 0.257. Both are the policy losing
    its grip on what it sees; only one was detected, which is why perception recall was 0.407 on
    `docs/classifier_eval.md`.

    Taking a running maximum fixes that while keeping the property that mattered: an episode
    whose confidence is *uniformly* low never fires, because there is no peak to have fallen
    from — only a decline relative to this policy's own demonstrated best counts.
    """
    vals = [c for c in confidences if c is not None]
    if len(vals) < CONF_BASELINE_WINDOW + 1:
        return HeuristicVerdict("perception", 0.0, signal_present=False)

    collapsed_depth = 0.0
    collapsed_frames = 0
    peak = sum(vals[:CONF_BASELINE_WINDOW]) / CONF_BASELINE_WINDOW
    for i in range(CONF_BASELINE_WINDOW, len(vals)):
        window = sum(vals[i - CONF_BASELINE_WINDOW : i]) / CONF_BASELINE_WINDOW
        # Update the peak from the trailing window, not the raw frame, so a single noisy spike
        # cannot raise the bar for the rest of the episode.
        peak = max(peak, window)
        if peak <= 0:
            continue
        ratio = vals[i] / peak
        if ratio < CONF_COLLAPSE_RATIO:
            collapsed_frames += 1
            collapsed_depth += (CONF_COLLAPSE_RATIO - ratio) / CONF_COLLAPSE_RATIO
    n = len(vals) - CONF_BASELINE_WINDOW
    frac = collapsed_frames / n if n else 0.0
    # Confidence blends how often it collapsed with how deep the collapses were.
    depth = collapsed_depth / collapsed_frames if collapsed_frames else 0.0
    confidence = _clamp(0.5 * min(1.0, frac * 2.0) + 0.5 * depth)
    return HeuristicVerdict(
        "perception",
        confidence,
        signal_present=True,
        evidence={
            "collapsed_frames": collapsed_frames,
            "fraction": round(frac, 3),
            "peak_confidence": round(peak, 3),
        },
    )


def contact_force_anomaly(forces: list[float | None]) -> HeuristicVerdict:
    """Motor. Flag spikes or unexpected flatlines in contact-force sensor readings."""
    vals = [f for f in forces if f is not None]
    if len(vals) < 4:
        return HeuristicVerdict("motor", 0.0, signal_present=False)

    srt = sorted(vals)
    median = srt[len(srt) // 2]
    deviations = sorted(abs(v - median) for v in vals)
    mad = deviations[len(deviations) // 2] or 1e-6  # median absolute deviation, guard div0
    scale = 1.4826 * mad  # MAD -> ~std for normal data

    max_z = max(abs(v - median) / scale for v in vals)
    spike_score = _clamp((max_z - FORCE_SPIKE_Z) / FORCE_SPIKE_Z) if max_z > FORCE_SPIKE_Z else 0.0

    spread = max(vals) - min(vals)
    # A dead-flat force trace *stuck near zero* signals lost contact — anomalous. A steady
    # non-zero force (e.g. holding an object) is normal and must not be flagged.
    flatline = spread < FORCE_FLATLINE_EPS and abs(median) < FORCE_FLATLINE_LEVEL
    flatline_score = 0.6 if flatline else 0.0

    confidence = _clamp(max(spike_score, flatline_score))
    return HeuristicVerdict(
        "motor",
        confidence,
        signal_present=True,
        evidence={"max_z": round(max_z, 2), "flatline": flatline},
    )


def replanning_frequency(subgoals: list[str | None]) -> HeuristicVerdict:
    """Grounding. Flag episodes where the policy re-issues the same sub-goal repeatedly.

    We collapse the sub-goal sequence into contiguous blocks, then count how many blocks
    are *re-entries* of a sub-goal already seen earlier (A B A -> the second A is a re-issue).
    Frequent re-issuing signals the policy cannot ground the instruction stably.
    """
    seq = [s for s in subgoals if s]
    if not seq:
        return HeuristicVerdict("grounding", 0.0, signal_present=False)

    blocks: list[str] = []
    for s in seq:
        if not blocks or blocks[-1] != s:
            blocks.append(s)
    if len(blocks) < REPLAN_MIN_BLOCKS:
        return HeuristicVerdict("grounding", 0.0, signal_present=True, evidence={"blocks": len(blocks)})

    seen: set[str] = set()
    reissues = 0
    for b in blocks:
        if b in seen:
            reissues += 1
        seen.add(b)
    # Normalize re-issues against the number of transitions.
    confidence = _clamp(reissues / (len(blocks) - 1))
    return HeuristicVerdict(
        "grounding",
        confidence,
        signal_present=True,
        evidence={"blocks": len(blocks), "reissues": reissues},
    )
