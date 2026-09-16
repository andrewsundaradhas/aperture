"""Combine the three heuristics into a single failure classification.

Rule: the highest-confidence surface wins. When no signal is strongly present the result is
deliberately low-confidence rather than false-certain — an ambiguous episode should read as
ambiguous.

**Unobservable channels.** Each heuristic reads exactly one signal, and real fleet logs drop
sensors: an episode with no sub-goal annotations gives the grounding heuristic nothing to read,
so grounding can neither be argued for nor ruled out. Tie-breaking such an episode toward motor
was the single largest error source measured in `docs/classifier_eval.md` — 32 of 40
grounding-mistaken-for-motor cases were episodes whose sub-goal channel was simply absent.

So when every signal that *is* observable comes back quiet, and exactly one surface's signal is
missing, that missing surface is the answer: the channels we can see are healthy, so the failure
is in the one we cannot. It is the same reasoning that makes `motor` the default when all three
are present and quiet — actuation failure is the surface a camera cannot see — generalised to
whichever channel happens to be unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from aperture.evaluation.heuristics import (
    HeuristicVerdict,
    action_confidence_collapse,
    contact_force_anomaly,
    replanning_frequency,
)
from aperture.ingestion.schemas import NormalizedFrame

# Tie-break priority: earlier = preferred when confidences are equal. Motor is most actionable.
_TIE_PRIORITY = {"motor": 0, "perception": 1, "grounding": 2}
_TIE_EPS = 1e-9
# Below this, a heuristic is treated as having found nothing at all.
_QUIET = 0.05
# Confidence attached to a verdict reached by elimination rather than by positive evidence. Low
# on purpose: the dashboard renders anything under 0.005 as "inconclusive", and this should read
# as a weak inference, not a finding.
_ELIMINATION_CONFIDENCE = 0.15


@dataclass
class Classification:
    surface: str
    confidence: float
    method: str
    details: dict


def classify_frames(frames: list[NormalizedFrame]) -> Classification:
    confidences = [f.action_confidence for f in frames]
    forces = [f.contact_force for f in frames]
    subgoals = [f.subgoal for f in frames]

    verdicts: list[HeuristicVerdict] = [
        action_confidence_collapse(confidences),
        replanning_frequency(subgoals),
        contact_force_anomaly(forces),
    ]

    # Winner: max confidence; ties prefer a surface whose signal was actually readable, then
    # fall back to _TIE_PRIORITY.
    winner = min(
        verdicts,
        key=lambda v: (-round(v.confidence, 9), not v.signal_present, _TIE_PRIORITY[v.surface]),
    )

    # Elimination: every readable channel is quiet and exactly one is unreadable.
    observable = [v for v in verdicts if v.signal_present]
    unobservable = [v for v in verdicts if not v.signal_present]
    eliminated = None
    if len(unobservable) == 1 and observable and all(v.confidence < _QUIET for v in observable):
        eliminated = unobservable[0]
        winner = eliminated

    details = {
        v.surface: {
            "confidence": round(v.confidence, 4),
            "signal_present": v.signal_present,
            "evidence": v.evidence,
        }
        for v in verdicts
    }
    # If the runner-up is within epsilon, note the contest so the UI can flag ambiguity.
    ranked = sorted(verdicts, key=lambda v: v.confidence, reverse=True)
    margin = ranked[0].confidence - ranked[1].confidence if len(ranked) > 1 else ranked[0].confidence
    details["_margin"] = round(margin, 4)

    confidence = winner.confidence
    if eliminated is not None:
        confidence = _ELIMINATION_CONFIDENCE
        details["_inferred_by_elimination"] = {
            "surface": winner.surface,
            "reason": (
                f"No {winner.surface} signal was present in this episode, and every signal that "
                f"was present came back quiet."
            ),
        }

    return Classification(
        surface=winner.surface,
        confidence=round(confidence, 4),
        method="heuristic",
        details=details,
    )
