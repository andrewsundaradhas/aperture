"""Combine the three heuristics into a single failure classification.

Rule: the highest-confidence surface wins; ties break toward motor-failure
since it's the most actionable. When no signal is strongly present the result is deliberately
low-confidence rather than false-certain — an ambiguous episode should read as ambiguous.
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

    # Winner: max confidence, ties broken by _TIE_PRIORITY.
    winner = min(
        verdicts,
        key=lambda v: (-round(v.confidence, 9), _TIE_PRIORITY[v.surface]),
    )

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

    return Classification(
        surface=winner.surface,
        confidence=round(winner.confidence, 4),
        method="heuristic",
        details=details,
    )
