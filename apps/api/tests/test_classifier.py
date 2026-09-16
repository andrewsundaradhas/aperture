"""Phase 2: the three obvious cases classify correctly; the ambiguous case is low-confidence."""

from __future__ import annotations

from aperture import fixtures
from aperture.evaluation.classifier import classify_frames
from aperture.ingestion.normalize import normalize


def _classify(raw: bytes, name: str):
    ne = normalize(raw, name)
    return classify_frames(ne.frames)


def test_perception_failure():
    c = _classify(fixtures.perception_failure_rlds(), "p.rlds.json")
    assert c.surface == "perception"
    assert c.confidence > 0.4


def test_motor_failure():
    c = _classify(fixtures.motor_failure_lerobot(), "m.lerobot.json")
    assert c.surface == "motor"
    assert c.confidence > 0.4


def test_grounding_failure():
    c = _classify(fixtures.grounding_failure_rlds(), "g.rlds.json")
    assert c.surface == "grounding"
    assert c.confidence > 0.4


def test_ambiguous_is_low_confidence():
    c = _classify(fixtures.ambiguous_lerobot(), "a.lerobot.json")
    # Not a false-certain verdict: confidence stays low even though a surface is picked.
    assert c.confidence < 0.4


def test_tie_breaks_toward_motor():
    from aperture.ingestion.schemas import NormalizedFrame

    # Craft frames with no discriminating signal -> all heuristics ~0 -> tie -> motor wins.
    frames = [NormalizedFrame(t=i) for i in range(6)]
    c = classify_frames(frames)
    assert c.surface == "motor"


def test_classify_endpoint(client, auth):
    up = client.post(
        "/v1/episodes/upload",
        files=[("files", ("p.rlds.json", fixtures.perception_failure_rlds(), "application/json"))],
        headers=auth,
    )
    eid = up.json()["episode_ids"][0]
    r = client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["surface"] == "perception"


# --- regressions the evaluation set exposed (see docs/classifier_eval.md) ---------------------


def test_a_gradual_confidence_decline_is_detected():
    """A trailing-window baseline drifts down with a slow decline and never sees it. This cost
    perception recall 0.407 before the baseline was anchored to the episode peak."""
    import numpy as np

    from aperture.evaluation.heuristics import action_confidence_collapse

    gradual = list(np.linspace(0.9, 0.3, 30))
    assert action_confidence_collapse(gradual).confidence > 0.3


def test_a_uniformly_low_confidence_trace_does_not_fire():
    """The property the peak anchor must not break: no peak to fall from means no collapse."""
    from aperture.evaluation.heuristics import action_confidence_collapse

    assert action_confidence_collapse([0.3] * 30).confidence < 0.05


def test_missing_subgoals_infer_grounding_rather_than_defaulting_to_motor():
    """The largest error source in the benchmark: an absent channel was tie-broken to motor."""
    from aperture.evaluation.classifier import classify_frames
    from aperture.ingestion.schemas import NormalizedFrame

    # Healthy confidence, healthy force, and no sub-goal annotations at all.
    frames = [
        NormalizedFrame(t=t, action_confidence=0.9, contact_force=1.0 + 0.01 * (t % 3), subgoal=None)
        for t in range(20)
    ]
    result = classify_frames(frames)
    assert result.surface == "grounding"
    assert "_inferred_by_elimination" in result.details


def test_an_inferred_verdict_is_low_confidence():
    """Inference from absence must not read as a finding."""
    from aperture.evaluation.classifier import classify_frames
    from aperture.ingestion.schemas import NormalizedFrame

    frames = [
        NormalizedFrame(t=t, action_confidence=0.9, contact_force=1.0 + 0.01 * (t % 3), subgoal=None)
        for t in range(20)
    ]
    assert classify_frames(frames).confidence < 0.25


def test_elimination_does_not_override_a_real_signal():
    """A present, firing heuristic must still win over an absent channel."""
    from aperture.evaluation.classifier import classify_frames
    from aperture.ingestion.schemas import NormalizedFrame

    # Force spikes hard (motor), and sub-goals are missing.
    forces = [1.0] * 10 + [60.0] + [1.0] * 9
    frames = [
        NormalizedFrame(t=t, action_confidence=0.9, contact_force=forces[t], subgoal=None)
        for t in range(20)
    ]
    result = classify_frames(frames)
    assert result.surface == "motor"
    assert "_inferred_by_elimination" not in result.details


def test_two_missing_channels_do_not_trigger_elimination():
    """Elimination needs exactly one unreadable channel — two is genuinely ambiguous."""
    from aperture.evaluation.classifier import classify_frames
    from aperture.ingestion.schemas import NormalizedFrame

    frames = [
        NormalizedFrame(t=t, action_confidence=0.9, contact_force=None, subgoal=None)
        for t in range(20)
    ]
    assert "_inferred_by_elimination" not in classify_frames(frames).details
