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
