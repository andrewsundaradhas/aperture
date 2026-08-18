"""Phase 1: RLDS + LeRobot normalize into identically-shaped episodes; upload round-trips."""

from __future__ import annotations

from aperture import fixtures
from aperture.ingestion.normalize import normalize


def test_both_formats_normalize_to_same_shape():
    rlds = normalize(fixtures.perception_failure_rlds(), "e.rlds.json")
    lerobot = normalize(fixtures.motor_failure_lerobot(), "e.lerobot.json")

    assert rlds.source_format == "rlds"
    assert lerobot.source_format == "lerobot"
    # Same internal representation: both have frames with the same signal fields populated.
    for ne in (rlds, lerobot):
        assert ne.outcome == "fail"
        assert ne.frames
        f = ne.frames[0]
        assert set(f.model_dump().keys()) == {"t", "action_confidence", "contact_force", "subgoal"}


def test_malformed_file_raises_clean_error():
    import pytest

    from aperture.ingestion.normalize import IngestionError

    with pytest.raises(IngestionError):
        normalize(b"{not json", "broken.rlds.json")
    with pytest.raises(IngestionError):
        normalize(b'{"steps": []}', "empty.rlds.json")


def test_upload_and_get_roundtrip(client, auth):
    files = [
        ("files", ("a.rlds.json", fixtures.perception_failure_rlds(), "application/json")),
        ("files", ("b.lerobot.json", fixtures.motor_failure_lerobot(), "application/json")),
    ]
    r = client.post("/v1/episodes/upload", files=files, headers=auth)
    assert r.status_code == 201, r.text
    ids = r.json()["episode_ids"]
    assert len(ids) == 2

    for eid in ids:
        g = client.get(f"/v1/episodes/{eid}", headers=auth)
        assert g.status_code == 200
        body = g.json()
        assert body["frames"]
        assert body["rlds_uri"]  # raw blob stored


def test_malformed_upload_returns_4xx_not_500(client, auth):
    files = [("files", ("bad.rlds.json", b"{oops", "application/json"))]
    r = client.post("/v1/episodes/upload", files=files, headers=auth)
    assert r.status_code == 422, r.text


def test_missing_api_key_rejected(client):
    files = [("files", ("a.rlds.json", fixtures.success_rlds(), "application/json"))]
    r = client.post("/v1/episodes/upload", files=files)
    assert r.status_code == 401
