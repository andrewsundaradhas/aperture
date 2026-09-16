"""Real RLDS / LeRobot v3 ingestion, end to end.

The product claim these protect: a customer uploads the format their fleet already produces,
and what comes back out of a dataset export is trainable. Both halves have to hold — parsing
actions that the export then drops is worth nothing.
"""

from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from sqlalchemy import select

from aperture.core.models import Episode, Organization
from aperture.datasets.export import build_export, serialize_rlds
from aperture.ingestion.archive import ArchiveError, unpack
from aperture.ingestion.normalize import IngestionError, normalize_many
from aperture.ingestion.service import persist_episode

from tests import _tfrecord_writer as w

ARM_DOF = 7
N_STEPS = 4


def _rlds_tfrecord(n_episodes: int = 2) -> bytes:
    """A real TFRecord stream of RLDS episodes, in the flattened layout TFDS emits."""
    payloads = []
    for ep in range(n_episodes):
        actions = [float(ep * 100 + t * 10 + j) for t in range(N_STEPS) for j in range(ARM_DOF)]
        states = [float(-(ep * 100 + t * 10 + j)) for t in range(N_STEPS) for j in range(ARM_DOF)]
        payloads.append(
            w.example(
                {
                    "steps/action": w.float_list(actions),
                    "steps/observation/state": w.float_list(states),
                    "steps/observation/image": w.bytes_list([f"IMG{ep}{t}".encode() for t in range(N_STEPS)]),
                    "steps/language_instruction": w.bytes_list([b"stack the blue cube"] * N_STEPS),
                    "steps/reward": w.float_list([0.0] * (N_STEPS - 1) + [1.0 if ep == 0 else 0.0]),
                }
            )
        )
    return w.tfrecord(payloads)


# --- RLDS TFRecord --------------------------------------------------------------------------


def test_tfrecord_yields_one_episode_per_record():
    episodes = normalize_many(_rlds_tfrecord(3), "shard.tfrecord")
    assert len(episodes) == 3
    assert all(e.source_format == "rlds" for e in episodes)


def test_tfrecord_actions_and_states_survive_parsing():
    """The flattened per-step vectors must be re-chunked to the right width, in order."""
    episode = normalize_many(_rlds_tfrecord(1), "shard.tfrecord")[0]
    assert len(episode.frames) == N_STEPS
    for t, frame in enumerate(episode.frames):
        assert frame.action == [float(t * 10 + j) for j in range(ARM_DOF)]
        assert frame.state == [float(-(t * 10 + j)) for j in range(ARM_DOF)]


def test_tfrecord_carries_instruction_and_outcome():
    episodes = normalize_many(_rlds_tfrecord(2), "shard.tfrecord")
    assert episodes[0].instruction == "stack the blue cube"
    assert episodes[0].outcome == "success"  # final reward 1.0
    assert episodes[1].outcome == "fail"     # final reward 0.0


def test_tfrecord_images_are_carried_through():
    episode = normalize_many(_rlds_tfrecord(1), "shard.tfrecord")[0]
    import base64

    assert base64.b64decode(episode.frames[0].image_b64) == b"IMG00"


def test_corrupt_tfrecord_is_a_clean_4xx_not_a_crash():
    with pytest.raises(IngestionError):
        normalize_many(b"\xff" * 64, "broken.tfrecord")


# --- the full round trip --------------------------------------------------------------------


def test_tfrecord_round_trips_to_a_trainable_export(db_session):
    """Ingest a real RLDS stream, persist it, export it — actions must survive every hop."""
    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()

    episodes = normalize_many(_rlds_tfrecord(1), "shard.tfrecord")
    stored = persist_episode(db_session, org, episodes[0])
    db_session.commit()
    stored = db_session.get(Episode, stored.id)

    # Survived the database.
    assert len(stored.frames) == N_STEPS
    assert stored.frames[0].action == [float(j) for j in range(ARM_DOF)]

    # Survived the export.
    exported = serialize_rlds(stored)
    assert [s["action"] for s in exported["steps"]] == [
        [float(t * 10 + j) for j in range(ARM_DOF)] for t in range(N_STEPS)
    ]

    doc = json.loads(build_export([stored], "rlds"))
    assert doc["readiness"]["trainable"] is True
    assert doc["readiness"]["frames_with_action"] == N_STEPS


def test_tfrecord_inside_an_archive_is_detected(db_session):
    """Uploads arrive as archives; the format is sniffed from the contents, not the filename."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("export/shard-00000.tfrecord", _rlds_tfrecord(2))

    episodes = normalize_many(buf.getvalue(), "rlds_export.zip")
    assert len(episodes) == 2
    assert episodes[0].frames[0].action


# --- archive safety ---------------------------------------------------------------------------


def test_zip_traversal_is_rejected(tmp_path):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escaped.txt", b"nope")
    with pytest.raises(ArchiveError, match="escapes the extraction root"):
        unpack(buf.getvalue(), "evil.zip", tmp_path / "out")


def test_tar_traversal_is_rejected(tmp_path):
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 4
        tf.addfile(info, BytesIO(b"nope"))
    with pytest.raises(ArchiveError, match="escapes the extraction root"):
        unpack(buf.getvalue(), "evil.tar", tmp_path / "out")


def test_tar_symlink_member_is_rejected(tmp_path):
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ArchiveError, match="link member"):
        unpack(buf.getvalue(), "evil.tar", tmp_path / "out")


def test_archive_without_a_known_dataset_is_rejected():
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"nothing useful here")
    with pytest.raises(IngestionError, match="neither a LeRobot v3 dataset"):
        normalize_many(buf.getvalue(), "junk.zip")


# --- legacy JSON still works -------------------------------------------------------------------


def test_legacy_json_still_ingests_as_a_single_episode():
    from aperture import fixtures

    episodes = normalize_many(fixtures.perception_failure_rlds(), "e.rlds.json")
    assert len(episodes) == 1
    assert episodes[0].frames[0].action


def test_unknown_extension_is_rejected():
    with pytest.raises(IngestionError, match="Cannot determine format"):
        normalize_many(b"{}", "mystery.bin")


# --- Loop closure accepts the same real formats as ingestion ---------------------------------


def test_verify_accepts_rlds_tfrecord_batch(client, auth):
    """A retrain batch arrives as the format the fleet produces, not as legacy JSON.

    Regression: `verify` used single-episode `normalize`, so every `.tfrecord` / archive the
    upload endpoint accepts was rejected here with "cannot determine format" — the one endpoint
    that proves a fix worked could not read real data.
    """
    from aperture import fixtures

    instr = "stack the blue cube"
    for i, raw in enumerate(fixtures.grounding_cluster(instr, n=4)):
        r = client.post(
            "/v1/episodes/upload",
            files=[("files", (f"vt{i}.rlds.json", raw, "application/json"))],
            headers=auth,
        )
        assert r.status_code == 201, r.text
        client.post(f"/v1/episodes/{r.json()['episode_ids'][0]}/classify", headers=auth)

    client.post("/v1/clusters/recompute?wait=true", headers=auth)
    cid = client.get("/v1/clusters", headers=auth).json()[0]["id"]

    r = client.post(
        f"/v1/clusters/{cid}/verify",
        files=[("files", ("batch.tfrecord", _rlds_tfrecord(3), "application/octet-stream"))],
        headers=auth,
    )
    assert r.status_code == 200, r.text
    # One .tfrecord carrying three episodes must count as three, not one and not zero.
    assert r.json()["post_n"] == 3


def test_verify_still_rejects_a_malformed_tfrecord(client, auth):
    """Widening the reader must not turn a corrupt upload into a 500."""
    from aperture import fixtures

    eid = client.post(
        "/v1/episodes/upload",
        files=[("files", ("vm.rlds.json", fixtures.grounding_failure_rlds(), "application/json"))],
        headers=auth,
    ).json()["episode_ids"][0]
    client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    client.post("/v1/clusters/recompute?wait=true", headers=auth)
    cid = client.get("/v1/clusters", headers=auth).json()[0]["id"]

    r = client.post(
        f"/v1/clusters/{cid}/verify",
        files=[("files", ("junk.tfrecord", b"not a tfrecord at all" * 8, "application/octet-stream"))],
        headers=auth,
    )
    assert r.status_code == 422, r.text
