"""Presigned upload: get a URL, PUT the bytes straight to storage, queue the parse.

The point is that a gigabyte LeRobot archive never enters the API process. `POST /v1/episodes/
upload` reads the whole file into memory to parse it, which is fine for a JSON episode and
impossible for real fleet video.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from aperture.core.models import Episode, Job, Organization
from aperture.jobs import queue

from tests import _tfrecord_writer as w


@pytest.fixture(autouse=True)
def _empty_queue(db_session):
    db_session.query(Job).delete()
    db_session.commit()
    yield


def _tfrecord(n_steps: int = 4, dof: int = 7) -> bytes:
    return w.tfrecord([
        w.example({
            "steps/action": w.float_list([float(t * 10 + j) for t in range(n_steps) for j in range(dof)]),
            "steps/observation/state": w.float_list([float(t) for t in range(n_steps * dof)]),
            "steps/language_instruction": w.bytes_list([b"lift the bracket"] * n_steps),
            "steps/reward": w.float_list([0.0] * n_steps),
        })
    ])


# --- presign ------------------------------------------------------------------------------------


def test_presign_returns_an_upload_target(client, auth):
    body = client.post("/v1/uploads/presign", headers=auth, json={"filename": "fleet.zip"}).json()
    assert body["method"] == "PUT"
    assert body["upload_url"]
    assert body["key"].startswith("acme-robotics/uploads/")
    assert body["key"].endswith("fleet.zip"), "the extension selects the reader, so it must survive"


def test_presign_keys_are_unique_per_call(client, auth):
    keys = {
        client.post("/v1/uploads/presign", headers=auth, json={"filename": "a.zip"}).json()["key"]
        for _ in range(5)
    }
    assert len(keys) == 5, "two uploads of the same filename must not collide"


def test_presign_strips_path_traversal_from_the_filename(client, auth):
    """A filename becomes a key, which becomes a path in the bucket."""
    body = client.post(
        "/v1/uploads/presign", headers=auth, json={"filename": "../../../etc/passwd"}
    ).json()
    assert ".." not in body["key"]
    assert body["key"].startswith("acme-robotics/uploads/")
    assert body["key"].endswith("passwd")


def test_presign_requires_authentication(client):
    assert client.post("/v1/uploads/presign", json={"filename": "x.zip"}).status_code == 401


# --- the direct PUT (local stand-in for a presigned URL) ------------------------------------------


def test_bytes_can_be_put_to_the_upload_url(client, auth):
    presigned = client.post("/v1/uploads/presign", headers=auth, json={"filename": "s.tfrecord"}).json()
    put = client.put(presigned["upload_url"], headers=auth, content=_tfrecord())
    assert put.status_code == 201
    assert put.json()["bytes"] > 0


def test_an_org_cannot_put_outside_its_own_prefix(client):
    """A presigned URL carries its own scoped grant; this local route has to enforce the same."""
    other = {"X-API-Key": "other-key"}
    res = client.put("/v1/blobs/acme-robotics/uploads/evil/x.bin", headers=other, content=b"x")
    assert res.status_code == 403


def test_put_requires_authentication(client):
    assert client.put("/v1/blobs/acme-robotics/uploads/x.bin", content=b"x").status_code == 401


# --- ingest -------------------------------------------------------------------------------------


def test_ingest_of_a_missing_key_is_rejected(client, auth):
    res = client.post("/v1/uploads/ingest", headers=auth, json={"key": "acme-robotics/uploads/nope/x.zip"})
    assert res.status_code == 404


def test_an_org_cannot_ingest_another_orgs_key(client, auth):
    """Otherwise one tenant could name another's object and pull it into their own org."""
    presigned = client.post("/v1/uploads/presign", headers=auth, json={"filename": "s.tfrecord"}).json()
    client.put(presigned["upload_url"], headers=auth, content=_tfrecord())

    res = client.post(
        "/v1/uploads/ingest", headers={"X-API-Key": "other-key"}, json={"key": presigned["key"]}
    )
    assert res.status_code == 403


def test_full_flow_presign_put_ingest_poll(client, auth, db_session):
    """The whole point, end to end."""
    presigned = client.post(
        "/v1/uploads/presign", headers=auth, json={"filename": "shard.tfrecord"}
    ).json()

    assert client.put(presigned["upload_url"], headers=auth, content=_tfrecord()).status_code == 201

    accepted = client.post("/v1/uploads/ingest", headers=auth, json={"key": presigned["key"]})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    # The API answered without parsing anything.
    pending = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert pending["status"] in ("queued", "running")

    queue.run_once(db_session)

    finished = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert finished["status"] == "done", finished["error"]
    assert finished["result"]["count"] == 1

    episode_id = finished["result"]["episode_ids"][0]
    episode = client.get(f"/v1/episodes/{episode_id}", headers=auth).json()
    assert episode["instruction"] == "lift the bracket"
    assert len(episode["frames"]) == 4

    # And the actions survived, so the export is trainable.
    stored = db_session.get(Episode, episode_id)
    assert stored.frames[0].action == [float(j) for j in range(7)]


def test_a_malformed_upload_fails_the_job_without_retrying(client, auth, db_session):
    """Bad input is permanent — it must not consume the worker three times over."""
    presigned = client.post(
        "/v1/uploads/presign", headers=auth, json={"filename": "broken.tfrecord"}
    ).json()
    client.put(presigned["upload_url"], headers=auth, content=b"\xff" * 64)
    job_id = client.post(
        "/v1/uploads/ingest", headers=auth, json={"key": presigned["key"]}
    ).json()["job_id"]

    queue.run_once(db_session)

    body = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert body["status"] == "failed"
    assert body["attempts"] == 1, "a malformed file should not be retried"
