"""The attribution-job queue: the contract an external GPU worker holds with the API.

The in-process learned path fills heatmaps directly, so these endpoints only matter when the
model is too big to run on the API host — which is exactly when nobody is watching them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from aperture.core.models import AttributionJob

# These tests upload and cluster nothing, but they do add episodes — so they run as `other-org`
# rather than the `acme-robotics` org the pipeline tests build their clusters in. Sharing one
# session-scoped database means an episode added here would otherwise change which cluster
# `test_pipeline` picks up.
WORKER_ORG = {"X-API-Key": "other-key"}
OUTSIDER = {"X-API-Key": "demo-key"}


@pytest.fixture
def auth() -> dict:
    return WORKER_ORG


@pytest.fixture
def episode_with_job(client, auth, db_session):
    """An uploaded episode plus the queued job its attribution run enqueued."""
    from aperture import fixtures

    res = client.post(
        "/v1/episodes/upload",
        headers=auth,
        files=[("files", ("e.rlds.json", fixtures.grounding_failure_rlds(), "application/json"))],
    )
    assert res.status_code == 201, res.text
    episode_id = res.json()["episode_ids"][0]

    client.post(f"/v1/episodes/{episode_id}/classify", headers=auth)
    client.post(f"/v1/episodes/{episode_id}/attribution", headers=auth)

    job = db_session.execute(
        select(AttributionJob)
        .where(AttributionJob.episode_id == episode_id)
        .order_by(AttributionJob.created_at.desc())
    ).scalars().first()
    assert job is not None
    # The local path completes jobs in-process; reset to queued to exercise the worker contract.
    job.status = "queued"
    job.attention_map_uri = None
    db_session.commit()
    return episode_id, job.id


def test_listing_returns_queued_jobs_with_their_instruction(client, auth, episode_with_job):
    episode_id, job_id = episode_with_job
    jobs = client.get("/v1/attribution_jobs", headers=auth).json()

    job = next(j for j in jobs if j["id"] == job_id)
    assert job["status"] == "queued"
    assert job["episode_id"] == episode_id
    # Denormalized so a worker can start from the listing alone.
    assert job["instruction"], "a worker needs the instruction to run the rollout"


def test_status_filter_narrows_the_listing(client, auth, episode_with_job):
    _, job_id = episode_with_job
    assert any(j["id"] == job_id for j in client.get("/v1/attribution_jobs?status=queued", headers=auth).json())
    assert not any(j["id"] == job_id for j in client.get("/v1/attribution_jobs?status=done", headers=auth).json())


def test_claim_moves_the_job_to_running(client, auth, episode_with_job):
    _, job_id = episode_with_job
    body = client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=auth).json()
    assert body["status"] == "running"


def test_second_claim_is_rejected(client, auth, episode_with_job):
    """Two workers must not burn GPU time on the same episode."""
    _, job_id = episode_with_job
    client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=auth)
    res = client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=auth)
    assert res.status_code == 409


def test_done_publishes_the_heatmap_onto_the_episode(client, auth, episode_with_job):
    """The result has to reach GET /attribution, or the worker's output never reaches a user."""
    episode_id, job_id = episode_with_job
    client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=auth)

    uri = "r2://aperture-blobs/rollouts/test.json"
    body = client.post(
        f"/v1/attribution_jobs/{job_id}/done", headers=auth, json={"attention_map_uri": uri}
    ).json()
    assert body["status"] == "done" and body["attention_map_uri"] == uri

    attribution = client.get(f"/v1/episodes/{episode_id}/attribution", headers=auth).json()
    assert attribution["attention_map_uri"] == uri
    assert attribution["job_status"] == "done"


def test_done_is_not_accepted_twice(client, auth, episode_with_job):
    _, job_id = episode_with_job
    payload = {"attention_map_uri": "r2://b/k.json"}
    client.post(f"/v1/attribution_jobs/{job_id}/done", headers=auth, json=payload)
    assert client.post(f"/v1/attribution_jobs/{job_id}/done", headers=auth, json=payload).status_code == 409


def test_done_requires_a_uri(client, auth, episode_with_job):
    _, job_id = episode_with_job
    assert client.post(
        f"/v1/attribution_jobs/{job_id}/done", headers=auth, json={"attention_map_uri": ""}
    ).status_code == 422


def test_failed_records_the_reason(client, auth, episode_with_job):
    """A dead worker must be distinguishable from a slow one."""
    _, job_id = episode_with_job
    client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=auth)
    body = client.post(
        f"/v1/attribution_jobs/{job_id}/failed", headers=auth, json={"error": "CUDA OOM"}
    ).json()
    assert body["status"] == "failed" and body["error"] == "CUDA OOM"


def test_another_org_can_neither_see_nor_complete_the_job(client, episode_with_job):
    """Tenant isolation on the worker path, same boundary as every other layer."""
    _, job_id = episode_with_job
    other = OUTSIDER

    assert not any(j["id"] == job_id for j in client.get("/v1/attribution_jobs", headers=other).json())
    assert client.post(f"/v1/attribution_jobs/{job_id}/claim", headers=other).status_code == 404
    assert client.post(
        f"/v1/attribution_jobs/{job_id}/done", headers=other, json={"attention_map_uri": "r2://b/k"}
    ).status_code == 404


def test_unknown_job_is_404(client, auth):
    assert client.post("/v1/attribution_jobs/does-not-exist/claim", headers=auth).status_code == 404


def test_queue_requires_authentication(client, episode_with_job):
    assert client.get("/v1/attribution_jobs").status_code == 401
