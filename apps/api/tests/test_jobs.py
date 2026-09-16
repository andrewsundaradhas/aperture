"""The background job queue: claiming, retries, isolation, and the two handlers.

Why it exists: cluster recompute is O(all failed episodes) with an HDBSCAN fit, and ingestion
decodes and stores every frame. Both ran inline in the request and would time out on a real
fleet's data.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from aperture.core.models import Job, Organization
from aperture.jobs import queue
from aperture.jobs.handlers import HANDLERS, PermanentJobError
from aperture.jobs.worker import drain


@pytest.fixture
def org(db_session) -> Organization:
    return db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()


@pytest.fixture
def other_org(db_session) -> Organization:
    return db_session.execute(
        select(Organization).where(Organization.slug == "other-org")
    ).scalar_one()


@pytest.fixture(autouse=True)
def _restore_handlers():
    original = dict(HANDLERS)
    yield
    HANDLERS.clear()
    HANDLERS.update(original)


@pytest.fixture(autouse=True)
def _empty_queue(db_session):
    """Start every test with an empty queue.

    The suite shares one session-scoped database, so jobs another module enqueued would
    otherwise be counted by `drain()` and handed to `claim_next` here.
    """
    db_session.query(Job).delete()
    db_session.commit()
    yield
    db_session.query(Job).delete()
    db_session.commit()


# --- enqueue / claim -------------------------------------------------------------------------


def test_enqueue_creates_a_queued_job(db_session, org):
    job = queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    assert job.status == "queued" and job.attempts == 0 and job.org_id == org.id


def test_unknown_kind_is_rejected_at_enqueue(db_session, org):
    """Fail where the caller can see it, not later inside a worker."""
    with pytest.raises(ValueError, match="unknown job kind"):
        queue.enqueue(db_session, org, "not_a_real_kind")


def test_claim_marks_running_and_counts_the_attempt(db_session, org):
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    claimed = queue.claim_next(db_session)
    assert claimed is not None and claimed.status == "running" and claimed.attempts == 1


def test_a_job_is_only_claimed_once(db_session, org):
    """The property that makes two workers safe: the second claim must not get the same row."""
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()

    first = queue.claim_next(db_session)
    second = queue.claim_next(db_session)
    assert first is not None
    assert second is None, "a running job was handed out a second time"


def test_claim_is_fifo(db_session, org):
    a = queue.enqueue(db_session, org, "cluster_recompute", {"n": 1})
    db_session.commit()
    b = queue.enqueue(db_session, org, "cluster_recompute", {"n": 2})
    db_session.commit()
    assert queue.claim_next(db_session).id == a.id
    assert queue.claim_next(db_session).id == b.id


def test_claim_can_filter_by_kind(db_session, org):
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    assert queue.claim_next(db_session, kinds=["episode_ingest"]) is None
    assert queue.claim_next(db_session, kinds=["cluster_recompute"]) is not None


def test_empty_queue_claims_nothing(db_session):
    assert queue.claim_next(db_session) is None


# --- execution and failure handling ------------------------------------------------------------


def test_a_successful_job_records_its_result(db_session, org):
    HANDLERS["cluster_recompute"] = lambda db, job: {"ok": True}
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()

    job = queue.run_once(db_session)
    assert job.status == "done" and job.result == {"ok": True}
    assert job.finished_at is not None and job.error is None


def test_a_transient_failure_is_retried_then_given_up_on(db_session, org):
    """A flaky dependency should get another go; an endlessly failing job must not pin the queue."""
    calls = []

    def always_fails(db, job):
        calls.append(1)
        raise RuntimeError("database went away")

    HANDLERS["cluster_recompute"] = always_fails
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()

    for _ in range(queue.MAX_ATTEMPTS):
        job = queue.run_once(db_session)

    assert len(calls) == queue.MAX_ATTEMPTS
    assert job.status == "failed"
    assert "database went away" in job.error
    # And it is not handed out again.
    assert queue.run_once(db_session) is None


def test_a_permanent_failure_is_not_retried(db_session, org):
    """Malformed input fails identically every time — retrying just wastes the worker."""
    calls = []

    def bad_input(db, job):
        calls.append(1)
        raise PermanentJobError("not a LeRobot dataset")

    HANDLERS["episode_ingest"] = bad_input
    queue.enqueue(db_session, org, "episode_ingest", {"uri": "local://x"})
    db_session.commit()

    job = queue.run_once(db_session)
    assert job.status == "failed" and job.attempts == 1
    assert queue.run_once(db_session) is None
    assert len(calls) == 1


def test_a_failing_handler_does_not_leave_partial_writes(db_session, org):
    """The handler must not own the transaction, or a crash mid-way persists half a dataset."""
    from aperture.core.models import Robot

    def writes_then_fails(db, job):
        db.add(Robot(org_id=job.org_id, embodiment_type="ghost", policy_name="ghost"))
        db.flush()
        raise RuntimeError("boom")

    HANDLERS["cluster_recompute"] = writes_then_fails
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    queue.run_once(db_session)

    ghosts = db_session.execute(select(Robot).where(Robot.embodiment_type == "ghost")).scalars().all()
    assert not ghosts, "a failed job left rows behind"


def test_a_missing_handler_fails_the_job_rather_than_looping(db_session, org):
    queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    del HANDLERS["cluster_recompute"]

    job = queue.run_once(db_session)
    assert job.status == "failed" and "no handler" in job.error


# --- the worker --------------------------------------------------------------------------------


def test_drain_runs_everything_queued(db_session, org):
    HANDLERS["cluster_recompute"] = lambda db, job: {"ok": True}
    for _ in range(3):
        queue.enqueue(db_session, org, "cluster_recompute")
        db_session.commit()

    assert drain() == 3
    assert drain() == 0


# --- the real handlers -------------------------------------------------------------------------


def test_cluster_recompute_handler_produces_clusters(client, auth, db_session, org):
    from aperture import fixtures

    for i, raw in enumerate(fixtures.grounding_cluster("queue the cable", n=4)):
        res = client.post(
            "/v1/episodes/upload", headers=auth,
            files=[("files", (f"q{i}.rlds.json", raw, "application/json"))],
        )
        for eid in res.json()["episode_ids"]:
            client.post(f"/v1/episodes/{eid}/classify", headers=auth)

    job = queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()
    done = queue.run_once(db_session)

    assert done.status == "done", done.error
    assert done.result["cluster_count"] >= 1
    assert done.result["episode_count"] >= 1


# --- isolation ----------------------------------------------------------------------------------


def test_an_org_cannot_see_another_orgs_jobs(client, db_session, org, other_org):
    job = queue.enqueue(db_session, org, "cluster_recompute")
    db_session.commit()

    assert client.get(f"/v1/jobs/{job.id}", headers={"X-API-Key": "demo-key"}).status_code == 200
    assert client.get(f"/v1/jobs/{job.id}", headers={"X-API-Key": "other-key"}).status_code == 404

    listed = client.get("/v1/jobs", headers={"X-API-Key": "other-key"}).json()
    assert job.id not in {j["id"] for j in listed}


def test_jobs_endpoint_requires_authentication(client):
    assert client.get("/v1/jobs").status_code == 401
