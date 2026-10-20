"""Ingested failures get classified without one API call per episode."""

from __future__ import annotations

from aperture import fixtures as fx
from aperture.jobs.worker import drain


def _surface(client, auth, eid):
    return (client.get(f"/v1/episodes/{eid}", headers=auth).json().get("classification") or {}).get("surface")


def test_upload_queues_classification_of_its_failures(client, auth):
    r = client.post(
        "/v1/episodes/upload", headers=auth,
        files=[("files", ("p.rlds.json", fx.perception_failure_rlds())), ("files", ("s.rlds.json", fx.success_rlds()))],
    ).json()
    assert r["classify_job_id"]
    drain()
    job = client.get(f"/v1/jobs/{r['classify_job_id']}", headers=auth).json()
    assert job["status"] == "done" and job["result"]["classified"] == 1
    fail_id, ok_id = r["episode_ids"]
    assert _surface(client, auth, fail_id) == "perception"
    assert _surface(client, auth, ok_id) is None  # successes are not failures to explain


def test_presigned_ingest_also_classifies(client, auth):
    p = client.post("/v1/uploads/presign", headers=auth, json={"filename": "m.lerobot.json"}).json()
    client.put(p["upload_url"], headers=auth, content=fx.motor_failure_lerobot())
    job = client.post("/v1/uploads/ingest", headers=auth, json={"key": p["key"]}).json()
    drain()
    done = client.get(f"/v1/jobs/{job['job_id']}", headers=auth).json()
    assert done["result"]["classify_job_id"]
    assert _surface(client, auth, done["result"]["episode_ids"][0]) == "motor"


def test_classify_pending_endpoint_backfills(client, auth):
    from aperture.core.config import get_settings

    s = get_settings()
    s.auto_classify = False
    try:
        eid = client.post(
            "/v1/episodes/upload", headers=auth, files=[("files", ("g.rlds.json", fx.grounding_failure_rlds()))]
        ).json()["episode_ids"][0]
        drain()
        assert _surface(client, auth, eid) is None
    finally:
        s.auto_classify = True
    r = client.post("/v1/episodes/classify-pending", headers=auth)
    assert r.status_code == 202
    drain()
    assert _surface(client, auth, eid) == "grounding"


def test_classify_does_not_500_when_the_worker_got_there_first(client, auth, db_session):
    """The race seen on Supabase: the verdict row appears between load and insert."""
    from aperture.core.db import SessionLocal
    from aperture.core.models import Episode, FailureClassification
    from aperture.evaluation.router import run_classification

    from aperture.core.config import get_settings

    s = get_settings()
    s.auto_classify = False
    try:
        eid = client.post(
            "/v1/episodes/upload", headers=auth, files=[("files", ("p.rlds.json", fx.perception_failure_rlds()))]
        ).json()["episode_ids"][0]
    finally:
        s.auto_classify = True

    mine = SessionLocal()
    ep = mine.get(Episode, eid)
    assert ep.classification is None  # loaded before the "worker" writes

    other = SessionLocal()  # the worker, in its own session, wins the race
    other.add(FailureClassification(episode_id=eid, surface="motor", confidence=0.1, method="heuristic", details={}))
    other.commit()
    other.close()

    row = run_classification(mine, ep)  # used to raise IntegrityError -> 500
    mine.commit()
    assert row.surface == "perception"
    mine.close()
    assert client.get(f"/v1/episodes/{eid}", headers=auth).json()["classification"]["surface"] == "perception"
