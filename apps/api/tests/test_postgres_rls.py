"""The production database path, end to end, against a real Postgres with RLS on.

The rest of the suite runs on SQLite, which has no row-level security, so it cannot see the
class of bug that only exists when the API connects as the non-owner `aperture_api` role — and
three such bugs shipped: signup could not insert an org, any read after `db.commit()` ran as no
tenant, and the job worker could see no jobs at all. Each one passed every SQLite test.

Opt-in: set `APERTURE_TEST_PG_URL` to a Postgres (16+, with pgvector) connection string for a
**superuser on a throwaway database** — the fixture drops and rebuilds its `public` schema.

    APERTURE_TEST_PG_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/aperture_test \\
        pytest tests/test_postgres_rls.py

CI runs this against a pgvector service container.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

PG_URL = os.environ.get("APERTURE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="set APERTURE_TEST_PG_URL to run the Postgres suite")

API_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = API_ROOT.parents[1] / "infra" / "supabase" / "migrations"
# Generated per run: the restricted role only needs *a* password for this throwaway database.
API_ROLE_PASSWORD = secrets.token_hex(16)


@pytest.fixture(scope="module")
def pg_sessionmaker():
    owner = create_engine(PG_URL, isolation_level="AUTOCOMMIT")
    with owner.connect() as c:
        c.execute(text("drop schema if exists public cascade"))
        c.execute(text("create schema public"))
    # Alembic binds its URL from settings at import time, so run it in its own process.
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=API_ROOT, check=True, env={**os.environ, "APERTURE_DATABASE_URL": PG_URL},
    )
    owner.dispose()
    # The same command a deploy runs (render.yaml, DEPLOY.md), so it is exercised here too.
    from aperture.dbsetup import apply

    apply(PG_URL, SQL_DIR, API_ROLE_PASSWORD)

    api_url = make_url(PG_URL).set(username="aperture_api", password=API_ROLE_PASSWORD)
    engine = create_engine(api_url, future=True)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    engine.dispose()


@pytest.fixture(scope="module")
def pg_client(pg_sessionmaker):
    """The real app, with every session — requests and the worker — on the restricted role."""
    from fastapi.testclient import TestClient

    from aperture.core.db import get_db
    from aperture.jobs import worker
    from aperture.main import app

    def _get_db():
        db = pg_sessionmaker()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db
    original = worker.SessionLocal
    worker.SessionLocal = pg_sessionmaker
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)
        worker.SessionLocal = original


def _signup(client, prefix: str) -> dict:
    slug = f"{prefix}-{uuid.uuid4().hex[:8]}"
    r = client.post("/v1/onboarding/signup", json={"slug": slug, "name": prefix})
    assert r.status_code == 201, r.text
    return {"X-API-Key": r.json()["api_key"], "_slug": slug}


def _h(org: dict) -> dict:
    return {"X-API-Key": org["X-API-Key"]}


def test_connected_as_a_role_rls_applies_to(pg_sessionmaker):
    db = pg_sessionmaker()
    try:
        assert db.execute(text("select current_user")).scalar() == "aperture_api"
        bypass = db.execute(
            text("select rolbypassrls or rolsuper from pg_roles where rolname = current_user")
        ).scalar()
        assert bypass is False
    finally:
        db.close()


def test_signup_creates_an_org_under_rls(pg_client):
    org = _signup(pg_client, "signup")
    assert pg_client.get("/v1/episodes", headers=_h(org)).status_code == 200
    dup = pg_client.post("/v1/onboarding/signup", json={"slug": org["_slug"], "name": "again"})
    assert dup.status_code == 409


def test_full_pipeline_runs_as_the_restricted_role(pg_client):
    from aperture import fixtures as fx
    from aperture.jobs.worker import drain

    org = _signup(pg_client, "pipeline")
    files = [("files", (f"g{i}.rlds.json", b)) for i, b in enumerate(fx.grounding_cluster("stack the cups", 5))]
    files += [("files", ("p.rlds.json", fx.perception_failure_rlds())), ("files", ("m.lerobot.json", fx.motor_failure_lerobot()))]
    r = pg_client.post("/v1/episodes/upload", headers=_h(org), files=files)
    assert r.status_code == 201, r.text
    ids = r.json()["episode_ids"]

    # Each of these commits and then reads the row back — the shape that 500'd before the
    # per-transaction tenant rebind.
    for eid in ids:
        assert pg_client.post(f"/v1/episodes/{eid}/classify", headers=_h(org)).status_code == 200
    assert pg_client.post(f"/v1/episodes/{ids[0]}/attribution", headers=_h(org)).status_code == 200

    job = pg_client.post("/v1/clusters/recompute", headers=_h(org)).json()
    assert drain() >= 1, "the worker found no jobs — it cannot see the queue under RLS"
    done = pg_client.get(f"/v1/jobs/{job['job_id']}", headers=_h(org)).json()
    assert done["status"] == "done", done

    clusters = pg_client.get("/v1/clusters", headers=_h(org)).json()
    assert clusters, "expected the five near-identical grounding failures to form a cluster"
    cid = clusters[0]["id"]
    export = pg_client.post(f"/v1/clusters/{cid}/dataset-export", headers=_h(org), json={"format": "lerobot"})
    assert export.status_code in (200, 201), export.text
    dl = pg_client.get(export.json()["download_url"].replace("/api/aperture", ""), headers=_h(org))
    assert dl.status_code == 200 and dl.content

    post = [("files", (f"post{i}.rlds.json", fx.success_rlds("stack the cups"))) for i in range(3)]
    assert pg_client.post(f"/v1/clusters/{cid}/verify", headers=_h(org), files=post).status_code in (200, 201)


def test_presigned_ingest_runs_through_the_worker(pg_client):
    from aperture import fixtures as fx
    from aperture.jobs.worker import drain

    org = _signup(pg_client, "ingest")
    p = pg_client.post("/v1/uploads/presign", headers=_h(org), json={"filename": "ep.rlds.json"}).json()
    assert pg_client.put(p["upload_url"], headers=_h(org), content=fx.perception_failure_rlds()).status_code == 201
    job = pg_client.post("/v1/uploads/ingest", headers=_h(org), json={"key": p["key"]}).json()
    drain()
    done = pg_client.get(f"/v1/jobs/{job['job_id']}", headers=_h(org)).json()
    assert done["status"] == "done", done
    assert done["result"]["count"] == 1


def test_worker_serves_every_org_and_never_mixes_them(pg_client):
    from aperture import fixtures as fx
    from aperture.jobs.worker import drain

    a, b = _signup(pg_client, "wa"), _signup(pg_client, "wb")
    for org in (a, b):
        pg_client.post("/v1/episodes/upload", headers=_h(org), files=[("files", ("p.rlds.json", fx.perception_failure_rlds()))])
    ja = pg_client.post("/v1/clusters/recompute", headers=_h(a)).json()["job_id"]
    jb = pg_client.post("/v1/clusters/recompute", headers=_h(b)).json()["job_id"]
    drain()
    assert pg_client.get(f"/v1/jobs/{ja}", headers=_h(a)).json()["status"] == "done"
    assert pg_client.get(f"/v1/jobs/{jb}", headers=_h(b)).json()["status"] == "done"
    assert pg_client.get(f"/v1/jobs/{ja}", headers=_h(b)).status_code == 404


def test_cross_tenant_access_is_blocked_at_the_api_and_the_database(pg_client, pg_sessionmaker):
    from aperture import fixtures as fx

    a, b = _signup(pg_client, "ta"), _signup(pg_client, "tb")
    eid = pg_client.post(
        "/v1/episodes/upload", headers=_h(a), files=[("files", ("p.rlds.json", fx.perception_failure_rlds()))]
    ).json()["episode_ids"][0]

    assert pg_client.get(f"/v1/episodes/{eid}", headers=_h(b)).status_code == 404
    assert eid not in {e["id"] for e in pg_client.get("/v1/episodes", headers=_h(b)).json()}

    db = pg_sessionmaker()
    try:
        # No tenant bound: RLS must hide everything, even from a query with no WHERE clause.
        assert db.execute(text("select count(*) from episodes")).scalar() == 0
        assert db.execute(text("select count(*) from organizations")).scalar() == 0
    finally:
        db.close()


def test_slug_cannot_nest_inside_another_orgs_storage_prefix(pg_client):
    victim = _signup(pg_client, "victim")
    r = pg_client.post("/v1/onboarding/signup", json={"slug": f"{victim['_slug']}/episodes", "name": "x"})
    assert r.status_code == 422


def test_admin_cli_provisions_under_rls(pg_client, pg_sessionmaker):
    """The operator path compose uses, as the restricted role, in its own process."""
    url = pg_sessionmaker.kw["bind"].url.render_as_string(hide_password=False)
    slug = f"cli-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "APERTURE_DATABASE_URL": url}
    run = lambda *a: subprocess.run([sys.executable, "-m", "aperture.admin", *a], cwd=API_ROOT, env=env, capture_output=True, text=True)  # noqa: E731
    first = run("ensure-org", "--slug", slug, "--name", "CLI")
    assert first.returncode == 0, first.stderr
    assert run("ensure-org", "--slug", slug, "--name", "CLI").returncode == 0
    issued = run("issue-key", "--slug", slug)
    assert issued.returncode == 0, issued.stderr
    key = issued.stdout.strip().split()[-1]
    assert pg_client.get("/v1/onboarding/me", headers={"X-API-Key": key}).json()["slug"] == slug


def test_orm_and_migrations_have_not_drifted(pg_sessionmaker):
    """`alembic check` fails if the models and the migrations disagree — drift that a later
    autogenerated migration would otherwise "fix" by dropping a live index or column."""
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "check"],
        cwd=API_ROOT, env={**os.environ, "APERTURE_DATABASE_URL": PG_URL}, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
