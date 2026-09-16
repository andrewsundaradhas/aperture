"""Test harness: isolate every run in a fresh temp SQLite db + temp blob dir.

Env vars must be set BEFORE aperture modules import (config/engine bind at import time), so
this runs at module top before any aperture import.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="aperture-test-"))
os.environ["APERTURE_DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["APERTURE_LOCAL_BLOB_DIR"] = str(_TMP / "blobs")
os.environ["APERTURE_API_KEYS"] = "demo-key:acme-robotics,other-key:other-org"
# The suite deliberately hammers endpoints; per-org budgets would make it flaky. The limiter
# itself is covered directly in test_ratelimit.py, which turns it back on.
os.environ["APERTURE_RATE_LIMIT_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from aperture.core.db import SessionLocal, init_db  # noqa: E402
from aperture.core.models import Organization  # noqa: E402
from aperture.main import app  # noqa: E402


def _ensure_org(slug: str, name: str) -> None:
    db = SessionLocal()
    try:
        if db.execute(select(Organization).where(Organization.slug == slug)).scalar_one_or_none() is None:
            db.add(Organization(slug=slug, name=name))
            db.commit()
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def _db():
    init_db()
    _ensure_org("acme-robotics", "Acme Robotics")
    _ensure_org("other-org", "Other Org")
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth() -> dict:
    return {"X-API-Key": "demo-key"}


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
