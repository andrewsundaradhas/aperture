"""Regression tests for two cross-boundary holes found in review.

1. Path traversal: `/v1/blobs/<slug>/%2e%2e/...` resolved outside the blob root on the local
   backend, so any API key could read (and PUT could write) arbitrary files on the server.
2. Prefix nesting: blob access checked `key.startswith(slug + "/")`, and slugs were not
   validated, so signing up as `acme-robotics/episodes` granted read access to Acme's episodes.
"""

from __future__ import annotations

import uuid

import pytest

from aperture.core.config import get_settings
from aperture.core.storage import InvalidKey, LocalStorage, key_belongs_to, validate_key


@pytest.mark.parametrize(
    "key",
    ["", "/etc/passwd", "acme/../other/x", "acme/./x", "acme//x", "acme\\..\\x", "..", "acme/x/.."],
)
def test_validate_key_rejects_anything_but_plain_relative_paths(key):
    with pytest.raises(InvalidKey):
        validate_key(key)


def test_local_storage_cannot_escape_its_root(tmp_path):
    store = LocalStorage(str(tmp_path / "blobs"))
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(InvalidKey):
        store.get_bytes("acme/../../secret.txt")
    with pytest.raises(InvalidKey):
        store.put_bytes("acme/../../planted.txt", b"x")
    assert not (tmp_path / "planted.txt").exists()


def test_key_ownership_is_by_exact_first_segment():
    assert key_belongs_to("acme/episodes/1/raw.json", "acme")
    assert not key_belongs_to("acme/episodes/1/raw.json", "acme/episodes")
    assert not key_belongs_to("acme-robotics/x", "acme")
    assert not key_belongs_to("acme", "acme")
    assert not key_belongs_to("acme/../other/x", "acme")


@pytest.mark.parametrize("path", [
    "acme-robotics/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/hosts",
    "acme-robotics/..%2f..%2f..%2f..%2fetc%2fhosts",
])
def test_blob_route_refuses_traversal(client, auth, path):
    r = client.get(f"/v1/blobs/{path}", headers=auth)
    assert r.status_code in (403, 404)
    assert b"localhost" not in r.content
    r = client.put(f"/v1/blobs/{path}", headers=auth, content=b"x")
    assert r.status_code in (403, 404)


@pytest.mark.parametrize("slug", ["acme-robotics/episodes", "../x", "Acme", "a", "has space", "x" * 60, "-lead"])
def test_signup_rejects_slugs_that_are_not_safe_path_segments(client, slug):
    assert client.post("/v1/onboarding/signup", json={"slug": slug, "name": "x"}).status_code == 422


def test_signup_ignores_a_requested_tier(client, db_session):
    from sqlalchemy import select

    from aperture.core.models import Organization

    slug = f"tier-{uuid.uuid4().hex[:8]}"
    r = client.post("/v1/onboarding/signup", json={"slug": slug, "name": "x", "tier": "enterprise"})
    assert r.status_code == 201
    org = db_session.execute(select(Organization).where(Organization.slug == slug)).scalar_one()
    assert org.tier == "pilot"


@pytest.fixture
def production_settings():
    s = get_settings()
    saved = (s.environment, s.signup_token)
    yield s
    s.environment, s.signup_token = saved


def test_signup_is_closed_in_production_without_a_token(client, production_settings):
    production_settings.environment, production_settings.signup_token = "production", None
    r = client.post("/v1/onboarding/signup", json={"slug": f"p-{uuid.uuid4().hex[:8]}", "name": "x"})
    assert r.status_code == 403


def test_signup_requires_the_configured_token(client, production_settings):
    production_settings.environment, production_settings.signup_token = "production", "s3cret-token"
    body = lambda: {"slug": f"t-{uuid.uuid4().hex[:8]}", "name": "x"}  # noqa: E731
    assert client.post("/v1/onboarding/signup", json=body()).status_code == 403
    assert client.post("/v1/onboarding/signup", json=body(), headers={"X-Signup-Token": "wrong"}).status_code == 403
    assert client.post("/v1/onboarding/signup", json=body(), headers={"X-Signup-Token": "s3cret-token"}).status_code == 201


# --- 3. archive expansion limits (zip bombs) -------------------------------------------------


def _zip(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def small_archive_limits():
    s = get_settings()
    saved = (s.max_extracted_bytes, s.max_archive_members)
    s.max_extracted_bytes, s.max_archive_members = 1024 * 1024, 5
    yield s
    s.max_extracted_bytes, s.max_archive_members = saved


def test_archive_that_expands_past_the_limit_is_refused(tmp_path, small_archive_limits):
    from aperture.ingestion.archive import ArchiveError, unpack

    bomb = _zip({"meta/info.json": b"{}", "data/zeros.bin": b"\0" * (8 * 1024 * 1024)})
    assert len(bomb) < 64 * 1024  # tiny on the wire
    with pytest.raises(ArchiveError, match="expands past"):
        unpack(bomb, "fleet.zip", tmp_path / "out")
    written = sum(p.stat().st_size for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert written <= small_archive_limits.max_extracted_bytes


def test_archive_with_too_many_files_is_refused(tmp_path, small_archive_limits):
    from aperture.ingestion.archive import ArchiveError, unpack

    many = _zip({f"f{i}.txt": b"x" for i in range(10)})
    with pytest.raises(ArchiveError, match="more than 5 files"):
        unpack(many, "many.zip", tmp_path / "out")


def test_ingest_refuses_objects_over_the_ingest_limit(client, auth):
    s = get_settings()
    saved = s.max_ingest_bytes
    s.max_ingest_bytes = 10
    try:
        p = client.post("/v1/uploads/presign", headers=auth, json={"filename": "big.rlds.json"}).json()
        # The upload route itself now refuses it…
        assert client.put(p["upload_url"], headers=auth, content=b"x" * 100).status_code == 413
        # …and on R2 the bytes bypass that route (presigned PUT to the bucket), so ingest must
        # still refuse an oversized object that is already in storage.
        from aperture.core.storage import get_storage

        get_storage().put_bytes(p["key"], b"x" * 100)
        assert client.post("/v1/uploads/ingest", headers=auth, json={"key": p["key"]}).status_code == 413
    finally:
        s.max_ingest_bytes = saved
