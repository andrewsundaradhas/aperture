"""Durable, hashed per-organization API keys.

What these protect: keys used to live in a mutated settings string, so one issued on a Render
replica was invisible to the others and every key died at restart. And a key that is stored in
the clear turns a leaked database backup into working credentials.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from aperture.core.apikeys import hash_key, issue_key, resolve_key, revoke_key
from aperture.core.models import ApiKey, Organization


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


# --- storage ---------------------------------------------------------------------------------


def test_raw_key_is_never_stored(db_session, org):
    """The whole point: a database dump must not yield usable credentials."""
    raw, row = issue_key(db_session, org)
    db_session.commit()

    assert row.key_hash != raw
    assert raw not in row.key_hash
    assert row.key_hash == hash_key(raw)
    # And nothing anywhere in the row carries the secret.
    stored = {row.key_hash, row.prefix, row.name}
    assert not any(raw in value for value in stored)


def test_prefix_is_a_recognisable_but_useless_fragment(db_session, org):
    raw, row = issue_key(db_session, org)
    assert raw.startswith(row.prefix)
    assert len(row.prefix) < len(raw) / 2, "the prefix must not be most of the key"


def test_issued_key_resolves_to_its_org(db_session, org):
    raw, _ = issue_key(db_session, org)
    db_session.commit()
    assert resolve_key(db_session, raw).id == org.id


def test_unknown_key_resolves_to_nothing(db_session):
    assert resolve_key(db_session, "ak_not-a-real-key") is None


def test_keys_are_distinct(db_session, org):
    keys = {issue_key(db_session, org)[0] for _ in range(20)}
    assert len(keys) == 20


# --- lifecycle -------------------------------------------------------------------------------


def test_revoked_key_stops_working(db_session, org):
    raw, row = issue_key(db_session, org)
    db_session.commit()
    assert resolve_key(db_session, raw) is not None

    assert revoke_key(db_session, row.id, org) is True
    db_session.commit()
    assert resolve_key(db_session, raw) is None


def test_revocation_keeps_the_row_for_audit(db_session, org):
    raw, row = issue_key(db_session, org)
    db_session.commit()
    revoke_key(db_session, row.id, org)
    db_session.commit()
    assert db_session.get(ApiKey, row.id) is not None
    assert db_session.get(ApiKey, row.id).revoked_at is not None


def test_an_org_cannot_revoke_another_orgs_key(db_session, org, other_org):
    _raw, row = issue_key(db_session, org)
    db_session.commit()
    assert revoke_key(db_session, row.id, other_org) is False
    db_session.commit()
    assert db_session.get(ApiKey, row.id).revoked_at is None


def test_rotation_leaves_the_old_key_working_until_revoked(db_session, org):
    """Rotation has to be doable without downtime: mint, cut traffic over, then revoke."""
    old, old_row = issue_key(db_session, org)
    new, _ = issue_key(db_session, org, name="rotated")
    db_session.commit()

    assert resolve_key(db_session, old) is not None
    assert resolve_key(db_session, new) is not None

    revoke_key(db_session, old_row.id, org)
    db_session.commit()
    assert resolve_key(db_session, old) is None
    assert resolve_key(db_session, new) is not None


def test_last_used_is_recorded(db_session, org):
    raw, row = issue_key(db_session, org)
    db_session.commit()
    assert row.last_used_at is None
    resolve_key(db_session, raw)
    db_session.commit()
    assert db_session.get(ApiKey, row.id).last_used_at is not None


# --- through the API -------------------------------------------------------------------------


def test_issued_key_authenticates_real_requests(client, auth):
    """A key minted through onboarding must work on an ordinary endpoint immediately."""
    created = client.post("/v1/onboarding/keys", headers=auth, json={"name": "ci"})
    assert created.status_code == 201, created.text
    raw = created.json()["api_key"]

    assert client.get("/v1/episodes", headers={"X-API-Key": raw}).status_code == 200


def test_revoked_key_is_rejected_by_the_api(client, auth):
    raw = client.post("/v1/onboarding/keys", headers=auth, json={"name": "temp"}).json()["api_key"]
    key_id = next(k["id"] for k in client.get("/v1/onboarding/keys", headers=auth).json() if k["name"] == "temp")

    assert client.delete(f"/v1/onboarding/keys/{key_id}", headers=auth).status_code == 204
    assert client.get("/v1/episodes", headers={"X-API-Key": raw}).status_code == 401


def test_listing_keys_never_exposes_a_secret(client, auth):
    raw = client.post("/v1/onboarding/keys", headers=auth, json={"name": "listing"}).json()["api_key"]
    body = client.get("/v1/onboarding/keys", headers=auth).text
    assert raw not in body, "the key listing leaked a usable secret"


def test_signup_returns_a_working_key(client):
    import uuid

    slug = f"acme-{uuid.uuid4().hex[:8]}"
    created = client.post("/v1/onboarding/signup", json={"slug": slug, "name": "Acme"})
    assert created.status_code == 201, created.text
    raw = created.json()["api_key"]
    # A brand-new org's key authenticates without any env-var change or restart.
    assert client.get("/v1/episodes", headers={"X-API-Key": raw}).status_code == 200


def test_env_var_keys_still_work_for_bootstrap(client):
    """The static map remains the way to get the first key onto a fresh deployment."""
    assert client.get("/v1/episodes", headers={"X-API-Key": "demo-key"}).status_code == 200
