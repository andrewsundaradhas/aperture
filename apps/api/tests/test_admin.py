"""The operator CLI: how orgs and keys get provisioned when public signup is closed."""

from __future__ import annotations

import re
import uuid

from aperture.admin import main


def _key(out: str) -> str:
    return re.search(r"(ak_\S+)", out).group(1)


def test_create_org_prints_a_working_key(client, capsys):
    slug = f"cli-{uuid.uuid4().hex[:8]}"
    assert main(["create-org", "--slug", slug, "--name", "CLI Org"]) == 0
    key = _key(capsys.readouterr().out)
    me = client.get("/v1/onboarding/me", headers={"X-API-Key": key}).json()
    assert me["slug"] == slug and me["tier"] == "pilot"
    assert main(["create-org", "--slug", slug, "--name", "again"]) == 1


def test_ensure_org_is_idempotent_and_prints_no_secret(capsys):
    slug = f"ens-{uuid.uuid4().hex[:8]}"
    assert main(["ensure-org", "--slug", slug, "--name", "E"]) == 0
    assert main(["ensure-org", "--slug", slug, "--name", "E"]) == 0
    assert "ak_" not in capsys.readouterr().out


def test_issue_key_for_existing_org(client, capsys):
    slug = f"iss-{uuid.uuid4().hex[:8]}"
    main(["ensure-org", "--slug", slug, "--name", "I"])
    capsys.readouterr()
    assert main(["issue-key", "--slug", slug, "--name", "ci"]) == 0
    assert client.get("/v1/onboarding/me", headers={"X-API-Key": _key(capsys.readouterr().out)}).status_code == 200
    assert main(["issue-key", "--slug", "no-such-org", "--name", "x"]) == 1


def test_rejects_unsafe_slugs():
    assert main(["create-org", "--slug", "a/b", "--name", "x"]) == 2
