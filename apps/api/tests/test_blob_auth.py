"""`GET /v1/blobs/{key}` must authenticate and stay inside the caller's org.

The route is registered unconditionally and reads through `get_storage()`, so it serves the
configured backend — R2 included. Unauthenticated, it was a read of any blob whose key was
known, and exported datasets are episode data.
"""

from __future__ import annotations

from aperture import fixtures

ACME = {"X-API-Key": "demo-key"}
OTHER = {"X-API-Key": "other-key"}


def _exported_blob_key(client) -> str:
    """Produce a real export and return the storage key behind its download URL.

    A whole cluster's worth of episodes, not one: clustering needs several members before it
    forms a group, and with no cluster there is nothing to export.
    """
    for i, raw in enumerate(fixtures.grounding_cluster("wipe the tray", n=4)):
        r = client.post(
            "/v1/episodes/upload",
            files=[("files", (f"b{i}.rlds.json", raw, "application/json"))],
            headers=ACME,
        )
        assert r.status_code == 201, r.text
        client.post(f"/v1/episodes/{r.json()['episode_ids'][0]}/classify", headers=ACME)

    client.post("/v1/clusters/recompute?wait=true", headers=ACME)
    clusters = client.get("/v1/clusters", headers=ACME).json()
    assert clusters, "no clusters formed, so there is nothing to export"
    cid = clusters[0]["id"]
    export = client.post(f"/v1/clusters/{cid}/dataset-export", json={"format": "lerobot"}, headers=ACME)
    assert export.status_code == 200, export.text
    return export.json()["download_url"].removeprefix("/v1/blobs/")


def test_blob_requires_a_key(client):
    key = _exported_blob_key(client)
    assert client.get(f"/v1/blobs/{key}").status_code == 401
    assert client.get(f"/v1/blobs/{key}", headers={"X-API-Key": "nope"}).status_code == 401


def test_blob_is_not_readable_by_another_org(client):
    key = _exported_blob_key(client)
    assert client.get(f"/v1/blobs/{key}", headers=OTHER).status_code == 403


def test_owner_can_still_download_its_export(client):
    key = _exported_blob_key(client)
    r = client.get(f"/v1/blobs/{key}", headers=ACME)
    assert r.status_code == 200, r.text
    assert r.json()["format"] == "lerobot"
