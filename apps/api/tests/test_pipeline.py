"""Phases 3–5 + isolation: attribution, clustering, export, verify, cross-org isolation."""

from __future__ import annotations

from aperture import fixtures


def _upload(client, auth, raw, name):
    r = client.post("/v1/episodes/upload", files=[("files", (name, raw, "application/json"))], headers=auth)
    assert r.status_code == 201, r.text
    return r.json()["episode_ids"][0]


def test_attribution_and_counterfactual(client, auth):
    eid = _upload(client, auth, fixtures.grounding_failure_rlds(), "g.rlds.json")
    client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    r = client.post(f"/v1/episodes/{eid}/attribution", headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["attention_map_uri"]  # simulated heatmap produced for the demo
    assert body["confidence_trace"]
    # Grounding failure -> counterfactual probe ran and shows instruction sensitivity.
    cf = body["counterfactual_result"]
    assert cf and cf["applicable"] is True


def test_motor_failure_has_no_counterfactual(client, auth):
    eid = _upload(client, auth, fixtures.motor_failure_lerobot(), "m.lerobot.json")
    client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    r = client.post(f"/v1/episodes/{eid}/attribution", headers=auth)
    assert r.json()["counterfactual_result"] is None


def test_clustering_groups_similar_failures(client, auth):
    for i, raw in enumerate(fixtures.grounding_cluster("stack the blue cup", n=4)):
        eid = _upload(client, auth, raw, f"c{i}.rlds.json")
        client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    # A dissimilar failure that should not join the grounding cluster.
    mid = _upload(client, auth, fixtures.motor_failure_lerobot("weld the seam"), "m.lerobot.json")
    client.post(f"/v1/episodes/{mid}/classify", headers=auth)

    r = client.post("/v1/clusters/recompute", headers=auth)
    assert r.status_code == 200, r.text
    clusters = r.json()
    assert clusters, "expected at least one cluster"
    top = clusters[0]
    assert top["episode_count"] >= 3
    assert top["dominant_surface"] == "grounding"


def test_dataset_export_scoped_to_cluster(client, auth):
    client.post("/v1/clusters/recompute", headers=auth)
    clusters = client.get("/v1/clusters", headers=auth).json()
    cid = clusters[0]["id"]
    detail = client.get(f"/v1/clusters/{cid}", headers=auth).json()
    n_members = len(detail["episodes"])

    r = client.post(f"/v1/clusters/{cid}/dataset-export", json={"format": "lerobot"}, headers=auth)
    assert r.status_code == 200, r.text
    export = r.json()
    assert export["episode_count"] == n_members  # exactly the cluster, not the whole fleet
    assert export["format"] == "lerobot"
    assert export["download_url"]


def test_loop_verification_improved_vs_unchanged(client, auth):
    instr = "route the cable"
    for i, raw in enumerate(fixtures.grounding_cluster(instr, n=4)):
        eid = _upload(client, auth, raw, f"lc{i}.rlds.json")
        client.post(f"/v1/episodes/{eid}/classify", headers=auth)
    client.post("/v1/clusters/recompute", headers=auth)
    clusters = client.get("/v1/clusters", headers=auth).json()
    cid = next(c["id"] for c in clusters if instr.lower() in c["label"] or c["dominant_surface"] == "grounding")

    # Improved batch: same task, now succeeds.
    improved = [
        ("files", (f"imp{i}.lerobot.json", fixtures.success_lerobot(instr), "application/json"))
        for i in range(3)
    ]
    r = client.post(f"/v1/clusters/{cid}/verify", files=improved, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["delta"] > 0.5

    # Unchanged batch: same task, still fails -> ~0 delta.
    unchanged = [
        ("files", (f"unc{i}.rlds.json", fixtures.grounding_failure_rlds(instr), "application/json"))
        for i in range(3)
    ]
    r2 = client.post(f"/v1/clusters/{cid}/verify", files=unchanged, headers=auth)
    assert abs(r2.json()["delta"]) < 0.2


def test_cross_org_isolation(client):
    # Upload as acme, then attempt to read it as another org's key -> 404 (not visible).
    eid = _upload(client, {"X-API-Key": "demo-key"}, fixtures.success_rlds(), "s.rlds.json")
    r = client.get(f"/v1/episodes/{eid}", headers={"X-API-Key": "other-key"})
    assert r.status_code == 404
    # Owner can read it.
    assert client.get(f"/v1/episodes/{eid}", headers={"X-API-Key": "demo-key"}).status_code == 200
