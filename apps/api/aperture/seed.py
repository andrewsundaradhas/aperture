"""Seed a fresh database so a clean checkout is demoable in under five minutes.

    python -m aperture.seed

Creates the demo org (matching the default demo API key), ingests a spread of synthetic
episodes across all three failure surfaces plus a recurring grounding cluster, classifies
every episode, and computes clusters. After running, start the API and open the dashboard.
"""

from __future__ import annotations

from sqlalchemy import select

from aperture.clustering.cluster import recompute_clusters
from aperture.core.db import SessionLocal, init_db
from aperture.core.models import Episode, Organization
from aperture.evaluation.router import run_classification
from aperture.ingestion.normalize import normalize
from aperture.ingestion.service import persist_episode
from aperture import fixtures

DEMO_ORG_SLUG = "acme-robotics"


def _ingest(db, org, raw: bytes, filename: str) -> Episode:
    ne = normalize(raw, filename)
    ep = persist_episode(db, org, ne)
    run_classification(db, ep)
    return ep


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        org = db.execute(
            select(Organization).where(Organization.slug == DEMO_ORG_SLUG)
        ).scalar_one_or_none()
        if org is None:
            org = Organization(slug=DEMO_ORG_SLUG, name="Acme Robotics", tier="pilot")
            db.add(org)
            db.flush()

        batch: list[tuple[bytes, str]] = [
            (fixtures.perception_failure_rlds(), "perception_fail.rlds.json"),
            (fixtures.motor_failure_lerobot(), "motor_fail.lerobot.json"),
            (fixtures.grounding_failure_rlds(), "grounding_fail.rlds.json"),
            (fixtures.ambiguous_lerobot(), "ambiguous.lerobot.json"),
            (fixtures.success_rlds(), "success.rlds.json"),
        ]
        for i, raw in enumerate(fixtures.grounding_cluster("move object to the shelf", n=4)):
            batch.append((raw, f"cluster_grounding_{i}.rlds.json"))

        count = 0
        for raw, name in batch:
            _ingest(db, org, raw, name)
            count += 1
        db.commit()

        clusters = recompute_clusters(db, org)
        print(f"Seeded org '{org.slug}' with {count} episodes; formed {len(clusters)} cluster(s).")
        for c in clusters:
            print(f"  - {c.label}: {c.episode_count} episodes (surface={c.dominant_surface})")
        print("\nDemo API key: demo-key   (header: X-API-Key: demo-key)")
        print("Start the API:  uvicorn aperture.main:app --reload")
    finally:
        db.close()


if __name__ == "__main__":
    main()
