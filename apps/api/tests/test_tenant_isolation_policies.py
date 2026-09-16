"""Tenant isolation, checked two ways.

**Statically**, against the Supabase migrations: every table the ORM defines must have a
row-level-security policy and `force row level security`. This is the check that would have
caught the original bug — `0001_init.sql` enabled RLS on 11 tables but wrote policies for only
4, which in Postgres means those 7 denied everything while the API (connected as the table
owner) bypassed RLS entirely and noticed nothing.

**Behaviourally**, against the running app: one org's key must never return another org's rows.
That is the guarantee RLS is defence-in-depth for; the application layer has to hold it too.

A live Postgres test — connecting as the non-owner `aperture_api` role and proving RLS blocks
cross-tenant reads at the database — is *not* here, because these tests run on SQLite, which has
no RLS. That check belongs in a Postgres-backed integration suite before GA.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import select

from aperture.core.db import Base
from aperture.core.models import Episode, Organization  # noqa: F401

MIGRATIONS = Path(__file__).resolve().parents[3] / "infra" / "supabase" / "migrations"


def _sql() -> str:
    return "\n".join(p.read_text() for p in sorted(MIGRATIONS.glob("*.sql")))


def _orm_tables() -> set[str]:
    return set(Base.metadata.tables)


# --- static: schema ownership and RLS coverage ------------------------------------------------


def _alembic_sql() -> str:
    versions = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    return "\n".join(p.read_text() for p in sorted(versions.glob("*.py")))


def test_alembic_owns_every_table():
    """Alembic is the single schema owner — it is what compose, Render and the tests run."""
    sql = _alembic_sql()
    missing = [t for t in _orm_tables() if f"'{t}'" not in sql]
    assert not missing, f"tables in the ORM that no Alembic migration creates: {missing}"


def test_supabase_sql_does_not_redefine_tables():
    """The bug this guards against cost a working deployment.

    The Supabase directory used to declare every table a second time, and the two definitions
    drifted: Alembic made `failure_clusters.embedding` a json column and ids varchar, the SQL
    made them vector(512) and uuid. Following DEPLOY.md applied both, and the second one dropped
    the column the ORM writes to.
    """
    sql = _sql()
    redefined = [t for t in _orm_tables() if re.search(rf"create table (if not exists )?{t}\b", sql)]
    assert not redefined, f"Supabase SQL re-creates tables Alembic owns: {redefined}"


def test_supabase_sql_never_drops_a_column_the_orm_uses():
    """Additive only. A `drop column` here silently breaks the running application."""
    sql = _sql()
    dropped = re.findall(r"alter table (\w+)\s+drop column (?:if exists )?(\w+)", sql, re.I)
    offending = [
        (table, column)
        for table, column in dropped
        if table in Base.metadata.tables and column in Base.metadata.tables[table].columns
    ]
    assert not offending, f"migration drops columns the ORM still maps: {offending}"


def test_every_table_has_row_level_security_forced_and_a_policy():
    """RLS enabled without a policy denies everything; a policy without FORCE is bypassed by the
    table owner. Both halves are required, on every tenant-scoped table."""
    sql = _sql()
    # The migration applies RLS through `foreach t in array [...]` loops, so read the table names
    # out of those arrays rather than looking for one literal statement per table.
    protected: set[str] = set()
    for block in re.findall(r"array\[(.*?)\]", sql, re.S):
        protected.update(re.findall(r"'(\w+)'", block))

    policied = set(re.findall(r"create policy \w+ on (\w+)", sql))

    missing_rls = _orm_tables() - protected
    assert not missing_rls, f"tables never given enable/force RLS: {sorted(missing_rls)}"
    missing_policy = _orm_tables() - policied
    assert not missing_policy, f"tables with no RLS policy: {sorted(missing_policy)}"
    assert "force row level security" in sql


def test_a_non_owner_role_exists_for_the_api():
    """RLS only applies to a role that is not the table owner."""
    sql = _sql()
    assert "create role aperture_api" in sql
    assert "grant select, insert, update, delete on all tables in schema public to aperture_api" in sql


def test_embedding_columns_match_the_dimensions_the_code_emits():
    """`vector(512)` matched neither embedding path; both real dimensions must be declared."""
    from aperture.clustering.embed import EMBED_DIM

    sql = _sql()
    assert f"embedding_signature vector({EMBED_DIM})" in sql
    assert re.search(r"embedding_visual\s+vector\(384\)", sql)

    # Comments discuss the old wrong dimension on purpose; only real statements matter.
    statements = "\n".join(line.split("--")[0] for line in sql.splitlines())
    assert "vector(512)" not in statements, "a vector(512) column declaration is still present"


def test_the_org_context_function_is_transaction_scoped():
    """`set_config(..., true)` is local to the transaction. Without that, a pooled connection
    would hand the previous request's tenant to the next one."""
    auth = (Path(__file__).resolve().parents[1] / "aperture" / "core" / "auth.py").read_text()
    assert "set_config('aperture.org_id', :org_id, true)" in auth


# --- behavioural: the application layer holds the boundary -----------------------------------


@pytest.fixture
def two_orgs(db_session):
    acme = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    other = db_session.execute(
        select(Organization).where(Organization.slug == "other-org")
    ).scalar_one()
    return acme, other


def test_an_orgs_episode_is_invisible_to_another_org(client, db_session, two_orgs):
    from aperture import fixtures
    from aperture.ingestion.normalize import normalize
    from aperture.ingestion.service import persist_episode

    acme, _other = two_orgs
    episode = persist_episode(
        db_session, acme, normalize(fixtures.perception_failure_rlds(), "iso.rlds.json")
    )
    db_session.commit()

    assert client.get(f"/v1/episodes/{episode.id}", headers={"X-API-Key": "demo-key"}).status_code == 200
    assert client.get(f"/v1/episodes/{episode.id}", headers={"X-API-Key": "other-key"}).status_code == 404


def test_listings_never_cross_the_boundary(client, db_session, two_orgs):
    """Not just 404 on a direct fetch — the other org's listing must not contain the row."""
    from aperture import fixtures
    from aperture.ingestion.normalize import normalize
    from aperture.ingestion.service import persist_episode

    acme, _ = two_orgs
    episode = persist_episode(
        db_session, acme, normalize(fixtures.grounding_failure_rlds(), "iso2.rlds.json")
    )
    db_session.commit()

    listed = client.get("/v1/episodes", headers={"X-API-Key": "other-key"}).json()
    assert episode.id not in {e["id"] for e in listed}


def test_every_episode_returned_belongs_to_the_caller(client, db_session):
    """Whatever each org can see, all of it must be theirs."""
    for key, slug in (("demo-key", "acme-robotics"), ("other-key", "other-org")):
        org = db_session.execute(
            select(Organization).where(Organization.slug == slug)
        ).scalar_one()
        for summary in client.get("/v1/episodes", headers={"X-API-Key": key}).json():
            assert db_session.get(Episode, summary["id"]).org_id == org.id
