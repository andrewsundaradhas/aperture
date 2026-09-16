"""Tenant isolation.

Two layers, and they are not redundant:

* **Application** — every request carries an API key that resolves to exactly one organization,
  and every query is scoped to that org's id. This is the layer that works everywhere, including
  the SQLite default.
* **Database** — on Postgres, row-level security enforces the same boundary in the engine, so a
  query in this codebase that *forgets* its `where org_id` clause returns nothing rather than
  another tenant's rows. See `infra/supabase/migrations/`.

A key resolves in two steps:

1. The `api_keys` table, where keys issued through onboarding live as SHA-256 hashes. The durable
   path — a key survives restarts and is visible to every replica.
2. `APERTURE_API_KEYS`, a static `"key:org_slug"` map from the environment. Kept for local
   development and for bootstrapping the first key onto a fresh deployment, where no row exists
   to authenticate against yet.

**Why the Postgres path looks different.** Under RLS, resolving identity is a chicken-and-egg
problem: every policy keys off `aperture.org_id`, but working out which org this request belongs
to means reading `api_keys` or `organizations` — both behind that same policy. Nothing matches,
authentication fails, and every request 403s. The two `security definer` lookups in the migration
are the way in: give them a credential, get back one org id, nothing else. Once we have the id we
set the transaction's org and the ordinary policies take over.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from aperture.core.apikeys import hash_key, resolve_key
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Organization

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid X-API-Key.",
    headers={"WWW-Authenticate": "ApiKey"},
)


def _is_postgres(db: Session) -> bool:
    return db.get_bind().dialect.name == "postgresql"


def bind_org_to_transaction(db: Session, org_id: str) -> None:
    """Tell Postgres which organization this transaction is for.

    `set_config(..., is_local => true)` is scoped to the transaction, which is what makes it safe
    on a pooled connection: the next request to borrow it cannot inherit this tenant's identity.
    No-op on SQLite, which has neither RLS nor `SET LOCAL`.
    """
    if _is_postgres(db):
        db.execute(text("select set_config('aperture.org_id', :org_id, true)"), {"org_id": org_id})


def _bootstrap_org_id(db: Session, api_key: str, key_map: dict[str, str]) -> str | None:
    """The org this key belongs to, resolved before RLS can be satisfied (Postgres only)."""
    org_id = db.execute(
        text("select resolve_api_key(:h)"), {"h": hash_key(api_key)}
    ).scalar()
    if org_id:
        return org_id

    slug = key_map.get(api_key)
    if slug:
        return db.execute(text("select resolve_org_slug(:s)"), {"s": slug}).scalar()
    return None


def resolve_org(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Organization:
    if not x_api_key:
        raise _UNAUTHORIZED

    key_map = get_settings().api_key_map()

    if _is_postgres(db):
        org_id = _bootstrap_org_id(db, x_api_key, key_map)
        if org_id is None:
            raise _UNAUTHORIZED
        # From here the policies can evaluate, so ordinary ORM access works.
        bind_org_to_transaction(db, org_id)
        org = db.get(Organization, org_id)
        if org is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key resolved to an organization that is not provisioned.",
            )
        # Touch last_used_at now that the row is visible to us.
        resolve_key(db, x_api_key)
        return org

    # SQLite: no RLS, so the straightforward lookups are the whole story.
    org = resolve_key(db, x_api_key)
    if org is not None:
        return org

    slug = key_map.get(x_api_key)
    if slug is None:
        raise _UNAUTHORIZED
    org = db.execute(select(Organization).where(Organization.slug == slug)).scalar_one_or_none()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key is valid but its organization '{slug}' is not provisioned.",
        )
    return org


CurrentOrg = Depends(resolve_org)
