"""Tenant isolation.

Production enforces single-tenant isolation with Supabase Auth + Postgres row-level-security
(one org can never SELECT another org's rows — see infra/supabase/migrations). Locally, and
for programmatic uploads, we enforce the same boundary at the application layer: every request
carries an API key that resolves to exactly one organization, and every query is scoped to
that org's id. There is no code path that returns rows for an org other than the caller's.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Organization


def resolve_org(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Organization:
    settings = get_settings()
    key_map = settings.api_key_map()
    if not x_api_key or x_api_key not in key_map:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    org_slug = key_map[x_api_key]
    org = db.execute(select(Organization).where(Organization.slug == org_slug)).scalar_one_or_none()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key is valid but its organization '{org_slug}' is not provisioned.",
        )
    return org


CurrentOrg = Depends(resolve_org)
