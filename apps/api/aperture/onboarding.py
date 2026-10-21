"""Pilot onboarding: org signup, robot registration, API-key issuance.

Minimal by design — enough for a design partner to self-serve programmatic uploads without a
call. In production this sits behind Supabase Auth.

Issued keys are stored in the `api_keys` table as SHA-256 hashes, so they survive a restart and
are valid on every replica. The raw key is returned **once**, in the signup response; nothing
can recover it afterwards, which is the point of storing only the hash.
"""

from __future__ import annotations

import hmac
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aperture.core.apikeys import issue_key, revoke_key
from aperture.core.auth import bind_org_to_transaction, resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.pagination import Page, page_params
from aperture.core.models import ApiKey, Organization, Robot

router = APIRouter(prefix="/v1/onboarding", tags=["onboarding"])


# The slug is the first segment of every storage key the org owns (`<slug>/episodes/...`), so it
# must never contain a path separator or dot segment: a slug like `acme/episodes` would put one
# tenant's keys *inside* another's prefix.
SLUG_PATTERN = r"^[a-z0-9](?:[a-z0-9-]{1,46}[a-z0-9])$"


class OrgSignup(BaseModel):
    slug: str = Field(pattern=SLUG_PATTERN, description="Lowercase letters, digits and hyphens; 3-48 chars.")
    name: str = Field(min_length=1, max_length=200)
    # No `tier` here on purpose: the plan an org is on is an operator decision, not something a
    # signup request gets to choose. New orgs start on "pilot".


class OrgSignupOut(BaseModel):
    org_id: str
    slug: str
    api_key: str
    note: str


class KeyCreate(BaseModel):
    name: str = "default"


class ApiKeyOut(BaseModel):
    id: str
    prefix: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class RobotRegister(BaseModel):
    embodiment_type: str
    policy_name: str


class RobotOut(BaseModel):
    id: str
    embodiment_type: str
    policy_name: str


def _check_signup_token(provided: str | None) -> None:
    settings = get_settings()
    expected = settings.signup_token
    if not expected:
        if settings.environment == "production":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Self-serve signup is closed. Set APERTURE_SIGNUP_TOKEN to enable it.",
            )
        return  # local/dev: open, so the quick-start works without setup
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing or invalid X-Signup-Token.")


def slug_taken(db: Session, slug: str) -> bool:
    if db.get_bind().dialect.name == "postgresql":
        # Under RLS this caller can see no organizations at all, so an ORM lookup would always
        # say "free". The definer-rights lookup answers exactly this one question.
        return db.execute(text("select resolve_org_slug(:s)"), {"s": slug}).scalar() is not None
    return db.execute(select(Organization.id).where(Organization.slug == slug)).first() is not None


def create_org(db: Session, slug: str, name: str, tier: str = "pilot") -> tuple[Organization, str]:
    """Create an org and its first API key; returns `(org, raw_key)`. The caller commits.

    Works under Postgres RLS: the new org's id is chosen first and bound to the transaction,
    because the organizations policy only admits a row whose id is the transaction's org.
    Raises IntegrityError if the slug is taken (unique index).
    """
    org_id = str(uuid.uuid4())
    bind_org_to_transaction(db, org_id)
    org = Organization(id=org_id, slug=slug, name=name, tier=tier)
    db.add(org)
    db.flush()
    raw, _row = issue_key(db, org)
    return org, raw


@router.post("/signup", response_model=OrgSignupOut, status_code=status.HTTP_201_CREATED)
def signup(
    body: OrgSignup,
    db: Session = Depends(get_db),
    x_signup_token: str | None = Header(default=None, alias="X-Signup-Token"),
) -> OrgSignupOut:
    _check_signup_token(x_signup_token)
    if slug_taken(db, body.slug):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Org slug '{body.slug}' already exists.")

    try:
        org, api_key = create_org(db, body.slug, body.name)
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent signup for the same slug.
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f"Org slug '{body.slug}' already exists.")

    return OrgSignupOut(
        org_id=org.id,
        slug=org.slug,
        api_key=api_key,
        note="Save this key — only its hash is stored, so it cannot be shown again.",
    )


class OrgOut(BaseModel):
    org_id: str
    slug: str
    name: str
    tier: str


@router.get("/me", response_model=OrgOut)
def whoami(org: Organization = Depends(resolve_org)) -> OrgOut:
    """The organization this API key belongs to. The dashboard uses it to validate a sign-in."""
    return OrgOut(org_id=org.id, slug=org.slug, name=org.name, tier=org.tier)


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[ApiKeyOut]:
    """This org's keys, by prefix. The secrets themselves are not recoverable."""
    rows = db.execute(select(ApiKey).where(ApiKey.org_id == org.id)).scalars().all()
    return [
        ApiKeyOut(
            id=r.id, prefix=r.prefix, name=r.name,
            created_at=r.created_at, last_used_at=r.last_used_at, revoked_at=r.revoked_at,
        )
        for r in rows
    ]


@router.post("/keys", response_model=OrgSignupOut, status_code=status.HTTP_201_CREATED)
def create_key(
    body: KeyCreate,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> OrgSignupOut:
    """Mint an additional key, so a key can be rotated without downtime: issue the new one,
    move traffic over, then revoke the old one."""
    raw, _row = issue_key(db, org, name=body.name)
    db.commit()
    return OrgSignupOut(
        org_id=org.id, slug=org.slug, api_key=raw,
        note="Save this key — only its hash is stored, so it cannot be shown again.",
    )


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_key(
    key_id: str,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> None:
    """Revoke a key. Revoking is a timestamp, not a delete, so the audit trail survives."""
    if not revoke_key(db, key_id, org):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Key not found for this organization.")
    db.commit()


@router.post("/robots", response_model=RobotOut, status_code=status.HTTP_201_CREATED)
def register_robot(
    body: RobotRegister,
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> RobotOut:
    robot = Robot(org_id=org.id, embodiment_type=body.embodiment_type, policy_name=body.policy_name)
    db.add(robot)
    db.commit()
    return RobotOut(id=robot.id, embodiment_type=robot.embodiment_type, policy_name=robot.policy_name)


@router.get("/robots", response_model=list[RobotOut])
def list_robots(
    page: Page = Depends(page_params),
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[RobotOut]:
    robots = db.execute(
        select(Robot).where(Robot.org_id == org.id).order_by(Robot.id)
        .limit(page.limit).offset(page.offset)
    ).scalars().all()
    return [RobotOut(id=r.id, embodiment_type=r.embodiment_type, policy_name=r.policy_name) for r in robots]
