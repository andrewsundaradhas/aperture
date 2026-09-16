"""Pilot onboarding: org signup, robot registration, API-key issuance.

Minimal by design — enough for a design partner to self-serve programmatic uploads without a
call. In production this sits behind Supabase Auth.

Issued keys are stored in the `api_keys` table as SHA-256 hashes, so they survive a restart and
are valid on every replica. The raw key is returned **once**, in the signup response; nothing
can recover it afterwards, which is the point of storing only the hash.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.apikeys import issue_key, revoke_key
from aperture.core.auth import resolve_org
from aperture.core.db import get_db
from aperture.core.pagination import Page, page_params
from aperture.core.models import ApiKey, Organization, Robot

router = APIRouter(prefix="/v1/onboarding", tags=["onboarding"])


class OrgSignup(BaseModel):
    slug: str
    name: str
    tier: str = "pilot"


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


@router.post("/signup", response_model=OrgSignupOut, status_code=status.HTTP_201_CREATED)
def signup(body: OrgSignup, db: Session = Depends(get_db)) -> OrgSignupOut:
    existing = db.execute(select(Organization).where(Organization.slug == body.slug)).scalar_one_or_none()
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Org slug '{body.slug}' already exists.")
    org = Organization(slug=body.slug, name=body.name, tier=body.tier)
    db.add(org)
    db.commit()

    api_key, _row = issue_key(db, org)
    db.commit()

    return OrgSignupOut(
        org_id=org.id,
        slug=org.slug,
        api_key=api_key,
        note="Save this key — only its hash is stored, so it cannot be shown again.",
    )


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
