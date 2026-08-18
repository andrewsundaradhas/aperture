"""Pilot onboarding: org signup, robot registration, API-key issuance.

Minimal by design — enough for a design partner to self-serve programmatic uploads without a
call. In production this sits behind Supabase Auth; here the issued API key is added to the
in-memory key map for the process lifetime and printed to the onboarding response so the
partner can start uploading immediately. Persist keys via APERTURE_API_KEYS for durability.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.db import get_db
from aperture.core.models import Organization, Robot

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

    api_key = f"ak_{secrets.token_urlsafe(24)}"
    # Register the key for this process. For durability, add "<key>:<slug>" to APERTURE_API_KEYS.
    # get_settings() is cached, so this mutation is visible to auth on the next request;
    # api_key_map() re-reads settings.api_keys each call.
    settings = get_settings()
    settings.api_keys = f"{settings.api_keys},{api_key}:{org.slug}"

    return OrgSignupOut(
        org_id=org.id,
        slug=org.slug,
        api_key=api_key,
        note="Save this key. For durability across restarts, add it to APERTURE_API_KEYS env var.",
    )


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
    org: Organization = Depends(resolve_org),
    db: Session = Depends(get_db),
) -> list[RobotOut]:
    robots = db.execute(select(Robot).where(Robot.org_id == org.id)).scalars().all()
    return [RobotOut(id=r.id, embodiment_type=r.embodiment_type, policy_name=r.policy_name) for r in robots]
