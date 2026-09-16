"""Issuing, hashing and verifying per-organization API keys.

Kept separate from `auth.py` so the *storage* of a credential and the *authorization* decision
it feeds are not tangled: this module never looks at a request, and `auth.py` never hashes.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from aperture.core.models import ApiKey, Organization

KEY_PREFIX = "ak_"
KEY_BYTES = 24          # 192 bits of entropy — not brute-forceable, so no work factor needed
PREFIX_DISPLAY_LEN = 7  # "ak_" + 4 chars, enough to tell two keys apart in a list


def hash_key(raw: str) -> str:
    """The stored representation of a key. Never reversible to the key itself."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_key() -> str:
    return f"{KEY_PREFIX}{secrets.token_urlsafe(KEY_BYTES)}"


def issue_key(db: Session, org: Organization, name: str = "default") -> tuple[str, ApiKey]:
    """Mint a key for `org`. Returns `(raw_key, row)` — the raw key is shown exactly once."""
    raw = generate_key()
    row = ApiKey(
        org_id=org.id,
        key_hash=hash_key(raw),
        prefix=raw[:PREFIX_DISPLAY_LEN],
        name=name,
    )
    db.add(row)
    db.flush()
    return raw, row


def resolve_key(db: Session, raw: str) -> Organization | None:
    """The organization a key belongs to, or None if it is unknown or revoked.

    Looks up by hash, so a timing difference can only reveal whether a *hash* exists, which the
    caller already knows the input for.
    """
    row = db.execute(
        select(ApiKey).where(ApiKey.key_hash == hash_key(raw))
    ).scalar_one_or_none()
    if row is None or row.revoked_at is not None:
        return None
    row.last_used_at = datetime.now(timezone.utc)
    return db.get(Organization, row.org_id)


def revoke_key(db: Session, key_id: str, org: Organization) -> bool:
    """Revoke one of `org`'s keys. Returns False if it is not theirs or is already revoked."""
    row = db.get(ApiKey, key_id)
    if row is None or row.org_id != org.id or row.revoked_at is not None:
        return False
    row.revoked_at = datetime.now(timezone.utc)
    return True
