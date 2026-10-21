"""Operator commands: provision organizations and keys from a shell.

Public signup is closed in production unless APERTURE_SIGNUP_TOKEN is set, so this is how an
operator onboards a customer:

    python -m aperture.admin create-org --slug acme --name "Acme Robotics"   # prints the key
    python -m aperture.admin issue-key  --slug acme --name ci-uploader        # another key
    python -m aperture.admin ensure-org --slug acme --name "Acme Robotics"   # idempotent, no key

`ensure-org` exists for deploy scripts (docker-compose runs it) so the org named in
APERTURE_API_KEYS exists before the API starts; it never prints a secret.

Uses APERTURE_DATABASE_URL like the API does, and works as the restricted `aperture_api` role.
"""

from __future__ import annotations

import argparse
import re
import sys

from sqlalchemy import text

from aperture.core.apikeys import issue_key
from aperture.core.auth import bind_org_to_transaction
from aperture.core.db import SessionLocal
from aperture.core.models import Organization
from aperture.onboarding import SLUG_PATTERN, create_org, slug_taken


def _org_id(db, slug: str) -> str | None:
    if db.get_bind().dialect.name == "postgresql":
        return db.execute(text("select resolve_org_slug(:s)"), {"s": slug}).scalar()
    org = db.query(Organization).filter(Organization.slug == slug).one_or_none()
    return org.id if org else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aperture.admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create-org", "ensure-org"):
        p = sub.add_parser(name)
        p.add_argument("--slug", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--tier", default="pilot", choices=["pilot", "growth", "enterprise"])
    p = sub.add_parser("issue-key")
    p.add_argument("--slug", required=True)
    p.add_argument("--name", default="default")
    args = ap.parse_args(argv)

    if not re.fullmatch(SLUG_PATTERN, args.slug):
        print(f"invalid slug {args.slug!r}: lowercase letters, digits and hyphens, 3-48 chars", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        if args.cmd in ("create-org", "ensure-org"):
            if slug_taken(db, args.slug):
                if args.cmd == "ensure-org":
                    print(f"org {args.slug} exists")
                    return 0
                print(f"org {args.slug} already exists", file=sys.stderr)
                return 1
            org, key = create_org(db, args.slug, args.name, args.tier)
            db.commit()
            if args.cmd == "create-org":
                print(f"created org {org.slug} ({org.id})")
                print(f"api key (shown once): {key}")
            else:
                # The key issued with it is discarded: deploy logs are not where secrets go.
                print(f"created org {org.slug}")
            return 0

        org_id = _org_id(db, args.slug)
        if org_id is None:
            print(f"no org {args.slug}", file=sys.stderr)
            return 1
        bind_org_to_transaction(db, org_id)
        raw, _ = issue_key(db, db.get(Organization, org_id), name=args.name)
        db.commit()
        print(f"api key for {args.slug} (shown once): {raw}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
