"""Apply the database security layer: `infra/supabase/migrations/*.sql`, in order.

Run after `alembic upgrade head`, as the database **owner** (the restricted `aperture_api` role
cannot create policies — that is the point of it):

    APERTURE_DATABASE_URL=<owner url> alembic upgrade head
    APERTURE_DATABASE_URL=<owner url> python -m aperture.dbsetup ../../infra/supabase/migrations

Every file is idempotent, so this is safe on every deploy.

Setting the `aperture_api` password is a separate, **one-time** step:

    APERTURE_DB_API_PASSWORD=... python -m aperture.dbsetup --set-api-password <dir>

It is opt-in because re-setting a password — even to the same value — makes a connection pooler
(Supabase's Supavisor) reject that role until it refreshes its credential cache, which took the
live API's database login down for minutes when it ran on every deploy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine


def apply(database_url: str, sql_dir: Path, api_role_password: str | None = None) -> list[str]:
    files = sorted(sql_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no .sql files in {sql_dir}")
    engine = create_engine(database_url)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        for f in files:
            cur.execute(f.read_text())
        if api_role_password:
            from psycopg2 import sql

            cur.execute(
                sql.SQL("alter role aperture_api with login password {}").format(sql.Literal(api_role_password))
            )
        raw.commit()
    finally:
        raw.close()
        engine.dispose()
    return [f.name for f in files]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    set_password = "--set-api-password" in argv
    argv = [a for a in argv if a != "--set-api-password"]
    default = Path(__file__).resolve().parents[3] / "infra" / "supabase" / "migrations"
    sql_dir = Path(argv[0]) if argv else default
    url = os.environ.get("APERTURE_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        print("APERTURE_DATABASE_URL must be a Postgres owner URL", file=sys.stderr)
        return 2
    password = os.environ.get("APERTURE_DB_API_PASSWORD") or None
    if set_password and not password:
        print("--set-api-password needs APERTURE_DB_API_PASSWORD", file=sys.stderr)
        return 2
    applied = apply(url, sql_dir, password if set_password else None)
    print("applied:", ", ".join(applied) + ("; aperture_api password set" if set_password else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
