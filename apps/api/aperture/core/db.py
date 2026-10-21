"""Database engine, session, and declarative base."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from aperture.core.config import get_settings

_settings = get_settings()

_is_sqlite = _settings.database_url.startswith("sqlite")
# SQLite needs check_same_thread=False for FastAPI's threaded request handling, and a busy
# timeout so a request that needs the write lock waits for a long ingest to commit instead of
# failing at once with "database is locked".
_connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}

engine = create_engine(_settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record) -> None:
        # WAL lets readers proceed while the worker holds a write transaction.
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()


@event.listens_for(Session, "after_begin")
def _rebind_org(session: Session, _transaction, connection) -> None:
    """Re-apply the tenant to every transaction a session opens, not just the first.

    Row-level security reads `aperture.org_id`, set with `set_config(..., is_local => true)` so a
    pooled connection can never leak one tenant's identity to the next borrower. The price is
    that it ends with the transaction: after `db.commit()` the next statement would run as no
    tenant and see nothing. `auth.bind_org_to_transaction` records the org in `session.info`;
    this puts it back each time a new transaction starts.
    """
    org_id = session.info.get("org_id")
    if org_id and connection.dialect.name == "postgresql":
        connection.execute(text("select set_config('aperture.org_id', :o, true)"), {"o": org_id})


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Alembic owns migrations in production; this is the local path."""
    # Import models so they register on the metadata before create_all.
    from aperture.core import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
