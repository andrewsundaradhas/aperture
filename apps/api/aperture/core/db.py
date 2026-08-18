"""Database engine, session, and declarative base."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from aperture.core.config import get_settings

_settings = get_settings()

# SQLite needs check_same_thread=False for FastAPI's threaded request handling.
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}

engine = create_engine(_settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


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
