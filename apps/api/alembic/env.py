"""Alembic environment — wired to the app's models metadata and env-driven DB URL.

Generate the initial migration once your Supabase DB URL is set:

    APERTURE_DATABASE_URL=postgresql+psycopg2://... alembic revision --autogenerate -m "init"
    APERTURE_DATABASE_URL=postgresql+psycopg2://... alembic upgrade head

(The production schema, including pgvector + RLS, is also captured directly in
 infra/supabase/migrations/0001_init.sql.)
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from aperture.core.config import get_settings
from aperture.core.db import Base
from aperture.core import models  # noqa: F401  (register tables on Base.metadata)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
