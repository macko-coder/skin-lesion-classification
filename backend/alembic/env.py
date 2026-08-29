"""Alembic migration environment: wires app.core.config.Settings.DATABASE_URL and app.db.base.Base.metadata as target_metadata."""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Repo root on sys.path so `backend.app...` imports resolve regardless of
# the working directory `alembic` is invoked from (matches the ml.src.xxx /
# backend.app.xxx import convention used elsewhere in the repo).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.db.base import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Base.metadata (Case included, via db/base.py's import) is what
# --autogenerate diffs the live DB schema against.
target_metadata = Base.metadata

# Overrides alembic.ini's blank sqlalchemy.url with the real one from
# Settings, so backend/.env's DATABASE_URL stays the single source of truth.
config.set_main_option("sqlalchemy.url", get_settings().database_url)


def run_migrations_offline() -> None:
    """Emit migrations as SQL statements without connecting to a live DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection — the normal, everyday case."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
