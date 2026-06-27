import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.conversation import Conversation  # noqa: E402
from src.models.paper import Paper  # noqa: E402
from src.models.researcher import ResearcherInterest, DailyBriefing  # noqa: E402
from src.db.interfaces.postgresql import Base  # noqa: E402

target_metadata = Base.metadata

# Override sqlalchemy.url from environment if available
db_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_DATABASE_URL")
if not db_url:
    # Try to load from .env file
    from src.config import get_settings
    try:
        settings = get_settings()
        db_url = settings.postgres_database_url
    except Exception:
        db_url = config.get_main_option("sqlalchemy.url")

if db_url:
    config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="rag_alembic_version",
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="rag_alembic_version",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
