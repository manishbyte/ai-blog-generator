
"""Initialize BlogForge database tables without Alembic."""

from database.base import Base
from database.connection import engine
import database.models  # noqa: F401 — register all models


async def init_db() -> None:
    """Create tables that do not already exist."""

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the async database engine."""

    await engine.dispose()
