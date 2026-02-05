from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.utils.config import get_db_url

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """
    Initialize database connection pool.

    Should be called during application startup.
    """
    global _engine, _async_session_factory

    if _engine is not None:
        logger.warning("Database already initialized")
        return

    db_url = get_db_url()
    logger.info(
        "Initializing database connection",
        extra={"db_url": db_url.split("@")[-1] if "@" in db_url else "***"},
    )

    # Create async engine with connection pooling
    # Using NullPool for pgbouncer (connection pooling handled by pgbouncer)
    _engine = create_async_engine(
        db_url,
        poolclass=NullPool,  # pgbouncer handles pooling
        echo=False,  # Set to True for SQL query logging
        connect_args={
            # Important with PgBouncer transaction pooling + asyncpg
            "statement_cache_size": 0,
        },
    )

    # Create session factory
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    logger.info("Database connection initialized")


async def shutdown_db() -> None:
    """
    Shutdown database connection pool.

    Should be called during application shutdown.
    """
    global _engine, _async_session_factory

    if _engine is None:
        return

    logger.info("Shutting down database connection")
    await _engine.dispose()
    _engine = None
    _async_session_factory = None
    logger.info("Database connection closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes to get database session.

    Yields:
        AsyncSession: Database session.

    Example:
        ```python
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_db_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
        ```
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Type alias for dependency injection
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
