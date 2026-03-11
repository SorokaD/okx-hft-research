"""
PostgreSQL/TimescaleDB connection via SQLAlchemy.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url


def build_postgres_url(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Build PostgreSQL connection URL from env or explicit args."""
    host = host or os.getenv("PGHOST", "localhost")
    port = port or int(os.getenv("PGPORT", "5432"))
    database = database or os.getenv("PGDATABASE", "okx_hft")
    user = user or os.getenv("PGUSER", "okx")
    password = password or os.getenv("PGPASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    pool_size: int = 5,
) -> Engine:
    """Create SQLAlchemy engine for Postgres."""
    url = build_postgres_url(host, port, database, user, password)
    try:
        engine = create_engine(
            url,
            pool_size=pool_size,
            pool_pre_ping=True,
        )
        return engine
    except Exception as e:
        raise RuntimeError(f"Failed to create engine: {e}") from e
