"""
PostgreSQL/TimescaleDB connection via SQLAlchemy.
Loads .env from project root. Use: from research.db import get_engine, engine
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env from project root (where pyproject.toml lives)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")


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


_engine: Optional[Engine] = None


def get_engine(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    pool_size: int = 5,
    use_cache: bool = True,
) -> Engine:
    """Get SQLAlchemy engine for Postgres. Uses cached engine when no overrides."""
    global _engine
    if use_cache and _engine is not None and all(x is None for x in (host, port, database, user, password)):
        return _engine
    url = build_postgres_url(host, port, database, user, password)
    try:
        eng = create_engine(url, pool_size=pool_size, pool_pre_ping=True)
        if use_cache and all(x is None for x in (host, port, database, user, password)):
            _engine = eng
        return eng
    except Exception as e:
        raise RuntimeError(f"Failed to create engine: {e}") from e


