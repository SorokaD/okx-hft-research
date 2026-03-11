"""
Database connection and data loaders for OKX HFT research.

Usage in notebooks/scripts:
    from research.db import get_engine, load_orderbook_snapshot

    engine = get_engine()  # reads PGHOST, PGPORT, etc. from .env
    df = load_orderbook_snapshot("BTC-USDT-SWAP", ts_start="2024-01-01", ts_end="2024-01-02")
"""

from research.db.connection import build_postgres_url, get_engine
from research.db.loaders import load_orderbook_snapshot, load_trades

__all__ = [
    "build_postgres_url",
    "get_engine",
    "load_orderbook_snapshot",
    "load_trades",
]
