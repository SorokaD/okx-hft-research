"""
Data loaders for orderbook, trades, and related tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from research.db.connection import get_engine
from research.db.queries import orderbook as _q_ob
from research.db.queries import trades as _q_tr


def load_orderbook_snapshot(
    inst_id: str,
    ts_start: Optional[str] = None,
    ts_end: Optional[str] = None,
    engine: Optional[Engine] = None,
    schema: str = "okx_core",
    table: str = "fact_orderbook_l10_snapshot",
) -> pd.DataFrame:
    """
    Load orderbook L10 snapshot rows for given instrument and time range.
    Returns DataFrame with bid_px_01..bid_sz_10, ask_px_01..ask_sz_10, mid_px, spread_px, etc.
    """
    eng = engine or get_engine()
    query = _q_ob.ORDERBOOK_SNAPSHOT.format(schema=schema, table=table)
    params = {"inst_id": inst_id, "ts_start": ts_start, "ts_end": ts_end}
    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df


def load_trades(
    inst_id: str,
    ts_start: Optional[str] = None,
    ts_end: Optional[str] = None,
    engine: Optional[Engine] = None,
    schema: str = "okx_core",
    table: str = "fact_trades_tick",
) -> pd.DataFrame:
    """
    Load trade ticks for given instrument and time range.
    Returns DataFrame with ts_event, price, size, side, etc.
    """
    eng = engine or get_engine()
    query = _q_tr.TRADES.format(schema=schema, table=table)
    params = {"inst_id": inst_id, "ts_start": ts_start, "ts_end": ts_end}
    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df
