#!/usr/bin/env python3
"""
Проверка подключения к БД и наличия raw orderbook данных.
Запуск: python scripts/check_raw_data.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

import pandas as pd
from sqlalchemy import text

from research.db.connection import get_engine


def main():
    engine = get_engine()
    schema = "okx_raw"
    inst_id = "BTC-USDT-SWAP"
    t0 = pd.Timestamp("2026-03-03 00:00:00", tz="UTC")
    t1 = pd.Timestamp("2026-03-03 01:00:00", tz="UTC")
    chunk_start_ms = int(t0.timestamp() * 1000)
    chunk_end_ms = int(t1.timestamp() * 1000)

    print("=== Check raw orderbook data ===\n")

    with engine.connect() as conn:
        # Snapshots before chunk
        q1 = text(
            f"""
            SELECT MAX(ts_event_ms) as max_ms, COUNT(*) as cnt
            FROM {schema}.orderbook_snapshots
            WHERE instid = :inst_id AND ts_event_ms < :chunk_start_ms
            """
        )
        r1 = pd.read_sql(q1, conn, params={"inst_id": inst_id, "chunk_start_ms": chunk_start_ms})
        print("Snapshots before 2026-03-03 00:00 UTC:")
        print(f"  max(ts_event_ms) = {r1['max_ms'].iloc[0]}")
        print(f"  count = {r1['cnt'].iloc[0]}")

        # Updates in first hour
        q2 = text(
            f"""
            SELECT COUNT(*) as cnt
            FROM {schema}.orderbook_updates
            WHERE instid = :inst_id
              AND ts_event_ms >= :start_ms AND ts_event_ms <= :end_ms
            """
        )
        r2 = pd.read_sql(
            q2,
            conn,
            params={"inst_id": inst_id, "start_ms": chunk_start_ms, "end_ms": chunk_end_ms},
        )
        print("\nUpdates in 2026-03-03 00:00-01:00 UTC:")
        print(f"  count = {r2['cnt'].iloc[0]}")

    print("\nOK - data exists. Run reconstruct with --check to verify loader.")


if __name__ == "__main__":
    main()
