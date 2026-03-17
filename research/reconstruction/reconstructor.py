"""
Order book reconstruction from snapshot + updates.

Supports two data sources:
- core: okx_core tables (fact_orderbook_l10_snapshot, fact_orderbook_update_level)
- raw: okx_raw tables (orderbook_snapshots, orderbook_updates) with OKX native format
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import pandas as pd

log = logging.getLogger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine

from research.reconstruction.book_state import OrderBookState
from research.reconstruction.sampler import grid_timestamps

# Valid SQL identifier pattern (schema, table, column)
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, kind: str = "identifier") -> None:
    """Raise ValueError if name is not a safe SQL identifier."""
    if not name or not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid {kind}: {name!r}. "
            "Use alphanumeric identifiers (letters, digits, underscore)."
        )


def find_anchor_snapshot(
    engine: Engine,
    snapshot_schema: str,
    snapshot_table: str,
    inst_id: str,
    time_column: str,
    chunk_start: str | pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Return the last snapshot row before chunk_start for the given instrument.

    Returns None if no such snapshot exists.
    """
    _validate_identifier(snapshot_schema, "schema")
    _validate_identifier(snapshot_table, "table")
    _validate_identifier(time_column, "time_column")

    sql = text(
        f"""
        SELECT *
        FROM {snapshot_schema}.{snapshot_table}
        WHERE inst_id = :inst_id
          AND {time_column} < :chunk_start
        ORDER BY {time_column} DESC
        LIMIT 1
        """
    )
    params = {"inst_id": inst_id, "chunk_start": chunk_start}
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return None
    return df.iloc[0]


def find_anchor_snapshot_raw(
    engine: Engine,
    schema: str,
    snapshot_table: str,
    inst_id: str,
    chunk_start: pd.Timestamp,
) -> Optional[pd.Series]:
    """
    Load anchor snapshot from raw orderbook_snapshots table.

    Raw format: one row per level (snapshot_id, instid, ts_event_ms, side, price, size, level).
    side=1 bid, side=2 ask. Returns a wide-format row compatible with OrderBookState.from_snapshot_row.
    """
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    chunk_start_ms = int(chunk_start.timestamp() * 1000)

    sql = text(
        f"""
        SELECT snapshot_id, instid, ts_event_ms, ts_ingest_ms, side, price, size, level
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms < :chunk_start_ms
          AND ts_event_ms = (
              SELECT MAX(ts_event_ms) FROM {schema}.{snapshot_table}
              WHERE instid = :inst_id AND ts_event_ms < :chunk_start_ms2
          )
        ORDER BY side, price DESC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "inst_id": inst_id,
                "chunk_start_ms": chunk_start_ms,
                "chunk_start_ms2": chunk_start_ms,
            },
        )

    if df.empty:
        return None

    # Take the latest snapshot (rows with max ts_event_ms)
    anchor_ms = int(df["ts_event_ms"].iloc[0])
    anchor_rows = df[df["ts_event_ms"] == anchor_ms]
    # If multiple snapshot_ids at same ts, pick the one with most levels
    if anchor_rows["snapshot_id"].nunique() > 1:
        snap_counts = anchor_rows.groupby("snapshot_id").size()
        snap_id = snap_counts.idxmax()
        anchor_rows = anchor_rows[anchor_rows["snapshot_id"] == snap_id]

    # Pivot to wide format: bid_px_01..bid_sz_10, ask_px_01..ask_sz_10
    bids = anchor_rows[anchor_rows["side"] == 1].sort_values("price", ascending=False).head(10)
    asks = anchor_rows[anchor_rows["side"] == 2].sort_values("price", ascending=True).head(10)

    record: dict[str, Any] = {
        "inst_id": inst_id,
        "ts_event": pd.Timestamp(anchor_ms, unit="ms", tz="utc"),
    }
    for i, (_, r) in enumerate(bids.iterrows(), start=1):
        record[f"bid_px_{i:02d}"] = float(r["price"])
        record[f"bid_sz_{i:02d}"] = float(r["size"])
    for i in range(len(bids) + 1, 11):
        record[f"bid_px_{i:02d}"] = None
        record[f"bid_sz_{i:02d}"] = None
    for i, (_, r) in enumerate(asks.iterrows(), start=1):
        record[f"ask_px_{i:02d}"] = float(r["price"])
        record[f"ask_sz_{i:02d}"] = float(r["size"])
    for i in range(len(asks) + 1, 11):
        record[f"ask_px_{i:02d}"] = None
        record[f"ask_sz_{i:02d}"] = None

    return pd.Series(record)


def load_updates_raw(
    engine: Engine,
    schema: str,
    updates_table: str,
    inst_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load orderbook updates from raw orderbook_updates table.

    Raw format: bids_delta and asks_delta are JSONB arrays [[price, size], ...] (OKX format).
    size=0 means remove level. Returns DataFrame with ts_event, side, price, size.
    """
    _validate_identifier(schema, "schema")
    _validate_identifier(updates_table, "table")

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    params = {"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms}
    with engine.connect() as conn:
        count_sql = text(
            f"""
            SELECT COUNT(*) AS cnt,
                   MIN(ts_event_ms) AS min_ms,
                   MAX(ts_event_ms) AS max_ms
            FROM {schema}.{updates_table}
            WHERE instid = :inst_id
              AND ts_event_ms >= :start_ms AND ts_event_ms <= :end_ms
            """
        )
        stats = pd.read_sql(count_sql, conn, params=params).iloc[0]
        cnt = int(stats["cnt"])
        if cnt == 0:
            log.warning(
                "load_updates_raw: 0 rows for inst_id=%s start_ms=%s end_ms=%s (check timezone/range)",
                inst_id,
                start_ms,
                end_ms,
            )
            return pd.DataFrame()

        log.info(
            "load_updates_raw: %d rows for inst_id=%s [%s .. %s]",
            cnt,
            inst_id,
            int(stats["min_ms"]) if not pd.isna(stats["min_ms"]) else None,
            int(stats["max_ms"]) if not pd.isna(stats["max_ms"]) else None,
        )

        sql = text(
            f"""
            SELECT instid, ts_event_ms, bids_delta, asks_delta
            FROM {schema}.{updates_table}
            WHERE instid = :inst_id
              AND ts_event_ms >= :start_ms
              AND ts_event_ms <= :end_ms
            ORDER BY ts_event_ms
            """
        )
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        log.warning(
            "load_updates_raw: df.empty after SELECT for inst_id=%s start_ms=%s end_ms=%s (cnt_from_stats=%d)",
            inst_id,
            start_ms,
            end_ms,
            cnt,
        )
        return pd.DataFrame()

    def _parse_level(level: Any, side: str) -> Optional[tuple[float, float]]:
        """Extract (price, size) from one level. Supports:
        - list/tuple: [price, size] (OKX raw)
        - dict: {"price": "x", "size": "y"} (collector format)
        """
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            try:
                return (float(level[0]), float(level[1]))
            except (ValueError, TypeError):
                return None
        if isinstance(level, dict):
            p, s = level.get("price"), level.get("size")
            if p is None or s is None:
                return None
            try:
                return (float(p), float(s))
            except (ValueError, TypeError):
                return None
        return None

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        ts_event = pd.Timestamp(int(r["ts_event_ms"]), unit="ms", tz="utc")
        for delta_col, side in [("bids_delta", "bid"), ("asks_delta", "ask")]:
            delta = r.get(delta_col)
            if delta is None:
                continue
            if isinstance(delta, str):
                delta = json.loads(delta) if delta and delta.strip() else []
            if not isinstance(delta, list):
                continue
            for level in delta:
                parsed = _parse_level(level, side)
                if parsed is None:
                    continue
                price, size = parsed
                rows.append({"ts_event": ts_event, "side": side, "price": price, "size": size})

    if not rows:
        log.warning(
            "load_updates_raw: %d db rows but 0 parsed deltas for inst_id=%s [%s .. %s]",
            len(df),
            inst_id,
            start_ms,
            end_ms,
        )
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values("ts_event", inplace=True)
    out.reset_index(drop=True, inplace=True)
    log.info("load_updates_raw: normalized %d levels (ts_event %s .. %s)", len(out), out["ts_event"].min(), out["ts_event"].max())
    return out


def load_updates(
    engine: Engine,
    updates_schema: str,
    updates_table: str,
    inst_id: str,
    time_column: str,
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    price_column: str = "price",
    size_column: str = "size",
) -> pd.DataFrame:
    """
    Load order book updates in the given time range, sorted by ts_event.

    Expects table columns: inst_id, ts_event (or time_column), side, price, size.
    If your table uses price_px/size_qty (e.g. fact_orderbook_update_level),
    pass price_column="price_px", size_column="size_qty".
    The returned DataFrame will have 'price' and 'size' columns for the reconstructor.
    """
    _validate_identifier(updates_schema, "schema")
    _validate_identifier(updates_table, "table")
    _validate_identifier(time_column, "time_column")
    _validate_identifier(price_column, "price_column")
    _validate_identifier(size_column, "size_column")

    price_select = f"{price_column} AS price" if price_column != "price" else "price"
    size_select = f"{size_column} AS size" if size_column != "size" else "size"

    sql = text(
        f"""
        SELECT inst_id, {time_column} AS ts_event, side, {price_select}, {size_select}
        FROM {updates_schema}.{updates_table}
        WHERE inst_id = :inst_id
          AND {time_column} >= :start_ts
          AND {time_column} <= :end_ts
        ORDER BY {time_column}
        """
    )
    params = {"inst_id": inst_id, "start_ts": start_ts, "end_ts": end_ts}
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df


def reconstruct_event_states(
    anchor_snapshot_row: pd.Series,
    updates_df: pd.DataFrame,
    chunk_start: str | pd.Timestamp,
    chunk_end: str | pd.Timestamp,
    depth: int = 10,
    inst_id_col: str = "inst_id",
    ts_col: str = "ts_event",
    side_col: str = "side",
    price_col: str = "price",
    size_col: str = "size",
) -> pd.DataFrame:
    """
    Reconstruct order book states at each update event.

    Initializes from anchor snapshot, applies updates in order, emits a record
    after each update (grouped by ts_event). Only includes states with
    ts_event >= chunk_start.
    """
    book = OrderBookState.from_snapshot_row(anchor_snapshot_row)
    inst_id = str(anchor_snapshot_row.get("inst_id", ""))
    anchor_ts = anchor_snapshot_row.get("ts_event")

    records: list[dict] = []
    last_ts = None

    for _, row in updates_df.iterrows():
        ts = row[ts_col]
        side = str(row[side_col]).strip().lower()
        price = float(row[price_col])
        size = float(row[size_col])

        if last_ts is not None and ts != last_ts:
            # New event boundary: emit state for previous event if in window
            if last_ts >= chunk_start:
                rec = book.to_record(
                    ts_event=last_ts,
                    inst_id=inst_id,
                    anchor_snapshot_ts=anchor_ts,
                    reconstruction_mode="event",
                    depth=depth,
                )
                records.append(rec)

        book.apply_update(side, price, size)
        last_ts = ts

    if last_ts is not None and last_ts >= chunk_start:
        rec = book.to_record(
            ts_event=last_ts,
            inst_id=inst_id,
            anchor_snapshot_ts=anchor_ts,
            reconstruction_mode="event",
            depth=depth,
        )
        records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()


def reconstruct_grid_states(
    anchor_snapshot_row: pd.Series,
    updates_df: pd.DataFrame,
    chunk_start: str | pd.Timestamp,
    chunk_end: str | pd.Timestamp,
    grid_ms: int = 50,
    depth: int = 10,
    inst_id_col: str = "inst_id",
    ts_col: str = "ts_event",
    side_col: str = "side",
    price_col: str = "price",
    size_col: str = "size",
) -> pd.DataFrame:
    """
    Reconstruct order book states on a regular time grid.

    Grid runs from chunk_start to chunk_end with step grid_ms.
    At each grid point, uses the last known state (after all updates with
    ts_event <= grid_time).
    """
    book = OrderBookState.from_snapshot_row(anchor_snapshot_row)
    inst_id = str(anchor_snapshot_row.get("inst_id", ""))
    anchor_ts = anchor_snapshot_row.get("ts_event")

    # Ensure datetime for grid
    if hasattr(chunk_start, "to_pydatetime"):
        start_dt = chunk_start.to_pydatetime()
    else:
        start_dt = pd.to_datetime(chunk_start).to_pydatetime()
    if hasattr(chunk_end, "to_pydatetime"):
        end_dt = chunk_end.to_pydatetime()
    else:
        end_dt = pd.to_datetime(chunk_end).to_pydatetime()

    records: list[dict] = []
    update_idx = 0
    updates_list = updates_df.to_dict("records")

    for grid_ts in grid_timestamps(start_dt, end_dt, grid_ms):
        # Apply all updates with ts_event <= grid_ts
        while update_idx < len(updates_list):
            row = updates_list[update_idx]
            ts = row[ts_col]
            if pd.to_datetime(ts).to_pydatetime() > grid_ts:
                break
            side = str(row[side_col]).strip().lower()
            price = float(row[price_col])
            size = float(row[size_col])
            book.apply_update(side, price, size)
            update_idx += 1

        rec = book.to_record(
            ts_event=grid_ts,
            inst_id=inst_id,
            anchor_snapshot_ts=anchor_ts,
            reconstruction_mode=f"grid_{grid_ms}ms",
            depth=depth,
        )
        records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()
