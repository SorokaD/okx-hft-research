"""
Order book reconstruction from snapshot + updates.

Supports two data sources:
- core: okx_core tables (fact_orderbook_l10_snapshot, fact_orderbook_update_level)
- raw: okx_raw tables (orderbook_snapshots, orderbook_updates) with OKX native format
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
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
    snapshot_max_depth: int | None = None,
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

    # Pivot to wide format: bid_px_01..bid_sz_XX, ask_px_01..ask_sz_XX
    # where XX is not hard-limited to 10. Optionally truncate with snapshot_max_depth.
    bids = anchor_rows[anchor_rows["side"] == 1].sort_values("level")
    asks = anchor_rows[anchor_rows["side"] == 2].sort_values("level")
    if snapshot_max_depth is not None and snapshot_max_depth > 0:
        bids = bids.head(snapshot_max_depth)
        asks = asks.head(snapshot_max_depth)
    max_levels = max(len(bids), len(asks))

    record: dict[str, Any] = {
        "inst_id": inst_id,
        "ts_event": pd.Timestamp(anchor_ms, unit="ms", tz="utc"),
    }
    for i in range(1, max_levels + 1):
        if i <= len(bids):
            r = bids.iloc[i - 1]
            record[f"bid_px_{i:02d}"] = float(r["price"])
            record[f"bid_sz_{i:02d}"] = float(r["size"])
        else:
            record[f"bid_px_{i:02d}"] = None
            record[f"bid_sz_{i:02d}"] = None

        if i <= len(asks):
            r = asks.iloc[i - 1]
            record[f"ask_px_{i:02d}"] = float(r["price"])
            record[f"ask_sz_{i:02d}"] = float(r["size"])
        else:
            record[f"ask_px_{i:02d}"] = None
            record[f"ask_sz_{i:02d}"] = None

    return pd.Series(record)


def _pivot_raw_snapshot_rows_to_wide(
    anchor_rows: pd.DataFrame,
    inst_id: str,
    anchor_ms: int,
    snapshot_max_depth: int | None = None,
) -> pd.Series:
    """Convert raw snapshot rows (one ts_event_ms + one snapshot_id) to wide format."""
    bids = anchor_rows[anchor_rows["side"] == 1].sort_values("level")
    asks = anchor_rows[anchor_rows["side"] == 2].sort_values("level")
    if snapshot_max_depth is not None and snapshot_max_depth > 0:
        bids = bids.head(snapshot_max_depth)
        asks = asks.head(snapshot_max_depth)
    max_levels = max(len(bids), len(asks))

    record: dict[str, Any] = {
        "inst_id": inst_id,
        "ts_event": pd.Timestamp(anchor_ms, unit="ms", tz="utc"),
    }

    for i in range(1, max_levels + 1):
        bid_px_key = f"bid_px_{i:02d}"
        bid_sz_key = f"bid_sz_{i:02d}"
        ask_px_key = f"ask_px_{i:02d}"
        ask_sz_key = f"ask_sz_{i:02d}"

        if i <= len(bids):
            r = bids.iloc[i - 1]
            record[bid_px_key] = float(r["price"])
            record[bid_sz_key] = float(r["size"])
        else:
            record[bid_px_key] = None
            record[bid_sz_key] = None

        if i <= len(asks):
            r = asks.iloc[i - 1]
            record[ask_px_key] = float(r["price"])
            record[ask_sz_key] = float(r["size"])
        else:
            record[ask_px_key] = None
            record[ask_sz_key] = None

    return pd.Series(record)


def find_anchor_snapshot_raw_at_or_before(
    engine: Engine,
    schema: str,
    snapshot_table: str,
    inst_id: str,
    chunk_start: pd.Timestamp,
    snapshot_max_depth: int | None = None,
) -> Optional[pd.Series]:
    """
    Raw anchor snapshot inclusive: last snapshot with ts_event_ms <= chunk_start.
    """
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    chunk_start_ms = int(chunk_start.timestamp() * 1000)

    sql = text(
        f"""
        SELECT snapshot_id, instid, ts_event_ms, ts_ingest_ms, side, price, size, level
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms = (
              SELECT MAX(ts_event_ms) FROM {schema}.{snapshot_table}
              WHERE instid = :inst_id AND ts_event_ms <= :chunk_start_ms2
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
                "chunk_start_ms2": chunk_start_ms,
            },
        )

    if df.empty:
        return None

    anchor_ms = int(df["ts_event_ms"].iloc[0])
    anchor_rows = df[df["ts_event_ms"] == anchor_ms]

    # If multiple snapshot_ids at same ts, pick the one with most levels
    if anchor_rows["snapshot_id"].nunique() > 1:
        snap_counts = anchor_rows.groupby("snapshot_id").size()
        snap_id = snap_counts.idxmax()
        anchor_rows = anchor_rows[anchor_rows["snapshot_id"] == snap_id]

    return _pivot_raw_snapshot_rows_to_wide(
        anchor_rows=anchor_rows,
        inst_id=inst_id,
        anchor_ms=anchor_ms,
        snapshot_max_depth=snapshot_max_depth,
    )


def load_snapshot_raw_by_ts(
    engine: Engine,
    schema: str,
    snapshot_table: str,
    inst_id: str,
    ts_event_ms: int,
    snapshot_max_depth: int | None = None,
) -> pd.Series:
    """Load a canonical wide snapshot row at an exact raw ts_event_ms."""
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    sql = text(
        f"""
        SELECT snapshot_id, instid, ts_event_ms, ts_ingest_ms, side, price, size, level
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms = :ts_event_ms
        ORDER BY side, price DESC
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"inst_id": inst_id, "ts_event_ms": ts_event_ms},
        )

    if df.empty:
        raise FileNotFoundError(
            f"No raw snapshot rows for inst_id={inst_id} ts_event_ms={ts_event_ms}"
        )

    anchor_ms = int(df["ts_event_ms"].iloc[0])
    anchor_rows = df[df["ts_event_ms"] == anchor_ms]

    if anchor_rows["snapshot_id"].nunique() > 1:
        snap_counts = anchor_rows.groupby("snapshot_id").size()
        snap_id = snap_counts.idxmax()
        anchor_rows = anchor_rows[anchor_rows["snapshot_id"] == snap_id]

    return _pivot_raw_snapshot_rows_to_wide(
        anchor_rows=anchor_rows,
        inst_id=inst_id,
        anchor_ms=anchor_ms,
        snapshot_max_depth=snapshot_max_depth,
    )


def load_snapshot_timestamps_raw(
    engine: Engine,
    schema: str,
    snapshot_table: str,
    inst_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    start_inclusive: bool,
    end_inclusive: bool,
) -> list[pd.Timestamp]:
    """Return sorted unique raw snapshot timestamps in a range."""
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    start_op = ">=" if start_inclusive else ">"
    end_op = "<=" if end_inclusive else "<"

    sql = text(
        f"""
        SELECT DISTINCT ts_event_ms
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms {start_op} :start_ms
          AND ts_event_ms {end_op} :end_ms
        ORDER BY ts_event_ms
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms},
        )

    if df.empty:
        return []

    return [pd.Timestamp(int(v), unit="ms", tz="utc") for v in df["ts_event_ms"].tolist()]


def _ensure_ts_event_in_row(row: pd.Series, time_column: str) -> pd.Series:
    """Reconstruction code expects anchor row to have key 'ts_event'."""
    if time_column != "ts_event" and time_column in row.index:
        row = row.copy()
        row["ts_event"] = row[time_column]
    return row


def find_anchor_snapshot_core_at_or_before(
    engine: Engine,
    snapshot_schema: str,
    snapshot_table: str,
    inst_id: str,
    time_column: str,
    chunk_start: str | pd.Timestamp,
) -> Optional[pd.Series]:
    """Core anchor snapshot inclusive: last snapshot with time_column <= chunk_start."""
    _validate_identifier(snapshot_schema, "schema")
    _validate_identifier(snapshot_table, "table")
    _validate_identifier(time_column, "time_column")

    sql = text(
        f"""
        SELECT *
        FROM {snapshot_schema}.{snapshot_table}
        WHERE inst_id = :inst_id
          AND {time_column} <= :chunk_start
        ORDER BY {time_column} DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"inst_id": inst_id, "chunk_start": chunk_start})

    if df.empty:
        return None

    return _ensure_ts_event_in_row(df.iloc[0], time_column)


def load_snapshot_timestamps_core(
    engine: Engine,
    snapshot_schema: str,
    snapshot_table: str,
    inst_id: str,
    time_column: str,
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    start_inclusive: bool,
    end_inclusive: bool,
) -> list[pd.Timestamp]:
    """Return sorted unique core snapshot timestamps in a range."""
    _validate_identifier(snapshot_schema, "schema")
    _validate_identifier(snapshot_table, "table")
    _validate_identifier(time_column, "time_column")

    start_op = ">=" if start_inclusive else ">"
    end_op = "<=" if end_inclusive else "<"

    sql = text(
        f"""
        SELECT DISTINCT {time_column} AS ts
        FROM {snapshot_schema}.{snapshot_table}
        WHERE inst_id = :inst_id
          AND {time_column} {start_op} :start_ts
          AND {time_column} {end_op} :end_ts
        ORDER BY {time_column}
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"inst_id": inst_id, "start_ts": start_ts, "end_ts": end_ts},
        )

    if df.empty:
        return []

    # Force UTC timestamps for boundary comparisons
    return [pd.to_datetime(v, utc=True) for v in df["ts"].tolist()]


def load_snapshot_core_by_ts(
    engine: Engine,
    snapshot_schema: str,
    snapshot_table: str,
    inst_id: str,
    time_column: str,
    ts_event: str | pd.Timestamp,
) -> pd.Series:
    """Load core snapshot row at exact timestamp boundary."""
    _validate_identifier(snapshot_schema, "schema")
    _validate_identifier(snapshot_table, "table")
    _validate_identifier(time_column, "time_column")

    sql = text(
        f"""
        SELECT *
        FROM {snapshot_schema}.{snapshot_table}
        WHERE inst_id = :inst_id
          AND {time_column} = :ts_event
        ORDER BY {time_column} DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"inst_id": inst_id, "ts_event": ts_event},
        )

    if df.empty:
        raise FileNotFoundError(
            f"No core snapshot rows for inst_id={inst_id} ts={ts_event} (time_column={time_column})"
        )

    return _ensure_ts_event_in_row(df.iloc[0], time_column)


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


def reconstruct_chunk_snapshot_to_snapshot(
    engine: Engine,
    *,
    source: str,
    # raw identifiers
    schema: str,
    raw_snapshot_table: str = "orderbook_snapshots",
    raw_updates_table: str = "orderbook_updates",
    # core identifiers
    snapshot_schema: str | None = None,
    snapshot_table: str | None = None,
    updates_schema: str | None = None,
    updates_table: str | None = None,
    time_column: str = "ts_event",
    price_column: str = "price",
    size_column: str = "size",
    # common
    inst_id: str,
    chunk_start: str | pd.Timestamp,
    chunk_end: str | pd.Timestamp,
    mode: str,
    grid_ms: int = 50,
    depth: int = 10,
    snapshot_max_depth: int | None = None,
    # diagnostics
    segment_diagnostics: bool = False,
    validate_segments: bool = False,
) -> pd.DataFrame:
    """
    Snapshot-to-snapshot reconstruction to minimize drift.

    Boundary rules (canonical semantics):
    - segment_start snapshot row is the truth state for the segment beginning
    - updates are applied from segment_start up to (but not including) segment_end
      for intermediate segments; for the final segment up to chunk_end
    - outputs are filtered so intermediate segments never emit ts_event == segment_end
    """
    if source not in {"raw", "core"}:
        raise ValueError(f"Unsupported source: {source!r}")
    if mode not in {"event", "grid"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if source == "raw":
        if not schema:
            raise ValueError("--schema is required for source=raw")

        anchor_snapshot_row = find_anchor_snapshot_raw_at_or_before(
            engine=engine,
            schema=schema,
            snapshot_table=raw_snapshot_table,
            inst_id=inst_id,
            chunk_start=pd.to_datetime(chunk_start, utc=True),
            snapshot_max_depth=snapshot_max_depth,
        )
        if anchor_snapshot_row is None:
            log.warning(
                "No anchor raw snapshot found for inst_id=%s at/before %s",
                inst_id,
                chunk_start,
            )
            return pd.DataFrame()

        anchor_ts = anchor_snapshot_row["ts_event"]
        # Next snapshot timestamps in (chunk_start, chunk_end]
        snapshot_ts_list = load_snapshot_timestamps_raw(
            engine=engine,
            schema=schema,
            snapshot_table=raw_snapshot_table,
            inst_id=inst_id,
            start_ts=pd.to_datetime(chunk_start, utc=True),
            end_ts=pd.to_datetime(chunk_end, utc=True),
            start_inclusive=False,
            end_inclusive=True,
        )
    else:
        if not (snapshot_schema and snapshot_table and updates_schema and updates_table):
            raise ValueError(
                "Core source requires snapshot_schema, snapshot_table, updates_schema, updates_table"
            )

        anchor_snapshot_row = find_anchor_snapshot_core_at_or_before(
            engine=engine,
            snapshot_schema=snapshot_schema,
            snapshot_table=snapshot_table,
            inst_id=inst_id,
            time_column=time_column,
            chunk_start=chunk_start,
        )
        if anchor_snapshot_row is None:
            log.warning(
                "No anchor core snapshot found for inst_id=%s at/before %s",
                inst_id,
                chunk_start,
            )
            return pd.DataFrame()

        anchor_ts = anchor_snapshot_row["ts_event"]
        snapshot_ts_list = load_snapshot_timestamps_core(
            engine=engine,
            snapshot_schema=snapshot_schema,
            snapshot_table=snapshot_table,
            inst_id=inst_id,
            time_column=time_column,
            start_ts=chunk_start,
            end_ts=chunk_end,
            start_inclusive=False,
            end_inclusive=True,
        )

    segments_starts: list[pd.Timestamp] = [anchor_ts] + snapshot_ts_list
    segment_count = len(segments_starts)

    out_parts: list[pd.DataFrame] = []

    chunk_start_ts = pd.to_datetime(chunk_start, utc=True)
    chunk_end_ts = pd.to_datetime(chunk_end, utc=True)

    # Used to keep grid timestamps globally aligned to original chunk_start.
    global_start_ms = int(chunk_start_ts.timestamp() * 1000)

    def _grid_aligned_start(seg_output_start: pd.Timestamp) -> pd.Timestamp:
        seg_ms = int(seg_output_start.timestamp() * 1000)
        if seg_ms <= global_start_ms:
            return chunk_start_ts
        steps = math.ceil((seg_ms - global_start_ms) / grid_ms)
        aligned_ms = global_start_ms + steps * grid_ms
        return pd.Timestamp(aligned_ms, unit="ms", tz="utc")

    for seg_idx, seg_start in enumerate(segments_starts):
        seg_end = snapshot_ts_list[seg_idx] if seg_idx < len(snapshot_ts_list) else chunk_end_ts

        out_start = max(chunk_start_ts, seg_start)
        out_end = min(chunk_end_ts, seg_end)

        if out_start > out_end:
            continue

        seg_t0 = time.perf_counter()
        if seg_idx == 0:
            seg_anchor_row = anchor_snapshot_row
        else:
            if source == "raw":
                seg_anchor_row = load_snapshot_raw_by_ts(
                    engine=engine,
                    schema=schema,
                    snapshot_table=raw_snapshot_table,
                    inst_id=inst_id,
                    ts_event_ms=int(seg_start.timestamp() * 1000),
                    snapshot_max_depth=snapshot_max_depth,
                )
            else:
                seg_anchor_row = load_snapshot_core_by_ts(
                    engine=engine,
                    snapshot_schema=snapshot_schema,
                    snapshot_table=snapshot_table,
                    inst_id=inst_id,
                    time_column=time_column,
                    ts_event=seg_start,
                )

        # Load updates in the segment interval.
        if source == "raw":
            updates_df = load_updates_raw(
                engine=engine,
                schema=schema,
                updates_table=raw_updates_table,
                inst_id=inst_id,
                start_ts=seg_start,
                end_ts=seg_end,
            )
        else:
            updates_df = load_updates(
                engine=engine,
                updates_schema=updates_schema,
                updates_table=updates_table,
                inst_id=inst_id,
                time_column=time_column,
                start_ts=seg_start,
                end_ts=seg_end,
                price_column=price_column,
                size_column=size_column,
            )

        # For intermediate segments, implement [seg_start, seg_end) by trimming boundary updates.
        is_last_segment = seg_idx == (segment_count - 1)
        if not is_last_segment and not updates_df.empty:
            updates_df = updates_df[updates_df["ts_event"] < seg_end]

        if segment_diagnostics:
            seg_updates_count = 0 if updates_df.empty else len(updates_df)
            log.info(
                "segment=%d start=%s end=%s anchor_ts=%s updates=%d mode=%s",
                seg_idx,
                seg_start,
                seg_end,
                seg_anchor_row.get("ts_event"),
                seg_updates_count,
                mode,
            )

        # Reconstruct inside the segment but emit only timestamps in the requested chunk window.
        if mode == "event":
            if updates_df.empty:
                # Edge case: no updates -> snapshot state should be emitted at out_start.
                book = OrderBookState.from_snapshot_row(seg_anchor_row)
                inst = str(seg_anchor_row.get("inst_id", "")) or inst_id
                rec = book.to_record(
                    ts_event=out_start.to_pydatetime(),
                    inst_id=inst,
                    anchor_snapshot_ts=seg_anchor_row.get("ts_event"),
                    reconstruction_mode="event",
                    depth=depth,
                )
                seg_df = pd.DataFrame([rec])
            else:
                seg_df = reconstruct_event_states(
                    seg_anchor_row,
                    updates_df,
                    chunk_start=out_start,
                    chunk_end=out_end,
                    depth=depth,
                )

            if not seg_df.empty:
                if not is_last_segment:
                    seg_df = seg_df[seg_df["ts_event"] < seg_end]
                else:
                    seg_df = seg_df[seg_df["ts_event"] <= chunk_end_ts]
                # Ensure requested chunk window only
                seg_df = seg_df[
                    (seg_df["ts_event"] >= chunk_start_ts)
                    & (seg_df["ts_event"] <= chunk_end_ts)
                ]

        else:
            # grid mode
            grid_start = _grid_aligned_start(out_start)
            seg_df = reconstruct_grid_states(
                seg_anchor_row,
                updates_df if not updates_df.empty else pd.DataFrame(columns=["ts_event", "side", "price", "size"]),
                chunk_start=grid_start,
                chunk_end=out_end,
                grid_ms=grid_ms,
                depth=depth,
            )
            if not seg_df.empty:
                if not is_last_segment:
                    seg_df = seg_df[seg_df["ts_event"] < seg_end]
                else:
                    seg_df = seg_df[seg_df["ts_event"] <= chunk_end_ts]
                seg_df = seg_df[
                    (seg_df["ts_event"] >= chunk_start_ts)
                    & (seg_df["ts_event"] <= chunk_end_ts)
                ]

        elapsed = time.perf_counter() - seg_t0
        if segment_diagnostics and not seg_df.empty:
            log.info(
                "segment=%d reconstructed_rows=%d elapsed_sec=%.1f",
                seg_idx,
                len(seg_df),
                elapsed,
            )

        if validate_segments and not is_last_segment and not seg_df.empty:
            # Compare the last reconstructed state vs next snapshot top levels.
            rec_row = seg_df.iloc[-1]
            if source == "raw":
                next_snap_row = load_snapshot_raw_by_ts(
                    engine=engine,
                    schema=schema,
                    snapshot_table=raw_snapshot_table,
                    inst_id=inst_id,
                    ts_event_ms=int(seg_end.timestamp() * 1000),
                    snapshot_max_depth=snapshot_max_depth,
                )
            else:
                next_snap_row = load_snapshot_core_by_ts(
                    engine=engine,
                    snapshot_schema=snapshot_schema,
                    snapshot_table=snapshot_table,
                    inst_id=inst_id,
                    time_column=time_column,
                    ts_event=seg_end,
                )

            tol = 1e-9
            top1_bid_match = (
                pd.isna(rec_row["bid_px_01"])
                or pd.isna(next_snap_row["bid_px_01"])
                or abs(float(rec_row["bid_px_01"]) - float(next_snap_row["bid_px_01"])) <= tol
            )
            top1_ask_match = (
                pd.isna(rec_row["ask_px_01"])
                or pd.isna(next_snap_row["ask_px_01"])
                or abs(float(rec_row["ask_px_01"]) - float(next_snap_row["ask_px_01"])) <= tol
            )

            def _max_abs_diff(cols: list[str]) -> float | None:
                diffs: list[float] = []
                for c in cols:
                    av = rec_row.get(c)
                    bv = next_snap_row.get(c)
                    if pd.isna(av) or pd.isna(bv):
                        continue
                    diffs.append(abs(float(av) - float(bv)))
                return max(diffs) if diffs else None

            px_cols = [f"bid_px_{i:02d}" for i in range(1, depth + 1)] + [
                f"ask_px_{i:02d}" for i in range(1, depth + 1)
            ]
            sz_cols = [f"bid_sz_{i:02d}" for i in range(1, depth + 1)] + [
                f"ask_sz_{i:02d}" for i in range(1, depth + 1)
            ]

            max_px_diff = _max_abs_diff(px_cols)
            max_sz_diff = _max_abs_diff(sz_cols)

            log.info(
                "validate segment=%d last_ts=%s next_snap_ts=%s top1_bid_match=%s top1_ask_match=%s max_px_diff=%s max_sz_diff=%s",
                seg_idx,
                rec_row.get("ts_event"),
                seg_end,
                top1_bid_match,
                top1_ask_match,
                max_px_diff,
                max_sz_diff,
            )

        if not seg_df.empty:
            out_parts.append(seg_df)

    if not out_parts:
        return pd.DataFrame()

    out = pd.concat(out_parts, ignore_index=True)
    # Sort + de-dup on canonical key.
    out.sort_values(["inst_id", "ts_event"], inplace=True)
    out.drop_duplicates(subset=["inst_id", "ts_event"], inplace=True)
    out.sort_values("ts_event", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
