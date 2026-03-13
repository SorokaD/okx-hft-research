"""
Order book reconstruction from snapshot + updates.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
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
